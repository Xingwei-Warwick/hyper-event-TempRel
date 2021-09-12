import matplotlib
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import os
import math

from TemporalDataSet import *
from event_poincare_emb import EventTempRel_poincare
from pairwise_ffnn_pytorch import VerbNet

from geoopt.optim import RiemannianAdam
from LM_Cache import LM_context_tokenizer

from argparse import ArgumentParser

seed_everything(13234)

class experiment:
    def __init__(self, device, embedding_dim, trainset, testset, output_labels,
                modelPath, args=None):
        self.model = EventTempRel_poincare(device=device, num_neg=args.neg,
                                                dim_out=args.nn_hid_dim, dim_in=embedding_dim, alpha=args.alpha)
        self.model.to(self.model.device)
        self.max_epoch = args.max_epoch
        self.testsetname = args.testsetname
        self.output_labels = output_labels
        self.exp_name = args.expname
        self.modelPath = "%s_%s" %(modelPath, self.exp_name)
        self.gen_output = args.gen_output

        self.dataset = trainset
        self.trainset, self.devset = trainset.temprel_ee, trainset.dev
        if args.debug:
            self.trainset = self.trainset[:10]
        self.dataset.build_matrix()

        self.context = args.context
        self.batch_size = args.batch
        self.lr = args.lr
        self.lm_lr = args.lm_lr
        self.gamma = args.gamma
        self.weight_decay = args.weight_decay
        self.step_size = args.step_size
        self.sd = args.sd
        self.testset = testset
        self.threshsss = np.arange(1e-3,1.0, 0.05)
        self.max_neg = self.model.num_neg
        if args.precision == 'double':
            self.float = False
        elif args.precision == 'float':
            self.float = True

        if self.testsetname == "matres":
            w2v_ser_dir = "ser/"
        else:
            w2v_ser_dir = "ser/TCR/"

        if self.context in ['roberta-base', 'roberta-large', 'bert-base-uncased']:
            self.lm = LM_context_tokenizer(model_name=self.context, device=self.model.device)
            if not args.loadcache:
                self.lm.encode([self.trainset, self.devset, self.testset])
                self.lm.save_as_cache()
            else:
                self.lm.load_cache()
        else:
            print('Unrecognized context encoder!')
            exit(0)

    def train(self):
        self.best_epoch = self.max_epoch-1
        print("------------Training and Development------------")
        # all_test_accuracies refer to performance on devset
        all_train_losses, all_test_accuracies, sta_dic_list, best_thresh_list  =\
            self.trainHelper(self.trainset,self.devset,self.max_epoch)
        
        # smooth within a window of +-2
        all_test_accuracies_smooth = [all_test_accuracies[0]]*2+all_test_accuracies+[all_test_accuracies[-1]]*2
        all_test_accuracies_smooth = [1.0/5*(all_test_accuracies_smooth[i-2]+all_test_accuracies_smooth[i-1]+all_test_accuracies_smooth[i]+all_test_accuracies_smooth[i+1]+all_test_accuracies_smooth[i+2]) for i in range(2,2+len(all_test_accuracies))]
        
        self.best_epoch, best_dev_acc = 0, 0
        print("Select epoch based on smoothed dev accuracy curve")
        for i,acc in enumerate(all_test_accuracies_smooth):
            print("Epoch %d,\tAcc %.4f" %(i,acc))
            if acc > best_dev_acc:
                best_dev_acc = acc
                self.best_epoch = i
                self.best_thresh = best_thresh_list[i]

        print("Best epoch=%d, best_dev_acc=%.4f/%.4f (before/after smoothing)" \
              % (self.best_epoch, all_test_accuracies[self.best_epoch], all_test_accuracies_smooth[self.best_epoch]))
        
        print("------------Testing with the best epoch number------------")
        self.model.load_state_dict(sta_dic_list[self.best_epoch])
        torch.save({'epoch': self.best_epoch,
                     'model_state_dict': self.model.state_dict(),
                     'best_thresh': self.best_thresh}, self.modelPath+"_selected")

        print("\n\n#####Summary#####")
        print("---Max Epoch (%d) Acc=%.4f" %(self.max_epoch-1,all_test_accuracies[self.max_epoch-1]))
        print("---Tuned Epoch (%d) Acc=%.4f" %(self.best_epoch,all_test_accuracies[self.best_epoch]))

    def trainHelper(self,trainset,testset,max_epoch):
        self.model.train()

        optimizer = RiemannianAdam(self.model.hyper_para, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        optimizer2 = torch.optim.Adam(self.model.euclid_para, lr=self.lm_lr)

        criterion = nn.CrossEntropyLoss()
        all_train_losses = []
        all_train_accuracies = []
        all_test_accuracies = []
        start = time.time()
        sta_dict_list = []
        best_thresh_list = []
        for epoch in range(max_epoch):
            print("epoch: %d/%d" % (epoch, max_epoch-1), flush=True)
            current_train_loss = 0
            random.shuffle(trainset)
            
            self.model.set_neg(math.ceil((epoch+1)/max_epoch*self.max_neg))

            current_batch = {'target':[],
                            'events':[],
                            'sentence':[],
                            'commonsense': [],
                            'sneg_sen':[],
                            'sneg_epos':[],
                            'sent_len': [],
                            'neg_sent_len': []}

            for i, temprel in enumerate(trainset):
                if self.output_labels[temprel.label] == 3:  # skip Vague pairs
                    continue

                relpair = temprel.docid+temprel.source+temprel.target
                e_pair = self.lm.get_events_in_sentence(relpair)    # [event1_idx, event2_idx]

                if self.output_labels[temprel.label] == 1:
                    events = [e_pair[1], e_pair[0]]
                    eid = temprel.docid+temprel.target
                else:
                    events = e_pair
                    eid = temprel.docid+temprel.source

                eid = self.dataset.key_to_eid[eid]
                negs = self.dataset.gen_neg_samples(eid, max_neg=self.model.num_neg)
                if len(negs) < self.model.num_neg:
                    continue

                current_batch['events'].append(events)
                sneg_epos = []
                neg_tk_cache = []
                for j in range(self.model.num_neg):
                    relid, is_source = self.dataset.eid_to_relid[negs[j]]
                    relpair = trainset[relid].docid+trainset[relid].source+trainset[relid].target
                    if is_source:
                        sneg_epos.append(self.lm.get_events_in_sentence(relpair)[0])
                    else:
                        sneg_epos.append(self.lm.get_events_in_sentence(relpair)[1])
                    neg_tks = self.lm.get_tokens_list(relpair)
                    neg_tk_cache.append(neg_tks)
                    current_batch['neg_sent_len'].append(len(neg_tks))
                current_batch['sneg_epos'].append(sneg_epos)
                
                relpair = temprel.docid+temprel.source+temprel.target
                tks = self.lm.get_tokens_list(relpair)

                current_batch['sneg_sen'].append(neg_tk_cache)

                # tks = [seq_len]
                current_batch['sentence'].append(tks)
                current_batch['sent_len'].append(len(tks))

                if len(current_batch['sentence']) >= self.batch_size:
                    seq_max_len = max(current_batch['sent_len'])

                    s_vectors = []
                    s_attention_mask = []
                    for line in current_batch['sentence']:
                        slen = len(line)
                        s_mask = [1] * slen
                        if slen != seq_max_len:
                            line += [1] * (seq_max_len-slen)
                            s_mask += [0] * (seq_max_len-slen)
                        s_vectors.append(torch.LongTensor(line))
                        s_attention_mask.append(torch.LongTensor(s_mask))
                    s = torch.stack(s_vectors).to(self.model.device)
                    s_a_mask = torch.stack(s_attention_mask).to(self.model.device)

                    mask1 = []
                    mask2 = []
                    for eve_pair in current_batch['events']:
                        tempmask = np.zeros((seq_max_len, 1), dtype=np.float64)
                        tempmask[eve_pair[0]] = 1.
                        mask1.append(torch.tensor(tempmask))

                        tempmask = np.zeros((seq_max_len, 1), dtype=np.float64)
                        tempmask[eve_pair[1]] = 1.
                        mask2.append(torch.tensor(tempmask))
                    u = torch.stack(mask1).to(self.model.device)# [b,seq_len, 1]
                    v = torch.stack(mask2).to(self.model.device)
                    if self.float:
                        u.float()
                        v.float()

                    neg_seq_max = max(current_batch['neg_sent_len'])
                    neg_sen_vectors = []
                    n_attention_mask = []
                    for negs in current_batch['sneg_sen']:
                        for neg in negs:
                            slen = len(neg)
                            n_mask = [1] * slen
                            
                            if slen < neg_seq_max:
                                neg += [1] * (neg_seq_max-slen)
                                n_mask += [0] * (neg_seq_max-slen)
                            elif slen > neg_seq_max:
                                neg = neg[:neg_seq_max]
                                n_mask = n_mask[:neg_seq_max]
                            neg_sen_vectors.append(torch.LongTensor(neg))
                            n_attention_mask.append(torch.LongTensor(n_mask))
                    neg_seq = torch.stack(neg_sen_vectors).to(self.model.device)
                    n_a_mask = torch.stack(n_attention_mask).to(self.model.device)

                    mask_u_neg = []
                    for negs in current_batch['sneg_epos']:
                        for neg in negs:
                            tempmask = np.zeros((neg_seq_max, 1), dtype=np.float64)
                            tempmask[neg] = 1.
                            mask_u_neg.append(torch.tensor(tempmask))
                    mask_u_neg = torch.stack(mask_u_neg).to(self.model.device)

                    optimizer.zero_grad()
                    optimizer2.zero_grad()

                    loss = self.model(s, s_a_mask, u, v, neg_seq, n_a_mask, mask_u_neg)

                    current_train_loss += loss.data
                    print("%d/%d: %s %.4f %.4f" % (i, len(trainset), timeSince(start), loss.data, current_train_loss), flush=True)
                    loss.backward()

                    optimizer.step()
                    optimizer2.step()

                    current_batch = {'target':[],
                            'events':[],
                            'sentence':[],
                            'commonsense': [],
                            'sneg_sen':[],
                            'sneg_epos':[],
                            'sent_len': [],
                            'neg_sent_len': []}

            scheduler.step()
            all_train_losses.append(current_train_loss)
            #current_train_acc, _, _ = self.eval(trainset)
            current_test_acc, confusion, best_thresh = self.eval(testset,True)
            best_thresh_list.append(best_thresh)
            all_test_accuracies.append(float(current_test_acc))
            print("Loss at epoch %d: %.4f" % (epoch, current_train_loss), flush=True)
            print("Dev/Test acc at epoch %d: %.4f" % (epoch, current_test_acc), flush=True)
            print(confusion, flush=True)
            prec,rec,f1 = confusion2prf(confusion)
            print("Prec=%.4f, Rec=%.4f, F1=%.4f" %(prec,rec,f1))

            sta_dict_list.append(self.model.state_dict())

        return all_train_losses, all_test_accuracies, sta_dict_list, best_thresh_list

    def test(self, thresh=0):
        self.model.eval()
        test_acc, test_confusion, _ = self.eval(self.testset,self.gen_output, is_test=True, thresh=thresh)
        test_prec = (test_confusion[0][0]+test_confusion[1][1]+test_confusion[2][2])/(np.sum(test_confusion)-np.sum(test_confusion,axis=0)[3])
        test_rec = (test_confusion[0][0]+test_confusion[1][1]+test_confusion[2][2])/(np.sum(test_confusion)-np.sum(test_confusion[3][:]))
        test_f1 = 2*test_prec*test_rec / (test_rec+test_prec)
        print("DATASET=%s" % self.testsetname)
        print("TEST ACCURACY=%.4f" % test_acc)
        print("TEST PRECISION=%.4f" % test_prec)
        print("TEST RECALL=%.4f" % test_rec)
        print("TEST F1=%.4f" % test_f1)
        print("CONFUSION MAT:")
        print(test_confusion)

    def eval(self, eval_on_set, gen_output=False, is_test=False, thresh=None):
        was_training = self.model.training
        self.model.eval()
        confusions = [np.zeros((len(self.output_labels), len(self.output_labels)), dtype=int) for i in range(len(self.threshsss))]
        output = {}
        softmax = nn.Softmax()
        current_batch = {'label':[],
                            'target':[],
                            'events':[],
                            'sentence':[],
                            'commonsense': [],
                            'docid':[],
                            'source':[],
                            'sent_len':[]}
        
        for it, ex in enumerate(eval_on_set):

            current_batch['label'].append(ex.label)
            current_batch['target'].append(ex.target)
            current_batch['source'].append(ex.source)
            current_batch['docid'].append(ex.docid)
            
            relpair = ex.docid+ex.source+ex.target
            tks = self.lm.get_tokens_list(relpair)
            current_batch['events'].append(self.lm.get_events_in_sentence(relpair))# [event1_idx, event2_idx]

            #[seq_len]
            current_batch['sentence'].append(tks)
            current_batch['sent_len'].append(len(tks))

            if len(current_batch['sentence']) > self.batch_size or it == len(eval_on_set)-1:
                seq_max_len = max(current_batch['sent_len'])
                
                s_vectors = []
                s_attention_mask = []
                for line in current_batch['sentence']:
                    slen = len(line)
                    mask = [1] * slen
                    if slen != seq_max_len:
                        line += [1] * (seq_max_len-slen)
                        mask += [0] * (seq_max_len-slen)
                    s_vectors.append(torch.LongTensor(line))
                    s_attention_mask.append(torch.LongTensor(mask))
                s = torch.stack(s_vectors).to(self.model.device)
                s_a_mask = torch.stack(s_attention_mask).to(self.model.device)
                
                mask1 = []
                mask2 = []
                for eve_pair in current_batch['events']:
                    tempmask = np.zeros((seq_max_len, 1), dtype=np.float64)
                    tempmask[eve_pair[0]] = 1.
                    mask1.append(torch.tensor(tempmask))

                    tempmask = np.zeros((seq_max_len, 1), dtype=np.float64)
                    tempmask[eve_pair[1]] = 1.
                    mask2.append(torch.tensor(tempmask))
                u = torch.stack(mask1).to(self.model.device)    # [b,seq_len, 1]
                v = torch.stack(mask2).to(self.model.device)
                if self.float:
                    u.float()
                    v.float()

                score = self.model.get_score(s, s_a_mask, u, v)
                #print(score)
                prediction = [[] for jjj in range(len(self.threshsss))]
                for iii in range(len(score)):
                    for kkk, thresh_t in enumerate(self.threshsss):
                        prediction[kkk].append(predict_with_score(score[iii], thresh_t))

                for oj in range(len(prediction)):
                    for nnn in range(len(prediction[oj])):
                        prediction_label = categoryFromOutput(prediction[oj][nnn])
                        confusions[oj][self.output_labels[current_batch['label'][nnn]]][prediction_label] += 1

                current_batch = {'label':[],
                                'target':[],
                                'events':[],
                                'sentence':[],
                                'commonsense': [],
                                'docid':[],
                                'source':[],
                                'sent_len':[]}

        if thresh is not None:
            self.best_thresh = thresh   # use the best validation thresh for testing

        if is_test:
            confusion = confusions[self.best_thresh]
            best_thresh= self.best_thresh
        else:
            best_f1 = 0.
            best_t = 0
            for oj in range(len(confusions)):
                prec,rec,f1 = confusion2prf(confusions[oj])
                if best_f1 < f1:
                    best_f1 = f1
                    best_t=oj
            print(f'Best thresh: %.2f' %(self.threshsss[best_t]))
            confusion = confusions[best_t]
            best_thresh = best_t    # save the best threshold on this validation

        if was_training:
            self.model.train()
        return 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion), confusion, best_thresh


def predict_with_score(score, thresh):
    if np.absolute(score) < 1e-5:           # equal
        predict=torch.tensor([0.,0.,1.,0.])
    elif np.absolute(score) <= thresh:      # vague
        predict=torch.tensor([0.,0.,0.,1.])
    elif score > 0:                         # before
        predict=torch.tensor([1.,0.,0.,0.])
    else:                                   # after
        predict=torch.tensor([0.,1.,0.,0.])
    return predict


if __name__ == "__main__":
    parser = ArgumentParser(description='Poincare Event Embedding')
    parser.add_argument('--cuda', help='Use GPU', type=int, default=1)
    parser.add_argument('--hid_dim', help='hidden state dimension', type=int, default=128)
    parser.add_argument('--nn_hid_dim', help='fully connected layer dimension', type=int, default=64)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-2)
    parser.add_argument('--step_size', help='step size', type=int, default=10)
    parser.add_argument('--max_epoch', help='max training epoch', type=int, default=30)
    parser.add_argument('--expname', help='save file name', type=str, default='test')
    parser.add_argument('--sd', help='random seed', type=int, default=13234)
    parser.add_argument('--neg', help='number of negative samples per positive', type=int, default=2)
    parser.add_argument('--batch', help='batch size', type=int, default=200)
    parser.add_argument('--dropout', help='dropout rate', type=float, default=0.1)
    parser.add_argument('--non_lin', help='non linear activation function', type=str, default='relu')
    parser.add_argument('--context', help='use which pre-trained language model', type=str, default='roberta-base')
    parser.add_argument('--testsetname', help='test set name', type=str, default='matres')
    parser.add_argument('--gamma', help='gamma', type=float, default=0.3)
    parser.add_argument('--lm_lr', help='The fine-tuning learning rate for RoBERTa', type=float, default=1e-5)
    parser.add_argument('--alpha', help='The weight to balance two loss terms', type=float, default=0.5)
    parser.add_argument('--precision', help='Use which precision data_type as parameters', type=str, default='double')

    parser.add_argument('--skiptraining', help='skip training', action='store_true')
    parser.add_argument('--loadcache', help='Load the representation cache of LM', action='store_true')
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--gen_output', help='generate output', action='store_true')
    
    args = parser.parse_args()
    print(args)

    if args.precision == 'double':
        torch.set_default_dtype(torch.float64) # Double precision is better than Float when it comes to hyperbolic embeddings
    elif args.precision == 'float':
        torch.set_default_dtype(torch.float32)
    else:
        print('Not support this datatype!')
        exit(0)

    seed_everything(args.sd)
    trainset = temprel_set("data/trainset-temprel.xml", is_train=args.sd)

    if args.testsetname == "matres":
        testset = temprel_set("data/testset-temprel.xml","matres")
        w2v_ser_dir = "ser/"
    else:
        testset = temprel_set("data/tcr-temprel.xml","tcr")
        w2v_ser_dir = "ser/TCR/"
    
    if args.context in ['roberta-base', 'bert-base-uncased']:
        embedding_dim = 768
        print(f"Using Pre-trained LM %s" % (args.context))
    elif args.context in ['roberta-large']:
        embedding_dim = 1024
        print(f"Using Pre-trained LM %s" % (args.context))
    
    output_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}

    if torch.cuda.is_available() and args.cuda > 0:
        device = f'cuda:%d' % (args.cuda-1)
    else:
        device = 'cpu'
    print('Using Device:', device)

    exp = experiment(device=device, embedding_dim=embedding_dim,\
                    trainset=trainset, testset=testset.temprel_ee,\
                    args=args, modelPath="models/ckpt", \
                    output_labels=output_labels)

    best_ttt=None
    if not args.skiptraining:
        exp.train()
    else:
        exp.model.load_state_dict(torch.load("models/ckpt_"+args.expname+"_selected", map_location=device)['model_state_dict'])
        best_ttt = torch.load("models/ckpt_"+args.expname+"_selected", map_location=device)['best_thresh']
    exp.test(thresh=best_ttt)


    