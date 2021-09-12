import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import os
from HGRU_model import EventTempRel_HGRU

from TemporalDataSet import *
from pairwise_ffnn_pytorch import VerbNet

from geoopt.optim import RiemannianAdam
from LM_Cache import LM_context_tokenizer

from argparse import ArgumentParser

seed_everything(13234)
torch.set_default_dtype(torch.float64) # Double precision is better than Float when it comes to hyperbolic embeddings


class experiment:
    def __init__(self, device, embedding_dim, trainset, testset, output_labels,
                modelPath, bigram, args=None):
        self.model = EventTempRel_HGRU(device=device, dim_in=embedding_dim, dim_hidden=args.hid_dim, 
                            dim_out=args.nn_hid_dim, dropout=args.dropout, non_lin=args.non_lin,
                            granularity=args.granularity, common_sense_emb_dim=args.common_sense_emb_dim,
                            bigramStats=args.bigramstats_dim, num_class=len(output_labels))

        self.model.to(device)
        self.max_epoch = args.max_epoch
        self.testsetname = args.testsetname
        self.output_labels = output_labels
        self.exp_name = args.expname
        self.modelPath = "%s_%s" %(modelPath,self.exp_name)
        self.gen_output = args.gen_output

        self.context = args.context
        self.batch_size = args.batch
        self.bigramGetter = bigram
        self.lr = args.lr
        self.lm_lr = args.lm_lr
        self.gamma = args.gamma
        self.write_report = args.write_report

        self.weight_decay = args.weight_decay
        self.step_size = args.step_size
        self.sd = args.sd
        self.trainset, self.devset = self.split_train_dev(trainset)

        if args.debug:
            self.trainset = self.trainset[:10]
        self.testset = testset

        if self.testsetname == "matres":
            w2v_ser_dir = "ser/"
        else:
            w2v_ser_dir = "ser/TCR/"

        if self.context in ['roberta-base', 'roberta-large', 'bert-base-uncased']:
            self.lm = LM_context_tokenizer(model_name=self.context, device=self.model.device)
            if args.loadcache:
                self.lm.load_cache()
            else:
                self.lm.encode([self.trainset, self.devset, self.testset])
                self.lm.save_as_cache()

        else:
            print('Unrecognized context encoder!')
            exit(0)

    def train(self):
        self.best_epoch = self.max_epoch-1
        print("------------Training and Development------------")
        # all_test_accuracies refer to performance on devset
        all_train_losses, all_test_accuracies, sta_dic_list =\
            self.trainHelper(self.trainset,self.devset,self.max_epoch)
        # smooth within a window of +-2
        all_test_accuracies_smooth = [all_test_accuracies[0]]*2+all_test_accuracies+[all_test_accuracies[-1]]*2
        all_test_accuracies_smooth = [1.0/5*(all_test_accuracies_smooth[i-2]+all_test_accuracies_smooth[i-1]+all_test_accuracies_smooth[i]+all_test_accuracies_smooth[i+1]+all_test_accuracies_smooth[i+2]) for i in range(2,2+len(all_test_accuracies))]
        
        self.best_epoch, best_dev_acc = 0,0
        print("Select epoch based on smoothed dev accuracy curve")
        for i,acc in enumerate(all_test_accuracies_smooth):
            print("Epoch %d,\tAcc %.4f" %(i,acc))
            if acc > best_dev_acc:
                best_dev_acc = acc
                self.best_epoch = i
        print("Best epoch=%d, best_dev_acc=%.4f/%.4f (before/after smoothing)" \
              % (self.best_epoch, all_test_accuracies[self.best_epoch], all_test_accuracies_smooth[self.best_epoch]))
        print("------------Testing with the best epoch number------------")
        self.model.load_state_dict(sta_dic_list[self.best_epoch])
        torch.save({'epoch': self.best_epoch,
                     'model_state_dict': self.model.state_dict()}, self.modelPath+"_selected")

        print("\n\n#####Summary#####")
        print("---Max Epoch (%d) Acc=%.4f" %(self.max_epoch-1,all_test_accuracies[self.max_epoch-1]))
        print("---Tuned Epoch (%d) Acc=%.4f" %(self.best_epoch,all_test_accuracies[self.best_epoch]))

        torch.cuda.empty_cache()
        self.test()

    def trainHelper(self,trainset,testset,max_epoch):
        self.model.train()
        optimizer1 = optim.Adam(self.model.euclid_para, lr=self.lr, weight_decay=self.weight_decay)
        optimizer2 = RiemannianAdam(self.model.hyper_para, lr=self.lr, weight_decay=self.weight_decay)
        optimizer3 = optim.Adam(self.model.plm_para, lr=self.lm_lr)
        
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=self.step_size, gamma=self.gamma)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=self.step_size, gamma=self.gamma)

        criterion = nn.CrossEntropyLoss()
        all_train_losses = []
        all_train_accuracies = []
        all_test_accuracies = []
        start = time.time()
        sta_dict_list = []
        for epoch in range(max_epoch):
            print("epoch: %d/%d" % (epoch, max_epoch-1), flush=True)
            current_train_loss = 0
            random.shuffle(trainset)
            

            current_batch = {'target':[],
                            'events':[],
                            'sentence':[],
                            'sent_len': [],
                            'commonsense': []}

            # prepare batches
            for i,temprel in enumerate(trainset):
                current_batch['target'].append(self.output_labels[temprel.label])
                
                relpair = temprel.docid+temprel.source+temprel.target
                current_batch['events'].append(self.lm.get_events_in_sentence(relpair)) # [event1_idx, event2_idx]
                embeds = self.lm.get_tokens_list(relpair)

                # embeds = [seq_len, dim]
                current_batch['sentence'].append(embeds)
                current_batch['sent_len'].append(len(embeds))

                # common sense embeddings
                bigramstats = self.bigramGetter.getBigramStatsFromTemprel(temprel).detach().cpu().numpy()
                commonsense = [min(int(1.0 / self.model.granularity) - 1, int(bigramstats[0][0] / self.model.granularity))]
                for k in range(1,self.model.bigramStats_dim):
                    commonsense.append((k - 1) * int(1.0 / self.model.granularity) +
                                        min(int(1.0 / self.model.granularity) - 1, 
                                        int(bigramstats[0][k] / self.model.granularity)))
                current_batch['commonsense'].append(commonsense)
                # [1,common_sense_emb_dim+bigramStats_dim]
                
                # feed batches
                if len(current_batch['sentence']) >= self.batch_size:
                    target = torch.LongTensor(current_batch['target']).to(self.model.device)
                    s_vectors = []
                    s_attention_mask = []
                    seq_max_len = max(current_batch['sent_len'])

                    for line in current_batch['sentence']:
                        slen = len(line)
                        s_mask = [1] * slen
                        if slen < seq_max_len:
                            #line = nn.functional.pad(line, pad=(0,0,0,seq_max_len-slen))
                            line += [1] * (seq_max_len-slen)
                            s_mask += [0] * (seq_max_len-slen)
                        elif slen > seq_max_len:
                            line = line[:seq_max_len]
                            s_mask = s_mask[:seq_max_len]

                        s_vectors.append(torch.LongTensor(line))
                        s_attention_mask.append(torch.LongTensor(s_mask))

                    s = torch.stack(s_vectors).to(self.model.device)
                    s_a_mask = torch.stack(s_attention_mask).to(self.model.device)

                    common_ids = torch.LongTensor(current_batch['commonsense']).to(self.model.device)
                    
                    mask1 = []
                    mask2 = []
                    for eve_pair in current_batch['events']:
                        temppp = np.zeros((seq_max_len, 1))
                        temppp[eve_pair[0]] = 1.
                        mask1.append(torch.tensor(temppp))

                        temppp = np.zeros((seq_max_len, 1))
                        temppp[eve_pair[1]] = 1.
                        mask2.append(torch.tensor(temppp))
                    u = torch.stack(mask1).to(self.model.device)# [b,seq_len, 1]
                    v = torch.stack(mask2).to(self.model.device)

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    optimizer3.zero_grad()

                    output = self.model(s, s_a_mask, u, v,common_ids)

                    loss = criterion(output, target)
                    current_train_loss += loss.data
                    print("%d/%d: %s %.4f %.4f" % (i, len(trainset), timeSince(start), loss, current_train_loss), flush=True)
                    loss.backward()

                    optimizer1.step()
                    optimizer2.step()
                    optimizer3.step()

                    current_batch = {'target':[],
                                    'events':[],
                                    'sentence':[],
                                    'commonsense':[],
                                    'sent_len': []}
            scheduler1.step()
            scheduler2.step()

            all_train_losses.append(current_train_loss)
            #current_train_acc, _, _ = self.eval(trainset)
            current_acc, current_f1, current_prec, current_rec, confusion, curr_output = self.eval(testset,True)
            all_test_accuracies.append(float(current_f1))

            print("Loss at epoch %d: %.4f" % (epoch, current_train_loss), flush=True)
            #print("Train acc at epoch %d: %.4f" % (epoch, current_train_acc), flush=True)
            print("Dev/Test acc at epoch %d: %.4f" % (epoch, current_acc), flush=True)
            print(confusion, flush=True)
            prec,rec,f1 = confusion2prf(confusion)
            print("Prec=%.4f, Rec=%.4f, F1=%.4f" %(prec,rec,f1))
            print("%s" % (timeSince(start)))

            sta_dict_list.append(self.model.state_dict())


        return all_train_losses, all_test_accuracies, sta_dict_list

    def test(self):
        self.model.eval()
        test_acc, test_f1, test_prec, test_rec, test_confusion, test_output = self.eval(self.testset,self.gen_output)
        print("DATASET=%s" % self.testsetname)
        print("TEST ACCURACY=%.4f" % test_acc)
        print("TEST PRECISION=%.4f" % test_prec)
        print("TEST RECALL=%.4f" % test_rec)
        print("TEST F1=%.4f" % test_f1)
        print("CONFUSION MAT:")
        print(test_confusion)
        if self.write_report:
            with open('experiment_report_HGRU.txt', 'a+') as f:
                out_str = self.exp_name
                out_str += f"\nTest Acc: %.4f, Test Prec: %.4f, Test Recall: %.4f, Test F1: %.4f" % (test_acc, test_prec, test_rec, test_f1)
                f.seek(2)
                f.write(out_str+'\n')

    # For evaluation epoches (Validation and Testing)
    def eval(self,eval_on_set, gen_output=False):
        was_training = self.model.training
        self.model.eval()
        confusion = np.zeros((len(self.output_labels), len(self.output_labels)), dtype=int)
        output = {}
        softmax = nn.Softmax()
        current_batch = {'label':[],
                            'target':[],
                            'events':[],
                            'sentence':[],
                            'commonsense': [],
                            'docid':[],
                            'source':[],
                            'sent_len': []}

        for it, ex in enumerate(eval_on_set):

            current_batch['label'].append(ex.label)
            current_batch['target'].append(ex.target)
            current_batch['source'].append(ex.source)
            current_batch['docid'].append(ex.docid)
            
            relpair = ex.docid+ex.source+ex.target
            embeds = self.lm.get_tokens_list(relpair)
            current_batch['events'].append(self.lm.get_events_in_sentence(relpair))# [event1_idx, event2_idx]

            #[seq_len, dim]
            current_batch['sentence'].append(embeds)
            current_batch['sent_len'].append(len(embeds))

            # common sense embeddings
            bigramstats = self.bigramGetter.getBigramStatsFromTemprel(ex).detach().cpu().numpy()
            commonsense = [min(int(1.0 / self.model.granularity) - 1, int(bigramstats[0][0] / self.model.granularity))]
            for k in range(1,self.model.bigramStats_dim):
                commonsense.append((k - 1) * int(1.0 / self.model.granularity) +
                                    min(int(1.0 / self.model.granularity) - 1, 
                                    int(bigramstats[0][k] / self.model.granularity)))
            current_batch['commonsense'].append(commonsense)

            if len(current_batch['sentence']) >= self.batch_size or it == len(eval_on_set)-1:
                s_vectors = []
                s_attention_mask = []

                seq_max_len = max(current_batch['sent_len'])
                for line in current_batch['sentence']:
                    slen = len(line)
                    s_mask = [1] * slen
                    if slen < seq_max_len:
                        #line = nn.functional.pad(line, pad=(0,0,0,seq_max_len-slen))
                        line += [1] * (seq_max_len-slen)
                        s_mask += [0] * (seq_max_len-slen)
                    elif slen > seq_max_len:
                        line = line[:seq_max_len]
                        s_mask = line[:seq_max_len]
                    s_vectors.append(torch.LongTensor(line))
                    s_attention_mask.append(torch.LongTensor(s_mask))
                s = torch.stack(s_vectors).to(self.model.device)
                s_a_mask = torch.stack(s_attention_mask).to(self.model.device)
                
                mask1 = []
                mask2 = []
                for eve_pair in current_batch['events']:
                    temppp = np.zeros((seq_max_len, 1))
                    temppp[eve_pair[0]] = 1.
                    mask1.append(torch.tensor(temppp))

                    temppp = np.zeros((seq_max_len, 1))
                    temppp[eve_pair[1]] = 1.
                    mask2.append(torch.tensor(temppp))
                u = torch.stack(mask1).to(self.model.device)# [b,seq_len, 1]
                v = torch.stack(mask2).to(self.model.device)

                common_ids = torch.LongTensor(current_batch['commonsense']).to(self.model.device)

                prediction = self.model(s, s_a_mask, u, v, common_ids)
                
                for nnn in range(len(prediction)):
                    prediction_label = categoryFromOutput(prediction[nnn])
                    if gen_output:
                        prediction_scores = softmax(prediction[nnn])
                        if current_batch['docid'][nnn] not in output:
                            output[current_batch['docid'][nnn]] = {}
                        output[current_batch['docid'][nnn]]["%s,%s" %(current_batch['source'][nnn],current_batch['target'][nnn])]\
                        = "%d,%f,%f,%f,%f" %(prediction_label,prediction_scores[0],prediction_scores[1],prediction_scores[2],prediction_scores[3])
                    confusion[self.output_labels[current_batch['label'][nnn]]][prediction_label] += 1

                current_batch = {'label':[],
                                'target':[],
                                'events':[],
                                'sentence':[],
                                'commonsense': [],
                                'docid':[],
                                'source':[],
                                'sent_len': []}

        if was_training:
            self.model.train()
        acc = 1.0 * np.sum([confusion[i][i] for i in range(confusion.shape[0])]) / np.sum(confusion)
        prec, rec, f1 = confusion2prf(confusion)

        return acc, f1, prec, rec, confusion, output

    def split_train_dev(self,trainset):
        train,dev = train_test_split(trainset,test_size=0.2,random_state=self.sd)
        return train,dev


# a class for getting external knowledge
class bigramGetter_fromNN:
    def __init__(self,device, emb_path, mdl_path, ratio=0.3, layer=1, emb_size=200, splitter=','):
        self.verb_i_map = {}
        self.device = device
        f = open(emb_path)
        lines = f.readlines()
        for i,line in enumerate(lines):
            self.verb_i_map[line.split(splitter)[0]] = i
        f.close()
        self.model = VerbNet(len(self.verb_i_map),hidden_ratio=ratio,emb_size=emb_size,num_layers=layer)
        self.model.to(self.device)
        checkpoint = torch.load(mdl_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    def eval(self,v1,v2):
        return self.model(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).to(self.device))
    def getBigramStatsFromTemprel(self,temprel):
        if type(temprel.lemma) == type((0,1)):
            v1 = temprel.lemma[0]
            v2 = temprel.lemma[1]
        else:
            v1,v2='',''
            for i,position in enumerate(temprel.position):
                if position == 'E1':
                    v1 = temprel.lemma[i]
                elif position == 'E2':
                    v2 = temprel.lemma[i]
                    break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.tensor([0,0]).view(1,-1).to(self.device)
        return torch.cat((self.eval(v1,v2),self.eval(v2,v1)),1).view(1,-1)
    def retrieveEmbeddings(self,temprel):
        if type(temprel.lemma) == type((0,1)):
            v1 = temprel.lemma[0]
            v2 = temprel.lemma[1]
        else:
            v1, v2 = '', ''
            for i, position in enumerate(temprel.position):
                if position == 'E1':
                    v1 = temprel.lemma[i]
                elif position == 'E2':
                    v2 = temprel.lemma[i]
                    break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.zeros_like(self.model.retrieveEmbeddings(torch.from_numpy(np.array([[0,0]])).to(self.device)).view(1,-1))
        return self.model.retrieveEmbeddings(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).to(self.device)).view(1,-1)


if __name__ == "__main__":
    parser = ArgumentParser(description='Hyperbolic GRU Model')
    parser.add_argument('--cuda', help='Use GPU', type=int, default=1)
    parser.add_argument('--hid_dim', help='hidden state dimension', type=int, default=128)
    parser.add_argument('--nn_hid_dim', help='fully connected layer dimension', type=int, default=64)
    parser.add_argument('--pos_emb_dim', help='fully connected layer dimension', type=int, default=32)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-1)
    parser.add_argument('--step_size', help='step size', type=int, default=10)
    parser.add_argument('--max_epoch', help='max training epoch', type=int, default=30)
    parser.add_argument('--expname', help='save file name', type=str, default='test')
    parser.add_argument('--sd', help='random seed', type=int, default=13234)
    parser.add_argument('--batch', help='batch size', type=int, default=100)
    parser.add_argument('--dropout', help='dropout rate', type=float, default=0.1)
    parser.add_argument('--non_lin', help='non linear activation function', type=str, default='relu')
    parser.add_argument('--context', help='use which pre-trained language model', type=str, default='roberta-base')
    parser.add_argument('--testsetname', help='test set name', type=str, default='matres')
    parser.add_argument('--gamma', help='gamma', type=float, default=0.3)
    parser.add_argument('--lm_lr', help='The fine-tuning learning rate for RoBERTa', type=float, default=1e-5)

    parser.add_argument('--granularity', help='granularity of discretize seame network output', type=float, default=0.5)
    parser.add_argument('--common_sense_emb_dim', help='common sense feature dimension', type=int, default=32)
    parser.add_argument('--bigramstats_dim', help='bigram feature dimension', type=int, default=1)

    parser.add_argument('--skiptraining', help='skip training', action='store_true')
    parser.add_argument('--loadcache', help='Load the representation cache of LM', action='store_true')
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--gen_output', help='generate output', action='store_true')
    parser.add_argument('--write_report', help='write report to file', action='store_true')
    
    args = parser.parse_args()
    print(args)

    seed_everything(args.sd)

    if args.context in ['roberta-base', 'bert-base-uncased']:
        embedding_dim = 768
        print(f"Using Pre-trained LM %s" % (args.context))
    elif args.context in ['roberta-large']:
        embedding_dim = 1024
        print(f"Using Pre-trained LM %s" % (args.context))
    else:
        print('This LM is not supportted.')

    # parameters for ELMo
    ratio = 0.3
    emb_size = 200
    layer = 1
    splitter = " "
    print("---------")
    print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
    emb_path = './ser/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
    mdl_path = './ser/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)

    if torch.cuda.is_available() and args.cuda > 0:
        device = f'cuda:%d' % (args.cuda-1)
    else:
        device = 'cpu'
    print('Using Device:', device)

    # external knowledge
    bigramGetter = bigramGetter_fromNN('cpu', emb_path, mdl_path, ratio, layer, emb_size, splitter=splitter)

    # labels from MATRES dataset
    output_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}
    trainset = temprel_set("data/trainset-temprel.xml")

    if args.testsetname == "matres":
        testset = temprel_set("data/testset-temprel.xml","matres")
        w2v_ser_dir = "ser/"
    else:
        testset = temprel_set("data/tcr-temprel.xml","tcr")
        w2v_ser_dir = "ser/TCR/"

    # declare experiment
    exp = experiment(device=device, embedding_dim=embedding_dim, trainset=trainset.temprel_ee, testset=testset.temprel_ee, 
                    modelPath="models/ckpt", output_labels=output_labels, bigram=bigramGetter, args=args)
    
    if not args.skiptraining:
        exp.train()
    else:
        exp.model.load_state_dict(torch.load("models/ckpt_"+args.expname+"_selected", map_location=device)['model_state_dict'])
        exp.test()




