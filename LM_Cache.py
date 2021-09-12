from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle as pkl


# This class can encode the text using the given LM model.
# Then, save as cache and can be used for static models in the paper.
class LM_context_encoder:
    def __init__(self, device, model_name='roberta-base'):
        self.model_name = model_name
        self.relpair2id = {}
        self.id2relpair = []
        self.encoded_context = []
        self.events_in_sentence = []
        self.device = device
    
    def get_encoded_context(self, relpair):
        return self.encoded_context[self.relpair2id[relpair]]
    
    def get_events_in_sentence(self, relpair):
        return self.events_in_sentence[self.relpair2id[relpair]]

    def encode(self, temprel_datasets):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        encoder_model = AutoModel.from_pretrained(self.model_name, return_dict=True)
        encoder_model.to(self.device)
        encoder_model.eval()

        for temprel_dataset in temprel_datasets:
            for i, temprel in tqdm(enumerate(temprel_dataset), desc='[Encoding Context]'):
                relpair = temprel.docid+temprel.source+temprel.target
                event_ix = temprel.event_ix
                tk_list = temprel.token

                #print(relpair, len(tk_list))
                if self.relpair2id.get(relpair) is None:
                    self.relpair2id[relpair] = len(self.id2relpair)
                    self.id2relpair.append(relpair)

                    lm_tks = tokenizer(' '.join(tk_list))
                    encoded = encoder_model(torch.LongTensor([lm_tks['input_ids']]).to(self.device),
                            torch.LongTensor([lm_tks['attention_mask']]).to(self.device)).last_hidden_state.detach().cpu()
                    # [1,seq_len,encode_dim]
                    self.encoded_context.append(encoded.squeeze(0))

                    tokens_before_e1 = tk_list[:event_ix[0]]

                    tokens_before_e1 = len(tokenizer.encode(' '.join(tokens_before_e1))) - 1 # the last token is </s>
                    e1 = tk_list[event_ix[0]]
                    # roberta tokenizer differentiate words with words start a sentence
                    if event_ix[0] != 0:
                        e1 = ' ' + e1
                    e1_len = len(tokenizer.encode(e1, add_special_tokens=False)) 
                    lm_tk_e1_pos = tokens_before_e1 + e1_len - 1 # last token of the verb

                    tokens_before_e2 = tk_list[:event_ix[1]]
                    tokens_before_e2 = len(tokenizer.encode(' '.join(tokens_before_e2))) - 1 # the last token is </s>
                    e2 = tk_list[event_ix[1]]
                    # roberta tokenizer differentiate words with words start a sentence
                    if event_ix[1] != 0:
                        e2 = ' ' + e2
                    e2_len = len(tokenizer.encode(e2, add_special_tokens=False))
                    lm_tk_e2_pos = tokens_before_e2 + e2_len - 1 # last token of the verb
                    self.events_in_sentence.append([lm_tk_e1_pos, lm_tk_e2_pos])
    
    def save_as_cache(self, path=''):
        path += self.model_name + '_cache.pkl'
        pkl.dump((self.model_name, self.relpair2id, self.id2relpair,
                self.encoded_context, self.events_in_sentence),
                open(path, "wb"))
    
    def load_cache(self, path=''):
        path += self.model_name + '_cache.pkl'
        self.model_name, self.relpair2id, self.id2relpair, self.encoded_context, self.events_in_sentence = pkl.load(open(path,"rb"))


# Context tokenizer for the input of HGRU model
class LM_context_tokenizer:
    def __init__(self, device, model_name='roberta-base'):
        self.model_name = model_name
        self.relpair2id = {}
        self.id2relpair = []
        self.tokens_list = []
        self.events_in_sentence = []
        self.device = device
    
    def get_tokens_list(self, relpair):
        return self.tokens_list[self.relpair2id[relpair]]
    
    def get_events_in_sentence(self, relpair):
        return self.events_in_sentence[self.relpair2id[relpair]]

    def encode(self, temprel_datasets):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        for temprel_dataset in temprel_datasets:
            for i, temprel in tqdm(enumerate(temprel_dataset), desc='[Encoding Context]'):
                relpair = temprel.docid+temprel.source+temprel.target
                event_ix = temprel.event_ix
                tk_list = temprel.token

                if self.relpair2id.get(relpair) is None:
                    self.relpair2id[relpair] = len(self.id2relpair)
                    self.id2relpair.append(relpair)

                    lm_tks = tokenizer(' '.join(tk_list))
                    self.tokens_list.append(lm_tks['input_ids'])

                    tokens_before_e1 = tk_list[:event_ix[0]]

                    tokens_before_e1 = len(tokenizer.encode(' '.join(tokens_before_e1))) - 1 # the last token is </s>
                    e1 = tk_list[event_ix[0]]
                    # roberta tokenizer differentiate words with words start a sentence
                    if event_ix[0] != 0:
                        e1 = ' ' + e1
                    e1_len = len(tokenizer.encode(e1, add_special_tokens=False)) 
                    lm_tk_e1_pos = tokens_before_e1 + e1_len - 1 # last token of the verb

                    tokens_before_e2 = tk_list[:event_ix[1]]
                    tokens_before_e2 = len(tokenizer.encode(' '.join(tokens_before_e2))) - 1 # the last token is </s>
                    e2 = tk_list[event_ix[1]]
                    # roberta tokenizer differentiate words with words start a sentence
                    if event_ix[1] != 0:
                        e2 = ' ' + e2
                    e2_len = len(tokenizer.encode(e2, add_special_tokens=False))
                    lm_tk_e2_pos = tokens_before_e2 + e2_len - 1 # last token of the verb
                    self.events_in_sentence.append([lm_tk_e1_pos, lm_tk_e2_pos])
    
    def save_as_cache(self, path=''):
        path += self.model_name + '_token_cache.pkl'
        pkl.dump((self.model_name, self.relpair2id, self.id2relpair,
                self.tokens_list, self.events_in_sentence),
                open(path, "wb"))
    
    def load_cache(self, path=''):
        path += self.model_name + '_token_cache.pkl'
        self.model_name, self.relpair2id, self.id2relpair, self.tokens_list, self.events_in_sentence = pkl.load(open(path,"rb"))

