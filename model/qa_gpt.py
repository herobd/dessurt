from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding
from model.layout_transformer import LayoutTransformer
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    from transformers import LayoutLMTokenizer, LayoutLMModel
except:
    pass
from collections import defaultdict

import timeit


class QAGPT(BaseModel):
    def __init__(self,config):
        super(QAGPT, self).__init__(config)
        d_model = config['decode_dim']
        dim_ff = config['dim_ff']
        nhead = config['decode_num_heads']
        num_layers = config['decode_layers']

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.SEP_TOKEN= 102
        self.CLS_TOKEN= 101

        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.1
        decoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_ff,dropout=dropout)
        self.decoder = nn.TransformerEncoder(decoder_layer,num_layers,nn.LayerNorm(d_model))
        self.answer_embedding = nn.Sequential(
                nn.Embedding(self.tokenizer.vocab_size, d_model),
                PositionalEncoding(d_model,dropout=0.1,max_len=1000)
                )
        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.tokenizer.vocab_size),
                nn.LogSoftmax(dim=-1) #except
                )



        #t#self.opt_history=defaultdict(list)#t#


    def forward(self,image,gtBBs,gtTrans,questions,answers=None):
        torch.autograd.set_detect_anomaly(True)
        #t#ticA=timeit.default_timer()#t#
        device = image.device
        total_strings = [' '.join(gtT) for gtT in gtTrans]


        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions

        repeat_docs = [[doc]*r for doc,r in zip(total_strings,repeats)]


        repeat_docs=[d for bd in repeat_docs for d in bd]
        questions=[q for bq in questions for q in bq]
        answers=[a for ba in answers for a in ba]

        #Append question before answer
        answers = [doc+':'+q+'[SEP]'+a for doc,q,a in zip(repeat_docs,questions,answers)]

        #run question+answer through decoder
        answers_t = self.tokenizer(answers,return_tensors="pt",padding=True)
        answers_to_emb = answers_t['input_ids'][:,:-1].detach().to(device)
        answers_emb = self.answer_embedding(answers_to_emb) #Remove end (SEP) token, as it doesn't need to predict anythin after that. emb needs to do position
        answers_emb = answers_emb.permute(1,0,2) #batch,len,feat -> len,batch,feat
        answer_padding_mask = (1-answers_t['attention_mask'][:,:-1]).bool().to(device)
        response = self.decoder(
                answers_emb,
                mask=nn.Transformer.generate_square_subsequent_mask(None,answers_emb.size(0)).to(device),
                src_key_padding_mask=answer_padding_mask
                )

        response_decoded = self.answer_decode(response.view(-1,response.size(2)))
        response_decoded = response_decoded.view(answers_emb.size(0),len(questions),-1)
        response_decoded = response_decoded.permute(1,0,2) #put batch dim first

        #t#time = timeit.default_timer()-ticA#t#
        #t#self.opt_history['transformers'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        response_greedy_tokens = response_decoded.argmax(dim=2)
        locs = (answers_t['input_ids']==self.SEP_TOKEN).nonzero(as_tuple=False)[::2,1] #get first SEP, not second
        mask_response = torch.ones(response_decoded.size())
        for b,loc in enumerate(locs):
            #for the question (before SEP)
            answers_t['input_ids'][b,:loc]=0 #zero GT where question is (we don't want to train that)
            #response_decoded[b,:loc]*=0 #zero pred
            mask_response[b,:loc]*=0
        response_decoded = response_decoded*mask_response.to(device)
        target_decoded = answers_t['input_ids'][:,1:]# This has the SEP tokens (and padding), but not CLS (start) token

        #decode the prediction to string
        string_response=[]
        for b in range(len(questions)):
            response_greedy_tokens_b = response_greedy_tokens[b]
            response_greedy_tokens_b = response_greedy_tokens_b[locs[b]:] #only take response
            pred_stop = response_greedy_tokens_b==self.SEP_TOKEN
            if pred_stop.any():
                stop_index = pred_stop.nonzero(as_tuple=False)[0][0].item()
                response_greedy_tokens_b[stop_index:]=self.SEP_TOKEN
            string_response.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(response_greedy_tokens_b,skip_special_tokens=True)))

        #reshape strings into batches
        batch_string_response=[]
        cur_pos=0
        for r in repeats:
            batch_string_response.append(string_response[cur_pos:cur_pos+r])
            cur_pos+=r

        #t#timeA = timeit.default_timer()-ticA#t#
        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['decode'].append(time)#t#
        #t#self.opt_history['full forward'].append(timeA)#t#
        #t#self.print_opt_times()#t#
        return response_decoded, target_decoded.to(device), batch_string_response

    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
