from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding, UniformRealEmbedding,PositiveRealEmbedding, ReturnPositionalEncoding
from model.transformer_encoder import RelativePositionTransformerEncoderLayer, PositionBiasedTransformerEncoder
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    from transformers import LayoutLMTokenizer, LayoutLMModel
except:
    pass
from collections import defaultdict

import timeit

MAX_DIST=1000
def normalize_bbox2(bbox, dwidth, dheight, twidth, theight):
     return [
         max(min(int(MAX_DIST * (bbox[0] / dwidth)),MAX_DIST),0),
         max(min(int(MAX_DIST * (bbox[1] / dheight)),MAX_DIST),0),
         max(min(int(MAX_DIST * (bbox[2] / twidth)),twidth),0),
         max(min(int(MAX_DIST * (bbox[3] / theight)),theight),0),
     ]

class QALayoutGPT(BaseModel):
    def __init__(self,config):
        super(QALayoutGPT, self).__init__(config)
        d_model = config['decode_dim']
        dim_ff = config['dim_ff']
        nhead = config['decode_num_heads']
        num_layers = config['decode_layers']

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.SEP_TOKEN= 102
        self.CLS_TOKEN= 101

        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.1
        encoder_layer = RelativePositionTransformerEncoderLayer(d_model,nhead,MAX_DIST,dim_ff,dropout=dropout)
        self.encoder = PositionBiasedTransformerEncoder(encoder_layer,num_layers,nn.LayerNorm(d_model))
        self.answer_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.pos_1d_enc = PositionalEncoding(d_model,dropout=dropout,max_len=1000)
        self.word_pos_enc = ReturnPositionalEncoding(d_model,dropout=dropout,max_len=1000)
        self.pos_emb_x = UniformRealEmbedding(d_model,0,MAX_DIST,100)
        self.pos_emb_y = UniformRealEmbedding(d_model,0,MAX_DIST,100)
        self.pos_emb_w = PositiveRealEmbedding(d_model,0,int(0.5*MAX_DIST),30)
        self.pos_emb_h = PositiveRealEmbedding(d_model,0,int(0.3*MAX_DIST),30)
                
        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.tokenizer.vocab_size),
                nn.LogSoftmax(dim=-1) #except
                )

d

        #t#self.opt_history=defaultdict(list)#t#


    def forward(self,image,gtBBs,gtTrans,questions,answers=None,useCurvedBBs=False):
        torch.autograd.set_detect_anomaly(True)
        #t#ticA=timeit.default_timer()#t#
        device = image.device
        image_size = image.size()

        all_input_bbs=[]
        all_input_pos=[]
        max_len=0
        for b,(words,bbs) in enumerate(zip(gtTrans,gtBBs)):
            input_bbs = [[0,0,0,0]]
            input_pos = [0]
            for i,(word,bb) in enumerate(zip(words,bbs)):
                if useCurvedBBs:
                    x1,y1,x2,y2,r=bb[:5]
                    h = y2-y1
                    w = x2-x1
                    xc = (x1+x2)/2
                    yc = (y1+y2)/2
                else:
                    xc,yc,r,h,w=bb[:5]
                    x1=xc-w
                    x2=xc+w
                    y1=yc-h
                    y2=yc+h
                #bb = normalize_bbox([x1,y1,x2,y2],image_size[3],image_size[2]) #x1y1x2y2
                bb = normalize_bbox2([xc,yc,w,h],image_size[3],image_size[2],500,300) #x1y1x2y2
                word_tokens = self.tokenizer.tokenize(word)
                #input_ids.extend(word_tokens)
                #total_string+=word+' '
                #word_token_map.append(range(len(input_bbs),len(input_bbs)+len(word_tokens)))
                input_bbs.extend([bb]*len(word_tokens))
                input_pos.extend(range(0,len(word_tokens)))
            input_bbs.append([0,0,0,0])
            input_pos.append(0)
            max_len = max(max_len,len(input_bbs))
            all_input_bbs.append(input_bbs)
            all_input_pos.append(input_pos)



        total_strings = [' '.join(gtT) for gtT in gtTrans]


        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions

        repeat_docs = [[doc]*r for doc,r in zip(total_strings,repeats)]


        repeat_docs=[d for bd in repeat_docs for d in bd]
        questions=[q for bq in questions for q in bq]
        answers=[a for ba in answers for a in ba]

        new_batch_size = len(questions)

        #Append question before answer
        input_str = [doc+':'+q+'[SEP]'+a for doc,q,a in zip(repeat_docs,questions,answers)]

        #run question+answer through decoder
        input_t = self.tokenizer(input_str,return_tensors="pt",padding=True)
        input_to_emb = input_t['input_ids'][:,:-1].detach().to(device) #Remove end (SEP) token, as it doesn't need to predict anythin after that. emb needs to do position
        input_emb = self.answer_embedding(input_to_emb) 

        input_len = input_emb.size(1)
        qa_mask = torch.FloatTensor(new_batch_size,input_len,1).fill_(1)
        bbs = torch.FloatTensor(new_batch_size,input_len,4).fill_(0)#float('NaN'))
        tok_pos = torch.LongTensor(new_batch_size,input_len).fill_(0)
        new_i=0
        for b,(input_bbs,input_pos) in enumerate(zip(all_input_bbs,all_input_pos)):
            len_input = len(input_bbs)
            input_bbs = torch.FloatTensor(input_bbs)[None,...]
            input_pos = torch.FloatTensor(input_pos)[None,...]
            r =repeats[b]
            bbs[new_i:new_i+r,0:len_input,:] = input_bbs
            tok_pos[new_i:new_i+r,0:len_input] = input_pos
            qa_mask[new_i:new_i+r,0:len_input] = 0
            new_i+=r
        bbs=bbs.to(device)
        xs=bbs[:,:,0]
        ys=bbs[:,:,1]
        ws=bbs[:,:,2]
        hs=bbs[:,:,3]
        qa_mask = qa_mask.to(device)
        doc_mask = 1-qa_mask

        input_emb = self.pos_1d_enc(input_emb,qa_mask) #This only applies the 1d position embedding to the q+a part of the input
        answer_padding_mask = (1-input_t['attention_mask'][:,:-1]).bool().to(device)

        input_emb += doc_mask*(self.pos_emb_x(xs) + self.pos_emb_y(ys) + self.pos_emb_w(ws) + self.pos_emb_h(hs) + self.word_pos_enc(tok_pos))

        locs = (input_t['input_ids']==self.SEP_TOKEN).nonzero(as_tuple=False)[::2,1] #get first SEP, not second
        att_mask = torch.BoolTensor(new_batch_size,input_len,input_len).fill_(1)
        for b,loc in enumerate(locs):
            att_mask[b,:loc+1,loc+1:]=0 #doc and question tokens cannot attend to answer tokens
            att_mask[b,loc+1:,loc+1:]=torch.tril(att_mask[b,loc+1:,loc+1:]) #causual attention for answer


        response = self.encoder(
                input_emb,#.permute(1,0,2),
                xs,
                ys,
                mask=att_mask.to(device), 
                src_key_padding_mask=answer_padding_mask,
                pos_mask=doc_mask
                )#.permute(1,0,2)



        response_decoded = self.answer_decode(response.reshape(-1,response.size(2)))
        response_decoded = response_decoded.view(new_batch_size,input_len,-1)


        #t#time = timeit.default_timer()-ticA#t#
        #t#self.opt_history['transformers'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        response_greedy_tokens = response_decoded.argmax(dim=2)
        locs = (input_t['input_ids']==self.SEP_TOKEN).nonzero(as_tuple=False)[::2,1] #get first SEP, not second
        mask_response = torch.ones(response_decoded.size())
        for b,loc in enumerate(locs):
            #for the question (before SEP)
            input_t['input_ids'][b,:loc]=0 #zero GT where question is (we don't want to train that)
            #response_decoded[b,:loc]*=0 #zero pred
            mask_response[b,:loc]*=0
        response_decoded = response_decoded*mask_response.to(device)
        target_decoded = input_t['input_ids'][:,1:]# This has the SEP tokens (and padding), but not CLS (start) token

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
