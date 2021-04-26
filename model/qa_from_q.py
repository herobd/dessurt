from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import  runLayoutLM
from model.pos_encode import PositionalEncoding
from model.layout_transformer import LayoutTransformer
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import LayoutLMTokenizer, LayoutLMModel
from collections import defaultdict

import timeit


class QAFromQ(BaseModel):
    def __init__(self,config):
        super(QAFromQ, self).__init__(config)
        d_model = config['decode_dim']
        dim_ff = config['dim_ff']
        nhead = config['decode_num_heads']
        num_layers = config['decode_layers']
        share_embeddings = config['share_embeddings'] if 'share_embeddings' in config else False

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.SEP_TOKEN= 102
        self.CLS_TOKEN= 101

        self.regress_from_question = config['regress_from_question'] if 'regress_from_question' in config else False
        dropout = 0 if 'no_dropout' in config and  config['no_dropout'] else 0.1
        decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_ff,dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers,nn.LayerNorm(d_model))


        #config['layout']['d_model']=d_model
        #config['layout']['dim_ff']=dim_ff
        num_e_layers = config['layout']['num_layers']
        #self.layout_model = LayoutTransformer(config['layout'],dropout)
        if share_embeddings:
            self.doc_tokenizer = self.tokenizer
        else:
            self.doc_tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.doc_embedding =  nn.Sequential(
                nn.Embedding(self.doc_tokenizer.vocab_size, d_model),
                PositionalEncoding(d_model,dropout=dropout,max_len=5000)
                )
        encoder_layer= nn.TransformerEncoderLayer(d_model,nhead,dim_ff,dropout=dropout)
        self.doc_encoder = nn.TransformerEncoder(encoder_layer,num_e_layers,nn.LayerNorm(d_model))

        if share_embeddings:
            self.answer_embedding = self.doc_embedding
        else:
            self.answer_embedding = nn.Sequential(
                    nn.Embedding(self.tokenizer.vocab_size, d_model),
                    PositionalEncoding(d_model,dropout=dropout,max_len=1000)
                    )
        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.tokenizer.vocab_size,bias=not share_embeddings),
                nn.LogSoftmax(dim=-1)
                )
        if share_embeddings:
            self.answer_decode[0].weight = self.doc_embedding[0].weight #num,dim
        #t#self.opt_history=defaultdict(list)#t#


    def forward(self,image,gtBBs,gtTrans,questions,answers=None):
        #t#ticA=timeit.default_timer()#t#
        device = image.device


        #layoutlm_feats,layout_padding = self.layout_model(image.size(),gtBBs[0],gtTrans,image.device)

        #Put document through the encoder
        total_strings = [' '.join(gtT) for gtT in gtTrans]
        inputs = self.doc_tokenizer(total_strings,return_tensors="pt",padding=True)
        embedded = self.doc_embedding(inputs['input_ids'].to(device))
        embedded = embedded.permute(1,0,2) #batch,len,feat -> len,batch,feat
        padding_mask = ~inputs['attention_mask'].bool().to(device)
        document_feats = self.doc_encoder(embedded,src_key_padding_mask=padding_mask).permute(1,0,2)
        document_padding = padding_mask


        #repeat output to number of questions
        repeats = [len(q) for q in questions] #different elements in batch may have different numbers of questions
        document_feats = torch.repeat_interleave(document_feats,torch.tensor(repeats).to(device),dim=0)
        document_padding = torch.repeat_interleave(document_padding,torch.tensor(repeats).to(device),dim=0)
        questions=[q for bq in questions for q in bq]
        answers=[a for ba in answers for a in ba]

        document_feats_len = document_feats.size(1)
        #document_feats = document_feats[None,...].expand(len(questions),-1,-1)
        memory_feats = document_feats
        memory_padding_mask = document_padding# torch.BoolTensor(len(questions),document_feats_len).zero_().to(device)


        #Append question before answer
        answers = [q+'[SEP]'+a for q,a in zip(questions,answers)]

        #run question+answer through decoder
        answers_t = self.tokenizer(answers,return_tensors="pt",padding=True)
        answers_emb = self.answer_embedding(answers_t['input_ids'][:,:-1].to(device)) #Remove end (SEP) token, as it doesn't need to predict anythin after that. emb needs to do position
        answers_emb = answers_emb.permute(1,0,2) #batch,len,feat -> len,batch,feat
        answer_padding_mask = (1-answers_t['attention_mask'][:,:-1]).bool().to(device)
        response = self.decoder(
                answers_emb,
                memory_feats, 
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(None,answers_emb.size(0)).to(device),
                tgt_key_padding_mask=answer_padding_mask,
                memory_key_padding_mask=memory_padding_mask
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
