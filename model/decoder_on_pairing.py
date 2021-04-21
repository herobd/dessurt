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


class DecoderOnPairing(BaseModel):
    def __init__(self,config):
        super(DecoderOnPairing, self).__init__(config)
        d_model = config['decode_dim']
        dim_ff = config['dim_ff']
        nhead = config['decode_num_heads']
        num_layers = config['decode_layers']
        num_e_layers = config['encode_layers']
        self.pairing_model = None

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.SEP_TOKEN= 102
        self.CLS_TOKEN= 101

        self.regress_from_question = config['regress_from_question'] if 'regress_from_question' in config else False
        if not self.regress_from_question:
            self.question_languagemodel = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.change_question = nn.Linear(768,d_model)

    
        decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_ff)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers,nn.LayerNorm(d_model))
        self.answer_embedding = nn.Sequential(
                nn.Embedding(self.tokenizer.vocab_size, d_model),
                PositionalEncoding(d_model,dropout=0.1,max_len=1000)
                )
        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.tokenizer.vocab_size),
                nn.LogSoftmax(dim=-1) #except
                )
        if num_e_layers>0:
            encoder_layer= nn.TransformerEncoderLayer(d_model,nhead,dim_ff)
            self.encoder = nn.TransformerEncoder(encoder_layer,num_e_layers,nn.LayerNorm(d_model))
        else:
            self.encoder = None

        if 'layout' not in config:
            self.layoutlm_tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            self.layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
            self.change_layoutlm = nn.Linear(768,d_model)
        else:
            config['layout']['d_model']=d_model
            config['layout']['dim_ff']=dim_ff
            self.layout_model = LayoutTransformer(config['layout'])
            self.layoutlm = None
            self.change_layoutlm = nn.Identity()

        self.half_mem_pos = config['half_mem_pos'] if 'half_mem_pos' in config else False
        self.no_mem_pos = config['no_mem_pos'] if 'no_mem_pos' in config else False
        if self.half_mem_pos:
            self.mem_pos_enc = PositionalEncoding(d_model//2,dropout=0.1,max_len=5000)
        elif not self.no_mem_pos:
            self.mem_pos_enc = PositionalEncoding(d_model,dropout=0.1,max_len=5000)

        self.doc_skip = config['doc_skip'] if 'doc_skip' in config else False
        self.q_skip = config['q_skip'] if 'q_skip' in config else False
        if self.doc_skip:
            self.skip_embedding = nn.Embedding(self.layoutlm_tokenizer.vocab_size, d_model)
            if self.q_skip:
                self.skip_q_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        else:
            assert not self.q_skip

        self.only_q = config['only_q'] if 'only_q' in config else False

        #def show_self_att(m,i,o):
        #    print('self attn in:')
        #    print(i)
        #    print('self attn out:')
        #    print(o)
        #def show_mem_att(m,i,o):
        #    print('self mem in:')
        #    print(i)
        #    print('self mem out:')
        #    print(o)
        #self.decoder.layers[0].self_attn.register_forward_hook(show_self_att)
        #self.decoder.layers[0].multihead_attn.register_forward_hook(show_mem_att)



    def forward(self,image,gtBBs,gtTrans,questions,answers=None):
        device = image.device
        if not self.only_q:
            if self.layoutlm is not None:
                layoutlm_feats = runLayoutLM(image.size(),gtBBs[0],gtTrans,image.device,self.layoutlm_tokenizer,self.layoutlm,keep_ends=True,all_tokens=True) #this assumes a single batch
            else:
                layoutlm_feats = self.layout_model(image.size(),gtBBs[0],gtTrans,image.device)

            layoutlm_feats =self.change_layoutlm(layoutlm_feats)
            #Maybe pos encode layoutlm as well, to account for times its been processed in two batches
            #acutally, it encodes x,y poisition...

            if self.pairing_model is not None:
                with torch.no_grad(): #I'm going to freeze it
                    graph_feats, node_cords = self.pairing_model(image,gtBBs=gtBBs,useGTBBs=True,gtTrans=gtTrans,return_feats=True)
                graph_feats = self.change_graph(graph_feats)
                #TODO position encoding, at least for graph
                graph_feats = self.graph_pos_enc(graph_feats,node_cords)

                document_feats = torch.cat((layoutlm_feats,graph_feats),dim=2)#place the graph_feats between so it gets put between the SEP and START tokens
            else:
                document_feats = layoutlm_feats
        


        if self.regress_from_question:
            document_feats_len = document_feats.size(0)
            document_feats = document_feats[None,...].expand(len(questions),-1,-1)
            memory_feats = document_feats
            memory_padding_mask = None#torch.BoolTensor(len(questions),document_feats_len).zero_().to(device)
        else:
            q_inputs = self.tokenizer(questions, return_tensors="pt", padding=True)
            q_inputs = {k:i.to(device) for k,i in q_inputs.items()}
            question_feats = self.question_languagemodel(**q_inputs).last_hidden_state
            question_feats = self.change_question(question_feats)
            if self.only_q:
                memory_feats = question_feats
                memory_padding_mask = ~q_inputs['attention_mask'].bool()
            else:
                document_feats_len = document_feats.size(0)
                document_feats = document_feats[None,...].expand(len(questions),-1,-1)
                sep_feat = torch.FloatTensor(len(questions),1,document_feats.size(2)).zero_().to(device)
                memory_feats = torch.cat((document_feats,sep_feat,question_feats),dim=1)
                memory_padding_mask = torch.cat((torch.BoolTensor(len(questions),document_feats_len+1).zero_().to(device),~q_inputs['attention_mask'].bool()),dim=1)

                if self.half_mem_pos:
                    memory_feats_h = self.mem_pos_enc(memory_feats[:,:,:memory_feats.size(2)//2])
                    memory_feats = torch.cat([memory_feats_h,memory_feats[:,:,memory_feats.size(2)//2:]],dim=2)
                elif not self.no_mem_pos:
                    memory_feats = self.mem_pos_enc(memory_feats)
            memory_feats = memory_feats.permute(1,0,2) #batch,len,feat -> len,batch,feat
        
            if self.encoder is not None:
                memory_feats = self.encoder(
                        memory_feats,
                        src_key_padding_mask=memory_padding_mask)

            if self.doc_skip:
                total_string = ' '.join(gtTrans)
                doc_tok = self.layoutlm_tokenizer(total_string,return_tensors="pt")
                assert self.pairing_model is None
                doc_emb = self.skip_embedding(doc_tok['input_ids'].to(device))
                doc_emb = doc_emb.expand(len(questions),-1,-1)
                doc_emb = doc_emb.permute(1,0,2)
                if self.q_skip:
                    q_emb = self.skip_q_embedding(q_inputs['input_ids'].to(device)).permute(1,0,2)
                    doc_emb = torch.cat([doc_emb,sep_feat, q_emb],dim=0)
                else:
                    diff_len = memory_feats.size(0)-doc_emb.size(0)
                    doc_emb = torch.cat([doc_emb, torch.zeros(diff_len,len(questions),doc_emb.size(2)).to(device)],dim=0)
                memory_feats += doc_emb



        if answers is not None: #we are training
            if self.regress_from_question:
                answers = [q+'[SEP]'+a for q,a in zip(questions,answers)]
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
            #response = answers_emb.contiguous()
            #response[0,0,0]=1
            #response[0,0,1]=0
            #response[0,1,0]=0
            #response[0,1,1]=1
            #response[1,0,0]=1
            #response[1,0,1]=0
            #response[1,1,0]=0
            #response[1,1,1]=1
            #response = (answers_emb+question_feats[:,1][None,...]).contiguous()

            response_decoded = self.answer_decode(response.view(-1,response.size(2)))
            response_decoded = response_decoded.view(answers_emb.size(0),len(questions),-1)
            response_decoded = response_decoded.permute(1,0,2) #put batch dim first
            #target_mask = answer_padding_mask[:,1:]
            response_greedy_tokens = response_decoded.argmax(dim=2)

            if self.regress_from_question:
                locs = (answers_t['input_ids']==self.SEP_TOKEN).nonzero(as_tuple=False)[::2,1] #get first SEP, not second
                for b,loc in enumerate(locs):
                    #for the question (before SEP)
                    answers_t['input_ids'][b,:loc]=0 #zero GT
                    #response[:loc,b]*=0 #zero pred
            target_decoded = answers_t['input_ids'][:,1:]# This has the SEP tokens (and padding), but not CLS (start) token

            string_response=[]
            for b in range(len(questions)):
                response_greedy_tokens_b = response_greedy_tokens[b]
                if self.regress_from_question:
                    response_greedy_tokens_b = response_greedy_tokens_b[locs[b]:]
                pred_stop = response_greedy_tokens_b==self.SEP_TOKEN
                if pred_stop.any():
                    stop_index = pred_stop.nonzero(as_tuple=False)[0][0].item()
                    response_greedy_tokens_b[stop_index:]=self.SEP_TOKEN
                string_response.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(response_greedy_tokens[b],skip_special_tokens=True)))
            return response_decoded, target_decoded.to(device), string_response
        else: #we need to do this autoregressively
            #This is not efficeint
            TODO() #how do I freeze the already run activations?
            p_answer_t = torch.LongTensor(len(questions),1).fill_(self.CLS_TOKEN).to(memory_feats.device)
            p_answer_emb = self.answer_embedding(p_answer_t)
            p_answers_emb = p_answers_emb.permute(1,0,2) #batch,len,feat -> len,batch,feat
            cont=True
            while cont:
                p_response = self.decoder(
                        p_answer_emb,
                        memory_feats_b,
                        memory_key_padding_mask=memory_padding_mask)
                p_decoded = self.answer_decode(p_response)
                last_token = p_decoded[-1].argmax(dim=2) #last token prob,  greedy
                if (last_token== self.SEP_TOKEN).all():
                    cont=False
                last_token = last_token[None,None]
                last_token_emb = self.answer_embedding(last_token,pos=p_answer_emb.size(-1))
                p_answer_emb = torch.cat((p_answer_emb,last_token_emb),dim=-1)

            #response_d = ?