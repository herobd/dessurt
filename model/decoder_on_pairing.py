from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_g_graph_layoutlm import PairingGGraphLayoutLM



class DecoderOnPairing(BaseModel):
    def __init__(self):
        self.pairing_model = None

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers,nn.LayerNorm(d_model))
        self.answer_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.answer_decode = nn.Sequential(
                nn.Linear(d_model,self.tokenizer.vocab_size),
                nn.Softmax(dim=-1)
                )

    def forward(image,gtBBs,gtTrans,questions,answers=None):
        layoutlm_feats = runLayoutLM(image.size(),gtBBs,gtTrans,image.device,self.layoutlm_tokenizer,self.layoutlm,keep_ends=True)

        if self.pairing_model is not None:
            with torch.no_grad(): #I'm going to freeze it
                graph_feats = self.pairing_model(image,gtBBs=gtBBs,useGTBBs=True,gtTrans=gtTrans,return_feats=True)
            graph_feats = self.change_graph(graph_feats)
            TODO position encoding, at least for graph

            document_feats = torch.cat((layoutlm_feats,graph_feats),dim=2)#place the graph_feats between so it gets put between the SEP and START tokens
        else:
            document_feats = layoutlm_feats
        


        layoutlm_feats =self.change_layoutlm(layoutlm_feats)
        Maybe pos encode layoutlm as well, to account for times its been processed in two batches

        for qi,question in enumerate(questions):
            answer = answers[qi] is answers is not None else None

            inputs = self.question_tokenizer(question, return_tensors="pt", padding=True)
            inputs = {k:i.to(device) for k,i in inputs.items()}
            question_feats = self.question_languagemodel(**inputs).last_hidden_state
            question_feats = self.change_question(question_feats)


            memory_feats = torch.cat((document_feats,question_feats),dim=2) #place the graph_feats between so it gets put between the SEP and START tokens

            memory_feats = self.encoder(memory_feats)

            if answer is not None: #we are training
                answer = self.tokenizer(answer,return_tensors="pt")['input_ids']
                answer_emb = self.answer_embedding(answer)
                response = self.decoder(
                        answer_emb,
                        memory_feats,
                        tgt_mask=self.decoder.generate_square_subsequent_mask(answer.size(0))
                        )
                        #We don't need padding as we only deal with batches of size 1
                        #memory_mask=None,
                        #tgt_key_padding_mask=question_padding_mask,
                        #memory_key_padding_mask=memory_padding_mask)
            else: #we need to do this autoregressively
                p_answer = torch.FloatTensor(1,1).fill_(self.CLS_TOKEN).to(memory_feats.device)
                p_answer_emb = self.answer_embedding(p_answer)
                while True:
                    p_response = self.decoder(
                            p_answer_emb,
                            memory_feats)
                    p_decoded = self.answer_decode(p_response)
                    last_token = p_decoded[-1,0].argmax() #last token prob, batchsize of 1, greedy
                    last_token = last_token[None,None]
                    last_token_emb = self.answer_embedding(last_token)
                    p_answer_emb = torch.cat((p_answer_emb,last_token_emb),dim=-1)


