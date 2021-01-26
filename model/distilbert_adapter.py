import gensim.downloader as api
from bpemb import BPEmb
import torch
import torch.nn as nn
import numpy as np
import re
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
except:
    pass

class DistilBertAdapter(nn.Module):
    def __init__(self,out_size):
        super(DistilBertAdapter, self).__init__()
        self.languagemodel_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.languagemodel = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.hidden_size = 768
        mid_size = (out_size+self.hidden_size)//2
        self.adaption = nn.Sequential(
                nn.Linear(self.hidden_size,mid_size),
                nn.ReLU(True),
                nn.Linear(mid_size,out_size),
                nn.ReLU(True),
        )

        self.size_limit=5000


    def forward(self,transcriptions,device):
        transcriptions = [t.strip() for t in transcriptions]
        #to save memory by preventing padding, we'll batch texts by their size. 
        #we'll have a max batch size (len x num)

        #sort
        sorttrans = [(i,trans) for i,trans in enumerate(transcriptions)]
        sorttrans.sort(key=lambda a:len(a[1]))
        allout=[]
        ti=0
        while ti<len(sorttrans):
            #create batch
            chars_in_batch=len(sorttrans[ti][1])
            max_len=len(sorttrans[ti][1])
            batch = [sorttrans[ti][1]]
            ti+=1
            while ti<len(sorttrans):
                #keep adding until it will overflow the size_limit
                max_len = max(len(sorttrans[ti][1]),max_len)
                chars_in_batch = (len(batch)+1)*max_len
                if chars_in_batch>self.size_limit:
                    break
                else:
                    batch.append(sorttrans[ti][1])
                    ti+=1

            #process batch
            inputs = self.languagemodel_tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k:i.to(device) for k,i in inputs.items()}
            outputs = self.languagemodel(**inputs)
            outputs = outputs.last_hidden_state[:,0] #take "pooled" node. DistilBert wasn't trained with the sentence task, but it probably still learned this node. Right?
            allout.append(outputs)
        #cat all results together and run linear layers
        emb = self.adaption(torch.cat(allout,dim=0))

        #put the embeddings batch back in the original order
        unsort = [None]*len(sorttrans)#[t[0] for t in sorttrans]
        for ii,(i,t) in enumerate(sorttrans):
            unsort[i]=ii
        emb=emb[unsort]
        assert transcriptions[0] == sorttrans[unsort[0]][1]
        #inputs = self.languagemodel_tokenizer(transcriptions, return_tensors="pt", padding=True)

        #inputs = {k:i.to(device) for k,i in inputs.items()}
        #outputs = self.languagemodel(**inputs)
        #outputs = outputs.last_hidden_state[:,0] #take "pooled" node. DistilBert wasn't trained with the sentence task, but it probably still learned this node. Right?
        #emb = self.adaption(outputs)
        return emb
        


#rather than processing each node/text at a time, this processes all the text at once, just putting seperators between them
#W
class DistilBertWholeAdapter(nn.Module):
    def __init__(self,out_size):
        super(DistilBertWholeAdapter, self).__init__()
        self.languagemodel_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        configuration = DistilBertConfig(max_position_embeddings=2048)
        self.languagemodel = DistilBertModel(configuration).from_pretrained('distilbert-base-uncased')
        self.hidden_size = 768
        mid_size = (out_size+self.hidden_size)//2
        self.adaption = nn.Sequential(
                nn.Linear(self.hidden_size,mid_size),
                nn.ReLU(True),
                nn.Linear(mid_size,out_size),
                nn.ReLU(True),
        )



    def forward(self,transcriptions,device):
        transcriptions = [t.strip() for t in transcriptions]
        alltrans = transcriptions[0]
        for t in transcriptions[1:]:
            alltrans+=self.languagemodel_tokenizer.sep_token+t
        inputs = self.languagemodel_tokenizer(alltrans, return_tensors="pt", padding=True)
        text_ends = (inputs['input_ids']==102).nonzero(as_tuple=True)[1] #102 is the [SEP] encoding
        #text_ends-=1

        inputs = {k:i.to(device) for k,i in inputs.items()}
        outputs = self.languagemodel(**inputs)
        outputs = outputs.last_hidden_state[0,text_ends] #we'll use the SEP token after each text as the location. It should be able to figure this out. Right?
        #we got rid of the batch, so text_ends creates the new batch dim
        emb = self.adaption(outputs)
        assert emb.size(0) == len(transcriptions)
        return emb
        


