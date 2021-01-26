import gensim.downloader as api
from bpemb import BPEmb
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import defaultdict
try:
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    from transformers import AutoTokenizer, AutoModel
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
        #configuration = DistilBertConfig(max_position_embeddings=2048)
        self.languagemodel = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.SEP=102
        self.CLS=101
        self.max_len=512
        #self.languagemodel_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        #self.languagemodel = AutoModel.from_pretrained("distilroberta-base")
        #self.SEP=2

        self.languagemodel.train()
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
        if inputs['input_ids'].size(1)<=self.max_len:
            text_ends = (inputs['input_ids']==self.SEP).nonzero(as_tuple=True)[1] 
            #text_ends-=1

            inputs = {k:i.to(device) for k,i in inputs.items()}
            outputs = self.languagemodel(**inputs)
            outputs = outputs.last_hidden_state[0,text_ends] #we'll use the SEP token after each text as the location. It should be able to figure this out. Right?
            #we got rid of the batch, so text_ends creates the new batch dim
        else:
            #We need to split the input into batches
            #We'll ensure at least self.max_len//2 overlap between batches to preserve context
            #Just average overlap outputs (would it be better to only take one though?)

            #Form batches
            batch_ids=[]
            batch_mask=[]
            start=-self.max_len//2 +1 # start in neg to allow first add, +1 to skip CLS token
            starts=[]
            text_ends=[]
            prev_end=-1
            end=0
            while end<inputs['input_ids'].size(1):
                start+=self.max_len//2
                end=start+self.max_len-1
                if end<=inputs['input_ids'].size(1): #is this the last batch?
                    #ensure we stop on a SEP
                    while end>start+2 and inputs['input_ids'][0,end-1]!=self.SEP:
                        end-=1
                    if end==start+2: #shouldn't happen, but we'll split the really long text
                        end=start+self.max_len-1
                else:
                    #last batch
                    end = inputs['input_ids'].size(1)
                    start = end-(self.max_len-1)
                #ensure we start on a SEP
                while start<prev_end and inputs['input_ids'][0,start-1]!=self.SEP: 
                    start+=1
                if start==prev_end:
                    start = end-(self.max_len-1)
             
                ids = inputs['input_ids'][:,start:end]
                ids = F.pad(ids,(0,(self.max_len-1)-ids.size(1))) #ensure all batches are same length
                mask = inputs['attention_mask'][:,start:end]
                mask = F.pad(mask,(0,(self.max_len-1)-mask.size(1)))

                batch_ids.append(ids)
                batch_mask.append(inputs['attention_mask'][:,start:end])
                starts.append(start)
                text_ends.append((ids==self.SEP).nonzero(as_tuple=True)[1])
                prev_end=end

            num_batches = len(starts)
            
            #cat the batch together, and add start token
            input_ids = torch.cat(batch_ids,dim=0)
            start_token = torch.LongTensor(num_batches,1).fill_(self.CLS)
            input_ids = torch.cat([start_token,input_ids],dim=1).to(device)
            attention_mask = torch.cat(batch_ids,dim=0)
            first_mask = torch.LongTensor(num_batches,1).fill_(1)
            attention_mask = torch.cat([first_mask,attention_mask],dim=1).to(device)
            #batch = {'input_ids':torch.cat(batch_ids,dim=0).to(device), 'attention_mask':torch.cat(batch_mask,dim=0).to(device)}
            outputs = self.languagemodel(input_ids=input_ids,attention_mask=attention_mask)

            collected_outputs = defaultdict(lambda :0)
            collected_outputs_count = defaultdict(lambda :0)
            for b in range(num_batches):
                start=starts[b]
                output = outputs.last_hidden_state[b]
                for text_end in text_ends[b]:
                    l=int(text_end+start)
                    collected_outputs[l]+=output[text_end]
                    collected_outputs_count[l]+=1
            locations = list(collected_outputs.keys())
            locations.sort() #actually it should be sorted already
            #outputs = torch.FloatTensor(len(locations),self.hidden_size,device=device)
            outputs = output.new_empty(size=(len(locations),self.hidden_size))
            for i,l in enumerate(locations):
                outputs[i] = collected_outputs[l]/collected_outputs_count[l]

        emb = self.adaption(outputs)
        assert emb.size(0) == len(transcriptions)
        return emb
        


