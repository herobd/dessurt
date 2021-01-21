import gensim.downloader as api
from bpemb import BPEmb
import torch
import torch.nn as nn
import numpy as np
import re
try:
    from transformers import DistilBertTokenizer, DistilBertModel
except:
    pass

class DistilBertAdapter(nn.Module):
    def __init__(self,out_size):
        super(DistilBertAdapter, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.languagemodel = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.hidden_size = 768
        mid_size = (out_size+self.hidden_size)//2
        self.adaption = nn.Sequential(
                nn.Linear(self.hidden_size,mid_size),
                nn.ReLU(True),
                nn.Linear(mid_size,out_size),
                nn.ReLU(True),
                )

    def forward(self,transcriptions):

        transcriptions = [t.strip() for t in transcriptions]
        inputs = tokenizer(transcriptions, return_tensors="pt", padding=True)
        outputs = self.languagemodel(inputs)
        outputs = .last_hidden_state.size()[:,0] #take "pooled" node. DistilBert wasn't trained with the sentence task, but it probably still learned this node. Right?
        emb = self.adaption(outputs)
        return emb
        


