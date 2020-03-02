import gensim.downloader as api
from bpemb import BPEmb
import torch
import torch.nn as nn
import numpy as np
import re

debug=False
if debug:
    wordmodel = 'glove-wiki-gigaword-100'
    vecsize = 300
    from collections import defaultdict
else:
    wordmodel = 'word2vec-google-news-300'
    vecsize = 300

class Word2VecAdapter(nn.Module):
    def __init__(self,out_size):
        super(Word2VecAdapter, self).__init__()
        self.wv = api.load(wordmodel)
        self.adaption1 = nn.Sequential(
                nn.Linear(vecsize,256),
                nn.ReLU(True),
                nn.Linear(256,256),
                nn.ReLU(True),
                )
        self.adaption2 = nn.Sequential(
                nn.Linear(256,256),
                nn.ReLU(True),
                nn.Linear(256,out_size),
                nn.ReLU(True),
                )

    def forward(self,transcriptions):

        words = [t.strip().split(' ') for t in transcriptions]
        wordCounts = [len(w) for w in words]
        allWords = []
        for ws in words:
            allWords += ws
        emb = np.cat([self.wv[w] for w in allWords],axis=0)
        emb = []
        for w in allWords:
            try:
                emb.append(self.wv[w])
            except KeyError:
                pass
        emb = torch.from_numpy(np.concatenate(emb)).to(self.adaption1.device)
        emb = self.adaption1(emb)
        c=0
        for count in wordCounts:
            embSent.append(emb[c:c+count].mean(dim=0))
            c+=count
        emb = torch.cat(embSent,dim=0)
        emb = self.adaption2(emb)
        return emb
        


class Word2VecAdapterShallow(nn.Module):
    def __init__(self,out_size):
        super(Word2VecAdapterShallow, self).__init__()
        if debug:
            self.wv = defaultdict(lambda : np.zeros(vecsize))
        else:
            self.wv = None#api.load(wordmodel)
        self.adaption = nn.Sequential(
                nn.Linear(vecsize,256),
                nn.ReLU(True),
                nn.Linear(256,out_size),
                nn.ReLU(True),
                )

    def forward(self,transcriptions):
        if len(transcriptions)==0:
            return torch.FloatTensor(0).to(self.adaption[0].weight.device)
        if self.wv is None:
            self.wv = api.load(wordmodel) #lazy
        emb=[]
        for t in transcriptions:
            vector = np.zeros(vecsize)
            count=0
            for w in t.strip().split(' '):
                w=re.sub(r'[^\w\s]','',w) #remove puncutation
                try:
                    vector+=self.wv[w]
                    count+=1
                except KeyError:
                    #Sometimes we needs '#' inplace of digits
                    w=re.sub(r'\d','#',w)
                    try:
                        vector+=self.wv[w]
                        count+=1
                    except KeyError:
                        pass
            if count>0:
                vector/=count
            emb.append(vector)
        emb = torch.from_numpy(np.stack(emb,axis=0)).float().to(self.adaption[0].weight.device)
        assert(not torch.isnan(emb).any())
        return self.adaption(emb)

        



class BPEmbAdapter(nn.Module):
    def __init__(self,out_size):
        super(BPEmbAdapter, self).__init__()
        self.embedder = None#api.load(wordmodel)
        self.vecsize=100
        #self.convs = nn.Sequential(
        #        nn.Conv1d(self.vecsize,256,kernel=3,padding=1)
        #        nn.ReLU(True),
        #        nn.Conv1d(256,256,kernel=3,padding=1),
        #        nn.ReLU(True),
        #        )
        self.rnn = nn.LSTM(100,128,2,dropout=0.2,bidirectional=True)
        self.linear = nn.Sequential(
                nn.Linear(256,out_size),
                nn.Dropout(0.1),
                nn.ReLU(True)
                )

    def forward(self,transcriptions):
        if self.embedder is None:
            self.embedder = BPEmb(lang="en",vs=100000)
        emb=[]
        max_len=0
        for t in transcriptions:
            #t=re.sub(r'[^\w\s]','',t) #remove puncutation, uhh it seems to have puncutation...
            t=re.sub(r'\d','0',t)
            bps = self.embedder.encode(t)
            vectors=[]
            for w in bps:
                try:
                    vectors.append(self.embedder[w])
                except KeyError:
                    vectors.append( np.zeros(self.vecsize) )
            max_len = max(max_len,len(vectors))
            if len(vectors)>0:
                vectors = np.stack(vectors,axis=0)
            else:
                #print('no embedding for: {}'.format(t))
                vectors = np.zeros((1,self.vecsize))
            emb.append(vectors)
        #pad all sequences to same length
        for i in range(len(emb)):
            if emb[i].shape[0]<max_len:
                diff = max_len-emb[i].shape[0]
                emb[i] = np.pad(emb[i],((0,diff),(0,0)))
        emb = torch.from_numpy(np.stack(emb,axis=1)).float().to(self.rnn._flat_weights[0].device)
        assert(not torch.isnan(emb).any())
        emb,_ = self.rnn(emb)
        emb = emb.mean(dim=0) #ehh, just average
        return self.linear(emb)
