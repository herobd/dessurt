import gensim.downloader as api
import torch
import torch.nn as nn
import numpy as np
import re

debug=False
if debug:
    wordmodel = 'glove-wiki-gigaword-100'
    vecsize = 100
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
            self.wv = api.load(wordmodel)
        self.adaption = nn.Sequential(
                nn.Linear(vecsize,256),
                nn.ReLU(True),
                nn.Linear(256,out_size),
                nn.ReLU(True),
                )

    def forward(self,transcriptions):
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
            vector/=count
            emb.append(vector)
        emb = torch.from_numpy(np.stack(emb,axis=0)).float().to(self.adaption[0].weight.device)
        return self.adaption(emb)

        


