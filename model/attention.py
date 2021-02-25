import math,copy
import torch
import torch.nn.functional as F
from torch import nn

#These are taken from the Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention)
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    ###
    #scores.fill_(0.1)
    ###
    if mask is not None:
        if torch.is_autocast_enabled():
            scores = scores.masked_fill(mask == 0, -1e4)
        else:
            scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if mask is not None:
        p_attn = scores.masked_fill(mask == 0, 0) #this is needed in casa node has no neigbors
        #will create a zero vector in those cases, instead of an average of all nodes
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
def learned_attention(query, key, value, mask=None, dropout=None,network=None):
    "Compute Attention using provided network"

    #naive "everywhere" implmenetation
    assert(len(query.size())==4)
    batch_size = query.size(0)
    heads = query.size(1)
    query_ex = query[:,:,:,None,:].expand(-1,-1,query.size(2),key.size(2),-1)
    key_ex = key[:,:,None,:,:].expand(-1,-1,query.size(2),key.size(2),-1)
    comb = torch.cat((query_ex,key_ex),dim=4)
    comb = comb.view(batch_size*heads,-1,comb.size(-1))
    scores = network(comb) #same function for each head
    scores = scores.view(batch_size,heads,query.size(2),key.size(2))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)
    if mask is not None:
        p_attn = scores.masked_fill(mask == 0, 0) #this is needed in casa node has no neigbors
        #will create a zero vector in those cases, instead of an average of all nodes
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, mod=None):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) #W_q W_k W_v W_o
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.mod=mod if mod else '' #learned: use network for attention instead of dot product, half: use only half of query/keys for dot product
        if 'learned' in self.mod:
            self.learned=True
            self.attNet = nn.Sequential(
                    #nn.GroupNorm(getGroupSize(self.d_k*2),self.d_k*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_k*2,self.d_k//4),
                    #nn.GroupNorm(getGroupSize(self.d_k//4),self.d_k//4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_k//4,1) 
                    )
        else:
            self.learned=False
        self.half = 'half' in self.mod
        self.none = 'none' in self.mod
        
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask[None,None,...]#mask.unsqueeze(1)
        nbatches = query.size(0)

        if self.none:
            key = torch.cat((key,torch.ones(key.size(0),1,key.size(2)).to(key.device)),dim=1)
            value = torch.cat((value,torch.zeros(value.size(0),1,value.size(2)).to(value.device)),dim=1)
            mask = torch.cat((mask,torch.ones(1,1,mask.size(2),1).to(mask.device)),dim=3)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        if self.half:
            x, self.attn = attention(query[...,:self.d_k//2], key[...,:self.d_k//2], value, mask=mask, 
                                     dropout=self.dropout)
        elif self.learned:
            x, self.attn = learned_attention(query, key, value, mask=mask, 
                                     dropout=self.dropout,network=self.attNet)
        else:
            x, self.attn = attention(query, key, value, mask=mask, 
                                     dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
