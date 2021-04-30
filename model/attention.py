import math,copy
import torch
import torch.nn.functional as F
from torch import nn
from .pos_encode import RealEmbedding

#These are taken from the Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention)
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, key_padding_mask=None, dropout=None,fixed=False,att_bias=None):
    #print(query.size()) #batch,heads,len,feats
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    #bsz, num_heads, src_len,d_k = key.size()
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    ###
    #scores.fill_(0.1)
    ###
    if att_bias is not None:
        scores+=att_bias
    if mask is not None:
        if torch.is_autocast_enabled():
            scores = scores.masked_fill(mask == 0, -1e4)
        else:
            scores = scores.masked_fill(mask == 0, -1e9)
    if key_padding_mask is not None:
        #tgt_len = query.size(2)
        #scores = scores.view(bsz, num_heads, tgt_len, src_len)
        scores = scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        #scores = scores.view(bsz * num_heads, tgt_len, src_len)

    p_attn = F.softmax(scores, dim = -1)
    
    if mask is not None and fixed:
        p_attn = p_attn.masked_fill(mask == 0, 0) #this is needed in casa node has no neigbors
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
        self.fixed= 'fixed' in self.mod
        
        
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
                                     dropout=self.dropout,fixed=self.fixed)
        elif self.learned:
            x, self.attn = learned_attention(query, key, value, mask=mask, 
                                     dropout=self.dropout,network=self.attNet)
        else:
            x, self.attn = attention(query, key, value, mask=mask, 
                                     dropout=self.dropout,fixed=self.fixed)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)




class PosBiasedMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, max_dist, dropout=0.1, mod=None):
        "Take in model size and number of heads."
        super(PosBiasedMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.pos_d_k = (d_model//4) // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) #W_q W_k W_v W_o
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.mod=mod if mod else '' #learned: use network for attention instead of dot product, half: use only half of query/keys for dot product

        assert d_model//4>=16
        self.x_emb = RealEmbedding(d_model//4,max_dist,20)
        self.y_emb = RealEmbedding(d_model//4,max_dist,20)
        self.bias_net = nn.Sequential(
                nn.Linear(self.pos_d_k,1),
                nn.Sigmoid()
                )
        
        
    def forward(self, query, key, value, query_x, query_y, key_x, key_y, mask=None, key_padding_mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask[None,None,...]#mask.unsqueeze(1)
        nbatches = query.size(0)

        nquery = query.size(1)
        nkey = key.size(1)
        x_diff = query_x[:,None,:].expand(-1,nkey,-1) - key_x[:,:,None].expand(-1,-1,nquery)
        y_diff = query_y[:,None,:].expand(-1,nkey,-1) - key_y[:,:,None].expand(-1,-1,nquery)
        pos_emb = self.x_emb(x_diff) + self.y_emb(y_diff)
        pos_emb = pos_emb.reshape((nbatches*nkey*nquery*self.h),self.pos_d_k) #reshape to heads
        att_bias = self.bias_net(pos_emb)
        att_bias = att_bias.view(nbatches,nkey,nquery,self.h).permute(0,3,1,2)
        #att_bias = torch.matmul(pos_emb, pos_emb.transpose(-2, -1)) \
        #         / math.sqrt(self.pos_d_k)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, key_padding_mask=key_padding_mask,
                                     dropout=self.dropout,fixed=True,att_bias=att_bias)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
