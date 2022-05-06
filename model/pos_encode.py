import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000,offset_start=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(offset_start, offset_start+max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x,mask=None,offset=0):
        if mask is None:
            x = x + self.pe[:, offset:offset+x.size(1)] 
        else:
            x = x + self.pe[:, offset:offset+x.size(1)]*mask 
        return self.dropout(x)

class ReturnPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000,offset_start=0):
        super(ReturnPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(offset_start, offset_start+max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self,x):
        x=self.pe[x]
        return self.dropout(x)

class ReturnPositionalEncodingSeq(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000,offset_start=0):
        super(ReturnPositionalEncodingSeq, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(offset_start, offset_start+max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self,length):
        x=self.pe[:length]
        return self.dropout(x)[None,...] #add batch dim

class PositiveRealEmbedding(nn.Module):
    "Embeds a real-value, with higher resolution on smaller values"
    def __init__(self,dim,min_v,max_v,resolution):
        super(PositiveRealEmbedding,self).__init__()
        self.linear = nn.Linear(resolution,dim)
        self.min_v = min_v
        self.max_v = max_v
        assert min_v >= 0

        self.division = torch.FloatTensor(resolution).fill_(1) / ((max_v-min_v)/torch.arange(resolution,0,-1))

        self.division = self.division[None,...]

    def forward(self, x):
        x_shape = x.size()
        x=x.view(-1,1)
        broken = (x-self.min_v)*self.division.to(x.device)
        broken = torch.clamp(broken,0,1) #negative values set to 0
        res = self.linear(broken)
        #res *= 1-torch.isnan(res)
        #res = torch.nan_to_num(res,0)
        new_shape = x_shape+(res.size(-1),)
        return res.view(new_shape)

class RealEmbedding(nn.Module):
    "a zero centered real-value embedding"
    def __init__(self,dim,max_mag,resolution):
        super(RealEmbedding,self).__init__()
        self.positive = PositiveRealEmbedding(dim,0,max_mag,resolution)
        self.negative = PositiveRealEmbedding(dim,0,max_mag,resolution)

    def forward(self,x):
        return (self.positive(x) + self.negative(-x))/2


class UniformRealEmbedding(nn.Module):
    "Embeds a real-value"
    def __init__(self,dim,min_v,max_v,resolution):
        super(UniformRealEmbedding,self).__init__()
        self.min_v = min_v
        self.max_v = max_v
        dist = max_v-min_v
        chunk = dist/resolution
        self.range1 = torch.arange(0,resolution).float()
        self.range2 = torch.arange(0,resolution+1).float()-0.5
        self.linear1 = nn.Linear(resolution,dim)
        self.linear2 = nn.Linear(resolution+1,dim)

        self.range1 = nn.Parameter(self.range1[None,...],requires_grad=False)
        self.range2 = nn.Parameter(self.range2[None,...],requires_grad=False)

    def forward(self, x):
        x_shape = x.size()
        x=x.view(-1,1)-self.min_v
        p1 = 1-torch.abs(self.range1-x)
        p1[p1<0]=0
        p2 = 1-torch.abs(self.range2-x)
        p2[p2<0]=0
        res = self.linear1(p1) + self.linear2(p2)
        #res *= 1-torch.isnan(res)
        #res = torch.nan_to_num(res,0)
        new_shape = x_shape+(res.size(-1),)
        return res.view(new_shape)

#From Huggingface transformers code
class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)
