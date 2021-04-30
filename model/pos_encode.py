import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)

class PositiveRealEmbedding(nn.Module):
    "Embeds a real-value"
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
