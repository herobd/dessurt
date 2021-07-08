import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRPooler(nn.Module):

    def __init__(self,d_model):
        super(OCRPooler, self).__init__()
        self.conv = nn.Conv1d(d_model,d_model,kernel_size=4,stride=2,padding=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=2,stride=2,count_include_pad=False)
        self.max_pool = nn.MaxPool1d(kernel_size=2,stride=2)

    def forward(self,tokens,pos,padding_mask):
        #To prevent the pos being averaged with the end zeros (for odd length)
        #get the location of first pad
        batch_size = tokens.size(0)

        #pad_first = torch.arange(batch_size)[None,:].expand(2,-1).clone().to(padding_mask.device)
        pad_first = []
        for b in range(batch_size):
            if (padding_mask[b]!=0).any():
                #pad_first[1,b] = padding_mask[b].nonzero(as_tuple=False)[0]
                pad_first.append((b,padding_mask[b].nonzero(as_tuple=False)[0]))
        if len(pad_first)>0:
            pad_first = torch.LongTensor(pad_first)
            
            pad_first_minus=pad_first[1,:]-1
            pos[pad_first[0,:],pad_first[1,:]]=pos[pad_first[0,:],pad_first_minus]

        tokens = self.conv(tokens.permute(0,2,1))
        pos = self.avg_pool(pos.permute(0,2,1))
        padding_mask = self.max_pool(padding_mask[:,None].float())>0

        return tokens.permute(0,2,1), pos.permute(0,2,1), padding_mask[:,0]

class QPooler(nn.Module):

    def __init__(self,d_model):
        super(QPooler, self).__init__()
        self.conv = nn.Conv1d(d_model,d_model,kernel_size=4,stride=2,padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2,stride=2)

    def forward(self,tokens,padding_mask):
        tokens = self.conv(tokens.permute(0,2,1))
        padding_mask = self.max_pool(padding_mask[:,None].float())>0

        return tokens.permute(0,2,1), padding_mask[:,0]


#class AttentionPrunning:#(nn.Module):
#    def __init__(self,size):
#        self.size=size
#
#    # tokens:   (batch,length,dim)
#    # pos:      (batch,length,2)
#    # attention:(length) 
#    def forward(self,tokens,pos,attention,size):
#        #tokens to 2d
#        att2d = attention.view(?,self.size[0],self.size[1])
#
#        #find keeping (with randomness)
        
