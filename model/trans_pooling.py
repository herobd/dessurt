import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRPooler(nn.Module):

    def __init__(self,d_model):
        super(OCRPooler, self).__init__()
        self.conv = nn.Conv1d(d_model,d_model,kernel_size=4,stride=2,padding=1)
        self.avg_pool = nn.AvgPool1d(kernel_size=2,stride=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2,stride=2)

    def forward(self,tokens,pos,padding_mask):
        tokens = self.conv(tokens.permute(0,2,1))
        pos = self.avg_pool(pos.permute(0,2,1))
        padding_mask = self.max_pool(padding_mask[:,None])

        return tokens.permute(0,2,1), pos.permute(0,2,1), padding_mask[:,0]

class QPooler(nn.Module):

    def __init__(self,d_model):
        super(QPooler, self).__init__()
        self.conv = nn.Conv1d(d_model,d_model,kernel_size=4,stride=2,padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2,stride=2)

    def forward(self,tokens,padding_mask):
        tokens = self.conv(tokens.permute(0,2,1))
        padding_mask = self.max_pool(padding_mask[:,None])

        return tokens.permute(0,2,1), padding_mask[:,0]
