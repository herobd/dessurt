
import torch
import torch.nn as nn
import numpy as np
import re


class HandCodeEmb(nn.Module):
    def __init__(self,out_size):
        super(HandCodeEmb, self).__init__()

        self.name_list = ??

    
    def forward(self,transcriptions):
        features = torch.FloatTensor(len(transcriptions),self.num_feats).zero_()
        for i, trans in enumerate(transcriptions):
            for j, check in enumerate(self.feature_checks):
                features[i,j] = check(trans)

        features = features.to(the device)
        return self.emb_layers(features)




def hasNumber(s):
    return re.search(r'\d',s) is not None

def hasDayNumber(s):
    return re.search(r'[\s,.:][012]?\d[\s,.:]',s) is not None

def hasYearNumber(s):
    return re.search(r'[\s,.:]((1[789])|20)\d\d[\s,.:]',s) is not None #TODO TEST
