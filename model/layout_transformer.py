from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pairing_grouping_graph import PairingGroupingGraph
#from model.cnn_lstm import CRNN, SmallCRNN
#import concurrent.futures
from skimage import draw
from model.net_builder import make_layers, getGroupSize
from utils.yolo_tools import non_max_sup_iou, non_max_sup_dist, non_max_sup_overseg, allIOU, allIO_clipU
from utils.util import decode_handwriting
from utils.bb_merging import TextLine, xyrwh_TextLine
#from utils.string_utils import correctTrans
import math, os
from collections import defaultdict
import utils.img_f as img_f

from model.pos_encode import PositionalEncoding

LLM_MAX_TOKEN_LEN = 512
LLM_SEP=102
LLM_CLS=101
try:
    from transformers import LayoutLMTokenizer, LayoutLMModel
except:
    pass
def normalize_bbox(bbox, width, height):
     return [
         max(min(int(1000 * (bbox[0] / width)),1000),0),
         max(min(int(1000 * (bbox[1] / height)),1000),0),
         max(min(int(1000 * (bbox[2] / width)),1000),0),
         max(min(int(1000 * (bbox[3] / height)),1000),0),
     ]

class LayoutTransformer(BaseModel):
    def __init__(self, config, dropout=0.1):
        super(LayoutTransformer, self).__init__(config) #call super of parent.

        d_model = config['d_model']
        dim_ff = config['dim_ff']
        num_e_layers = config['num_layers']
        nhead = config['nhead']

        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

        encoder_layer= nn.TransformerEncoderLayer(d_model,nhead,dim_ff,dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_e_layers,nn.LayerNorm(d_model))

        self.embedding =  nn.Sequential(
                nn.Embedding(self.tokenizer.vocab_size, d_model),
                PositionalEncoding(d_model,dropout=0.1,max_len=5000)
                )

    def forward(self,image_size,gtBBs,gtTrans,device):
        #input_ids = []
        #input_bbs = [[0,0,0,0]]
        #total_string=''
        #word_token_map=[]
        #for i,(word,bb) in enumerate(zip(gtTrans,gtBBs)):
        #    if useCurvedBBs:
        #        x1,y1,x2,y2,r=bb[:5]
        #    else:
        #        xc,yc,r,h,w=bb[:5]
        #        x1=xc-w
        #        x2=xc+w
        #        y1=yc-h
        #        y2=yc+h
        #    bb = normalize_bbox([x1,y1,x2,y2],image_size[3],image_size[2]) #x1y1x2y2
        #    word_tokens = tokenizer.tokenize(word)
        #    #input_ids.extend(word_tokens)
        #    total_string+=word+' '
        #    word_token_map.append(range(len(input_bbs),len(input_bbs)+len(word_tokens)))
        #    input_bbs.extend([bb]*len(word_tokens))
        #input_bbs.append([1000,1000,1000,1000])
        #inputs = tokenizer(total_string,return_tensors="pt")
        #input_bbs = torch.LongTensor([input_bbs])

        total_strings = [' '.join(gtT) for gtT in gtTrans]

        inputs = self.tokenizer(total_strings,return_tensors="pt",padding=True)

        embedded = self.embedding(inputs['input_ids'].to(device))

        embedded = embedded.permute(1,0,2) #batch,len,feat -> len,batch,feat
        padding_mask = ~inputs['attention_mask'].bool().to(device)

        return (
                self.encoder(embedded,src_key_padding_mask=padding_mask).permute(1,0,2), 
                padding_mask) #len,batch,feat -> batch,len,feat
