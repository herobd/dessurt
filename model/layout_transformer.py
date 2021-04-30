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
def normalize_bbox2(bbox, dwidth, dheight, twidth, theight):
     return [
         max(min(int(1000 * (bbox[0] / dwidth)),1000),0),
         max(min(int(1000 * (bbox[1] / dheight)),1000),0),
         max(min(int(1000 * (bbox[2] / twidth)),twidth),0),
         max(min(int(1000 * (bbox[3] / theight)),theight),0),
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

        self.embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        
        self.pos_enc = PositionalEncoding(d_model,dropout=0.1,max_len=5000)
        self.pos_emb_x = PositiveRealEmbedding(d_model,0,1000,100)
        self.pos_emb_y = PositiveRealEmbedding(d_model,0,1000,100)
        self.pos_emb_w = PositiveRealEmbedding(d_model,0,500,30)
        self.pos_emb_h = PositiveRealEmbedding(d_model,0,300,30)
                

    def forward(self,image_size,gtBBs,gtTrans,device):
        #input_ids = []
        #total_string=''
        #word_token_map=[]
        all_input_bbs=[]
        max_len=0
        for b,(words,bbs) in enumerate(zip(gtTrans,gtBBs)):
            input_bbs = [[0,0,0,0]]
            for i,(word,bb) in enumerate(zip(words,bbs)):
                if useCurvedBBs:
                    x1,y1,x2,y2,r=bb[:5]
                    h = y2-y1
                    w = x2-x1
                    xc = (x1+x2)/2
                    yc = (y1+y2)/2
                else:
                    xc,yc,r,h,w=bb[:5]
                    x1=xc-w
                    x2=xc+w
                    y1=yc-h
                    y2=yc+h
                #bb = normalize_bbox([x1,y1,x2,y2],image_size[3],image_size[2]) #x1y1x2y2
                bb = normalize_bbox2([xc,yc,w,h],image_size[3],image_size[2],500,300) #x1y1x2y2
                word_tokens = tokenizer.tokenize(word)
                #input_ids.extend(word_tokens)
                #total_string+=word+' '
                #word_token_map.append(range(len(input_bbs),len(input_bbs)+len(word_tokens)))
                input_bbs.extend([bb]*len(word_tokens))
            input_bbs.append([0,0,0,0])
            max_len = max(max_len,len(input_bbs))
            all_input_bbs.append(input_bbs)

        bbs = torch.FloatTensor(batch_size,max_len,4).zero_()
        for b,input_bbs in enumerate(all_input_bbs):
            bbs[b,0:len(input_bbs),:] = torch.FloatTensor(input_bbs)
        bbs=bbs.to(device)
        xs=bbs[:,0]
        ys=bbs[:,1]
        ws=bbs[:,2]
        hs=bbs[:,3]

        #inputs = tokenizer(total_string,return_tensors="pt")
        #input_bbs = torch.LongTensor([input_bbs])

        total_strings = [' '.join(gtT) for gtT in gtTrans]

        inputs = self.tokenizer(total_strings,return_tensors="pt",padding=True)

        embedded = self.embedding(inputs['input_ids'].to(device))
        #embedded = self.pos_enc(embedded)
        embedded += self.pos_emb_x(xs) + self.pos_emb_y(ys) + self.pos_emb_w(ws) + self.pos_emb_h(hs)

        embedded = embedded.permute(1,0,2) #batch,len,feat -> len,batch,feat
        padding_mask = ~inputs['attention_mask'].bool().to(device)

        encoded = self.encoder(embedded,src_key_padding_mask=padding_mask).permute(1,0,2) #len,batch,feat -> batch,len,feat

        return (
                encoded, 
                padding_mask)
