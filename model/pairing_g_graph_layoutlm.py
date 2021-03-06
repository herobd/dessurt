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

import timeit
import torch.autograd.profiler as profiler
from .resnet import resnet50

try:
    from transformers import LayoutLMTokenizer, LayoutLMModel
except:
    pass

def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]

class PairingGGraphLayoutLM(PairingGroupingGraph):
    def __init__(self, config):
        super(PairingGroupingGraph, self).__init__(config) #call super of parent.
        self.legacy=False
        self.useCurvedBBs = config['use_curved_bbs'] if 'use_curved_bbs' in config else False
        self.rotation = config['rotation'] if 'rotation' in config else False

        self.text_rec=None
        self.detector_predNumNeighbors=False
        self.numBBTypes=0

        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased") #out feats 768
        self.numTextFeats=256
        self.reduce_lm = nn.Sequential(
                    nn.Linear(768,self.numTextFeats),
                    nn.GroupNorm(getGroupSize(self.numTextFeats),self.numTextFeats),
                    nn.ReLU(True))

        checkpoint = torch.load(config['publaynet_model'], map_location='cpu')
        self.resnet = resnet50(pretrained=False)
	#has five downsamples, 
        model_state_dict = self.resnet.state_dict()
        new_state_dict={}
        prefix = 'backbone.body.'
        for key,value in checkpoint['model'].items():
            if key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = value
        self.resnet.load_state_dict(new_state_dict)
        feats_size =  1024+512
        feats_scale = 16
        self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(2048,512,2,2,0),
                    nn.GroupNorm(16,512),
                    nn.ReLU(True))

        self.detector_frozen=False

        #def save_feats(module,input,output):
        #    self.saved_features=output
        #self.resnet.some_layer.register_forward_hook(save_feats)

        self.use2ndFeatures= config['use_2_feat_levels']
        if self.use2ndFeatures:
            feats2_size = 512
            feats2_scale = 4
            #def save_feats2(module,input,output):
            #    self.saved_features2=output
            #self.resnet.some_layer2.register_forward_hook(save_feats2)


        self.buildNet(config,feats_size,feats2_size,feats_scale,feats2_scale)

    def unfreeze(self): 
        if self.detector_frozen:
            for param in self.layoutlm.parameters(): 
                param.requires_grad=param.will_use_grad 
            self.detector_frozen=False
            print('Unfroze LayoutLM')
        

    def forward(self, image, gtBBs=None, gtNNs=None, useGTBBs=False, otherThresh=None, otherThreshIntur=None, hard_detect_limit=5000, debug=False,old_nn=False,gtTrans=None,merge_first_only=False, gtGroups=None):
        num_words = len(gtTrans)
        assert useGTBBs and gtTrans is not None and len(gtTrans)==gtBBs.size(1)
        assert image.size(0)==1  #implementation designed for batch size of 1. Should work to do data parallelism, since each copy of the model will get a batch size of 1
        self.merges_performed=0 #just tracking to see if it's working
        gtBBs=gtBBs[0]
        #input_ids = []
        input_bbs = [[0,0,0,0]]
        total_string=''
        for word,bb in zip(gtTrans,gtBBs):
            if self.useCurvedBBs:
                x1,y1,x2,y2,r=bb[:5]
            else:
                xc,yc,r,h,w=bb[:5]
                x1=xc-w
                x2=xc+w
                y1=yc-h
                y2=yc+h
            bb = normalize_bbox([x1,y1,x2,y2],image.size(3),image.size(2)) #x1y1x2y2
            word_tokens = self.tokenizer.tokenize(word)
            #input_ids.extend(word_tokens)
            total_string+=word+' '
            input_bbs.extend([bb]*len(word_tokens))
        input_bbs.append([1000,1000,1000,1000])
        inputs = self.tokenizer(total_string)
        #input_ids = torch.LongTensor([input_ids])
        input_bbs = torch.LongTensor([input_bbs])
        lm_out = self.layoutlm(**inputs,bbox=input_bbs)
        lm_out = self.reduce_lm(lm_out)



        init_class_pred = self.init_class_layer(lm_out)

        useBBs = torch.cat([
            torch.ones(num_words,1),
            gtBBs,
            init_class_pred.detach()], dim=1)


        if self.useCurvedBBs:
            useBBs = [TextLine(bb,step_size=self.text_line_smoothness) for bb in useBBs] #self.x1y1x2y2rToCurved(useBBs)


        if len(useBBs)>0:
            x2,x3,x4,x5 = self.resnet(image)
            x5 = self.upsample(x5)
            if x5.size(2)<x4.size(2):
                assert x4.size(2)-x5.size(2)<2
                x4 = x4[:,:,:x5.size(2)]
            elif x5.size(2)>x4.size(2):
                assert x5.size(2)-x4.size(2)<2
                x5 = x5[:,:,:x4.size(2)]
            if x5.size(3)<x4.size(3):
                assert x4.size(3)-x5.size(3)<2
                x4 = x4[:,:,:,:x5.size(3)]
            elif x5.size(3)>x4.size(3):
                assert x5.size(3)-x4.size(3)<2
                x5 = x5[:,:,:,:x4.size(3)] 
            saved_features=torch.cat([x4,x5],dim=1)
            saved_features2=x2
            x3=x4=None

            embeddings=lm_out
            bbTrans = gtTran

            allOutputBoxes, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final = self.runGraph(
                    gtGroups,
                    gtTrans,
                    image,
                    useBBs,
                    saved_features,
                    saved_features2,
                    bbTrans,
                    embeddings,
                    merge_first_only)

            return allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final
        else:
            if not self.useCurvedBBs and self.detector.predNumNeighbors:
                #Discard NN prediction. We don't use it anymore
                bbPredictions = torch.cat([bbPredictions[:,:6],bbPredictions[:,7:]],dim=1)
                useBBs = torch.cat([useBBs[:,:6],useBBs[:,7:]],dim=1)
            return [bbPredictions], offsetPredictions, None, None, None, None, None, None, (useBBs if self.useCurvedBBs else useBBs.cpu().detach(),None,None,transcriptions)

