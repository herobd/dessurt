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
         max(min(int(1000 * (bbox[0] / width)),1000),0),
         max(min(int(1000 * (bbox[1] / height)),1000),0),
         max(min(int(1000 * (bbox[2] / width)),1000),0),
         max(min(int(1000 * (bbox[3] / height)),1000),0),
     ]

class PairingGGraphLayoutLM(PairingGroupingGraph):
    def __init__(self, config):
        super(PairingGroupingGraph, self).__init__(config) #call super of parent.
        self.legacy=False
        self.useCurvedBBs = config['use_curved_bbs'] if 'use_curved_bbs' in config else False
        self.rotation = config['rotation'] if 'rotation' in config else False

        self.text_rec=None
        self.detector_predNumNeighbors=False
        self.numBBTypes=config['num_classes']

        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased") #out feats 768
        self.max_token_len = 512
        self.SEP=102
        self.CLS=101
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


        self.init_class_layer = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(self.numTextFeats,64*self.numBBTypes),
                nn.ReLU(True),
                nn.Dropout(0.2),
                nn.Linear(64*self.numBBTypes,self.numBBTypes)
                )

        #def save_feats(module,input,output):
        #    self.saved_features=output
        #self.resnet.some_layer.register_forward_hook(save_feats)

        self.use2ndFeatures= config['use_2_feat_levels']
        if self.use2ndFeatures:
            feats2_size = 256
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
        device = image.device
        num_words = len(gtTrans)
        assert useGTBBs and gtTrans is not None and len(gtTrans)==gtBBs.size(1)
        assert image.size(0)==1  #implementation designed for batch size of 1. Should work to do data parallelism, since each copy of the model will get a batch size of 1
        self.merges_performed=0 #just tracking to see if it's working
        gtBBs=gtBBs[0]

        lm_out = self.runLayoutLM(image.size(),gtBBs,gtTrans,device)

        init_class_pred = self.init_class_layer(lm_out)

        useBBs = torch.cat([
            torch.ones(num_words,1).to(device),
            gtBBs[:,:5],
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
            bbTrans = gtTrans

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

            return allOutputBoxes, init_class_pred, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final
        else:
            if not self.useCurvedBBs and self.detector.predNumNeighbors:
                #Discard NN prediction. We don't use it anymore
                bbPredictions = torch.cat([bbPredictions[:,:6],bbPredictions[:,7:]],dim=1)
                useBBs = torch.cat([useBBs[:,:6],useBBs[:,7:]],dim=1)
            return [bbPredictions], init_class_pred, None, None, None, None, None, None, (useBBs if self.useCurvedBBs else useBBs.cpu().detach(),None,None,transcriptions)


    def runLayoutLM(self,image_size,gtBBs,gtTrans,device):
        #input_ids = []
        input_bbs = [[0,0,0,0]]
        total_string=''
        word_token_map=[]
        for i,(word,bb) in enumerate(zip(gtTrans,gtBBs)):
            if self.useCurvedBBs:
                x1,y1,x2,y2,r=bb[:5]
            else:
                xc,yc,r,h,w=bb[:5]
                x1=xc-w
                x2=xc+w
                y1=yc-h
                y2=yc+h
            bb = normalize_bbox([x1,y1,x2,y2],image_size[3],image_size[2]) #x1y1x2y2
            word_tokens = self.tokenizer.tokenize(word)
            #input_ids.extend(word_tokens)
            total_string+=word+' '
            word_token_map.append(range(len(input_bbs),len(input_bbs)+len(word_tokens)))
            input_bbs.extend([bb]*len(word_tokens))
        input_bbs.append([1000,1000,1000,1000])
        inputs = self.tokenizer(total_string,return_tensors="pt")
        input_bbs = torch.LongTensor([input_bbs])

        if inputs['input_ids'].size(1)<self.max_token_len:
            inputs = {k:i.to(device) for k,i in inputs.items()}
            input_bbs = input_bbs.to(device)
            #assert inputs['input_ids'].size(1)==input_bbs.size(1)
            #print('input {} {} {}'.format(inputs['input_ids'].size(),inputs['input_ids'].max(),inputs['input_ids'].min()))
            #print('bb    {} {} {}'.format(input_bbs.size(),input_bbs.max(),input_bbs.min()))
            lm_out = self.layoutlm(**inputs,bbox=input_bbs).last_hidden_state[0] #get rid of batch dim

        else:

            #We need to split the input into batches
            #We'll ensure at least self.max_len//2 overlap between batches to preserve context
            #Just average overlap outputs (would it be better to only take one though?)

            #Form batches
            batch_ids=[]
            batch_mask=[]
            start=-self.max_token_len//2 +1 # start in neg to allow first add, +1 to skip CLS token
            starts=[]
            ends=[]
            batch_bbs=[]
            prev_end=-1
            end=0
            while end<inputs['input_ids'].size(1)-1:
                start+=self.max_token_len//2
                end=start+self.max_token_len-2 #-2 for CLS and SEP token that need added
                if end<=inputs['input_ids'].size(1)-1: #is this the last batch? (-1 to ignore SEP token)
                    pass
                else:
                    #last batch
                    end = inputs['input_ids'].size(1)-1 #-1, we'll clip the SEP token here 
                    start = end-(self.max_token_len-2) 
             
                ids = inputs['input_ids'][:,start:end]
                ids = F.pad(ids,(0,(self.max_token_len-2)-ids.size(1))) #ensure all batches are same length
                mask = inputs['attention_mask'][:,start:end]
                mask = F.pad(mask,(0,(self.max_token_len-2)-mask.size(1)))

                b_bbs = input_bbs[:,start:end]
                b_bbs = F.pad(b_bbs,(0,0,0,(self.max_token_len-2)-b_bbs.size(1)))

                batch_ids.append(ids)
                batch_mask.append(mask)#inputs['attention_mask'][:,start:end])
                batch_bbs.append(b_bbs)
                starts.append(start)
                ends.append(end)
                prev_end=end

            num_batches = len(starts)
            print('split to {} batches'.format(num_batches))
            
            #cat the batch together, and add start/end CLS and SEP tokens
            input_ids = torch.cat(batch_ids,dim=0)
            start_token = torch.LongTensor(num_batches,1).fill_(self.CLS)
            end_token = torch.LongTensor(num_batches,1).fill_(self.SEP)
            input_ids = torch.cat([start_token,input_ids,end_token],dim=1).to(device)

            attention_mask = torch.cat(batch_ids,dim=0)
            clssep_mask = torch.LongTensor(num_batches,1).fill_(1)
            attention_mask = torch.cat([clssep_mask,attention_mask,clssep_mask],dim=1).to(device)

            batch_bbs = torch.cat(batch_bbs,dim=0)
            cls_bbs = torch.LongTensor(num_batches,1,4).fill_(0)
            sep_bbs = torch.LongTensor(num_batches,1,4).fill_(1000)
            batch_bbs = torch.cat([cls_bbs,batch_bbs,sep_bbs],dim=1).to(device)

            print('input_ids:{}, attention_mask:{}, batch_bbs:{}'.format(input_ids.size(),attention_mask.size(),batch_bbs.size()))

            #batch = {'input_ids':torch.cat(batch_ids,dim=0).to(device), 'attention_mask':torch.cat(batch_mask,dim=0).to(device)}
            outputs = self.layoutlm(input_ids=input_ids,attention_mask=attention_mask,bbox=batch_bbs).last_hidden_state

            attention_mask=batch_bbs=batch_ids=None

            #Now we'll reshape the output to be a single batch again
            full_output = [] 
            #this requires averaging overlap areas

            prev_out=None
            prev_start=None
            print('begin remerge')
            for b in range(num_batches):
                start=starts[b]
                end=ends[b]
                if b+1<num_batches:
                    next_start = starts[b+1]
                else:
                    next_start = None
                if b>0:
                    prev_end = ends[b-1]
                else:
                    prev_end = None
                output = outputs[b]

                if prev_end is None:
                    full_output.append(output[0:next_start-start+1])# this includes first CLS token (+1 is to account for it)
                    prev_out_overlap = output[next_start-start+1:-1] #save overlap with next (-1 to account to SEP token)
                elif next_start is not None:
                    mean_prev_overlap = (prev_out_overlap + output[1:prev_end-start+1])/2 #average overlap of prev
                    full_output.append(mean_prev_overlap)
                    assert prev_end>=next_start #be sure we're not leaving tokens out
                    prev_out_overlap = output[next_start-start+1:-1] #save overlap with next
                else:
                    mean_prev_overlap = (prev_out_overlap + output[1:prev_end-start+1])/2
                    full_output.append(mean_prev_overlap)
                    full_output.append(output[prev_end-start+1:]) #this includes last SEP token
            lm_out = torch.cat(full_output,dim=0)
            assert lm_out.size(0) == inputs['input_ids'].size(1)
            print('finished remerge')


        #We'll now average the features for tokens from the same word-bb
        #  This also discards the class and sep token's features.
        new_lm_out = lm_out.new_zeros(len(word_token_map),lm_out.size(1))
        for i,parts in enumerate(word_token_map):
            if len(parts)>0:
                new_lm_out[i] = torch.stack([lm_out[t] for t in parts],dim=0).mean(dim=0)
        lm_out = self.reduce_lm(new_lm_out)
        return lm_out
