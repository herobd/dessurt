from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import *
from model.graph_net import GraphNet
from model.meta_graph_net import MetaGraphNet
from model.binary_pair_net import BinaryPairNet
from model.binary_pair_real import BinaryPairReal
#from model.roi_align.roi_align import RoIAlign
#from model.roi_align import ROIAlign as RoIAlign
from torchvision.ops import RoIAlign
from model.cnn_lstm import CRNN, SmallCRNN
from model.tesseract_wrap import TesseractWrap
import concurrent.futures
from model.word2vec_adapter import Word2VecAdapter, Word2VecAdapterShallow, BPEmbAdapter
from model.distilbert_adapter import DistilBertAdapter, DistilBertWholeAdapter
from model.hand_code_emb import HandCodeEmb
from skimage import draw
from model.net_builder import make_layers, getGroupSize
from utils.yolo_tools import non_max_sup_iou, non_max_sup_dist, non_max_sup_overseg, allIOU, allIO_clipU
from utils.util import decode_handwriting
from utils.bb_merging import TextLine, xyrwh_TextLine
#from utils.string_utils import correctTrans
import editdistance
import math, os
import random
import json
from collections import defaultdict
import utils.img_f as img_f

import timeit
import torch.autograd.profiler as profiler

MAX_CANDIDATES=700 #these are only used for line-of-sight selection
MAX_GRAPH_SIZE=750

def minAndMaxXY(boundingRects):
    min_X,min_Y,max_X,max_Y = np.array(boundingRects).transpose(1,0)
    return min_X.min(),max_X.max(),min_Y.min(),max_Y.max()
def combineShapeFeats(feats):
    if len(feats)==1:
        return torch.FloatTensor(feats[0])
    feats.sort(key=lambda x: x[17]) #sort into read order
    feats = torch.FloatTensor(feats)
    easy_feats = feats[:,0:6].mean(dim=0)
    tl=feats[0,6:8]
    tr=feats[0,8:10]
    br=feats[-1,10:12]
    bl=feats[-1,12:14]
    easy2_feats=feats[:,14:].mean(dim=0)
    return torch.cat((easy_feats,tl,tr,br,bl,easy2_feats),dim=0)
def combineShapeFeatsTensor(feats):
    feats = torch.stack(feats,dim=0)
    return feats.mean(dim=0)
def groupRect(corners):
    corners=np.array(corners)
    return corners[:,0].min(), corners[:,1].min(), corners[:,2].max(), corners[:,3].max()
def correctTrans(pred,predBB,gt,gtBB):
    thresh=100
    #x0,y0,r0,h0,w0 = bb0[locIdx:classIdx]
    #x1,y1,r1,h1,w1 = bb1[locIdx:classIdx]

    new_pred=[]
    for i,p in enumerate(pred):
        xp,yp,rp,hp,wp = predBB[i,0:5]
        assert(rp==0)
        tx0=xp
        ty0=yp-hp
        lx0=xp-wp
        ly0=yp

        ds=[]
        for j,g in enumerate(gt):
            xg,yg,rg,hg,wg = gtBB[0,j,0:5]
            assert(rg==0)
            tx1=xg
            ty1=yg-hg
            lx1=xg-wg
            ly1=yg
            d = math.sqrt((tx0-tx1)**2 + (ty0-ty1)**2) + math.sqrt((lx0-lx1)**2 + (ly0-ly1)**2)
            if d<thresh:
                ds.append([d,j,g])
        ds.sort(key=lambda x:x[0])

        if len(ds)==0:
            new_pred.append(p)
        elif len(p)<3:
            new_pred.append(ds[0][2]) #This isn't a long enough string, so base on distance alone
        else:
            min_d = ds[0][0]
            best_score=None
            for d,j,g in ds[:10]:
                dis = editdistance.eval(p,g)/len(p)
                score = dis*(d/min_d)
                if best_score is None or score<best_score:
                    best_score=score
                    best_g = g
            new_pred.append(best_g)

    return new_pred

class PairingGroupingGraph(BaseModel):
    def __init__(self, config):
        super(PairingGroupingGraph, self).__init__(config)
        self.useCurvedBBs=False
        self.legacy= 'legacy' in config and config['legacy']
        self.legacy_read= 'legacy_read' in config and config['legacy_read']

        if 'detector_checkpoint' in config:
            if os.path.exists(config['detector_checkpoint']):
                checkpoint = torch.load(config['detector_checkpoint'], map_location=lambda storage, location: storage)
                checkpoint['config']['model']['arch'] = checkpoint['config']['arch']
            else:
                checkpoint = None
                print('Warning: unable to load {}'.format(config['detector_checkpoint']))
            detector_config = json.load(open(config['detector_config']))['model'] if 'detector_config' in config else checkpoint['config']['model']
            if checkpoint is None:
                self.detector = eval(detector_config['arch'])(detector_config)
                for p in self.detector.parameters():
                    import pdb;pdb.set_trace()
                    p.something = float('nan') #ensure this gets changed
            elif 'state_dict' in checkpoint:
                #self.detector = eval(checkpoint['config']['arch'])(detector_config)
                self.detector = eval(detector_config['arch'])(detector_config)
                self.detector.load_state_dict(checkpoint['state_dict'])
                #config['detector_config'] = checkpoint['config']['model']
            else:
                self.detector = checkpoint['model']
            self.useCurvedBBs = 'OverSeg' in checkpoint['config']['arch']
        else:
            detector_config = config['detector_config']
            self.detector = eval(detector_config['arch'])(detector_config)
            self.useCurvedBBs = 'OverSeg' in config['detector_config']['arch']

        if 'pretrained_backbone_checkpoint' in config:
            if os.path.exists(config['pretrained_backbone_checkpoint']):
                checkpoint = torch.load(config['pretrained_backbone_checkpoint'], map_location=lambda storage, location: storage)
                detector_state_dict={}
                for name,data in checkpoint['state_dict'].items():
                    if name.startswith('detector.'):
                        detector_state_dict[name[9:]]=data
                self.detector.load_state_dict(detector_state_dict)
            elif 'DONT_NEED_TO_LOAD_PRETRAINED' not in config or not config['DONT_NEED_TO_LOAD_PRETRAINED']:
                raise FileNotFoundError('Could not find pretrained backbone: {}'.format(config['pretrained_backbone_checkpoint']))


        self.detector_predNumNeighbors=self.detector.predNumNeighbors

        self.use_overseg_non_max_sup = config['overseg_non_max_sup'] if 'overseg_non_max_sup' in config else False


        useBeginningOfLast = config['use_beg_det_feats'] if 'use_beg_det_feats' in config else False
        useFeatsLayer = config['use_detect_layer_feats'] if 'use_detect_layer_feats' in config else -1
        useFeatsScale = config['use_detect_scale_feats'] if 'use_detect_scale_feats' in config else -2
        useFLayer2 = config['use_2nd_detect_layer_feats'] if 'use_2nd_detect_layer_feats' in config else None
        useFScale2 = config['use_2nd_detect_scale_feats'] if 'use_2nd_detect_scale_feats' in config else None
        detectorSavedFeatSize = config['use_detect_feats_size'] if 'use_detect_feats_size' in config else self.detector.last_channels
        assert((useFeatsScale==-2) or ('use_detect_feats_size' in config))
        detectorSavedFeatSize2 = config['use_2nd_detect_feats_size'] if 'use_2nd_detect_feats_size' in config else None
        

        self.use2ndFeatures = useFLayer2 is not None
            
        self.detector.setForGraphPairing(useBeginningOfLast,useFeatsLayer,useFeatsScale,useFLayer2,useFScale2)

        if 'detect_save_scale' in config:
            detect_save_scale = config['detect_save_scale']
        elif useBeginningOfLast:
            detect_save_scale = self.detector.scale[0]
        else:
            detect_save_scale = self.detector.save_scale
        if 'detect_save2_scale' in config:
            detect_save2_scale = config['detect_save2_scale']
        elif self.use2ndFeatures:
            detect_save2_scale = self.detector.save2_scale
        else:
            detect_save2_scale = None
        #splitScaleDiff = config['split_features_scale_diff'] if 'split_features_scale_diff' in config else None

        self.no_grad_feats = config['no_grad_feats'] if 'no_grad_feats' in config else False

        if (config['start_frozen'] if 'start_frozen' in config else False):
            for param in self.detector.parameters(): 
                param.will_use_grad=param.requires_grad 
                param.requires_grad=False 
            self.detector_frozen=True
        else:
            self.detector_frozen=False


        self.numBBTypes = self.detector.numBBTypes
        self.rotation = self.detector.rotation
        self.scale = self.detector.scale
        self.anchors = self.detector.anchors
        if 'detect_conf_thresh' in config:
            self.detect_conf_thresh = config['detect_conf_thresh'] 
        elif 'conf_thresh' in config:
            self.detect_conf_thresh = config['conf_thresh'] 
        else:
            self.detect_conf_thresh = 0.5
        self.useHardConfThresh = config['use_hard_conf_thresh'] if 'use_hard_conf_thresh' in config else True


        if type(self.detector.scale[0]) is int:
            assert(self.detector.scale[0]==self.detector.scale[1])
        else:
            for level_sc in self.detector.scale:
                assert(level_sc[0]==level_sc[1])

        self.set_detect_params = (useBeginningOfLast,useFeatsLayer,useFeatsScale,useFLayer2,useFScale2)


        #HWR stuff
        if 'text_rec' in config:
            self.numTextFeats = config['text_rec']['num_feats']
            self.use_tesseract=False
            self.padATRy=3
            self.padATRx=10
            self.atr_batch_size = config['text_rec']['batch_size']
            self.pad_text_height = config['text_rec']['pad_text_height'] if 'pad_text_height' in config['text_rec'] else False
            if 'CRNN' in config['text_rec']['model']:
                self.hw_channels = config['text_rec']['num_channels'] if 'num_channels' in config['text_rec'] else 1
                norm = config['text_rec']['norm'] if 'norm' in config['text_rec'] else 'batch'
                use_softmax = config['text_rec']['use_softmax'] if 'use_softmax' in config['text_rec'] else True
                if 'Small' in config['text_rec']['model']:
                    self.text_rec = SmallCRNN(config['text_rec']['num_char'],self.hw_channels,norm=norm,use_softmax=use_softmax)
                else:
                    self.text_rec = CRNN(config['text_rec']['num_char'],self.hw_channels,norm=norm,use_softmax=use_softmax)
                    
                print('WARNING, is text_rec set to frozen?')
                self.text_rec.eval()
                #self.text_rec = self.text_rec.cuda()
                if 'hw_with_style_file' in config['text_rec']:
                    state=torch.load(config['text_rec']['hw_with_style_file'], map_location=lambda storage, location: storage)['state_dict']
                    hwr_state_dict={}
                    for key,value in  state.items():
                        if key[:4]=='hwr.':
                            hwr_state_dict[key[4:]] = value
                    self.text_rec.load_state_dict(hwr_state_dict)
                elif 'file' in config['text_rec']:
                    hwr_state_dict=torch.load(config['text_rec']['file'])['state_dict']
                    self.text_rec.load_state_dict(hwr_state_dict)

                self.hw_input_height = config['text_rec']['input_height']
                with open(config['text_rec']['char_set']) as f:
                    char_set = json.load(f)
                self.idx_to_char = {}
                for k,v in char_set['idx_to_char'].items():
                    self.idx_to_char[int(k)] = v
            elif 'esseract' in config['text_rec']['model']:
                self.text_rec = TesseractWrap(config['text_rec'])
                self.hw_input_height = config['text_rec']['height']
                self.use_tesseract=True
                self.trans_threads = config['text_rec']['trans_threads'] if 'trans_threads' in config['text_rec'] else 3
            else:
                raise NotImplementedError('Unknown ATR model: {}'.format(config['text_rec']['model']))
            
            if 'embedding' in config['text_rec']:
                if 'word2vec' in config['text_rec']['embedding']:
                    if 'shallow' in config['text_rec']['embedding']:
                        self.embedding_model = Word2VecAdapterShallow(self.numTextFeats)
                    else:
                        self.embedding_model = Word2VecAdapter(self.numTextFeats)
                elif 'BP' in config['text_rec']['embedding']:
                    self.embedding_model = BPEmbAdapter(self.numTextFeats)
                elif 'hand' in config['text_rec']['embedding']:
                    self.embedding_model = HandCodeEmb(self.numTextFeats)
                elif 'DistilBertWhole' in config['text_rec']['embedding']:
                    self.embedding_model = DistilBertWholeAdapter(self.numTextFeats)
                elif 'DistilBert' in config['text_rec']['embedding']:
                    self.embedding_model = DistilBertAdapter(self.numTextFeats)
                else:
                    raise NotImplementedError('Unknown text embedding method: {}'.format(config['text_rec']['embedding']))
            else:
                self.embedding_model = lambda x: None 

            if type(config['graph_config']) is list:
                graph_in_channels = config['graph_config'][0]['in_channels'] if 'in_channels' in config['graph_config'][0] else 1
            else:
                graph_in_channels = config['graph_config']['in_channels'] if 'in_channels' in config['graph_config'] else 1

            if self.legacy_read:
                self.merge_embedding_layer = nn.Sequential(nn.ReLU(True),nn.Linear(graph_in_channels+self.numTextFeats,graph_in_channels))
        else:
            self.text_rec=None
            self.numTextFeats = 0


        self.add_noise_to_word_embeddings = config['add_noise_to_word_embeddings'] if 'add_noise_to_word_embeddings' in config else 0
        self.blind_during_gt = config['blind_during_gt'] if 'blind_during_gt' in config else None
        self.no_text_during_no_gt = config['no_text_during_no_gt'] if 'no_text_during_no_gt' in config else None

        self.buildNet(config,detectorSavedFeatSize,detectorSavedFeatSize2,detect_save_scale,detect_save2_scale)
    
    def buildNet(self,config,backboneSavedFeatSize,backboneSavedFeatSize2,backbone_save_scale,backbone_save2_scale):
        self.all_grad=False
        self.splitFeatures= config['split_features_scale'] if 'split_features_scale' in config else False
        if self.use2ndFeatures and not self.splitFeatures:
            backboneSavedFeatSize += backboneSavedFeatSize2

        self.predNN = config['pred_nn'] if 'pred_nn' in config else False
        assert(not self.predNN)
        self.predClass = config['pred_class'] if 'pred_class' in config else False

        self.text_line_smoothness = config['text_line_smoothness'] if 'text_line_smoothness' in config else 'original' #200
        self.prevent_vert_merges = config['prevent_vert_merges'] if 'prevent_vert_merges' in config else False
        self.fully_connected = config['fully_connected'] if 'fully_connected' in config else False
        self.graph_min_degree = config['graph_min_degree'] if 'graph_min_degree' in config else None
        self.graph_four_connected = config['graph_four_connected'] if 'graph_four_connected' in config else None
        assert(bool(self.fully_connected) + bool(self.graph_min_degree) + bool(self.graph_four_connected) <=1)

        self.merge_first = config['merge_first'] if 'merge_first' in config else False
        self.merge_use_mask = config['merge_use_mask'] if 'merge_use_mask' in config else self.merge_first

        self.nodeIdxConf = 0
        self.nodeIdxClass = 1
        self.nodeIdxClassEnd = self.nodeIdxClass+self.numBBTypes

        if type(config['graph_config']) is list:
            graph_in_channels = config['graph_config'][0]['in_channels'] if 'in_channels' in config['graph_config'][0] else 1
        else:
            graph_in_channels = config['graph_config']['in_channels'] if 'in_channels' in config['graph_config'] else 1
        self.useBBVisualFeats=True
        if (type(config['graph_config']) is str and config['graph_config']['arch'][:10]=='BinaryPair' and not self.predNN) or ('noBBVisualFeats' in config and config['noBBVisualFeats']):
            self.useBBVisualFeats=False
        #self.includeRelRelEdges= config['use_rel_rel_edges'] if 'use_rel_rel_edges' in config else True
        #rel_channels = config['graph_config']['rel_channels']

        if 'use_rel_shape_feats' in config:
             config['use_shape_feats'] =  config['use_rel_shape_feats']
        self.useShapeFeats= config['use_shape_feats'] if 'use_shape_feats' in config else False

        if self.useShapeFeats!='only':
            if self.useShapeFeats!='only for edge':
                self.pool_h = config['featurizer_start_h']
                self.pool_w = config['featurizer_start_w']
                self.pool2_h=self.pool_h
                self.pool2_w=self.pool_w

            self.poolBB_h = config['featurizer_bb_start_h'] if 'featurizer_bb_start_h' in config else 2
            self.poolBB_w = config['featurizer_bb_start_w'] if 'featurizer_bb_start_w' in config else 3

            self.poolBB2_h=self.poolBB_h
            self.poolBB2_w=self.poolBB_w

        self.merge_pool_h = self.merge_pool2_h = config['merge_featurizer_start_h'] if 'merge_featurizer_start_h' in config else None
        self.merge_pool_w = self.merge_pool2_w = config['merge_featurizer_start_w'] if 'merge_featurizer_start_w' in config else None

        self.reintroduce_features = config['reintroduce_features'] if 'reintroduce_features' in config else  (config['reintroduce_visual_features'] if 'reintroduce_visual_features' in config else False) #"fixed map"

        self.splitFeatures= config['split_features_scale'] if 'split_features_scale' in config else False

        self.usePositionFeature = config['use_position_feats'] if 'use_position_feats' in config else False
        assert(not self.usePositionFeature or self.useShapeFeats)
        self.normalizeHorz=config['normalize_horz'] if 'normalize_horz' in config else 400
        self.normalizeVert=config['normalize_vert'] if 'normalize_vert' in config else 50
        self.normalizeDist=(self.normalizeHorz+self.normalizeVert)/2
        

        if self.useShapeFeats:
           self.shape_feats_normal = config['shape_feats_normal'] if 'shape_feats_normal' in config else True
           if self.useCurvedBBs:
               self.numShapeFeats=20+2*self.numBBTypes
               if self.shape_feats_normal:
                   self.numShapeFeatsBB=6+self.numBBTypes
               else:
                   self.numShapeFeatsBB=3+self.numBBTypes
           else:
               self.numShapeFeats=8+2*self.numBBTypes #we'll append some extra feats
               self.numShapeFeatsBB=3+self.numBBTypes
           if self.useShapeFeats!='old' and not self.useCurvedBBs:
               self.numShapeFeats+=4
           if self.detector_predNumNeighbors and not self.useCurvedBBs:
               self.numShapeFeats+=2
               self.numShapeFeatsBB+=1
           if self.usePositionFeature:
               if not self.useCurvedBBs:
                   self.numShapeFeats+=4
               self.numShapeFeatsBB+=2
        else:
           self.numShapeFeats=0
           self.numShapeFeatsBB=0



        if type(config['graph_config']) is list:
            for graphconfig in config['graph_config']:
                graphconfig['num_shape_feats']=self.numShapeFeats
        else:
            config['graph_config']['num_shape_feats']=self.numShapeFeats
        featurizer_fc = config['featurizer_fc'] if 'featurizer_fc' in config else []
        if self.useShapeFeats!='only':
            if self.merge_first:
                self.expandedMergeContextY,self.expandedMergeContextX = config['expand_merge_context']

            self.expandedBBContext = config['expand_bb_context'] if 'expand_bb_context' in config else None
            if self.expandedBBContext is not None:
                bbMasks_bb=2
            else:
                bbMasks_bb=0

            self.splitFeatureRes = config['split_feature_res'] if 'split_feature_res' in config else False

            feat_norm = config['feat_norm'] if 'feat_norm' in config else 'group_norm' #detector_config['norm_type'] #if 'norm_type' in detector_config else None
            if self.useShapeFeats!='only for edge':
                self.expandedRelContext = config['expand_rel_context'] if 'expand_rel_context' in config else None
                if self.expandedRelContext is not None:
                    bbMasks=3
                else:
                    bbMasks=2
                featurizer_conv = config['featurizer_conv'] if 'featurizer_conv' in config else [512,'M',512]
                if self.splitFeatures:
                    featurizer_conv2 = config['featurizer_conv_first'] if 'featurizer_conv_first' in config else None
                    featurizer_conv2 = [backboneSavedFeatSize2+bbMasks] + featurizer_conv2 #bbMasks are appended
                    scaleX=1
                    scaleY=1
                    for a in featurizer_conv2:
                        if a=='M' or (type(a) is str and a[0]=='D'):
                            scaleX*=2
                            scaleY*=2
                        elif type(a) is str and a[0]=='U':
                            scaleX/=2
                            scaleY/=2
                        elif type(a) is str and a[0:4]=='long': #long pool
                            scaleX*=3
                            scaleY*=2
                    assert(scaleX==scaleY)
                    splitScaleDiff=scaleX
                    self.pool_h = self.pool_h//splitScaleDiff
                    self.pool_w = self.pool_w//splitScaleDiff
                    layers, last_ch_relC = make_layers(featurizer_conv2,norm=feat_norm,dropout=True)
                    self.relFeaturizerConv2 = nn.Sequential(*layers)

                    featurizer_conv = [backboneSavedFeatSize+last_ch_relC] + featurizer_conv
                else:
                    featurizer_conv = [backboneSavedFeatSize+bbMasks] + featurizer_conv #bbMasks are appended
                scaleX=1
                scaleY=1
                for a in featurizer_conv:
                    if a=='M' or (type(a) is str and a[0]=='D'):
                        scaleX*=2
                        scaleY*=2
                    elif type(a) is str and a[0]=='U':
                        scaleX/=2
                        scaleY/=2
                    elif type(a) is str and a[0:4]=='long': #long pool
                        scaleX*=3
                        scaleY*=2
                #self.scale=(scaleX,scaleY) this holds scale for detector
                fsizeX = self.pool_w//scaleX
                fsizeY = self.pool_h//scaleY
                if 'featurizer_conv_auto' in config and config['featurizer_conv_auto']:
                    featurizer_conv.append(graph_in_channels-self.numShapeFeats)
                    assert featurizer_fc is None
                layers, last_ch_relC = make_layers(featurizer_conv,norm=feat_norm,dropout=True) 
                if featurizer_fc is None: #we don't have a FC layer, so channels need to be the same as graph model expects
                    if last_ch_relC+self.numShapeFeats!=graph_in_channels:
                        new_layer = [last_ch_relC,'k1-{}'.format(graph_in_channels-self.numShapeFeats)]
                        print('WARNING: featurizer_conv did not line up with graph_in_channels, adding layer k1-{}'.format(graph_in_channels-self.numShapeFeats))
                        #new_layer = last_ch_relC,'C3-{}'.format(graph_in_channels-self.numShapeFeats)]
                        new_layer, last_ch_relC = make_layers(new_layer,norm=feat_norm,dropout=True) 
                        layers+=new_layer
                layers.append( nn.AvgPool2d((fsizeY,fsizeX)) )
                self.relFeaturizerConv = nn.Sequential(*layers)
                rel_featurizer_conv_last = last_ch_relC

                #self.roi_align = RoIAlign(self.pool_h,self.pool_w,1.0/backbone_save_scale) Facebook implementation
                self.roi_align = RoIAlign((self.pool_h,self.pool_w),1.0/backbone_save_scale,-1)
                if self.use2ndFeatures:
                    #self.roi_align2 = RoIAlign(self.pool2_h,self.pool2_w,1.0/backbone_save2_scale)
                    self.roi_align2 = RoIAlign((self.pool2_h,self.pool2_w),1.0/backbone_save2_scale,-1)
                else:
                    last_ch_relC=0
            else:
                rel_featurizer_conv_last = 0
                last_ch_relC=0
                self.expandedRelContext=None
        else:
            rel_featurizer_conv_last = 0

        if self.merge_first:
            if self.splitFeatures:
                raise NotImplementedError('split feature embedding not implemented for merge_first model')
            merge_featurizer_conv = config['merge_featurizer_conv']
            if self.merge_use_mask:
                extra = bbMasks
            else:
                extra = 0
            merge_featurizer_conv = [backboneSavedFeatSize+extra] + merge_featurizer_conv #bbMasks are appended
            layers, last_ch_relC = make_layers(merge_featurizer_conv,norm=feat_norm,dropout=True) 
            #if last_ch_relC+self.numShapeFeats!=graph_in_channels:
            #    new_layer = [last_ch_relC,'k1-{}'.format(graph_in_channels-self.numShapeFeats)]
            #    print('WARNING: merge_featurizer_conv did not line up with graph_in_channels, adding layer k1-{}'.format(graph_in_channels-self.numShapeFeats))
            #    new_layer, last_ch_relC = make_layers(new_layer,norm=feat_norm,dropout=True) 
            #    layers+=new_layer
            scaleX=1
            scaleY=1
            for a in merge_featurizer_conv:
                if a=='M' or (type(a) is str and a[0]=='D'):
                    scaleX*=2
                    scaleY*=2
                elif type(a) is str and a[0]=='U':
                    scaleX/=2
                    scaleY/=2
                elif type(a) is str and a[0:4]=='long': #long pool
                    scaleX*=3
                    scaleY*=2
            fsizeX = self.merge_pool_w//scaleX
            fsizeY = self.merge_pool_h//scaleY
            layers.append( nn.AvgPool2d((fsizeY,fsizeX)) )
            self.mergeFeaturizerConv = nn.Sequential(*layers)
            if 'merge_pred_net' in config:
                merge_pred_desc = config['merge_pred_net']#TODO
                if self.reintroduce_features=='map':
                    merge_pred_desc = [last_ch_relC+self.numShapeFeats]+merge_pred_desc+['FCnR1']
                else:
                    merge_pred_desc = [last_ch_relC+self.numShapeFeats,'ReLU']+merge_pred_desc+['FCnR1']
                layers, last_ch = make_layers(merge_pred_desc,norm=feat_norm,dropout=True)
                self.mergepred  = nn.Sequential(*layers)
            else:
                #merge_pred_desc = ['FC{}'.format(last_ch_relC+self.numShapeFeats)]
                layers = [
                        nn.Linear(last_ch_relC+self.numShapeFeats,last_ch_relC+self.numShapeFeats),
                        nn.ReLU(True),
                        nn.Linear(last_ch_relC+self.numShapeFeats,1)
                        ]
                if self.reintroduce_features!='map':
                    layers = [nn.ReLU(True)]+layers
                self.mergepred = nn.Sequential(*layers)

            self.merge_roi_align = RoIAlign((self.merge_pool_h,self.merge_pool_w),1.0/backbone_save_scale,-1)
            if self.use2ndFeatures:
                self.merge_roi_align2 = RoIAlign((self.merge_pool2_h,self.merge_pool2_w),1.0/backbone_save2_scale,-1)


        #if config['graph_config']['arch'][:10]=='BinaryPair' or self.useShapeFeats=='only':
        #    feat_norm_fc=None
        feat_norm_fc = 'group_norm'

        if featurizer_fc is not None:
            if type(self.reintroduce_features) is str and 'map' in self.reintroduce_features:
                featurizer_fc = [rel_featurizer_conv_last+self.numShapeFeats] + featurizer_fc + ['FC{}'.format(graph_in_channels)]
            else:
                featurizer_fc = [rel_featurizer_conv_last+self.numShapeFeats] + featurizer_fc + ['FCnR{}'.format(graph_in_channels)]
            layers, last_ch_rel = make_layers(featurizer_fc,norm=feat_norm_fc,dropout=True) 
            self.relFeaturizerFC = nn.Sequential(*layers)
        else:
            self.relFeaturizerFC = None

        if self.useBBVisualFeats:
            featurizer = config['bb_featurizer_conv'] if 'bb_featurizer_conv' in config else None
            featurizer_fc = config['bb_featurizer_fc'] if 'bb_featurizer_fc' in config else None
            if self.useShapeFeats!='only':
                if featurizer_fc is None:
                    convOut=graph_in_channels-(self.numShapeFeatsBB+self.numTextFeats)
                else:
                    convOut=featurizer_fc[0]-(self.numShapeFeatsBB+self.numTextFeats)
                assert convOut>100,'There should be sufficient visual features. May need to increase graph (in) channels'
                if featurizer is None:
                    convlayers = [ nn.Conv2d(backboneSavedFeatSize+bbMasks_bb,convOut,kernel_size=(2,3)) ]
                    if featurizer_fc is not None:
                        convlayers+=[   nn.GroupNorm(getGroupSize(convOut),convOut),
                                        nn.Dropout2d(p=0.1,inplace=True),
                                        nn.ReLU(inplace=True)
                                    ]
                else:
                    if self.splitFeatures:
                        featurizer_conv2 = config['bb_featurizer_conv_first'] if 'bb_featurizer_conv_first' in config else None
                        featurizer_conv2 = [backboneSavedFeatSize2+bbMasks_bb] + featurizer_conv2 #bbMasks are appended
                        scaleX=1
                        scaleY=1
                        for a in featurizer_conv2:
                            if a=='M' or (type(a) is str and a[0]=='D'):
                                scaleX*=2
                                scaleY*=2
                            elif type(a) is str and a[0]=='U':
                                scaleX/=2
                                scaleY/=2
                            elif type(a) is str and a[0:4]=='long': #long pool
                                scaleX*=3
                                scaleY*=2
                        assert(scaleX==scaleY)
                        splitScaleDiff=scaleX
                        self.poolBB_h = self.poolBB_h//splitScaleDiff
                        self.poolBB_w = self.poolBB_w//splitScaleDiff
                        layers, last_ch_relC = make_layers(featurizer_conv2,norm=feat_norm,dropout=True)
                        self.bbFeaturizerConv2 = nn.Sequential(*layers)

                        featurizer_conv = [backboneSavedFeatSize+last_ch_relC] + featurizer_conv
                    else:
                        featurizer_conv = [backboneSavedFeatSize+bbMasks_bb] + featurizer
                    if featurizer_fc is None:
                        if type(self.reintroduce_features) is str and 'map' in self.reintroduce_features:
                            featurizer_conv += [convOut]
                        else:
                            featurizer_conv += ['C3-{}'.format(convOut)]
                    else:
                         featurizer_conv += [convOut]
                    convlayers, _  = make_layers(featurizer_conv,norm=feat_norm,dropout=True)
                    scaleX=1
                    scaleY=1
                    for a in featurizer_conv:
                        if a=='M' or (type(a) is str and a[0]=='D'):
                            scaleX*=2
                            scaleY*=2
                        elif type(a) is str and a[0]=='U':
                            scaleX/=2
                            scaleY/=2
                        elif type(a) is str and a[0:4]=='long': #long pool
                            scaleX*=3
                            scaleY*=2
                    #self.scale=(scaleX,scaleY) this holds scale for detector
                    fsizeX = self.poolBB_w//scaleX
                    fsizeY = self.poolBB_h//scaleY
                    convlayers.append( nn.AvgPool2d((fsizeY,fsizeX)) )
                self.bbFeaturizerConv = nn.Sequential(*convlayers)

                #self.roi_alignBB = RoIAlign(self.poolBB_h,self.poolBB_w,1.0/backbone_save_scale)
                self.roi_alignBB = RoIAlign((self.poolBB_h,self.poolBB_w),1.0/backbone_save_scale,-1)
                if self.use2ndFeatures:
                    #self.roi_alignBB2 = RoIAlign(self.poolBB2_h,self.poolBB2_w,1.0/backbone_save2_scale)
                    self.roi_alignBB2 = RoIAlign((self.poolBB2_h,self.poolBB2_w),1.0/backbone_save2_scale,-1)
            else:
                featurizer_fc = [self.numShapeFeatsBB+self.numTextFeats]+featurizer_fc
            if featurizer_fc is not None:
                if type(self.reintroduce_features) is str and 'map' in self.reintroduce_features:
                    featurizer_fc = featurizer_fc + ['FC{}'.format(graph_in_channels)] #the noRelu is handeled in remap
                else:
                    featurizer_fc = featurizer_fc + ['FCnR{}'.format(graph_in_channels)]
                layers, last_ch_node = make_layers(featurizer_fc,norm=feat_norm_fc)
                self.bbFeaturizerFC = nn.Sequential(*layers)
            else:
                self.bbFeaturizerFC = None


        #self.pairer = GraphNet(config['graph_config'])
        if type(config['graph_config']) is list:
            self.useMetaGraph = True
            self.graphnets=nn.ModuleList()
            if self.merge_first:
                self.mergeThresh=[config['init_merge_thresh']]
                self.groupThresh=[None]
                self.keepEdgeThresh=[config['init_merge_thresh']] #This is the one actually used, as we only have 1 value predicted by initail merging
            else:
                self.mergeThresh=[]
                self.groupThresh=[]
                self.keepEdgeThresh=[]

            for graphconfig in config['graph_config']:
                self.graphnets.append( eval(graphconfig['arch'])(graphconfig) )
                #self.relThresh.append(graphconfig['rel_thresh'] if 'rel_thresh' in graphconfig else 0.6)
                self.mergeThresh.append(graphconfig['merge_thresh'] if 'merge_thresh' in graphconfig else 0.6)
                self.groupThresh.append(graphconfig['group_thresh'] if 'group_thresh' in graphconfig else 0.6)
                self.keepEdgeThresh.append(graphconfig['keep_edge_thresh'] if 'keep_edge_thresh' in graphconfig else 0.4)
            self.pairer = None

            if type(self.reintroduce_features) is str and 'map' in self.reintroduce_features:
                self.reintroduce_node_visual_maps = nn.ModuleList()
                self.reintroduce_edge_visual_maps = nn.ModuleList()
                self.reintroduce_node_visual_maps.append(nn.Linear(graph_in_channels,graph_in_channels))
                self.reintroduce_edge_visual_maps.append(nn.Linear(graph_in_channels,graph_in_channels))
                for i in range(len(self.graphnets)-1):
                    self.reintroduce_node_visual_maps.append(nn.Linear(graph_in_channels*2,graph_in_channels))
                    self.reintroduce_edge_visual_maps.append(nn.Linear(graph_in_channels*2,graph_in_channels))
                if 'fixed' in self.reintroduce_features:
                    self.reintroduce_node_visual_activations =nn.ModuleList()
                    self.reintroduce_node_visual_activations.append(None)
                    self.reintroduce_edge_visual_activations =nn.ModuleList()
                    self.reintroduce_edge_visual_activations.append(None)
                    for i in range(len(self.graphnets)-1):
                        self.reintroduce_node_visual_activations.append(nn.Sequential(nn.GroupNorm(getGroupSize(graph_in_channels),graph_in_channels),nn.LeakyReLU(0.01,True)))
                        self.reintroduce_edge_visual_activations.append(nn.Sequential(nn.GroupNorm(getGroupSize(graph_in_channels),graph_in_channels),nn.LeakyReLU(0.01,True)))
            else:
                self.reintroduce_node_visual_maps = None
                self.reintroduce_edge_visual_maps = None
            
            if 'group_node_method' not in config or config['group_node_method']=='mean':
                self.groupNodeFunc = lambda l: torch.stack(l,dim=0).mean(dim=0)
            else:
                raise NotImplementedError('Error, unknown node group method: {}'.format(config['group_node_method']))
            if 'group_edge_method' not in config or config['group_edge_method']=='mean':
                self.groupEdgeFunc = lambda l: torch.stack(l,dim=0).mean(dim=0)
            else:
                raise NotImplementedError('Error, unknown edge group method: {}'.format(config['group_edge_method']))
        else:
            self.pairer = eval(config['graph_config']['arch'])(config['graph_config'])
            self.useMetaGraph = type(self.pairer) is MetaGraphNet
        self.fixBiDirection= config['fix_bi_dir'] #should be True unless this is a really old model
        if 'max_graph_size' in config:
            MAX_GRAPH_SIZE = config['max_graph_size']

        self.useOldDecay = config['use_old_len_decay'] if 'use_old_len_decay' in config else False

        self.relationshipProposal= config['relationship_proposal'] if 'relationship_proposal' in config else 'line_of_sight'
        self.include_bb_conf=False if self.legacy else True
        if self.relationshipProposal=='feature_nn':
            self.include_bb_conf=True
            #num_classes = config['num_class']
            num_bb_feat = self.numBBTypes + (1 if self.detector_predNumNeighbors else 0) #config['graph_config']['bb_out']
            prop_feats = 30+2*num_bb_feat
            if self.legacy:
                prop_feats = 26+2*num_bb_feat
            self.no_betweenPixels = not config['use_betweenPixels'] if 'use_betweenPixels' in config else False
            if self.useCurvedBBs:
                prop_feats += 8
                if self.shape_feats_normal:
                    prop_feats += 1

            self.prop_with_text_emb = config['prop_with_text_emb'] if 'prop_with_text_emb' in config else False
            
            if self.prop_with_text_emb:
                prop_feats+= 2*self.numTextFeats
            prop_num_hidden = config['prop_num_hidden'] if 'prop_num_hidden' in config else 64
            self.rel_prop_nn = nn.Sequential(
                                nn.Linear(prop_feats,prop_num_hidden),
                                nn.Dropout(0.25),
                                nn.ReLU(True),
                                nn.Linear(prop_num_hidden,1)
                                )
            if self.merge_first:
                
                self.merge_prop_nn = nn.Sequential(
                                    nn.Linear(prop_feats,64),
                                    nn.Dropout(0.25),
                                    nn.ReLU(True),
                                    nn.Linear(64,1)
                                    )
            self.rel_merge_hard_thresh = config['rel_merge_hard_thresh'] if 'rel_merge_hard_thresh' in config else None
            self.rel_hard_thresh = config['rel_hard_thresh'] if 'rel_hard_thresh' in config else None
            self.percent_rel_to_keep = config['percent_rel_to_keep'] if 'percent_rel_to_keep' in config else 0.2
            self.max_rel_to_keep = config['max_rel_to_keep'] if 'max_rel_to_keep' in config else 3000
            self.max_merge_rel_to_keep = config['max_merge_rel_to_keep'] if 'max_merge_rel_to_keep' in config else 5000
            self.roi_batch_size = config['roi_batch_size'] if 'roi_batch_size' in config else 300


        self.blankRelFeats = config['blankRelFeats'] if 'blankRelFeats' in config else False

        if 'DEBUG' in config:
            self.detector.setDEBUG()
            self.setDEBUG()
            self.debug=True
        else:
            self.debug=False

        if 'change_relationship_proposal' in config:
            self.relationshipProposal = config['change_relationship_proposal']

        #t#self.opt_history=defaultdict(list)#t#

        if type(self.pairer) is BinaryPairReal and type(self.pairer.shape_layers) is not nn.Sequential:
            print("Shape feats aligned to feat dataset.")



    def unfreeze(self): 
        if self.detector_frozen:
            for param in self.detector.parameters(): 
                param.requires_grad=param.will_use_grad 
            self.detector_frozen=False
            print('Unfroze detector')
        

    def forward(self, image, gtBBs=None, gtNNs=None, useGTBBs=False, otherThresh=None, otherThreshIntur=None, hard_detect_limit=5000, debug=False,old_nn=False,gtTrans=None,merge_first_only=False, gtGroups=None):
        assert(image.size(0)==1) #implementation designed for batch size of 1. Should work to do data parallelism, since each copy of the model will get a batch size of 1
        self.merges_performed=0 #just tracking to see if it's working
        #t###tic=timeit.default_timer()#t#
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        if not self.detector.forGraphPairing:
            self.detector.setForGraphPairing(*self.set_detect_params)

        bbPredictions, offsetPredictions, _,_,_,_ = self.detector(image)
        _=None
        #print('detector')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        if self.detector.saved_features is None:
            self.detector.setForGraphPairing(*self.set_detect_params)
            bbPredictions, offsetPredictions, _,_,_,_ = self.detector(image)
        saved_features=self.detector.saved_features
        self.detector.saved_features=None

        if self.use2ndFeatures:
            saved_features2=self.detector.saved_features2
        else:
            saved_features2=None
        #t###print('   detector: {}'.format(timeit.default_timer()-tic))#t#

        if saved_features is None:
            print('ERROR:no saved features!')
            import pdb;pdb.set_trace()

        
        #t###tic=timeit.default_timer()#t#
        if self.useHardConfThresh:
            self.used_threshConf = self.detect_conf_thresh
        else:
            maxConf = bbPredictions[:,:,0].max().item()
            if otherThreshIntur is None:
                confThreshMul = self.detect_conf_thresh
            else:
                confThreshMul = self.detect_conf_thresh*(1-otherThreshIntur) + otherThresh*otherThreshIntur
            self.used_threshConf = max(maxConf*confThreshMul,0.5)

        if self.training:
            self.used_threshConf += np.random.normal(0,0.1) #we'll tweak the threshold around to make training more robust

        ###
        #print('THresh: {}'.format(self.used_threshConf))
        ###

        #t###print('   process boxes: {}'.format(timeit.default_timer()-tic))#t#
        #bbPredictions should be switched for GT for training? Then we can easily use BCE loss. 
        #Otherwise we have to to alignment first



        #if not useGTBBs:
        if self.useCurvedBBs:
            #TODO make this actually check for overseg...
            threshed_bbPredictions = [bbPredictions[0,bbPredictions[0,:,0]>self.used_threshConf].cpu()]
            if self.use_overseg_non_max_sup:
                threshed_bbPredictions[0] = non_max_sup_overseg(threshed_bbPredictions[0])
            bbPredictions = threshed_bbPredictions
        else:
            bbPredictions = non_max_sup_iou(bbPredictions.cpu(),self.used_threshConf,0.4,hard_detect_limit)
        #print(bbPredictions[0].size())

        #I'm assuming batch size of one
        assert(len(bbPredictions)==1)
        bbPredictions=bbPredictions[0]
        if self.no_grad_feats:
            bbPredictions=bbPredictions.detach()


        if useGTBBs and  gtBBs is not None:
            useBBs, gtBBs, gtGroups, gt_to_new = self.alignGTBBs(useGTBBs,gtBBs,gtGroups,bbPredictions)
            if self.training and self.blind_during_gt is not None and self.blind_during_gt>random.random():
                saved_features = saved_features.new_zeros(saved_features.size())
                if saved_features2 is not None:
                    saved_features2 = saved_features2.new_zeros(saved_features2.size())
        else:
            if bbPredictions.size(0)==0:
                return [bbPredictions], offsetPredictions, None, None, None, None, None, None, (None,None,None,None)
            if self.include_bb_conf or self.useCurvedBBs: 
                useBBs = bbPredictions
            else:
                useBBs = bbPredictions[:,1:] #remove confidence score

        useBBs=useBBs.detach()
        #if useGTBBs and self.useCurvedBBs:
        #    useBBs = xyrwhToCurved(useBBs)
        #el
        if self.useCurvedBBs:
            useBBs = [TextLine(bb,step_size=self.text_line_smoothness) for bb in useBBs] #self.x1y1x2y2rToCurved(useBBs)

        if self.text_rec is not None:
            if useGTBBs and gtTrans is not None: # and len(gtTrans)==useBBs.size[0]:
                assert 'word_bbs' not in useGTBBs and not self.useCurvedBBs
                #transcriptions = gtTrans
                transcriptions = ['']*useBBs.size(0)
                for i,trans in enumerate(gtTrans):
                    transcriptions[gt_to_new[i]]=trans
                #transcriptions = [gtTrans[new_to_gt[newi]] if newi in new_to_gt else '' for newi in range(useBBs.size(0))] 
            elif not self.merge_first: #skip if oversegmented, for speed
                transcriptions = self.getTranscriptions(useBBs,image)
                if gtTrans is not None:
                    if self.include_bb_conf:
                        justBBs = useBBs[:,1:]
                    else:
                        justBBs = useBBs
                    transcriptions=correctTrans(transcriptions,justBBs,gtTrans,gtBBs)
            else:
                transcriptions = None
        else:
            transcriptions=None


        if len(useBBs):#useBBs.size(0)>1:
            if transcriptions is not None:
                embeddings = self.embedding_model(transcriptions,saved_features.device)
                if self.add_noise_to_word_embeddings:
                    embeddings += torch.randn_like(embeddings).to(embeddings.device)*self.add_noise_to_word_embeddings*embeddings.mean()
            else:
                embeddings=None


            if not self.useMetaGraph:
                raise NotImplementedError('Simple pairing not implemented for new grouping stuff')

            bbTrans = transcriptions

            allOutputBoxes, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final = self.runGraph(
                    gtGroups,
                    gtTrans,
                    image,
                    useBBs,
                    saved_features,
                    saved_features2,
                    bbTrans,
                    embeddings,
                    merge_first_only,
                    zero_embeddings = self.training and not useGTBBs and self.no_text_during_no_gt is not None and self.no_text_during_no_gt>random.random())

            return allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final

        else:
            if not self.useCurvedBBs and self.detector_predNumNeighbors:
                #Discard NN prediction. We don't use it anymore
                bbPredictions = torch.cat([bbPredictions[:,:6],bbPredictions[:,7:]],dim=1)
                useBBs = torch.cat([useBBs[:,:6],useBBs[:,7:]],dim=1)
            return [bbPredictions], offsetPredictions, None, None, None, None, None, None, (useBBs if self.useCurvedBBs else useBBs.cpu().detach(),None,None,transcriptions)


    #This ROIAligns features and creates mask images for each edge and node, and runs the embedding convnet and [appends?] these features to the graph... This is only neccesary if a node has been updated...
    #perhaps we need a saved visual feature. If the node/edge is updated, it is recomputed. It is appended  to the graphs current features at each call of a GCN
    def appendVisualFeatures(self,
            giter,
            bbs,
            graph,
            groups,
            edge_indexes,
            features,
            features2,
            text_emb,
            image_height,
            image_width,
            same_node_map,
            prev_node_visual_feats,
            prev_edge_visual_feats,
            prev_edge_indexes,
            merge_only=False,
            debug_image=None,
            good_edges=None,
            flip=None):

        node_features, _edge_indexes, edge_features, universal_features = graph
        #same_node_map, maps the old node id (index) to the new one

        node_visual_feats = torch.FloatTensor(node_features.size(0),prev_node_visual_feats.size(1)).to(node_features.device)
        has_feat = [False]*node_features.size(0)
        for old_id,new_id in same_node_map.items():
            has_feat[new_id]=True
            node_visual_feats[new_id] = prev_node_visual_feats[old_id]

        if not all(has_feat):
            #assert (text_emb is not None) == (self.text_rec is not None)
            if text_emb is not None: #self.text_rec is not None:    
                need_new_ids,need_groups,need_text_emb = zip(* [(i,g,t) for i,(has,g,t) in enumerate(zip(has_feat,groups,text_emb)) if not has])
                need_text_emb = torch.stack(need_text_emb,dim=0)
            else:
                need_new_ids,need_groups = zip(* [(i,g) for i,(has,g) in enumerate(zip(has_feat,groups)) if not has])
                need_text_emb = None
            if len(need_new_ids)>0:
                need_new_ids=list(need_new_ids)
                need_new_ids=list(need_new_ids)
                if self.useShapeFeats!='only':
                    allMasks=self.makeAllMasks(image_height,image_width,bbs)
                else:
                    allMasks=None
                node_visual_feats[need_new_ids] = self.computeNodeVisualFeatures(features,features2,image_height,image_width,bbs,need_groups,need_text_emb,allMasks,merge_only,debug_image)

        new_to_old_ids = {v:k for k,v in same_node_map.items()}
        edge_visual_feats = torch.FloatTensor(len(edge_indexes),prev_edge_visual_feats.size(1)).to(edge_features.device)
        if self.fully_connected:
            edge_visual_feats.zero_()
        need_edge_ids=[]
        need_edge_node_ids=[]
        for ei,(n0,n1) in enumerate(edge_indexes):
            if n0 in new_to_old_ids and n1 in new_to_old_ids:
                old_id0 = new_to_old_ids[n0]
                old_id1 = new_to_old_ids[n1]
                try:
                    old_ei =  prev_edge_indexes.index((min(old_id0,old_id1),max(old_id0,old_id1)))
                    edge_visual_feats[ei]=prev_edge_visual_feats[old_ei]
                except ValueError:
                    print('{ERROR ERROR ERROR')
                    print('Edge {} could not be found in prev edges, but is in new as {}'.format((min(old_id0,old_id1),max(old_id0,old_id1)),(n0,n1)))
                    print('ERROR ERROR ERROR}')
                    if not self.fully_connected or good_edges[ei]:
                        need_edge_ids.append(ei)
                        need_edge_node_ids.append((n0,n1))
            elif not self.fully_connected or good_edges[ei]:
                need_edge_ids.append(ei)
                need_edge_node_ids.append((n0,n1))
        if len(need_edge_ids)>0:
            edge_visual_feats[need_edge_ids] = self.computeEdgeVisualFeatures(features,features2,image_height,image_width,bbs,groups,need_edge_node_ids,allMasks,flip,merge_only,debug_image)

        #for now, we'll just sum the features.
        #new_graph = (torch.cat((node_features,node_visual_feats),dim=1),edge_indexes,torch.cat((edge_features,edge_visual_feats),dim=1),universal_features)
        if self.reintroduce_features=='fixed map':
            node_features_old=self.reintroduce_node_visual_activations[giter](node_features)
            edge_features_old=self.reintroduce_edge_visual_activations[giter](edge_features)
            cat_node_f = torch.cat((node_features_old,node_visual_feats),dim=1)
            node_features = self.reintroduce_node_visual_maps[giter](cat_node_f)
            if edge_features.size(1)==0:
                edge_features = edge_visual_feats
                #assert(edge_features.size(0)==0 or edge_features.max()<900)
            elif edge_features.size(0)==edge_visual_feats.size(0)*2:
                edge_features = self.reintroduce_edge_visual_maps[giter](torch.cat((edge_features_old,edge_visual_feats.repeat(2,1)),dim=1))
                #assert(edge_features.size(0)==0 or edge_features.max()<900)

            else:
                edge_features = self.reintroduce_edge_visual_maps[giter](torch.cat((edge_features_old,edge_visual_feats),dim=1))
                #assert(edge_features.size(0)==0 or edge_features.max()<900)
            #assert(node_features.max()<900)
            
        elif self.reintroduce_features=='map':
            node_features_old=node_features
            edge_features_old=edge_features
            cat_node_f = torch.cat((node_features_old,node_visual_feats),dim=1)
            node_features = self.reintroduce_node_visual_maps[giter](cat_node_f)
            if edge_features.size(1)==0:
                edge_features = edge_visual_feats
            elif edge_features.size(0)==edge_visual_feats.size(0)*2:
                edge_features = self.reintroduce_edge_visual_maps[giter](torch.cat((edge_features_old,edge_visual_feats.repeat(2,1)),dim=1))

            else:
                edge_features = self.reintroduce_edge_visual_maps[giter](torch.cat((edge_features_old,edge_visual_feats),dim=1))
        else:
            node_features += node_visual_feats
            if edge_features.size(1)==0:
                edge_features = edge_visual_feats
            elif edge_features.size(0)==edge_visual_feats.size(0)*2:
                edge_features = edge_features+edge_visual_feats.repeat(2,1)
            else:
                edge_features = edge_features+edge_visual_feats
        
        new_graph = (node_features,_edge_indexes,edge_features,universal_features)
        #edge features get repeated for bidirectional graph
        return new_graph, node_visual_feats, edge_visual_feats

    #This rewrites the confidence and class predictions based on the (re)predictions from the graph network
    def updateBBs(self,bbs,groups,nodeOuts):
        if self.useCurvedBBs:
            nodeConfPred = torch.sigmoid(nodeOuts[:,-1,self.nodeIdxConf]).cpu().detach()
            nodeClassPred = torch.sigmoid(nodeOuts[:,-1,self.nodeIdxClass:self.nodeIdxClassEnd].detach()).cpu().detach()
            for i,group in enumerate(groups):
                for bbId in group:
                    bbs[bbId].conf= nodeConfPred[i].numpy()
                    bbs[bbId].cls = nodeClassPred[i].numpy()
                    bbs[bbId].all_conf=None
                    bbs[bbId].all_cls=None
        else:
            if len(bbs)>1:
                nodeConfPred = torch.sigmoid(nodeOuts[:,-1,self.nodeIdxConf:self.nodeIdxConf+1]).cpu()
                bbConfPred = nodeConfPred.new_empty((bbs.size(0),1))#torch.FloatTensor(bbs.size(0),1)
                for i,group in enumerate(groups):
                    bbConfPred[group] = nodeConfPred[i].detach()
                if self.include_bb_conf:
                    bbs[:,0:1] = bbConfPred
                else:
                    bbs = torch.cat((bbConfPred,bbs.cpu()),dim=1) #this is bad, its adding conf when I assume it's not there
                    assert(self.legacy)
            elif bbs.size(0)==1 and not self.include_bb_conf:
                bbs = torch.cat((torch.FloatTensor(1,1).fill_(1).to(bbs.device),bbs),dim=2)

            if self.predNN:
                raise NotImplementedError('Have not implemented num neighbor pred for new graph method')
                
            if self.predClass:
                #if not useGTBBs:
                nodeClassPred = torch.sigmoid(nodeOuts[:,-1,self.nodeIdxClass:self.nodeIdxClassEnd].detach()).cpu()
                bbClasPred = nodeClassPred.new_empty((bbs.size(0),nodeClassPred.size(1)))#torch.FloatTensor(bbs.size(0),nodeClassPred.size(1))
                for i,group in enumerate(groups):
                    bbClasPred[group] = nodeClassPred[i].detach()
                if self.numBBTypes==nodeClassPred.size(1):
                    bbs[:,-self.numBBTypes:] = bbClasPred
                else:
                    diff = self.numBBTypes-nodeClassPred.size(1)
                    bbs[:,-self.numBBTypes:-diff] = bbClasPred
        return bbs

    #This merges two bounding box predictions, assuming they were oversegmented
    def mergeBB(self,bb0,bb1):
        #Get encompassing rectangle for actual bb
        #REctify curved line for ATR
        #scale = self.hw_input_height/crop.size(2)
        #scaled_w = int(crop.size(3)*scale)
        #line[i,:,:,0:scaled_w] = F.interpolate(crop, size=(self.hw_input_height,scaled_w), mode='bilinear')#.to(crop.device)
        #imm[i] = line[i].cpu().numpy().transpose([1,2,0])
        #imm[i] = 256*(2-imm[i])/2

        #if line.size(1)==1 and self.hw_channels==3:
            #line = lines.expand(-1,3,-1,-1)

        if self.rotation:
            raise NotImplementedError('Rotation not implemented for merging bounding boxes')
        else:
            if self.include_bb_conf:
                locIdx=1
                classIdx=6 #if self.legacy else -self.numBBTypes
                conf = (bb0[0:1]+bb1[0:1])/2
            else:
                locIdx=0
                classIdx=5
                #classIdx includes num neighbor pred
            x0,y0,r0,h0,w0 = bb0[locIdx:classIdx]
            x1,y1,r1,h1,w1 = bb1[locIdx:classIdx]
            minX = min(x0-w0,x1-w1)
            maxX = max(x0+w0,x1+w1)
            minY = min(y0-h0,y1-h1)
            maxY = max(y0+h0,y1+h1)

            newW = (maxX-minX)/2
            newH = (maxY-minY)/2
            newX = (maxX+minX)/2
            newY = (maxY+minY)/2

            newClass = (bb0[classIdx:]+bb1[classIdx:])/2

            loc = torch.FloatTensor([newX,newY,0,newH,newW])

            minX=int(minX.item())
            minY=int(minY.item())
            maxX=int(maxX.item())
            maxY=int(maxY.item())

            if self.include_bb_conf:
                bb = torch.cat((conf,loc,newClass),dim=0)
            else:
                bb = torch.cat((loc,newClass),dim=0)
            #if self.text_rec is not None:
            #    crop = image[:,:,minY:maxY+1,minX:maxX+1]
            #    scale = self.hw_input_height/crop.size(2)
            #    line = F.interpolate(crop,scale=scale,mode='bilinear')
            #else:
            #    line=None

        return bb

        #resBatch = self.text_rec(lines)

    #Use the graph network's predictions to merge oversegmented detections and group nodes into a single node
    def mergeAndGroup(self,
            mergeThresh,
            keepEdgeThresh,
            groupThresh,
            oldEdgeIndexes,
            edgePredictions,
            oldGroups,
            oldNodeFeats,
            oldEdgeFeats,
            oldUniversalFeats,
            oldBBs,
            oldBBTrans,
            old_text_emb,
            image,
            skip_rec=False,
            merge_only=False,
            good_edges=None,
            keep_edges=None,
            gt_groups=None,
            final=False):
        #assert(len(oldBBs)==0 or type(oldBBs[0]) is TextLine)
        assert(oldNodeFeats is None or oldGroups is None or oldNodeFeats.size(0)==len(oldGroups))
        oldNumGroups=len(oldGroups)
        #changedNodeIds=set()
        if self.useCurvedBBs:
            bbs={i:TextLine(clone=v) for i,v in enumerate(oldBBs)}
        else:
            oldBBs=oldBBs.cpu()
            bbs={i:v for i,v in enumerate(oldBBs)}
        if self.text_rec is not None and oldBBTrans is not None:
            bbTrans={i:v for i,v  in enumerate(oldBBTrans)}
        else:
            bbTrans=None
        oldToNewBBIndexes={i:i for i in range(len(oldBBs))}
        #newBBs_line={}
        newBBIdCounter=0
        #toMergeBBs={}
        if not merge_only:
            if not final:
                edgePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach() #keep edge pred
            else:
                edgePreds = torch.sigmoid(edgePredictions[:,-1,1]).cpu().detach() #rel pred
            mergePreds = torch.sigmoid(edgePredictions[:,-1,2]).cpu().detach()
            groupPreds = torch.sigmoid(edgePredictions[:,-1,3]).cpu().detach()
            if gt_groups:
                #just rewrite the predictions
                gt_groups_map={}
                for i,group in enumerate(gt_groups):
                    for n in group:
                        gt_groups_map[n]=i
                for i,(n0,n1) in enumerate(oldEdgeIndexes):
                    if gt_groups_map[n0] == gt_groups_map[n1]:
                        groupPreds[i]=1
                    else:
                        groupPreds[i]=0
        else:
            mergePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach()

        if gt_groups is not None:
            mergeThresh=6

        mergedTo=set()
        #check for merges, where we will combine two BBs into one
        for i,(n0,n1) in enumerate(oldEdgeIndexes):
            #mergePred = edgePreds[i,-1,1]
            
            if mergePreds[i]>mergeThresh: #TODO condition this on whether it is correct. and GT?:
                if self.training and random.random()<0.001: #randomly don't merge for robustness in training
                    continue

                if len(oldGroups[n0])==1 and len(oldGroups[n1])==1: #can only merge ungrouped nodes. This assumption is used later in the code WXS
                    #changedNodeIds.add(n0)
                    #changedNodeIds.add(n1)
                    bbId0 = oldGroups[n0][0]
                    bbId1 = oldGroups[n1][0]
                    newId0 = oldToNewBBIndexes[bbId0]
                    bb0ToMerge = bbs[newId0]

                    newId1 = oldToNewBBIndexes[bbId1]
                    bb1ToMerge = bbs[newId1]

                    if self.prevent_vert_merges:
                        #This will introduce slowdowns as we are computing each partail merge instead of waiting till all merges are found
                        angle = (bb0ToMerge.medianAngle()+bb1ToMerge.medianAngle())/2
                        h0 = bb0ToMerge.getHeight()
                        r0 = bb0ToMerge.getReadPosition(angle)
                        h1 = bb1ToMerge.getHeight()
                        r1 = bb1ToMerge.getReadPosition(angle)
                        
                        #if they are horz (read orientation) offset too much (half height), don't merge
                        #x,y = bb0ToMerge.getCenterPoint()
                        #if y>990 and y<1110 and x>800 and x<1380:
                        #    print('{},{}    h0={}, h1={}, r0={}, r1={}, D: {}'.format(int(x),int(y),h0,h1,r0,r1,abs(r0-r1)<(h0+h1)/4))
                        #    print('rot0={}, rot1={}'.format(bb0ToMerge.medianAngle(),bb1ToMerge.medianAngle()))
                        if abs(r0-r1)>(h0+h1)/4:
                            continue



                    if newId0!=newId1:
                        if self.useCurvedBBs:
                            bb0ToMerge.merge(bb1ToMerge) # "
                        else:
                            bbs[newId0]= self.mergeBB(bb0ToMerge,bb1ToMerge)
                        #merge two merged bbs
                        oldToNewBBIndexes = {k:(v if v!=newId1 else newId0) for k,v in oldToNewBBIndexes.items()}
                        del bbs[newId1]
                        #if self.text_rec is not None and not skip_rec:
                        if bbTrans is not None:
                            del bbTrans[newId1]
                        mergedTo.add(newId0)
                        self.merges_performed+=1


        oldBBIdToNew = oldToNewBBIndexes
                

        if self.text_rec is not None and len(bbs)>0 and not skip_rec:
            if merge_only :
                doTransIndexes = [idx for idx in bbs] #everything, since we skip recognition for speed
            else:
                doTransIndexes = [idx for idx in mergedTo if idx in bbs]
            if len(doTransIndexes)>0:
                doBBs = [bbs[idx] for idx in doTransIndexes]
                if not self.useCurvedBBs:
                    doBBs = torch.stack(doBBs,dim=0)
                newTrans = self.getTranscriptions(doBBs,image)
                for i,idx in enumerate(doTransIndexes):
                    bbTrans[idx] = newTrans[i]
        if merge_only:
            newBBs=[]
            newBBTrans=[] if self.text_rec is not None else None
            for bbId,bb in bbs.items():
                newBBs.append(bb)
                if self.text_rec is not None and not skip_rec:
                    newBBTrans.append(bbTrans[bbId])
            return newBBs, newBBTrans

        #rewrite groups with merged instances
        assignedGroup={} #this will allow us to remove merged instances
        oldGroupToNew={}
        workGroups =  {}#{i:v for i,v in enumerate(oldGroups)}
        changedGroups = []
        for id,bbIds in enumerate(oldGroups):
            newGroup = [oldBBIdToNew[oldId] for oldId in bbIds]
            if len(newGroup)==1 and newGroup[0] in assignedGroup: #WXS
                oldGroupToNew[id]=assignedGroup[newGroup[0]]
                changedGroups.append(newGroup[0])
                #nothing else needs done, since the group has the ID,
            else:
                workGroups[id] = newGroup
                for bbId in newGroup:
                    assignedGroup[bbId]=id
    
        newGroupToOldMerge=defaultdict(list) #tracks what has been merged
        for k,v in oldGroupToNew.items():
            newGroupToOldMerge[v].append(k)

        #D#
        for i in range(oldNumGroups):
            assert(i in oldGroupToNew or i in workGroups)

        #We'll adjust the edges to acount for merges as well as prune edges and get ready for grouping
        #temp = oldEdgeIndexes
        #oldEdgeIndexes = []

        #Prune and adjust the edges (to groups)
        groupEdges=[]

        D_numOldEdges=len(oldEdgeIndexes)
        D_numOldAboveThresh=(edgePreds>keepEdgeThresh).sum()
        prunedOldEdgeIndexes=[]
        for i,(n0,n1) in enumerate(oldEdgeIndexes):
            if not merge_only and self.fully_connected and edgePreds[i]>keepEdgeThresh:
                good_edges.append(i)
            if ((keep_edges is not None and i in keep_edges) or 
                    ((not self.fully_connected or not merge_only) and edgePreds[i]>keepEdgeThresh)):
                old_n0=n0
                old_n1=n1
                if n0 in oldGroupToNew:
                    n0 = oldGroupToNew[n0]
                if n1 in oldGroupToNew:
                    n1 = oldGroupToNew[n1]

                assert(n0 in workGroups and n1 in workGroups)
                if n0!=n1:
                    #oldEdgeIndexes.append((n0,n1))
                    groupEdges.append((groupPreds[i].item(),n0,n1))
                #else:
                #    It disapears
                prunedOldEdgeIndexes.append((i,old_n0,old_n1))
            #else: #D#
            #    old_n0=n0
            #    old_n1=n1
            #    if n0 in oldGroupToNew:
            #        n0 = oldGroupToNew[n0]
            #    if n1 in oldGroupToNew:
            #        n1 = oldGroupToNew[n1]
            #    print('pruned [{},{}] n([{},{}])'.format(old_n0,old_n1,n0,n1))

        #print('!D! original edges:{}, above thresh:{}, kept edges:{}'.format(D_numOldEdges,D_numOldAboveThresh,len(groupEdges)))
             



        #Find nodes that should be grouped
        ##NEWER, just merge the groups with the highest score between them. when merging edges, sum the scores
        #newNodeFeats = {i:[oldNodeFeats[i]] for i in range(oldNodeFeats.size(0))}
        oldGroupToNewGrouping = {i:i for i in workGroups.keys()}
        while len(groupEdges)>0:
            groupEdges.sort(key=lambda x:x[0])
            score, g0, g1 = groupEdges.pop()
            assert(g0!=g1)
            if score<groupThresh:
                groupEdges.append((score, g0, g1))
                break
            
            new_g0 = oldGroupToNewGrouping[g0]
            new_g1 = oldGroupToNewGrouping[g1]
            if new_g0!=new_g1:
                workGroups[new_g0] += workGroups[new_g1]
                oldGroupToNewGrouping = {k:(v if v!=new_g1 else new_g0) for k,v in oldGroupToNewGrouping.items()}

                del workGroups[new_g1]



        #D#
        for i in range(oldNumGroups):
            assert(i in oldGroupToNewGrouping or i in oldGroupToNew)


        if gt_groups is not None:
            #check the produced groups to see if they match gt groups
            fix_gg = [] #gt groups not in workGroups (because no edge existed)
            for gg in gt_groups:
                match_found=False
                for id,group in workGroups.items():
                    is_match=True
                    for bb in gg:
                        if bb not in group:
                            is_match=False
                            break
                    if is_match:
                        match_found=True
                        break
                #assert match_found
                if not match_found:
                    fix_gg.append(gg)

            #fix
            for gg in fix_gg:
                w_groups=[]
                for new_g,w_group in workGroups.items():
                    for g_bb in gg:
                        if g_bb in w_group:
                            w_groups.append(new_g)
                            break
                assert len(w_groups)>1
                root_new_g = w_groups[0]
                for new_g in w_groups[1:]:
                    if new_g in workGroups:
                        workGroups[root_new_g] += workGroups[new_g]
                        oldGroupToNewGrouping = {k:(v if v!=new_g else root_new_g) for k,v in oldGroupToNewGrouping.items()}
                        del workGroups[new_g]

            #recheck
            for gg in gt_groups:
                match_found=False
                for id,group in workGroups.items():
                    is_match=True
                    for bb in gg:
                        if bb not in group:
                            is_match=False
                            break
                    if is_match:
                        match_found=True
                        break
                assert match_found


        #Actually change bbs to list,  we'll adjusting appropriate values in groups as we convert groups to list
        bbIdToPos={}
        newBBs=[]
        newBBTrans=[]
        for i,(bbId,bb) in enumerate(bbs.items()):
            bbIdToPos[bbId]=i
            newBBs.append(bb)
            if self.text_rec is not None:
                newBBTrans.append(bbTrans[bbId])

        ##pull the features together for nodes
        #Actually change workGroups to list
        newGroupToOldGrouping=defaultdict(list) #tracks what has been merged
        for k,v in oldGroupToNewGrouping.items():
            newGroupToOldGrouping[v].append(k)
        if oldNodeFeats is not None:
            newNodeFeats = torch.FloatTensor(len(workGroups),oldNodeFeats.size(1)).to(oldNodeFeats.device)
        else:
            newNodeFeats = None
        if old_text_emb is not None and self.text_rec is None:
            new_text_emb = torch.FloatTensor(len(workGroups),old_text_emb.size(1)).to(old_text_emb.device)
        else:
            new_text_emb = None
        oldToNewNodeIds_unchanged={}
        oldToNewIds_all={}
        newGroups=[]
        groupNodeTrans=[]
        for i,(idx,bbIds) in enumerate(workGroups.items()):
            newGroups.append([bbIdToPos[bbId] for bbId in bbIds])
            featsToCombine=[]
            embeddings_to_combine=[]
            for oldNodeIdx in newGroupToOldGrouping[idx]:
                oldToNewIds_all[oldNodeIdx]=i
                featsToCombine.append(oldNodeFeats[oldNodeIdx] if oldNodeFeats is not None else None)
                embeddings_to_combine.append(old_text_emb[oldNodeIdx] if old_text_emb is not None else None)
                if oldNodeIdx in newGroupToOldMerge:
                    for mergedIdx in newGroupToOldMerge[oldNodeIdx]:
                        featsToCombine.append(oldNodeFeats[mergedIdx] if oldNodeFeats is not None else None)
                        embeddings_to_combine.append(old_text_emb[mergedIdx] if old_text_emb is not None else None)
                        oldToNewIds_all[mergedIdx]=i

            if len(featsToCombine)==1:
                oldToNewNodeIds_unchanged[oldNodeIdx]=i
                if oldNodeFeats is not None:
                    newNodeFeats[i]=featsToCombine[0]
                if new_text_emb is not None and self.text_rec is None:
                    new_text_emb[i]=embeddings_to_combine[0]
            else:
                if oldNodeFeats is not None:
                    newNodeFeats[i]=self.groupNodeFunc(featsToCombine)
                if new_text_emb is not None and self.text_rec is None:
                    new_text_emb[i]=torch.stack(embeddings_to_combine,dim=0).mean(dim=0)

            #if oldNodeFeats is not None:
            #    #TEST#
            #    newNodeFeats=newNodeFeats.detach()


            if self.text_rec is not None:
                if self.useCurvedBBs:
                    groupTrans = [(bbs[bbId].getReadPosition(),bbTrans[bbId]) for bbId in bbIds]
                else:
                    groupTrans = [(bbs[bbId][2],bbTrans[bbId]) for bbId in bbIds] #by center y
                groupTrans.sort(key=lambda a:a[0])
                groupNodeTrans.append(' '.join([t[1] for t in groupTrans]))
        #D#
        assert(all([x in oldToNewIds_all for x in range(oldNumGroups)]))

        
        #find overlapped edges and combine
        #first change all node ids to their new ones
        D_newToOld = {v:k for k,v in oldToNewNodeIds_unchanged.items()}
        newEdges_map=defaultdict(list)
        for i,n0,n1  in  prunedOldEdgeIndexes:
            new_n0 = oldToNewIds_all[n0]
            new_n1 = oldToNewIds_all[n1]
            if new_n0 != new_n1:
                newEdges_map[(min(new_n0,new_n1),max(new_n0,new_n1))].append(i)

            #D#
            if new_n0 in D_newToOld and new_n1 in D_newToOld:
                o0 = D_newToOld[new_n0]
                o1 = D_newToOld[new_n1]
                assert( (min(o0,o1),max(o0,o1)) in oldEdgeIndexes )
        #This leaves some old edges pointing to the same new edge, so combine their features
        newEdges=[]
        if oldEdgeFeats is not None:
            newEdgeFeats=torch.FloatTensor(len(newEdges_map),oldEdgeFeats.size(1)).to(oldEdgeFeats.device)
        else:
            newEdgeFeats = None
        if self.fully_connected:
            good_edges_copy = good_edges.copy()
            good_edges.clear() #yes, I'm returning data by chaning the internal state of an argument object. Sorry.
        if keep_edges is not None:
            old_keep_edges=keep_edges
            keep_edges=set()
        for edge,oldIds in newEdges_map.items():
            if oldEdgeFeats is not None:
                if len(oldIds)==1:
                    newEdgeFeats[len(newEdges)]=oldEdgeFeats[oldIds[0]]
                else:
                    newEdgeFeats[len(newEdges)]=self.groupEdgeFunc([oldEdgeFeats[oId] for oId in oldIds])
            if keep_edges is not None:
                if any([oId in old_keep_edges for oId in oldIds]):
                    keep_edges.add(len(newEdges))
            newEdges.append(edge)
            if self.fully_connected:
                good_edges.append(any([good_edges_copy[oId] for oId in oldIds]))



        if self.text_rec is not None and oldNodeFeats is not None:
            newNodeEmbeddings = self.embedding_model(groupNodeTrans,oldNodeFeats.device)
            if self.add_noise_to_word_embeddings>0:
                newNodeEmbeddings += torch.randn_like(newNodeEmbeddings).to(newNodeEmbeddings.device)*self.add_noise_to_word_text_emb
            if self.legacy_read:
                newNodeFeats = self.merge_embedding_layer(torch.cat((newNodeFeats,newNodeEmbeddings),dim=1))
                new_text_emb=old_text_emb
            else:
                new_text_emb = newNodeEmbeddings

        edges = newEdges
        newEdges = list(newEdges) + [(y,x) for x,y in newEdges] #add reverse edges so undirected/bidirectional
        if len(newEdges)>0:
            newEdgeIndexes = torch.LongTensor(newEdges).t()
            if oldEdgeFeats is not None:
                newEdgeIndexes= newEdgeIndexes.to(oldEdgeFeats.device)
        else:
            newEdgeIndexes = torch.LongTensor(0)
        if oldEdgeFeats is not None:
            newEdgeFeats = newEdgeFeats.repeat(2,1)

        newGraph = (newNodeFeats, newEdgeIndexes, newEdgeFeats, oldUniversalFeats)

        ###DEBUG###
        newToOld = {v:k for k,v in oldToNewNodeIds_unchanged.items()}
        for n0,n1 in edges:
            if n0 in newToOld and n1 in newToOld:
                o0 = newToOld[n0]
                o1 = newToOld[n1]
                assert( (min(o0,o1),max(o0,o1)) in oldEdgeIndexes )
        #print('!D! final edges: {}'.format(len(edges)))
        ##D###

        if not self.useCurvedBBs:
            newBBs = torch.stack(newBBs,dim=0)

        return newBBs, newGraph, newGroups, edges, newBBTrans if self.text_rec is not None else None, new_text_emb,  oldToNewNodeIds_unchanged, keep_edges

    def mergeAndGroupLegacy(self,mergeThresh,keepEdgeThresh,groupThresh,oldEdgeIndexes,edgePredictions,oldGroups,oldNodeFeats,oldEdgeFeats,oldUniversalFeats,oldBBs,bbTrans,image,skip_rec=False,merge_only=False,good_edges=None,keep_edges=None):
        if self.useCurvedBBs:
            return self.mergeAndGroupCurved(mergeThresh,keepEdgeThresh,groupThresh,oldEdgeIndexes,edgePredictions,oldGroups,oldNodeFeats,oldEdgeFeats,oldUniversalFeats,oldBBs,bbTrans,image,skip_rec,merge_only,good_edges,keep_edges)
        if keep_edges is not None:
            raise NotImplementedError('graph edge mainining not implemented for normal bbs')
        assert(not self.fully_connected)
        changedNodeIds=set()
        newBBs={}
        #newBBs_line={}
        newBBIdCounter=0
        #toMergeBBs={}
        oldToNewBBIndexes={}
        if not merge_only:
            edgePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach()
            #relPreds = torch.sigmoid(edgePredictions[:,-1,1]).cpu().detach()
            if not self.legacy:
                mergePreds = torch.sigmoid(edgePredictions[:,-1,2]).cpu().detach()
                groupPreds = torch.sigmoid(edgePredictions[:,-1,3]).cpu().detach()
            else:
                mergePreds = torch.sigmoid(edgePredictions[:,-1,1]).cpu().detach()
                groupPreds = torch.sigmoid(edgePredictions[:,-1,2]).cpu().detach()
        else:
            mergePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach()
        ##Prevent all nodes from merging during first iterations (bad init):
    
        #check for merges, where we will combine two BBs into one
        for i,(n0,n1) in enumerate(oldEdgeIndexes):
            #mergePred = edgePreds[i,-1,1]
            
            if mergePreds[i]>mergeThresh: #TODO condition this on whether it is correct. and GT?:
                if len(oldGroups[n0])==1 and len(oldGroups[n1])==1: #can only merge ungrouped nodes. This assumption is used later in the code WXS
                    #changedNodeIds.add(n0)
                    #changedNodeIds.add(n1)
                    bbId0 = oldGroups[n0][0]
                    bbId1 = oldGroups[n1][0]
                    if bbId0 in oldToNewBBIndexes:
                        mergeNewId0 = oldToNewBBIndexes[bbId0]
                        bb0 = newBBs[mergeNewId0]
                    else:
                        mergeNewId0 = None
                        bb0 = oldBBs[bbId0].cpu()
                    if bbId1 in oldToNewBBIndexes:
                        mergeNewId1 = oldToNewBBIndexes[bbId1]
                        bb1 = newBBs[mergeNewId1]
                    else:
                        mergeNewId1 = None
                        bb1 = oldBBs[bbId1].cpu()

                    newBB= self.mergeBB(bb0,bb1)

                    if mergeNewId0 is None and mergeNewId1 is None:
                        oldToNewBBIndexes[bbId0]=newBBIdCounter
                        oldToNewBBIndexes[bbId1]=newBBIdCounter
                        newBBs[newBBIdCounter]=newBB
                        newBBIdCounter+=1
                    elif mergeNewId0 is None:
                        oldToNewBBIndexes[bbId0]=mergeNewId1
                        newBBs[mergeNewId1]=newBB
                    elif mergeNewId1 is None:
                        oldToNewBBIndexes[bbId1]=mergeNewId0
                        newBBs[mergeNewId0]=newBB
                    elif mergeNewId0!=mergeNewId1:
                        #merge two merged bbs
                        oldToNewBBIndexes[bbId1]=mergeNewId0
                        for old,new in oldToNewBBIndexes.items():
                            if new == mergeNewId1:
                                oldToNewBBIndexes[old]=mergeNewId0
                        newBBs[mergeNewId0]=newBB
                        #print('merge {} and {} (d), because of {} and {}'.format(mergeNewId0,mergeNewId1,bbId0,bbId1))
                        del newBBs[mergeNewId1]


        #Actually rewrite bbs
        if len(newBBs)==0:
            bbs = oldBBs
            oldBBIdToNew=list(range(len(oldBBs)))
        else:
            device = oldBBs.device
            bbs=[]
            oldBBIdToNew={}
            if self.text_rec is not None:
                bbTransTmp=[]
            for i in range(len(oldBBs)):
                if i not in oldToNewBBIndexes:
                    oldBBIdToNew[i]=len(bbs)
                    bbs.append(oldBBs[i])
                    if self.text_rec is not None:
                        bbTransTmp.append(bbTrans[i])
            if self.text_rec is not None:
                bbTrans = bbTransTmp
            #oldBBs=oldBBs.cpu()
            for id,bb in newBBs.items():
                for old,new in oldToNewBBIndexes.items():
                    if new==id:
                        oldBBIdToNew[old]=len(bbs)
                bbs.append(bb.to(device))
            bbs=torch.stack(bbs,dim=0)

                

        if self.text_rec is not None and len(newBBs)>0:
            newTrans = self.getTranscriptions(bbs[-len(newBBs):],image)
            #newEmbeddings = self.embedding_model(newTrans)
            #now we need to embed and append these and the old trans to node features
            bbTrans += newTrans
        if merge_only:
            return bbs, bbTrans

        #rewrite groups with merged instances
        assignedGroup={} #this will allow us to remove merged instances
        oldGroupToNew={}
        workGroups =  {}#{i:v for i,v in enumerate(oldGroups)}
        for id,bbIds in enumerate(oldGroups):
            newGroup = [oldBBIdToNew[oldId] for oldId in bbIds]
            if len(newGroup)==1 and newGroup[0] in assignedGroup: #WXS
                oldGroupToNew[id]=assignedGroup[newGroup[0]]
                #del workGroups[id]
            else:
                workGroups[id] = newGroup
                #alreadyInGroup.update(newGroup)
                for bbId in newGroup:
                    assignedGroup[bbId]=id
                #for bbId in bbIds:
                #    assignedGroup[bbId]=id
    

        #rewrite the graph to reflect merged instances
        newNodeFeats=[]
        oldGroups=[]
        newIdToPos={}
        newGroupToOld=defaultdict(list)
        for k,v in oldGroupToNew.items():
            newGroupToOld[v].append(k)
        for id,group in workGroups.items():
            newIdToPos[id]=len(oldGroups)
            oldGroups.append(group)
            if id in newGroupToOld:
                oldNodes = newGroupToOld[id]+[id]
                #random.shuffle(oldNodes) 
                #newNodeFeat = self.groupNodeFunc(oldNodeFeats[oldNodes[0]],oldNodeFeats[oldNodes[1]])
                #for oldNode in oldNodes[2:]:
                #    newNodeFeat = self.groupNodeFunc(newNodeFeat,oldNodeFeats[oldNode])
                if oldNodeFeats is not None:
                    newNodeFeat = self.groupNodeFunc( [oldNodeFeats[on] for on in oldNodes] )
                    newNodeFeats.append(newNodeFeat)
                changedNodeIds.update(oldNodes)
            elif oldNodeFeats is not None:
                newNodeFeats.append(oldNodeFeats[id])
        if oldNodeFeats is not None:
            oldNodeFeats = torch.stack(newNodeFeats,dim=0)

        #We'll adjust the edges to acount for merges as well as prune edges and get ready for grouping
        #temp = oldEdgeIndexes
        #oldEdgeIndexes = []

        #Prune and adjust the edges (to groups)
        groupEdges=[]
        edgeFeats = []

        #D_numOldEdges=len(oldEdgeIndexes)
        #D_numOldAboveThresh=(edgePreds>keepEdgeThresh).sum()
        for i,(n0,n1) in enumerate(oldEdgeIndexes):
            if edgePreds[i]>keepEdgeThresh:
                if n0 in oldGroupToNew:
                    n0=newIdToPos[oldGroupToNew[n0]]
                else:
                    n0 = newIdToPos[n0]
                if n1 in oldGroupToNew:
                    n1=newIdToPos[oldGroupToNew[n1]]
                else:
                    n1 = newIdToPos[n1]
                assert(n0<len(bbs) and n1<len(bbs))
                if n0!=n1:
                    #oldEdgeIndexes.append((n0,n1))
                    groupEdges.append((groupPreds[i].item(),n0,n1))
                    if oldEdgeFeats is not None:
                        edgeFeats.append([oldEdgeFeats[i]])
                #else:
                #    It disapears
        #oldEdgeIndexes=None

        #print('!D! original edges:{}, above thresh:{}, kept edges:{}'.format(D_numOldEdges,D_numOldAboveThresh,len(edgeFeats)))
         
        #else:
        #    #skipping merging
        #    bbs=oldBBs
        #    groupEdges=[]
        #    edgeFeats = []
        #    for i,(n0,n1) in enumerate(oldEdgeIndexes):
        #        if edgePreds[i]>keepEdgeThresh:
        #            groupEdges.append((groupPreds[i].item(),n0,n1))
        #            edgeFeats.append([oldEdgeFeats[i]])
        #    #oldEdgeIndexes=None



        #Find nodes that should be grouped
        ##NEWER, just merge the groups with the highest score between them. when merging edges, sum the scores
        #groupEdges=[]
        #edgeFeats = [[oldEdgeFeats[i]] for i in range(oldEdgeFeats.size(0))]
        if oldNodeFeats is not None:
            newNodeFeats = {i:[oldNodeFeats[i]] for i in range(oldNodeFeats.size(0))}
        workingGroups = {i:v for i,v in enumerate(oldGroups)}
        #for i,(g0,g1) in enumerate(oldEdgeIndexes):
        #    groupEdges.append((groupPreds[i].item(),g0,g1))
        while len(groupEdges)>0:
            groupEdges.sort(key=lambda x:x[0])
            score, g0, g1 = groupEdges.pop()
            if g0==g1:
                continue
            if score<groupThresh:
                groupEdges.append((score, g0, g1))
                break

            changedNodeIds.add(g0)
            changedNodeIds.add(g1)

            workingGroups[g0] += workingGroups[g1]
            del workingGroups[g1]

            #Now combine the edges which the two nodes share (edges going to the same node)
            newGroupEdges=[]
            newEdgeFeats=[]
            mEdges=defaultdict(list)
            mEdgeFeats=defaultdict(list)
            for i,(scoreE,g0E,g1E) in enumerate(groupEdges):
                if g0E==g1E or ( (g0E==g1 or g0E==g0) and (g1E==g1 or g1E==g0) ):
                    continue
                #assert(not( (g0E==g1 or g0E==g0) and (g1E==g1 or g1E==g0) ))
                if g0E==g1 or g0E==g0:
                    mEdges[g1E].append(scoreE)
                    if oldEdgeFeats is not None:
                        mEdgeFeats[g1E] += edgeFeats[i]
                elif g1E==g1 or g1E==g0:
                    mEdges[g0E].append(scoreE)
                    if oldEdgeFeats is not None:
                        mEdgeFeats[g0E] += edgeFeats[i]
                else:
                    newGroupEdges.append((scoreE,g0E,g1E))
                    if oldEdgeFeats is not None:
                        newEdgeFeats.append(edgeFeats[i])

            for g,scores in mEdges.items():
                newGroupEdges.append((np.mean(scores),g0,g))
                newEdgeFeats.append(mEdgeFeats[g])
                #if len(mEdgeFeats[g])>1:
                #    newEdgeFeat = self.groupEdgeFunc(mEdgeFeats[g][0],mEdgeFeats[g][1])
                #    assert(len(mEdgeFeats[g])==2)
                #    #for feat in mEdgeFeats[g][2:]:
                #    #    newEdgeFeat = self.groupEdgeFunc(newEdgeFeat,feat)
                #    newEdgeFeats.append(newEdgeFeat)
                #else:
                #    newEdgeFeats.append(mEdgeFeats[g][0])

            groupEdges=newGroupEdges
            edgeFeats=newEdgeFeats
            assert(len(newEdgeFeats)==len(groupEdges) or oldEdgeFeats is None)

            if oldNodeFeats is not None:
                newNodeFeats[g0] += newNodeFeats[g1] #self.groupNodeFunc(newNodeFeats[g0],newNodeFeats[g1])
                del newNodeFeats[g1]
        #print('!D! num edges after grouping {}'.format(len(groupEdges)))
        
        if oldEdgeFeats is not None:
            newEdgeFeats = [self.groupEdgeFunc(feats) for feats in edgeFeats]

        newNodeFeatsD=newNodeFeats
        newNodeFeats=[]
        if self.text_rec is not None:
            newNodeTrans=[]
        newGroups=[]
        oldToIdx={}
        if self.include_bb_conf:
            yIndex=2
        else:
            yIndex=1
        oldToNewNodeIds={}
        for oldG,bbIds in workingGroups.items():
            oldToIdx[oldG]=len(newGroups)
            newGroups.append(bbIds)
            if oldNodeFeats is not None:
                newNodeFeats.append( self.groupNodeFunc(newNodeFeatsD[oldG]) )
            if oldG not in changedNodeIds:
                oldToNewNodeIds[oldG]=len(newGroups)-1
            if self.text_rec is not None:
                newTrans = ''
                #Something to get read-order correct, assuming groups only vertical, so sorting by y-position
                groupTrans = [(bbs[bbId,yIndex].item(),bbTrans[bbId]) for bbId in bbIds]
                groupTrans.sort(key=lambda a:a[0])
                newNodeTrans.append(' '.join([t[1] for t in groupTrans]))
        newEdges = [(oldToIdx[g0],oldToIdx[g1]) for s,g0,g1 in groupEdges]
        assert(oldEdgeFeats is None or len(newEdgeFeats)==len(newEdges))


        ##import pdb;pdb.set_trace()
        #oldIdToNew = {i:i for i in range(bbs.size(0))}
        ##newIdToOld=defaultdict(list)
        #newIdToOld = {i:[i] for i in range(bbs.size(0))}
        ##newGroups=defaultdict(list) #map new node id->bbIds (new bbs)
        ##for i,group in enumerate(oldGroups):
        ##    newGroups[i]=group
        #newGroups = {i:group for i,group in enumerate(oldGroups)}
        #for i,(n0,n1) in enumerate(oldEdgeIndexes):
        #    groupPred = edgePreds[i,-1,2]
        #    if torch.sigmoid(groupPred)>self.groupThresh:
        #        n0Id=oldIdToNew[n0]
        #        n1Id=oldIdToNew[n1]
        #        if n0Id!=n1Id:
        #            if len(newGroups[n0Id])>=len(newGroups[n1Id]):
        #                newGroups[n0Id] += newGroups[n1Id]
        #                del newGroups[n1Id]
        #                #oldIdToNew[n1]=n0Id
        #                for oldId in newIdToOld[n1Id]:
        #                    oldIdToNew[oldId]=n0Id
        #                    newIdToOld[n0Id].append(oldId)
        #                del newIdToOld[n1Id]
        #                #mergedOrder[n0Id].append(n1)
        #            else:
        #                newGroups[n1Id] += newGroups[n0Id]
        #                del newGroups[n0Id]
        #                #oldIdToNew[n0]=n1Id
        #                for oldId in newIdToOld[n0Id]:
        #                    oldIdToNew[oldId]=n1Id
        #                    newIdToOld[n1Id].append(oldId)
        #                del newIdToOld[n0Id]



        ##Create new graph
        ##(nodeFeatures, edgeIndexes, edgeFeatures, universalFeatures)
        #newIdToOld=defaultdict(list)
        #for old,new in oldIdToNew.items():
        #    newIdToOld[new].append(old)
        #newNodeFeats=[]
        #newNodeIdToPos={}
        #tempGroups=newGroups
        #newGroups=[]
        #for newNode,oldNodes in newIdToOld.items():
        #    if len(oldNodes)>1:
        #        random.shuffle(oldNodes) #TODO something other than random
        #        newNodeFeat = self.groupNodeFunc(oldNodeFeats[oldNodes[0]],oldNodeFeats[oldNodes[1]])
        #        for oldNode in oldNodes[2:]:
        #            newNodeFeat = self.groupNodeFunc(newNodeFeat,oldNodeFeats[oldNode])
        #    else:
        #        newNodeFeat = oldNodeFeats[oldNodes[0]]
        #    newNodeIdToPos[newNode]=len(newNodeFeats)
        #    newNodeFeats.append(newNodeFeat)
        #    newGroups.append(tempGroups[newNode])
        #newNodeFeats = torch.stack(newNodeFeats,dim=0)

        #newEdges=[]
        #newEdgeFeats=[]
        #oldEdgeIds=defaultdict(list)
        #for i,(n0,n1) in enumerate(oldEdgeIndexes):
        #    n0Id=newNodeIdToPos[oldIdToNew[n0]]
        #    n1Id=newNodeIdToPos[oldIdToNew[n1]]
        #    if n0Id!=n1Id:
        #        pair = (min(n0Id,n1Id),max(n0Id,n1Id))
        #        oldEdgeIds[pair].append(i)
        #        #newEdges.add(min(n0Id,n1Id),max(n0Id,n1Id))
        #for newEdge,oldIds in oldEdgeIds.items():
        #    newEdges.append(newEdge)
        #    if len(oldIds)>1:
        #        random.shuffle(oldIds) #TODO something other than random
        #        newEdgeFeat = self.groupEdgeFunc(oldEdgeFeats[oldIds[0]],oldEdgeFeats[oldIds[1]])
        #        for oldEdge in oldIds[2:]:
        #            newEdgeFeat = self.groupEdgeFunc(newEdgeFeat,oldEdgeFeats[oldEdge])
        #    else:
        #        newEdgeFeat = oldEdgeFeats[oldIds[0]]
        #    newEdgeFeats.append(newEdgeFeat)

        if oldNodeFeats is not None:
            newNodeFeats = torch.stack(newNodeFeats,dim=0)
            if self.text_rec is not None:
                newNodeEmbeddings = self.embedding_model(newNodeTrans,oldNodeFeats.device)
                if self.add_noise_to_word_embeddings>0:
                    newNodeEmbeddings += torch.randn_like(newNodeEmbeddings).to(newNodeEmbeddings.device)*self.add_noise_to_word_embeddings
                newNodeFeats = self.merge_embedding_layer(torch.cat((newNodeFeats,newNodeEmbeddings),dim=1))

        if oldEdgeFeats is not None and len(newEdgeFeats)>0:
            newEdgeFeats = torch.stack(newEdgeFeats,dim=0)
        else:
            newEdgeFeats = torch.FloatTensor(0)
        edges = newEdges
        newEdges = list(newEdges) + [(y,x) for x,y in newEdges] #add reverse edges so undirected/bidirectional
        if len(newEdges)>0:
            newEdgeIndexes = torch.LongTensor(newEdges).t()
            if oldEdgeFeats is not None:
                newEdgeIndexes = newEdgeIndexes.to(oldEdgeFeats.device)
        else:
            newEdgeIndexes = torch.LongTensor(0)
        newEdgeFeats = newEdgeFeats.repeat(2,1)

        newGraph = (newNodeFeats, newEdgeIndexes, newEdgeFeats, oldUniversalFeats)

        ###DEBUG###
        #newToOld = {v:k for k,v in oldToNewNodeIds.items()}
        #for n0,n1 in edges:
        #    if n0 in newToOld and n1 in newToOld:
        #        o0 = newToOld[n0]
        #        o1 = newToOld[n1]
        #        assert( (min(o0,o1),max(o0,o1)) in oldEdgeIndexes )

        ##D###

        return bbs, newGraph, newGroups, edges, bbTrans,  None, keep_edges#oldToNewNodeIds


                



    def createGraph(self,bbs,features,features2,imageHeight,imageWidth,text_emb=None,flip=None,debug_image=None,image=None,merge_only=False):
        #t#tic=timeit.default_timer()#t#
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        if self.relationshipProposal == 'line_of_sight':
            assert(not merge_only)
            candidates = self.selectLineOfSightEdges(bbs,imageHeight,imageWidth)
            rel_prop_scores = None
        elif self.relationshipProposal == 'feature_nn':
            candidates, rel_prop_scores = self.selectFeatureNNEdges(bbs,imageHeight,imageWidth,image,features.device,merge_only=merge_only,text_emb=text_emb)
            if self.legacy:
                bbs=bbs[:,1:] #discard confidence, we kept it so the proposer could see them

        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['candidates per bb'].append(time/len(bbs))#t#
        #t#self.opt_history['candidates /bb^2'].append(time/(len(bbs)**2))#t#
        #t#if merge_only:#t#
            #t#self.opt_history['candidates m1st'].append(time)#t#
        if len(candidates)==0:
            if merge_only:
                return None,None,None
            if self.useMetaGraph:
                return None, None, None, None, None, None
            else:
                return None,None,None,None,None, None, None
        if self.training:
            random.shuffle(candidates)

        if not merge_only:
            if self.graph_min_degree:
                candidates,keep_edges = self.makeGraphMinDegree(candidates,bbs)
            elif self.graph_four_connected:
                candidates,keep_edges = self.makeGraphFourConnected(candidates,bbs)
            else:
                keep_edges=None
        #print('proposed relationships: {}  (bbs:{})'.format(len(candidates),len(bbs)))
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        #print('------prop------')

        if (not merge_only and  self.useShapeFeats!='only') or self.merge_use_mask:
            allMasks=self.makeAllMasks(imageHeight,imageWidth,bbs,merge_only)
        else:
            allMasks=None
        groups=[[i] for i in range(len(bbs))]
        edge_vis_features = self.computeEdgeVisualFeatures(features,features2,imageHeight,imageWidth,bbs,groups,candidates,allMasks,flip,merge_only,debug_image)

        #if self.useShapeFeats=='sp
        #print('rel features built')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        #print('------rel------')
        if merge_only:
            return edge_vis_features, candidates, rel_prop_scores #we won't build the graph
        if self.reintroduce_edge_visual_maps is not None:
            rel_features = self.reintroduce_edge_visual_maps[0](edge_vis_features) #this is an extra linear layer to prep the features for the graph (which expects non-activated values)
        else:
            rel_features = edge_vis_features
    
        #compute features for the bounding boxes by themselves
        #This will be replaced with/appended to some type of word embedding
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        node_vis_features = self.computeNodeVisualFeatures(features,features2,imageHeight,imageWidth,bbs,groups,text_emb,allMasks,merge_only,debug_image)
        if self.reintroduce_node_visual_maps is not None:
            #print('node_vis_features: {}'.format(node_vis_features.size()))
            if node_vis_features.size(0)==0:
                print(node_vis_features.size())
            try:
                bb_features = self.reintroduce_node_visual_maps[0](node_vis_features) #this is an extra linear layer to prep the features for the graph (which expects non-activated values)
            except RuntimeError as e:
                print('text_emb = {}'.format(text_emb))
                print('node_vis_features: {}, layer: {}'.format(node_vis_features.size(),self.reintroduce_node_visual_maps[0]))
                raise e
        else:
            bb_features = node_vis_features
        #rint('node features built')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        #print('------node------')
        
        #We're not adding diagonal (self-rels) here!
        #Expecting special handeling during graph conv
        #candidateLocs = torch.LongTensor(candidates).t().to(relFeats.device)
        #ones = torch.ones(len(candidates)).to(relFeats.device)
        #adjacencyMatrix = torch.sparse.FloatTensor(candidateLocs,ones,torch.Size([bbs.size(0),bbs.size(0)]))

        #assert(relFeats.requries_grad)
        #rel_features = torch.sparse.FloatTensor(candidateLocs,relFeats,torch.Size([bbs.size(0),bbs.size(0),relFeats.size(1)]))
        #assert(rel_features.requries_grad)
        relIndexes=candidates
        numBB = len(bbs)
        numRel = len(candidates)
        if self.useMetaGraph:
            nodeFeatures= bb_features
            if self.fully_connected and not merge_only:
                edges = [(a,b) for b in range(numBB) for a in range(b)]
                edgeFeatures= torch.FloatTensor(len(edges),rel_features.size(1)).zero_().to(rel_features.device)
                #TODO this could be optimized
                for i,e in enumerate(candidates):
                    j = edges.index(e)
                    edgeFeatures[j]=rel_features[i]
            else:
                edgeFeatures= rel_features
                edges=candidates

            edges += [(y,x) for x,y in edges] #add backward edges for undirected graph
            edgeIndexes = torch.LongTensor(edges).t().to(rel_features.device)
            #now we need to also replicate the edgeFeatures
            edgeFeatures = edgeFeatures.repeat(2,1)

            #features
            universalFeatures=None

            #t##time = timeit.default_timer()-tic#t#
            #print('   create graph: {}'.format(time)) #old 0.37, new 0.16
            ##self.opt_createG.append(time)
            #t##if len(self.opt_createG)>17:#t#
            #t##    print('   create graph running mean: {}'.format(np.mean(self.opt_createG)))#t#
            #t##    if len(self.opt_createG)>30:#t#
            #t##        self.opt_createG = self.opt_createG[1:]#t#
            return (nodeFeatures, edgeIndexes, edgeFeatures, universalFeatures), relIndexes, rel_prop_scores, node_vis_features,edge_vis_features, keep_edges
        else:
            if bb_features is None:
                numBB=0
                bbAndRel_features=relFeats
                adjacencyMatrix = None
                numOfNeighbors = None
            else:
                bbAndRel_features = torch.cat((bb_features,relFeats),dim=0)
                numOfNeighbors = torch.ones(len(bbs)+len(candidates)) #starts at one for yourself
                edges=[]
                i=0
                for bb1,bb2 in candidates:
                    edges.append( (bb1,numBB+i) )
                    edges.append( (bb2,numBB+i) )
                    numOfNeighbors[bb1]+=1
                    numOfNeighbors[bb2]+=1
                    numOfNeighbors[numBB+i]+=2
                    i+=1
                if self.includeRelRelEdges:
                    relEdges=set()
                    i=0
                    for bb1,bb2 in candidates:
                        j=0
                        for bbA,bbB in candidates[i:]:
                            if i!=j and bb1==bbA or bb1==bbB or bb2==bbA or bb2==bbB:
                                relEdges.add( (numBB+i,numBB+j) ) #i<j always
                            j+=1   
                        i+=1
                    relEdges = list(relEdges)
                    for r1, r2 in relEdges:
                        numOfNeighbors[r1]+=1
                        numOfNeighbors[r2]+=1
                    edges += relEdges
                #add reverse edges
                edges+=[(y,x) for x,y in edges]
                #add diagonal (self edges)
                for i in range(bbAndRel_features.size(0)):
                    edges.append((i,i))

                edgeLocs = torch.LongTensor(edges).t().to(relFeats.device)
                ones = torch.ones(len(edges)).to(relFeats.device)
                adjacencyMatrix = torch.sparse.FloatTensor(edgeLocs,ones,torch.Size([bbAndRel_features.size(0),bbAndRel_features.size(0)]))
                #numOfNeighbors is for convienence in tracking the normalization term
                numOfNeighbors=numOfNeighbors.to(relFeats.device)

            #rel_features = (candidates,relFeats)
            #adjacencyMatrix = None

            return bbAndRel_features, (adjacencyMatrix,numOfNeighbors), numBB, numRel, relIndexes, rel_prop_scores, keep_edges

    def makeAllMasks(self,imageHeight,imageWidth,bbs,merge_only=False):
        if not self.useCurvedBBs:
            if not self.legacy and self.include_bb_conf:
                bbs=bbs[:,1:]
            #get corners from bb predictions
            x = bbs[:,0]
            y = bbs[:,1]
            r = bbs[:,2]
            h = bbs[:,3]
            w = bbs[:,4]
            cos_r = torch.cos(r)
            sin_r = torch.sin(r)
            tlX = -w*cos_r + -h*sin_r +x
            tlY =  w*sin_r + -h*cos_r +y
            trX =  w*cos_r + -h*sin_r +x
            trY = -w*sin_r + -h*cos_r +y
            brX =  w*cos_r + h*sin_r +x
            brY = -w*sin_r + h*cos_r +y
            blX = -w*cos_r + h*sin_r +x
            blY =  w*sin_r + h*cos_r +y

            tlX = tlX.cpu()
            tlY = tlY.cpu()
            trX = trX.cpu()
            trY = trY.cpu()
            blX = blX.cpu()
            blY = blY.cpu()
            brX = brX.cpu()
            brY = brY.cpu()
        #build all-mask image, may want to move this up and use for relationship proposals
        if self.expandedRelContext is not None or self.expandedBBContext is not None:
            allMasks = torch.zeros(imageHeight,imageWidth)
            if merge_only:
                #since each bb fragment is an axis aligned rect, we'll speed things up
                for bb_id in range(len(bbs)):
                    rect=bbs[bb_id].all_primitive_rects[0]
                    lx = max(0,int(rect[0][0]))
                    rx = min(imageWidth,int(rect[1][0]+1))
                    ty = max(0,int(rect[0][1]))
                    by = min(imageHeight,int(rect[2][1]+1))
                    allMasks[ty:by,lx:rx]=1
            else:
                for bbIdx in range(len(bbs)):
                    if self.useCurvedBBs:
                        rr, cc = draw.polygon(bbs[bbIdx].polyYs(),bbs[bbIdx].polyXs(), [imageHeight,imageWidth])
                    else:
                        rr, cc = draw.polygon([tlY[bbIdx],trY[bbIdx],brY[bbIdx],blY[bbIdx]],[tlX[bbIdx],trX[bbIdx],brX[bbIdx],blX[bbIdx]], [imageHeight,imageWidth])
                    allMasks[rr,cc]=1
            return allMasks
        else:
            return None

    def computeEdgeVisualFeatures(self,features,features2,imageHeight,imageWidth,bbs,groups,edges,allMasks,flip,merge_only,debug_image):
        if merge_only:
            pool_h=self.merge_pool_h
            pool_w=self.merge_pool_w
            pool2_h=self.merge_pool2_h
            pool2_w=self.merge_pool2_w
        elif self.useShapeFeats != 'only' and self.useShapeFeats != 'only for edge':
            pool_h=self.pool_h
            pool_w=self.pool_w
            pool2_h=self.pool2_h
            pool2_w=self.pool2_w
        #t##groups_index1 = [ [bbs[b] for b in groups[c[0]]] for c in edges ] #t#
        #t##groups_index2 = [ [bbs[b] for b in groups[c[1]]] for c in edges ] #t#
        #t##tic=timeit.default_timer()#t#
        #t##debug=[groupRect([bb.boundingRect() for bb in group]) for group in groups_index1]#t#
        #t##self.opt_history['computeEdgeFs bb compute{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic) #t#
        #t#tic=timeit.default_timer()#t#

        #stackedEdgeFeatWindows = torch.FloatTensor((len(edges),features.size(1)+2,self.relWindowSize,self.relWindowSize)).to(features.device())

        if not self.useCurvedBBs:
            #get corners from bb predictions
            if self.legacy or not self.include_bb_conf:
                x = bbs[:,0]
                y = bbs[:,1]
                r = bbs[:,2]
                h = bbs[:,3]
                w = bbs[:,4]
            else:
                x = bbs[:,1]
                y = bbs[:,2]
                r = bbs[:,3]
                h = bbs[:,4]
                w = bbs[:,5]
            cos_r = torch.cos(r)
            sin_r = torch.sin(r)
            tlX = -w*cos_r + -h*sin_r +x
            tlY =  w*sin_r + -h*cos_r +y
            trX =  w*cos_r + -h*sin_r +x
            trY = -w*sin_r + -h*cos_r +y
            brX =  w*cos_r + h*sin_r +x
            brY = -w*sin_r + h*cos_r +y
            blX = -w*cos_r + h*sin_r +x
            blY =  w*sin_r + h*cos_r +y

            tlX = tlX.cpu()
            tlY = tlY.cpu()
            trX = trX.cpu()
            trY = trY.cpu()
            blX = blX.cpu()
            blY = blY.cpu()
            brX = brX.cpu()
            brY = brY.cpu()


        if debug_image is not None:
            debug_images=[]
            debug_masks=[]





        groups_index1 = [ [bbs[b] for b in groups[c[0]]] for c in edges ]
        groups_index2 = [ [bbs[b] for b in groups[c[1]]] for c in edges ]
        if not self.useCurvedBBs:
            assert(not self.rotation)
            groupIs_index1 = [ [b for b in groups[c[0]]] for c in edges ]
            groupIs_index2 = [ [b for b in groups[c[1]]] for c in edges ]

        if self.useShapeFeats!='only' and self.useShapeFeats != 'only for edge':
            #get axis aligned rectangle from corners
            #t#tic2=timeit.default_timer()#t#
            rois = torch.zeros((len(edges),5)).to(features.device) #(batchIndex,x1,y1,x2,y2) as expected by ROI Align

            #t#self.opt_history['computeEdgeFs ROISETUP setup{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic2) #t#
            if self.useCurvedBBs:

                #t#tic2=timeit.default_timer()#t#

                min_X1,min_Y1,max_X1,max_Y1 = torch.IntTensor([groupRect([bb.boundingRect() for bb in group]) for group in groups_index1]).permute(1,0).to(features.device).chunk(4,dim=0)
                min_X2,min_Y2,max_X2,max_Y2 = torch.IntTensor([groupRect([bb.boundingRect() for bb in group]) for group in groups_index2]).permute(1,0).to(features.device).chunk(4,dim=0)
                min_X = torch.min(min_X1,min_X2)
                min_Y = torch.min(min_Y1,min_Y2)
                max_X = torch.max(max_X1,max_X2)
                max_Y = torch.max(max_Y1,max_Y2)

                #t#self.opt_history['computeEdgeFs ROISETUP minmax{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic2) #t#



            else:

                min_X1,min_Y1,max_X1,max_Y1 = torch.IntTensor([groupRect([[tlX[b],tlY[b],brX[b],brY[b]] for b in group]) for group in groupIs_index1]).permute(1,0)
                min_X2,min_Y2,max_X2,max_Y2 = torch.IntTensor([groupRect([[tlX[b],tlY[b],brX[b],brY[b]] for b in group]) for group in groupIs_index2]).permute(1,0)
                min_X = torch.min(min_X1,min_X2).to(features.device)
                min_Y = torch.min(min_Y1,min_Y2).to(features.device)
                max_X = torch.max(max_X1,max_X2).to(features.device)
                max_Y = torch.max(max_Y1,max_Y2).to(features.device)
            if merge_only:
                padX = self.expandedMergeContextX
                padY = self.expandedMergeContextY
            else:
                if type(self.expandedRelContext) is list:
                    padY,padX = self.expandedRelContext
                else:
                    padX=padY=  self.expandedRelContext

            D_xs = min_X<=max_X
            D_ys = min_Y<=max_Y
            if not D_xs.all():
                print('bad x')
                print(min_X[~D_xs])
                print(max_X[~D_xs])
            if not D_ys.all():
                print('bad y')
                print(min_Y[~D_ys])
                print(max_Y[~D_ys])
            assert((D_xs).all())
            assert((D_ys).all())

            #t#tic2=timeit.default_timer()#t#
            oneT = torch.FloatTensor([1]).to(features.device)
            zeroT = torch.FloatTensor([1]).to(features.device)
            max_X = torch.max(torch.min((max_X+padX).float(),torch.FloatTensor([imageWidth-1]).to(features.device)),oneT)
            min_X = torch.max(torch.min((min_X-padX).float(),torch.FloatTensor([imageWidth-2]).to(features.device)),zeroT.to(features.device))
            max_Y = torch.max(torch.min((max_Y+padY).float(),torch.FloatTensor([imageHeight-1]).to(features.device)),oneT)
            min_Y = torch.max(torch.min((min_Y-padY).float(),torch.FloatTensor([imageHeight-2]).to(features.device)),zeroT)
            zeroT=oneT=None
            #min_X = torch.max(min_X-padX,torch.IntTensor([0]))
            #max_Y = torch.min(max_Y+padY,torch.IntTensor([imageHeight-1]))
            #min_Y = torch.max(min_Y-padY,torch.IntTensor([0]))
            rois[:,1]=min_X
            rois[:,2]=min_Y
            rois[:,3]=max_X
            rois[:,4]=max_Y
            #t#self.opt_history['computeEdgeFs ROISETUP finalize{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic2) #t#
            
            #t#self.opt_history['computeEdgeFs rois setup{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic) #t#


            ###DEBUG
            if debug_image is not None:
                feature_w = rois[:,3]-rois[:,1] +1
                feature_h = rois[:,4]-rois[:,2] +1
                w_m = pool2_w/feature_w
                h_m = pool2_h/feature_h
                for i in range(4):
                    index1,index2 = edges[i]
                    minY = min_Y[i]
                    minX = min_X[i]
                    maxY = max_Y[i]
                    maxX = max_X[i]
                    #print('crop {}: ({},{}), ({},{})'.format(i,minX.item(),maxX.item(),minY.item(),maxY.item()))
                    #print(bbs[index1])
                    #print(bbs[index2])
                    assert(False)#wrong, not drawing group
                    crop = debug_image[0,:,int(minY):int(maxY),int(minX):int(maxX)+1].cpu()
                    crop = (2-crop)/2
                    if crop.size(0)==1:
                        crop = crop.expand(3,crop.size(1),crop.size(2))
                    if self.useCurvedBBs:
                        rr, cc = draw.polygon(
                                (bbs[index1].polyYs()-rois[i,2].item())*h_m[i].item(),
                                (bbs[index1].polyXs()-rois[i,1].item())*w_m[i].item(), 
                                [pool2_h,pool2_w])
                        crop[0,rr,cc]*=0.5
                        rr, cc = draw.polygon(
                                (bbs[index2].polyYs()-rois[i,2].item())*h_m[i].item(),
                                (bbs[index2].polyXs()-rois[i,1].item())*w_m[i].item(), 
                                [pool2_h,pool2_w])
                        crop[1,rr,cc]*=0.5
                    else:
                        crop[0,int(tlY[index1].item()-minY):int(brY[index1].item()-minY)+1,int(tlX[index1].item()-minX):int(brX[index1].item()-minX)+1]*=0.5
                        crop[1,int(tlY[index2].item()-minY):int(brY[index2].item()-minY)+1,int(tlX[index2].item()-minX):int(brX[index2].item()-minX)+1]*=0.5
                    crop = crop.numpy().transpose([1,2,0])
                    #img_f.imshow('crop {}'.format(i),crop)
                    debug_images.append(crop)
                    #import pdb;pdb.set_trace()
            ###
        #if debug_image is not None:
        #    img_f.waitKey()

        #build all-mask image, may want to move this up and use for relationship proposals
        if (not merge_only and self.useShapeFeats!='only' and self.useShapeFeats != 'only for edge') or (merge_only and self.merge_use_mask):
            if self.expandedRelContext is not None:
                #We're going to add a third mask for all bbs
                numMasks=3
            else:
                numMasks=2
        else:
            numMasks=0

        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        relFeats=[] #where we'll store the feature of each batch
        
        if self.useShapeFeats=='only' or self.useShapeFeats=='only for edge':
            batch_size = len(edges)
        elif merge_only:
            batch_size = 2*self.roi_batch_size
        else:
            batch_size = self.roi_batch_size

        innerbatches = [(s,min(s+batch_size,len(edges))) for s in range(0,len(edges),batch_size)]
        #crop from feats, ROI pool
        for ib,(b_start,b_end) in enumerate(innerbatches): #we can batch extracting computing the feature vector from rois to save memory
            #t#tic=timeit.default_timer()#t#
            if ib>0 and not self.all_grad:
                torch.set_grad_enabled(False)
            if (self.useShapeFeats!='only' and self.useShapeFeats != 'only for edge') or merge_only:
                b_rois = rois[b_start:b_end]
            b_edges = edges[b_start:b_end]
            b_groups_index1 = groups_index1[b_start:b_end]
            b_groups_index2 = groups_index2[b_start:b_end]

            if not self.useCurvedBBs:
                b_groupIs_index1 = groupIs_index1[b_start:b_end]
                b_groupIs_index2 = groupIs_index2[b_start:b_end]

            if self.useShapeFeats:
                shapeFeats = torch.FloatTensor(len(b_edges),self.numShapeFeats)
            if self.detector_predNumNeighbors:
                extraPred=1
            else:
                extraPred=0

            #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            if self.useShapeFeats!='only' and self.useShapeFeats != 'only for edge':
                if merge_only:
                    #o#stackedEdgeFeatWindows = self.merge_roi_align(features,b_rois.to(features.device))
                    stackedEdgeFeatWindows = self.merge_roi_align(features,b_rois)
                else:
                    stackedEdgeFeatWindows = self.roi_align(features,b_rois.to(features.device))
                if features2 is not None:
                    if merge_only:
                        stackedEdgeFeatWindows2 = self.merge_roi_align2(features2,b_rois.to(features.device))
                    else:
                        stackedEdgeFeatWindows2 = self.roi_align2(features2,b_rois.to(features.device))
                    if not self.splitFeatures:
                        stackedEdgeFeatWindows = torch.cat( (stackedEdgeFeatWindows,stackedEdgeFeatWindows2), dim=1)
                        stackedEdgeFeatWindows2=None
                #t#self.opt_history['computeEdge    iter roialign{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic) #t#
                #print('{} roi profile'.format('merge' if merge_only else 'full'))
                #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

                #create and add masks
                if not merge_only or self.merge_use_mask:
                    masks = torch.zeros(stackedEdgeFeatWindows.size(0),numMasks,pool2_h,pool2_w)


                #make instance specific masks and make shape (spatial) features
                if self.useShapeFeats!='only'  and self.useShapeFeats != 'only for edge':
                    if (random.random()<0.5 and flip is None and  not self.debug) or flip:
                        pass
                        #TODO
                    feature_w = b_rois[:,3]-b_rois[:,1] +1
                    feature_h = b_rois[:,4]-b_rois[:,2] +1
                    w_m = pool2_w/feature_w
                    h_m = pool2_h/feature_h

                    #if not self.useCurvedBBs:
                    #    tlX1 = (tlX_index1-b_rois[:,1])*w_m
                    #    trX1 = (trX_index1-b_rois[:,1])*w_m
                    #    brX1 = (brX_index1-b_rois[:,1])*w_m
                    #    blX1 = (blX_index1-b_rois[:,1])*w_m
                    #    tlY1 = (tlY_index1-b_rois[:,2])*h_m
                    #    trY1 = (trY_index1-b_rois[:,2])*h_m
                    #    brY1 = (brY_index1-b_rois[:,2])*h_m
                    #    blY1 = (blY_index1-b_rois[:,2])*h_m
                    #    tlX2 = (tlX_index2-b_rois[:,1])*w_m
                    #    trX2 = (trX_index2-b_rois[:,1])*w_m
                    #    brX2 = (brX_index2-b_rois[:,1])*w_m
                    #    blX2 = (blX_index2-b_rois[:,1])*w_m
                    #    tlY2 = (tlY_index2-b_rois[:,2])*h_m
                    #    trY2 = (trY_index2-b_rois[:,2])*h_m
                    #    brY2 = (brY_index2-b_rois[:,2])*h_m
                    #    blY2 = (blY_index2-b_rois[:,2])*h_m

                if not merge_only or self.merge_use_mask:
                    #t#tic2=timeit.default_timer()#t#
                    for i,(index1, index2) in enumerate(b_edges):
                        if self.useShapeFeats!='only' and self.useShapeFeats != 'only for edge':
                            if self.useCurvedBBs:
                                if merge_only:
                                    #since each bb fragment is an axis aligned rect, we'll speed things up
                                    for bb_id in groups[index1]:
                                        rect=bbs[bb_id].all_primitive_rects[0]
                                        lx = max(0,int(rect[0][0]))
                                        rx = min(pool2_w,int(rect[1][0]+1))
                                        ty = max(0,int(rect[0][1]))
                                        by = min(pool2_h,int(rect[2][1]+1))
                                        masks[i,0,ty:by,lx:rx]=1
                                    for bb_id in groups[index2]:
                                        rect=bbs[bb_id].all_primitive_rects[0]
                                        lx = max(0,int(rect[0][0]))
                                        rx = min(pool2_w,int(rect[1][0]+1))
                                        ty = max(0,int(rect[0][1]))
                                        by = min(pool2_h,int(rect[2][1]+1))
                                        masks[i,1,ty:by,lx:rx]=1
                                else:
                                    for bb_id in groups[index1]:
                                        rr, cc = draw.polygon(
                                                (bbs[bb_id].polyYs()-b_rois[i,2].item())*h_m[i].item(),
                                                (bbs[bb_id].polyXs()-b_rois[i,1].item())*w_m[i].item(), 
                                                [pool2_h,pool2_w])
                                        masks[i,0,rr,cc]=1
                                    for bb_id in groups[index2]:
                                        rr, cc = draw.polygon(
                                                (bbs[bb_id].polyYs()-b_rois[i,2].item())*h_m[i].item(),
                                                (bbs[bb_id].polyXs()-b_rois[i,1].item())*w_m[i].item(), 
                                                [pool2_h,pool2_w])
                                        masks[i,1,rr,cc]=1
                            else:
                                #TODO groups
                                for bb_id in groups[index1]:
                                    rr, cc = draw.polygon(
                                                [round((tlY[bb_id].item()-b_rois[i,2].item())*h_m[i].item()),
                                                 round((trY[bb_id].item()-b_rois[i,2].item())*h_m[i].item()),
                                                 round((brY[bb_id].item()-b_rois[i,2].item())*h_m[i].item()),
                                                 round((blY[bb_id].item()-b_rois[i,2].item())*h_m[i].item())],
                                                [round((tlX[bb_id].item()-b_rois[i,1].item())*w_m[i].item()),
                                                 round((trX[bb_id].item()-b_rois[i,1].item())*w_m[i].item()),
                                                 round((brX[bb_id].item()-b_rois[i,1].item())*w_m[i].item()),
                                                 round((blX[bb_id].item()-b_rois[i,1].item())*w_m[i].item())], 
                                                [pool2_h,pool2_w])
                                    masks[i,0,rr,cc]=1

                                for bb_id in groups[index2]:
                                    rr, cc = draw.polygon(
                                                [round((tlY[bb_id].item()-b_rois[i,2].item())*h_m[i].item()),
                                                 round((trY[bb_id].item()-b_rois[i,2].item())*h_m[i].item()),
                                                 round((brY[bb_id].item()-b_rois[i,2].item())*h_m[i].item()),
                                                 round((blY[bb_id].item()-b_rois[i,2].item())*h_m[i].item())],
                                                [round((tlX[bb_id].item()-b_rois[i,1].item())*w_m[i].item()),
                                                 round((trX[bb_id].item()-b_rois[i,1].item())*w_m[i].item()),
                                                 round((brX[bb_id].item()-b_rois[i,1].item())*w_m[i].item()),
                                                 round((blX[bb_id].item()-b_rois[i,1].item())*w_m[i].item())], 
                                                [pool2_h,pool2_w])
                                    masks[i,1,rr,cc]=1

                            if self.expandedRelContext is not None:
                                cropArea = allMasks[round(b_rois[i,2].item()):round(b_rois[i,4].item())+1,round(b_rois[i,1].item()):round(b_rois[i,3].item())+1]
                                if len(cropArea.shape)==0:
                                    raise ValueError("RoI is bad: {}:{},{}:{} for size {}".format(round(b_rois[i,2].item()),round(b_rois[i,4].item())+1,round(b_rois[i,1].item()),round(b_rois[i,3].item())+1,allMasks.shape))
                                masks[i,2] = F.interpolate(cropArea[None,None,...], size=(pool2_h,pool2_w), mode='bilinear',align_corners=False)[0,0]
                                #masks[i,2] = img_f.resize(cropArea,(stackedEdgeFeatWindows.size(2),stackedEdgeFeatWindows.size(3)))
                                if debug_image is not None:
                                    debug_masks.append(cropArea)
                    
                    #t#self.opt_history['computeEdge    iter mask create{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic2) #t#
                #t#tic2=timeit.default_timer()#t#
        

            if self.useShapeFeats:
                if self.useCurvedBBs:

                    #conf, x,y,r,h,w,tlx,tly,trx,try,brx,bry,blx,bly,r_left,r_right,r_std,read_pos,classFeats = bb.getFeatureInfo()
                    allFeats1 = torch.stack([combineShapeFeats([bb.getFeatureInfo() for bb in group]) for group in b_groups_index1],dim=0)
                    allFeats2 = torch.stack([combineShapeFeats([bb.getFeatureInfo() for bb in group]) for group in b_groups_index2],dim=0)
                    shapeFeats[:,0] = allFeats1[:,4]/self.normalizeVert #height
                    shapeFeats[:,1] = allFeats2[:,4]/self.normalizeVert
                    shapeFeats[:,2] = allFeats1[:,5]/self.normalizeHorz #width
                    shapeFeats[:,3] = allFeats2[:,5]/self.normalizeHorz
                    shapeFeats[:,4] = allFeats1[:,3]/math.pi
                    shapeFeats[:,5] = allFeats2[:,3]/math.pi
                    leftX1 = (allFeats1[:,6]+allFeats1[:,12])/2
                    leftY1 = (allFeats1[:,7]+allFeats1[:,13])/2
                    leftX2 = (allFeats2[:,6]+allFeats2[:,12])/2
                    leftY2 = (allFeats2[:,7]+allFeats2[:,13])/2
                    rightX1 = (allFeats1[:,8]+allFeats1[:,10])/2
                    rightY1 = (allFeats1[:,9]+allFeats1[:,11])/2
                    rightX2 = (allFeats2[:,8]+allFeats2[:,10])/2
                    rightY2 = (allFeats2[:,9]+allFeats2[:,11])/2
                    shapeFeats[:,6] = torch.sqrt( (leftX1-leftX2)**2 + (leftY1-leftY2)**2 )/self.normalizeDist
                    shapeFeats[:,7] = torch.sqrt( (rightX1-rightX2)**2 + (rightY1-rightY2)**2 )/self.normalizeDist
                    shapeFeats[:,8] = torch.sqrt( (leftX1-rightX2)**2 + (leftY1-rightY2)**2 )/self.normalizeDist
                    shapeFeats[:,9] = torch.sqrt( (rightX1-leftX2)**2 + (rightY1-leftY2)**2 )/self.normalizeDist
                    centriodX1 = allFeats1[:,1]
                    centriodY1 = allFeats1[:,2]
                    centriodX2 = allFeats2[:,1]
                    centriodY2 = allFeats2[:,2]
                    shapeFeats[:,10] = torch.sqrt( (centriodX1-centriodX2)**2 + (centriodY1-centriodY2)**2 )/self.normalizeDist
                    shapeFeats[:,11] = torch.sqrt( (centriodX1-leftX2)**2 + (centriodY1-leftY2)**2 )/self.normalizeDist
                    shapeFeats[:,12] = torch.sqrt( (centriodX1-rightX2)**2 + (centriodY1-rightY2)**2 )/self.normalizeDist
                    shapeFeats[:,13] = torch.sqrt( (centriodX2-leftX1)**2 + (centriodY2-leftY1)**2 )/self.normalizeDist
                    shapeFeats[:,14] = torch.sqrt( (centriodX2-rightX1)**2 + (centriodY2-rightY1)**2 )/self.normalizeDist
                    shapeFeats[:,15] = (centriodX1-centriodX2)/self.normalizeDist
                    shapeFeats[:,16] = (centriodY1-centriodY2)/self.normalizeDist
                    shapeFeats[:,17] = allFeats1[:,16] #std angle
                    shapeFeats[:,18] = allFeats2[:,16]

                    #shapeFeats[:,19] = torch.FloatTensor([bb1.getReadPosition()-bb2.getReadPosition() for bb1,bb2 in zip(b_bbs_index1,b_bbs_index2)])/self.normalizeDist #read pos
                    shapeFeats[:,19] = (allFeats1[:,17]-allFeats2[:,17])/self.normalizeDist 
                    if self.numBBTypes==3 and allFeats1[:,18:].size(1)==2:
                        shapeFeats[:,20:20+self.numBBTypes-1] = allFeats1[:,18:]
                        shapeFeats[:,20+self.numBBTypes-1]=0
                        shapeFeats[:,20+self.numBBTypes:20+2*self.numBBTypes-1] = allFeats2[:,18:]
                        shapeFeats[:,20+2*self.numBBTypes-1]=0
                    else:
                        shapeFeats[:,20:20+self.numBBTypes] = allFeats1[:,18:]
                        shapeFeats[:,20+self.numBBTypes:20+2*self.numBBTypes] = allFeats2[:,18:]
                    assert(not torch.isnan(shapeFeats).any())



                else:
                    if type(self.pairer) is BinaryPairReal and type(self.pairer.shape_layers) is not nn.Sequential:
                        #The index specification is to allign with the format feat nets are trained with
                        ixs=[0,1,2,3,3+self.numBBTypes,3+self.numBBTypes,4+self.numBBTypes,5+self.numBBTypes,6+self.numBBTypes,6+2*self.numBBTypes,6+2*self.numBBTypes,7+2*self.numBBTypes]
                    else:
                        ixs=[4,6,2,8,8+self.numBBTypes,5,7,3,8+self.numBBTypes,8+self.numBBTypes+self.numBBTypes,0,1]
                    
                    allFeats1 = torch.stack([combineShapeFeatsTensor([bb for bb in group]) for group in b_groups_index1],dim=0)
                    allFeats2 = torch.stack([combineShapeFeatsTensor([bb for bb in group]) for group in b_groups_index2],dim=0)
                    if self.include_bb_conf and not self.legacy:
                        allFeats1 = allFeats1[:,1:] #discard conf
                        allFeats2 = allFeats2[:,1:] #discard conf

                    shapeFeats[:,ixs[0]] = 2*allFeats1[:,3]/self.normalizeVert #bb preds half height/width
                    shapeFeats[:,ixs[1]] = 2*allFeats1[:,4]/self.normalizeHorz
                    shapeFeats[:,ixs[2]] = allFeats1[:,2]/math.pi
                    if self.legacy:
                        shapeFeats[:,ixs[3]:ixs[4]] = allFeats1[:,extraPred+5:]
                    else:
                        shapeFeats[:,ixs[3]:ixs[4]] = allFeats1[:,-self.numBBTypes:]# torch.sigmoid(allFeats1[:,extraPred+5:])

                    shapeFeats[:,ixs[5]] = 2*allFeats2[:,3]/self.normalizeVert
                    shapeFeats[:,ixs[6]] = 2*allFeats2[:,4]/self.normalizeHorz
                    shapeFeats[:,ixs[7]] = allFeats2[:,2]/math.pi
                    if self.legacy:
                        shapeFeats[:,ixs[8]:ixs[9]] = allFeats2[:,extraPred+5:]
                    else:
                        shapeFeats[:,ixs[8]:ixs[9]] = allFeats2[:,-self.numBBTypes:]#torch.sigmoid(allFeats2[:,extraPred+5:])

                    shapeFeats[:,ixs[10]] = (allFeats1[:,0]-allFeats2[:,0])/self.normalizeHorz
                    shapeFeats[:,ixs[11]] = (allFeats1[:,1]-allFeats2[:,1])/self.normalizeVert
                    if self.useShapeFeats!='old':
                        assert(not self.rotation)
                        tlX_index1=blX_index1 = torch.stack([min([tlX[b] for b in group]) for group in b_groupIs_index1],dim=0)
                        trX_index1=brX_index1 = torch.stack([max([trX[b] for b in group]) for group in b_groupIs_index1],dim=0)
                        tlY_index1=trY_index1 = torch.stack([min([tlY[b] for b in group]) for group in b_groupIs_index1],dim=0)
                        blY_index1=brY_index1 = torch.stack([max([brY[b] for b in group]) for group in b_groupIs_index1],dim=0)

                        tlX_index2=blX_index2 = torch.stack([min([tlX[b] for b in group]) for group in b_groupIs_index2],dim=0)
                        trX_index2=brX_index2 = torch.stack([max([trX[b] for b in group]) for group in b_groupIs_index2],dim=0)
                        tlY_index2=trY_index2 = torch.stack([min([tlY[b] for b in group]) for group in b_groupIs_index2],dim=0)
                        blY_index2=brY_index2 = torch.stack([max([brY[b] for b in group]) for group in b_groupIs_index2],dim=0)

                        startCorners = 8+self.numBBTypes+self.numBBTypes
                        shapeFeats[:,startCorners +0] = torch.sqrt( (tlX_index1-tlX_index2)**2 + (tlY_index1-tlY_index2)**2 )/self.normalizeDist
                        shapeFeats[:,startCorners +1] = torch.sqrt( (trX_index1-trX_index2)**2 + (trY_index1-trY_index2)**2 )/self.normalizeDist
                        shapeFeats[:,startCorners +3] = torch.sqrt( (brX_index1-brX_index2)**2 + (brY_index1-brY_index2)**2 )/self.normalizeDist
                        shapeFeats[:,startCorners +2] = torch.sqrt( (blX_index1-blX_index2)**2 + (blY_index1-blY_index2)**2 )/self.normalizeDist
                        startNN =startCorners+4
                    else:
                        startNN = 8+self.numBBTypes+self.numBBTypes
                    if self.detector_predNumNeighbors:
                        shapeFeats[:,startNN +0] = allFeats1[:,5]
                        shapeFeats[:,startNN +1] = allFeats2[:,5]
                        startPos=startNN+2
                    else:
                        startPos=startNN
                    if self.usePositionFeature:
                        if self.usePositionFeature=='absolute':
                            shapeFeats[:,startPos +0] = (allFeats1[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                            shapeFeats[:,startPos +1] = (allFeats1[:,1]-imageHeight/2)/(10*self.normalizeVert)
                            shapeFeats[:,startPos +2] = (allFeats2[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                            shapeFeats[:,startPos +3] = (allFeats2[:,1]-imageHeight/2)/(10*self.normalizeVert)
                        else:
                            shapeFeats[:,startPos +0] = (allFeats1[:,0]-imageWidth/2)/(imageWidth/2)
                            shapeFeats[:,startPos +1] = (allFeats1[:,1]-imageHeight/2)/(imageHeight/2)
                            shapeFeats[:,startPos +2] = (allFeats2[:,0]-imageWidth/2)/(imageWidth/2)
                            shapeFeats[:,startPos +3] = (allFeats2[:,1]-imageHeight/2)/(imageHeight/2)

            #t#self.opt_history['computeEdge    iter loc feats{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic2) #t#
            ###DEBUG
            if debug_image is not None:
                for i in range(4):
                    img_f.imshow('b{}-{} crop rel {}'.format(b_start,b_end,i),debug_images[i])
                    img_f.imshow('b{}-{} masks rel {}'.format(b_start,b_end,i),masks[i].numpy().transpose([1,2,0]))
                    img_f.imshow('b{}-{} mask all rel {}'.format(b_start,b_end,i),debug_masks[i].numpy())
                img_f.waitKey()
                debug_images=[]


            #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            if self.useShapeFeats!='only' and self.useShapeFeats != 'only for edge':
                if self.splitFeatures:
                    if not merge_only or self.merge_use_mask:
                        stackedEdgeFeatWindows2 = torch.cat((stackedEdgeFeatWindows2,masks.to(stackedEdgeFeatWindows2.device)),dim=1)
                    if merge_only:
                        b_relFeats = self.mergeFeaturizerConv2(stackedEdgeFeatWindows2)
                    else:
                        b_relFeats = self.relFeaturizerConv2(stackedEdgeFeatWindows2)
                    stackedEdgeFeatWindows = torch.cat((stackedEdgeFeatWindows,b_relFeats),dim=1)
                else:
                    if not merge_only or self.merge_use_mask:
                        stackedEdgeFeatWindows = torch.cat((stackedEdgeFeatWindows,masks.to(stackedEdgeFeatWindows.device)),dim=1)
                    #import pdb; pdb.set_trace()
                if merge_only:
                    b_relFeats = self.mergeFeaturizerConv(stackedEdgeFeatWindows) #preparing for graph feature size
                else:
                    b_relFeats = self.relFeaturizerConv(stackedEdgeFeatWindows) #preparing for graph feature size
                b_relFeats = b_relFeats.view(b_relFeats.size(0),b_relFeats.size(1))
                #THESE ARE THE VISUAL FEATURES FOR EDGES, but do we also want to include shape feats?
            #print('{} append, net profile'.format('merge' if merge_only else 'full'))
            #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            #print('b_relFeats {}'.format(b_relFeats.size()))
            #print('shapeFeats {}',format(shapeFeats.size()))
            if self.useShapeFeats:
                if self.useShapeFeats=='only' or self.useShapeFeats=='only for edge':
                    b_relFeats = shapeFeats.to(features.device)
                else:
                    b_relFeats = torch.cat((b_relFeats,shapeFeats.to(features.device)),dim=1)
            assert(not torch.isnan(b_relFeats).any())
            relFeats.append(b_relFeats)
            
            #t#self.opt_history['computeEdgeFs feat iter{}'.format(' m1st' if merge_only else '')].append(timeit.default_timer()-tic) #t#
        if self.training:
            torch.set_grad_enabled(True)
        relFeats = torch.cat(relFeats,dim=0)
        stackedEdgeFeatWindows=None
        stackedEdgeFeatWindows2=None
        b_relFeats=None

        if self.blankRelFeats:
            assert(False and 'code this better')
            relFeats = relFeats.zero_()
        if self.relFeaturizerFC is not None:
            relFeats = self.relFeaturizerFC(relFeats)
        return relFeats


    def computeNodeVisualFeatures(self,features,features2,imageHeight,imageWidth,bbs,groups,text_emb,allMasks,merge_only,debug_image):
        if self.useBBVisualFeats and not merge_only:
            assert(features.size(0)==1)
            if self.useShapeFeats:
                node_shapeFeats=torch.FloatTensor(len(groups),self.numShapeFeatsBB)
            if self.useShapeFeats != "only" and self.expandedBBContext:
                masks = torch.zeros(len(groups),2,self.poolBB2_h,self.poolBB2_w)

            if self.useShapeFeats != "only":
                rois = torch.zeros((len(groups),5))
                if self.useCurvedBBs:
                    min_X,max_X,min_Y,max_Y=torch.IntTensor([
                            minAndMaxXY([bbs[b].boundingRect() for b in group])
                            for group in groups]).permute(1,0)
                else:
                    assert(not self.rotation)
                    if self.legacy or not self.include_bb_conf:
                        x = bbs[:,0]
                        y = bbs[:,1]
                        h = bbs[:,3]
                        w = bbs[:,4]
                    else:
                        x = bbs[:,1]
                        y = bbs[:,2]
                        h = bbs[:,3]
                        w = bbs[:,4]
                    tlX = -w+x
                    tlY = -h+y
                    brX = w+x
                    brY = h+y

                    tlX=blX = tlX.cpu()
                    tlY=trY = tlY.cpu()
                    brX=trX = brX.cpu()
                    brY=blY = brY.cpu()
                    min_X,min_Y,max_X,max_Y = torch.IntTensor([
                            groupRect([[tlX[b],tlY[b],brX[b],brY[b]] for b in group]) 
                            for group in groups]).permute(1,0)

                if self.expandedBBContext is not None:
                    #max_X = torch.min(max_X+self.expandedBBContext,torch.IntTensor([imageWidth-1]))
                    #min_X = torch.max(min_X-self.expandedBBContext,torch.IntTensor([0]))
                    #max_Y = torch.min(max_Y+self.expandedBBContext,torch.IntTensor([imageHeight-1]))
                    #min_Y = torch.max(min_Y-self.expandedBBContext,torch.IntTensor([0]))
                    if type(self.expandedBBContext) is list:
                        padY,padX=self.expandedBBContext
                    else:
                        padY=padX=self.expandedBBContext
                    max_X = torch.max(torch.min(max_X+padX,torch.IntTensor([imageWidth-1])),torch.IntTensor([1]))
                    min_X = torch.max(torch.min(min_X-padX,torch.IntTensor([imageWidth-2])),torch.IntTensor([0]))
                    max_Y = torch.max(torch.min(max_Y+padY,torch.IntTensor([imageHeight-1])),torch.IntTensor([1]))
                    min_Y = torch.max(torch.min(min_Y-padY,torch.IntTensor([imageHeight-2])),torch.IntTensor([0]))
                rois[:,1]=min_X
                rois[:,2]=min_Y
                rois[:,3]=max_X
                rois[:,4]=max_Y

            if self.useShapeFeats:
                if self.useCurvedBBs:
                    allFeats = torch.stack([combineShapeFeats([bbs[bb_id].getFeatureInfo() for bb_id in group]) for group in groups],dim=0)
                    if self.shape_feats_normal:
                        node_shapeFeats[:,0]= allFeats[:,0]
                        node_shapeFeats[:,1]= (allFeats[:,3]+math.pi)/(2*math.pi)
                        node_shapeFeats[:,2]=allFeats[:,4]/self.normalizeVert
                        node_shapeFeats[:,3]=allFeats[:,5]/self.normalizeHorz
                        node_shapeFeats[:,4]=torch.sqrt( ((allFeats[:,6:8]+allFeats[:,8:10])/2 - (allFeats[:,12:14]+allFeats[:,10:12])/2).pow(2).sum(dim=1))/self.normalizeVert
                        node_shapeFeats[:,5]=torch.sqrt( ((allFeats[:,6:8]+allFeats[:,12:14])/2 - (allFeats[:,8:10]+allFeats[:,10:12])/2).pow(2).sum(dim=1))/self.normalizeHorz
                        if self.numBBTypes==3 and allFeats[:,18:].size(1)==2: #catch error in blank detection (but not graph pred)
                            node_shapeFeats[:,6:6+self.numBBTypes-1]=torch.sigmoid(allFeats[:,18:])
                            node_shapeFeats[:,6+self.numBBTypes-1]=0
                        else:
                            node_shapeFeats[:,6:6+self.numBBTypes]=torch.sigmoid(allFeats[:,18:])
                        if self.usePositionFeature:
                            if self.usePositionFeature=='absolute':
                                node_shapeFeats[:,self.numBBTypes+6] = (allFeats[:,1]-imageWidth/2)/(5*self.normalizeHorz)
                                node_shapeFeats[:,self.numBBTypes+7] = (allFeats[:,2]-imageHeight/2)/(10*self.normalizeVert)
                            else:
                                node_shapeFeats[:,self.numBBTypes+6] = (allFeats[:,1]-imageWidth/2)/(imageWidth/2)
                                node_shapeFeats[:,self.numBBTypes+7] = (allFeats[:,2]-imageHeight/2)/(imageHeight/2)
                    else:
                        node_shapeFeats[:,0]= (allFeats[:,3]+math.pi)/(2*math.pi)
                        node_shapeFeats[:,1]=allFeats[:,4]/self.normalizeVert
                        node_shapeFeats[:,2]=allFeats[:,5]/self.normalizeHorz
                        node_shapeFeats[:,3:3+self.numBBTypes]=torch.sigmoid(allFeats[:,18:])
                        assert(not self.usePositionFeature)


                else:
                    allFeats = torch.stack([combineShapeFeatsTensor([bbs[bb_id] for bb_id in group]) for group in groups],dim=0)
                    if self.include_bb_conf and not self.legacy:
                        allFeats=allFeats[:,1:]
                    node_shapeFeats[:,0]= (allFeats[:,2]+math.pi)/(2*math.pi)
                    node_shapeFeats[:,1]=allFeats[:,3]/self.normalizeVert
                    node_shapeFeats[:,2]=allFeats[:,4]/self.normalizeHorz
                    if self.detector_predNumNeighbors:
                        if self.legacy:
                            node_shapeFeats[:,3]=allFeats[:,5]
                        else:
                            node_shapeFeats[:,3]=allFeats[:,-(1+self.numBBTypes)]
                        extraPred=1
                    else:
                        extraPred=0
                    if self.legacy:
                        node_shapeFeats[:,3+extraPred:self.numBBTypes+3+extraPred]=torch.sigmoid(allFeats[:,5+extraPred:self.numBBTypes+5+extraPred])
                    else:
                        node_shapeFeats[:,3+extraPred:self.numBBTypes+3+extraPred]=torch.sigmoid(allFeats[:,-self.numBBTypes:])
                    if self.usePositionFeature:
                        if self.usePositionFeature=='absolute':
                            node_shapeFeats[:,self.numBBTypes+3+extraPred] = (allFeats[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                            node_shapeFeats[:,self.numBBTypes+4+extraPred] = (allFeats[:,1]-imageHeight/2)/(10*self.normalizeVert)
                        else:
                            node_shapeFeats[:,self.numBBTypes+3+extraPred] = (allFeats[:,0]-imageWidth/2)/(imageWidth/2)
                            node_shapeFeats[:,self.numBBTypes+4+extraPred] = (allFeats[:,1]-imageHeight/2)/(imageHeight/2)
            if self.useShapeFeats != "only" and self.expandedBBContext:
                #Add detected BB masks
                #warp to roi space
                feature_w = rois[:,3]-rois[:,1] +1
                feature_h = rois[:,4]-rois[:,2] +1
                w_m = self.poolBB2_w/feature_w
                h_m = self.poolBB2_h/feature_h


                for i in range(len(groups)):
                    if self.useCurvedBBs:
                        for bb_id in groups[i]:
                            rr, cc = draw.polygon(
                                    (bbs[bb_id].polyYs()-rois[i,2].item())*h_m[i].item(),
                                    (bbs[bb_id].polyXs()-rois[i,1].item())*w_m[i].item(), 
                                    [self.poolBB2_h,self.poolBB2_w])
                    else:
                        for bb_id in groups[i]:
                            rr, cc = draw.polygon(
                                        [round((tlY[bb_id].item()-rois[i,2].item())*h_m[i].item()),
                                         round((trY[bb_id].item()-rois[i,2].item())*h_m[i].item()),
                                         round((brY[bb_id].item()-rois[i,2].item())*h_m[i].item()),
                                         round((blY[bb_id].item()-rois[i,2].item())*h_m[i].item())],
                                        [round((tlX[bb_id].item()-rois[i,1].item())*w_m[i].item()),
                                         round((trX[bb_id].item()-rois[i,1].item())*w_m[i].item()),
                                         round((brX[bb_id].item()-rois[i,1].item())*w_m[i].item()),
                                         round((blX[bb_id].item()-rois[i,1].item())*w_m[i].item())], 
                                        [self.poolBB2_h,self.poolBB2_w])
                    masks[i,0,rr,cc]=1
                    if self.expandedBBContext is not None:
                        cropArea = allMasks[round(rois[i,2].item()):round(rois[i,4].item())+1,round(rois[i,1].item()):round(rois[i,3].item())+1]
                        masks[i,1] = F.interpolate(cropArea[None,None,...], size=(self.poolBB2_h,self.poolBB2_w), mode='bilinear',align_corners=False)[0,0]
            
                    ###DEBUG
                    if debug_image is not None and i<5:
                        assert(self.rotation==False)
                        crop = debug_image[0,:,int(minY):int(maxY),int(minX):int(maxX)+1].cpu()
                        crop = (2-crop)/2
                        if crop.size(0)==1:
                            crop = crop.expand(3,crop.size(1),crop.size(2))
                        crop[0,int(tlY[i].item()-minY):int(brY[i].item()-minY)+1,int(tlX[i].item()-minX):int(brX[i].item()-minX)+1]*=0.5
                        crop = crop.numpy().transpose([1,2,0])
                        img_f.imshow('crop bb {}'.format(i),crop)
                        img_f.imshow('masks bb {}'.format(i),torch.cat((masks[i],torch.zeros(1,self.poolBB2_h,self.poolBB2_w)),dim=0).numpy().transpose([1,2,0]))
                        #debug_images.append(crop)

            if debug_image is not None:
                img_f.waitKey()
            if self.useShapeFeats != "only":
                #node_features[i]= F.avg_pool2d(features[0,:,minY:maxY+1,minX:maxX+1], (1+maxY-minY,1+maxX-minX)).view(-1)
                node_features = self.roi_alignBB(features,rois.to(features.device))
                assert(not torch.isnan(node_features).any())
                if features2 is not None:
                    node_features2 = self.roi_alignBB2(features2,rois.to(features.device))
                    if not self.splitFeatures:
                        node_features = torch.cat( (node_features,node_features2), dim=1)
                if self.expandedBBContext:
                    if self.splitFeatures:
                        node_features2 = torch.cat( (node_features2,masks.to(node_features2.device)) ,dim=1)
                        node_features2 = self.bbFeaturizerConv2(node_features2)
                        node_features = torch.cat( (node_features,node_features2), dim=1)
                    else:
                        node_features = torch.cat( (node_features,masks.to(node_features.device)) ,dim=1)
                node_features = self.bbFeaturizerConv(node_features)
                node_features = node_features.view(node_features.size(0),node_features.size(1))
                #THESE ARE THE VISUAL FEATURES FOR NODES
                
                if self.useShapeFeats:
                    node_features = torch.cat( (node_features,node_shapeFeats.to(node_features.device)), dim=1 )
            else:
                assert(self.useShapeFeats)
                node_features = node_shapeFeats.to(features.device)

            if text_emb is not None: #I'll assume the text_emb is just left off if not wanted
                node_features = torch.cat( (node_features,text_emb), dim=1 )

            assert(not torch.isnan(node_features).any())
            if self.bbFeaturizerFC is not None:
                node_features = self.bbFeaturizerFC(node_features) #if uncommented, change rot on node_shapeFeats, maybe not
            assert(not torch.isnan(node_features).any())
        elif text_emb is not None:
            node_features = text_emb
        else:
            node_features = None
        return node_features




    def selectFeatureNNEdges(self,bbs,imageHeight,imageWidth,image,device,merge_only=False,text_emb=False):
        if len(bbs)<2:
            return [], None
        #t#tic=timeit.default_timer()#t#
        
        if self.useCurvedBBs:
            #0: tlXDiff
            #1: trXDiff
            #2: brXDiff
            #3: blXDiff
            #4: centerXDiff
            #5: w1
            #6: w2
            #7: tlYDiff
            #8: trYDiff
            #9: brYDiff
            #10: blYDiff
            #11: centerYDiff
            #12: h1
            #13: h2
            #14: tlDist
            #15: trDist
            #16: brDist
            #17: blDist
            #18: centDist
            #19: rel pos X1
            #20: rel pos Y1
            #21: rel pos X2
            #22: rel pos Y2
            #23: line of sight
            #24: conf1
            #25: conf2
            #26-n: classpred1
            #n+1-m: classpred2

            if merge_only or self.no_betweenPixels:
                line_counts=0
            else:
                #t#tic2=timeit.default_timer()#t#
                line_counts = self.betweenPixels(bbs,image)
                #t#time=timeit.default_timer()-tic2#t#
                #t#self.opt_history['candidates betweenPixels{}'.format(' m1st' if merge_only else '')].append(time) #t#
            numClassFeat = bbs[0].getCls().shape[0]
            
            #conf, x,y,r,h,w,tl, tr, br, bl = torch.FloatTensor([bb.getFeatureInfo() for bb in bbs]).permute(1,0)
            #conf, x,y,r,h,w,tlx,tly,trx,try,brx,bry,blx,bly,r_left,r_rightA,classFeats = bb.getFeatureInfo()

            #t#tic2=timeit.default_timer()#t#
            allFeats = torch.FloatTensor([bb.getFeatureInfo() for bb in bbs])
            #t#time=timeit.default_timer()-tic2#t#
            #t#self.opt_history['candidates getFeatureInfo{}'.format(' m1st' if merge_only else '')].append(time) #t#
            #t#tic2=timeit.default_timer()#t#
            num_bb = allFeats.size(0)
            conf1 = allFeats[:,None,0].expand(-1,num_bb)
            conf2 = allFeats[None,:,0].expand(num_bb,-1)
            x1 = allFeats[:,None,1].expand(-1,num_bb)
            x2 = allFeats[None,:,1].expand(num_bb,-1)
            y1 = allFeats[:,None,2].expand(-1,num_bb)
            y2 = allFeats[None,:,2].expand(num_bb,-1)
            #r1 = allFeats[:,None,3].expand(-1,num_bb)
            #r2 = allFeats[None,:,3].expand(num_bb,-1)
            h1 = allFeats[:,None,4].expand(-1,num_bb)
            h2 = allFeats[None,:,4].expand(num_bb,-1)
            w1 = allFeats[:,None,5].expand(-1,num_bb)
            w2 = allFeats[None,:,5].expand(num_bb,-1)
            tlX1 = allFeats[:,None,6].expand(-1,num_bb)
            tlX2 = allFeats[None,:,6].expand(num_bb,-1)
            tlY1 = allFeats[:,None,7].expand(-1,num_bb)
            tlY2 = allFeats[None,:,7].expand(num_bb,-1)
            trX1 = allFeats[:,None,8].expand(-1,num_bb)
            trX2 = allFeats[None,:,8].expand(num_bb,-1)
            trY1 = allFeats[:,None,9].expand(-1,num_bb)
            trY2 = allFeats[None,:,9].expand(num_bb,-1)
            brX1 = allFeats[:,None,10].expand(-1,num_bb)
            brX2 = allFeats[None,:,10].expand(num_bb,-1)
            brY1 = allFeats[:,None,11].expand(-1,num_bb)
            brY2 = allFeats[None,:,11].expand(num_bb,-1)
            blX1 = allFeats[:,None,12].expand(-1,num_bb)
            blX2 = allFeats[None,:,12].expand(num_bb,-1)
            blY1 = allFeats[:,None,13].expand(-1,num_bb)
            blY2 = allFeats[None,:,13].expand(num_bb,-1)
            classFeat1 = allFeats[:,None,18:].expand(-1,num_bb,-1)
            classFeat2 = allFeats[None,:,18:].expand(num_bb,-1,-1)
            sin_r = torch.sin(allFeats[:,3])
            cos_r = torch.cos(allFeats[:,3])
            cos_r1 = cos_r[:,None].expand(-1,cos_r.size(0))
            cos_r2 = cos_r[None,:].expand(cos_r.size(0),-1)
            sin_r1 = sin_r[:,None].expand(-1,sin_r.size(0))
            sin_r2 = sin_r[None,:].expand(sin_r.size(0),-1)
            sin_r_left = torch.sin(allFeats[:,14])
            cos_r_left = torch.cos(allFeats[:,14])
            sin_r_right = torch.sin(allFeats[:,15])
            cos_r_right = torch.cos(allFeats[:,15])
            cos_r_left1 = cos_r_left[:,None].expand(-1,cos_r_left.size(0))
            cos_r_left2 = cos_r_left[None,:].expand(cos_r_left.size(0),-1)
            sin_r_left1 = sin_r_left[:,None].expand(-1,sin_r_left.size(0))
            sin_r_left2 = sin_r_left[None,:].expand(sin_r_left.size(0),-1)
            cos_r_right1 = cos_r_right[:,None].expand(-1,cos_r_right.size(0))
            cos_r_right2 = cos_r_right[None,:].expand(cos_r_right.size(0),-1)
            sin_r_right1 = sin_r_right[:,None].expand(-1,sin_r_right.size(0))
            sin_r_right2 = sin_r_right[None,:].expand(sin_r_right.size(0),-1)
            ro1 = allFeats[:,None,17].expand(-1,num_bb)
            ro2 = allFeats[None,:,17].expand(num_bb,-1)
            read_order_diff=ro1-ro2

            #t#time=timeit.default_timer()-tic2#t#
            #t#self.opt_history['candidates expand features{}'.format(' m1st' if merge_only else '')].append(time) #t#

        else:
            #features: tlXDiff,trXDiff,brXDiff,blXDiff,tlYDiff,trYDiff,brYDiff,blYDiff, centerXDiff, centerYDiff, absX, absY, h1, w1, h2, w2, classpred1, classpred2, line of sight (binary)

            #0: tlXDiff
            #1: trXDiff
            #2: brXDiff
            #3: blXDiff
            #4: centerXDiff
            #5: w1
            #6: w2
            #7: tlYDiff
            #8: trYDiff
            #9: brYDiff
            #10: blYDiff
            #11: centerYDiff
            #12: h1
            #13: h2
            #14: tlDist
            #15: trDist
            #16: brDist
            #17: blDist
            #18: centDist
            #19: rel pos X1
            #20: rel pos Y1
            #21: rel pos X2
            #22: rel pos Y2
            #23: line of sight
            #24: conf1
            #25: conf2
            #26: sin r 1
            #27: sin r 2
            #28: cos r 1
            #29: cos r 2
            #30-n: classpred1
            #n-m: classpred2
            #if curvedBB:
            #m:m+8: left and right sin/cos

            conf = bbs[:,0]
            x = bbs[:,1]
            y = bbs[:,2]
            r = bbs[:,3]
            h = bbs[:,4]
            w = bbs[:,5]
            classFeat = bbs[:,6:] #this is meant to capture num neighbor pred
            numClassFeat = classFeat.size(1)
            cos_r = torch.cos(r)
            sin_r = torch.sin(r)
            tlX = -w*cos_r + -h*sin_r +x
            tlY =  w*sin_r + -h*cos_r +y
            trX =  w*cos_r + -h*sin_r +x
            trY = -w*sin_r + -h*cos_r +y
            brX =  w*cos_r + h*sin_r +x
            brY = -w*sin_r + h*cos_r +y
            blX = -w*cos_r + h*sin_r +x
            blY =  w*sin_r + h*cos_r +y

            #t#tic=timeit.default_timer()#t#
            line_of_sight = self.selectLineOfSightEdges(bbs,imageHeight,imageWidth,return_all=True)
            #t#print('   candidates line-of-sight: {}'.format(timeit.default_timer()-tic))#t#
            #t#tic=timeit.default_timer()#t#
            conf1 = conf[:,None].expand(-1,conf.size(0))
            conf2 = conf[None,:].expand(conf.size(0),-1)
            x1 = x[:,None].expand(-1,x.size(0))
            x2 = x[None,:].expand(x.size(0),-1)
            y1 = y[:,None].expand(-1,y.size(0))
            y2 = y[None,:].expand(y.size(0),-1)
            r1 = r[:,None].expand(-1,r.size(0))
            r2 = r[None,:].expand(r.size(0),-1)
            h1 = h[:,None].expand(-1,h.size(0))
            h2 = h[None,:].expand(h.size(0),-1)
            w1 = w[:,None].expand(-1,w.size(0))
            w2 = w[None,:].expand(w.size(0),-1)
            classFeat1 = classFeat[:,None].expand(-1,classFeat.size(0),-1)
            classFeat2 = classFeat[None,:].expand(classFeat.size(0),-1,-1)
            cos_r1 = cos_r[:,None].expand(-1,cos_r.size(0))
            cos_r2 = cos_r[None,:].expand(cos_r.size(0),-1)
            sin_r1 = sin_r[:,None].expand(-1,sin_r.size(0))
            sin_r2 = sin_r[None,:].expand(sin_r.size(0),-1)
            tlX1 = tlX[:,None].expand(-1,tlX.size(0))
            tlX2 = tlX[None,:].expand(tlX.size(0),-1)
            tlY1 = tlY[:,None].expand(-1,tlY.size(0))
            tlY2 = tlY[None,:].expand(tlY.size(0),-1)
            trX1 = trX[:,None].expand(-1,trX.size(0))
            trX2 = trX[None,:].expand(trX.size(0),-1)
            trY1 = trY[:,None].expand(-1,trY.size(0))
            trY2 = trY[None,:].expand(trY.size(0),-1)
            brX1 = brX[:,None].expand(-1,brX.size(0))
            brX2 = brX[None,:].expand(brX.size(0),-1)
            brY1 = brY[:,None].expand(-1,brY.size(0))
            brY2 = brY[None,:].expand(brY.size(0),-1)
            blX1 = blX[:,None].expand(-1,blX.size(0))
            blX2 = blX[None,:].expand(blX.size(0),-1)
            blY1 = blY[:,None].expand(-1,blY.size(0))
            blY2 = blY[None,:].expand(blY.size(0),-1)

        #t#tic2 = timeit.default_timer()#t#
        if not self.legacy:
            num_feats = 30+numClassFeat*2
        else:
            num_feats = 26+numClassFeat*2
        if self.useCurvedBBs:
            num_feats+=9
            if not self.shape_feats_normal:
                num_feats-=1
        
        if self.prop_with_text_emb:
            num_feats += 2*self.numTextFeats
        features = torch.FloatTensor(len(bbs),len(bbs), num_feats)
        features[:,:,0] = tlX1-tlX2
        features[:,:,1] = trX1-trX2
        features[:,:,2] = brX1-brX2
        features[:,:,3] = blX1-blX2
        features[:,:,4] = x1-x2
        features[:,:,5] = w1
        features[:,:,6] = w2
        features[:,:,7] = tlY1-tlY2
        features[:,:,8] = trY1-trY2
        features[:,:,9] = brY1-brY2
        features[:,:,10] = blY1-blY2
        features[:,:,11] = y1-y2
        features[:,:,12] = h1
        features[:,:,13] = h2
        features[:,:,14] = torch.sqrt((tlY1-tlY2)**2 + (tlX1-tlX2)**2)
        features[:,:,15] = torch.sqrt((trY1-trY2)**2 + (trX1-trX2)**2)
        features[:,:,16] = torch.sqrt((brY1-brY2)**2 + (brX1-brX2)**2)
        features[:,:,17] = torch.sqrt((blY1-blY2)**2 + (blX1-blX2)**2)
        features[:,:,18] = torch.sqrt((y1-y2)**2 + (x1-x2)**2)
        features[:,:,19] = x1/imageWidth
        features[:,:,20] = y1/imageHeight
        features[:,:,21] = x2/imageWidth
        features[:,:,22] = y2/imageHeight
        #features[:,:,23] = 1 if (index1,index2) in line_of_sight else 0
        if self.useCurvedBBs:
            features[:,:,23] = line_counts
        else:
            features[:,:,23].zero_()
            for index1,index2 in line_of_sight:
                features[index1,index2,23]=1
                features[index2,index1,23]=1
        features[:,:,24] = conf1
        features[:,:,25] = conf2
        if not self.legacy:
            features[:,:,26] = sin_r1
            features[:,:,27] = sin_r2
            features[:,:,28] = cos_r1
            features[:,:,29] = cos_r2
            features[:,:,30:30+numClassFeat] = classFeat1
            features[:,:,30+numClassFeat:30+2*numClassFeat] = classFeat2
        else:
            features[:,:,26:26+numClassFeat] = classFeat1
            features[:,:,26+numClassFeat:] = classFeat2
        if self.useCurvedBBs:
            features[:,:,30+2*numClassFeat] = sin_r_left1
            features[:,:,31+2*numClassFeat] = sin_r_left2
            features[:,:,32+2*numClassFeat] = cos_r_left1
            features[:,:,33+2*numClassFeat] = cos_r_left2
            features[:,:,34+2*numClassFeat] = sin_r_right1
            features[:,:,35+2*numClassFeat] = sin_r_right2
            features[:,:,36+2*numClassFeat] = cos_r_right1
            features[:,:,37+2*numClassFeat] = cos_r_right2
            if self.shape_feats_normal:
                features[:,:,38+2*numClassFeat] = read_order_diff


        #normalize distance features
        features[:,:,0:7]/=self.normalizeHorz
        features[:,:,7:14]/=self.normalizeVert
        features[:,:,14:19]/=(self.normalizeVert+self.normalizeHorz)/2

        if self.prop_with_text_emb:
            reduced_emb = text_emb#self.reduce_text_emb_for_prop(text_emb)

            features[:,:,-2*reduced_emb.size(1):-reduced_emb.size(1)] = reduced_emb[None,:,:]
            features[:,:,-reduced_emb.size(1):] = reduced_emb[:,None,:]

        features = features.view(len(bbs)**2,num_feats) #flatten

        #t#time=timeit.default_timer()-tic2#t#
        #t#self.opt_history['candidates place features{}'.format(' m1st' if merge_only else '')].append(time) #t#
        #t###time = timeit.default_timer()-tic#t#
        #t###print('   candidates feats: {}'.format(time))#t#
        ##self.opt_cand.append(time)
        #t###if len(self.opt_cand)>30:#t#
        #t###    print('   candidates feats running mean: {}'.format(np.mean(self.opt_cand)))#t#
        #t###    self.opt_cand = self.opt_cand[1:]#t#
        #t#tic=timeit.default_timer()#t#
        if merge_only:
            rel_pred = self.merge_prop_nn(features.to(device))
            #features=features.to(device)
            #rel_pred = self.merge_prop_nn(features)
            ##HARD CODED RULES FOR EARLY TRAINING
            #avg_h = features[:,12:14].mean()
            #avg_w = features[:,5:6].mean()
            ##could_merge = ((y1-y2).abs()<4*avg_h).logical_and((x1-x2).abs()<10*avg_w)
            ##could_merge = could_merge.view(-1)[:,None]
            #could_merge = (features[:,11].abs()<4*avg_h).logical_and((features[:,4]).abs()<10*avg_w)[:,None]
            #features=features.cpu()
            #full_rel_pred = rel_pred
            #minV=rel_pred.min()
            #rel_pred=torch.where(could_merge,rel_pred,minV)
            #could_merge=could_merge.cpu()
        else:
            rel_pred = self.rel_prop_nn(features.to(device))

        if self.rel_hard_thresh is not None:
            rel_pred = torch.sigmoid(rel_pred)


        rel_pred2d = rel_pred.view(len(bbs),len(bbs)) #unflatten
        rel_pred2d_comb = (torch.triu(rel_pred2d,diagonal=1)+torch.tril(rel_pred2d,diagonal=-1).permute(1,0))/2
        rel_coords=torch.triu_indices(len(bbs),len(bbs),offset=1)
        rel_pred = rel_pred2d_comb[rel_coords.tolist()]
        #I need to convert to tuples so that later "(x,y) in rels" works
        rel_coords = [(i,j) for i,j in rel_coords.permute(1,0).tolist()]#rel_coords.permute(1,0).tolist()
        #rel_coords = [(i.item(),j.item()) for i,j in rel_coords.permute(1,0)]
        rels_ordered = list(zip(rel_pred.cpu().tolist(),rel_coords))

        #DDDD
        #actual_rels = [(i,j) for i in range(len(bbs)) for j in range(i+1,len(bbs))]
        #rels_ordered_D = [ ((rel_pred2d[rel[0],rel[1]].item()+rel_pred2d[rel[1],rel[0]].item())/2,rel) for rel in actual_rels ]
        #for (score,rel),(scoreD,relD) in zip(rels_ordered,rels_ordered_D):
        #    assert(abs(score-scoreD)<0.00001 and rel==relD)
        #DDDD

        #t#tic=timeit.default_timer()#t#

        if merge_only:
            rel_hard_thresh = self.rel_merge_hard_thresh
        else:
            rel_hard_thresh = self.rel_hard_thresh


        if rel_hard_thresh is not None:
            if self.training:
                rels_ordered.sort(key=lambda x: x[0], reverse=True)
            keep_rels = [r[1] for r in rels_ordered if r[0]>rel_hard_thresh]
            if merge_only:
                max_rel_to_keep = self.max_merge_rel_to_keep
            else:
                max_rel_to_keep = self.max_rel_to_keep
            if self.training:
                max_rel_to_keep *= 4
            keep_rels = keep_rels[:max_rel_to_keep]
            implicit_threshold = rel_hard_thresh
        else:
            rels_ordered.sort(key=lambda x: x[0], reverse=True)
            #t#time=timeit.default_timer()-tic#t#
            #t#self.opt_history['candidates sort{}'.format(' m1st' if merge_only else '')].append(time) #t#

            keep = math.ceil(self.percent_rel_to_keep*len(rels_ordered))
            if merge_only:
                max_rel_to_keep = self.max_merge_rel_to_keep
            else:
                max_rel_to_keep = self.max_rel_to_keep
            if not self.training:
                max_rel_to_keep *= 3
            keep = min(keep,max_rel_to_keep)
            #print('keeping {} of {}'.format(keep,len(rels_ordered)))
            keep_rels = [r[1] for r in rels_ordered[:keep]]
            #if merge_only:
                #print('total rels:{}, keeping:{}, max:{}'.format(len(rels_ordered),keep,max_rel_to_keep))
            if keep<len(rels_ordered):
                implicit_threshold = rels_ordered[keep][0]
            else:
                implicit_threshold = rels_ordered[-1][0]-0.1 #We're taking everything


        #t###print('   candidates net and thresh: {}'.format(timeit.default_timer()-tic))#t#
        return keep_rels, (rel_pred,rel_coords, implicit_threshold)


    def betweenPixels(self,bbs,image):
        #instead just read in mask image?
        TIME_getCenter=[]
        TIME_draw_line=[]
        TIME_sum_pixels=[]
        image=image.cpu() #This will run faster in CPU
        values = torch.FloatTensor(len(bbs),len(bbs)).zero_()
        for i,bb1 in enumerate(bbs[:-1]):
            for j,bb2 in zip(range(i+1,len(bbs)),bbs[i+1:]):
                #t#tic=timeit.default_timer()#t#
                x1,y1 = bb1.getCenterPoint()
                x2,y2 = bb2.getCenterPoint()

                x1 = min(image.size(3)-1,max(0,x1))
                x2 = min(image.size(3)-1,max(0,x2))
                y1 = min(image.size(2)-1,max(0,y1))
                y2 = min(image.size(2)-1,max(0,y2))
                #t#TIME_getCenter.append(timeit.default_timer()-tic)#t#
                #t#tic=timeit.default_timer()#t#
                rr,cc = draw.line(int(round(y1)),int(round(x1)),int(round(y2)),int(round(x2)))
                #t#TIME_draw_line.append(timeit.default_timer()-tic)#t#
                #t#tic=timeit.default_timer()#t#
                v = image[0,:,rr,cc].mean()#.cpu()
                #t#TIME_sum_pixels.append(timeit.default_timer()-tic)#t#
                values[i,j] = v
                values[j,i] = v
        #t###print('    candidates, betweenPixels, getCenter:{}, draw.line:{}, sum pixels:{}'.format(np.mean(TIME_getCenter),np.mean(TIME_draw_line),np.mean(TIME_sum_pixels)))#t#
        return values


    def selectLineOfSightEdges(self,bbs,imageHeight,imageWidth, return_all=False):
        if bbs.size(0)<2:
            return []
        #return list of index pairs

        if self.include_bb_conf and not self.legacy:
            bbs = bbs[:,1:] #remove conf


        sin_r = torch.sin(bbs[:,2])
        cos_r = torch.cos(bbs[:,2])
        #lx = bbs[:,0] - cos_r*bbs[:,4] 
        #ly = bbs[:,1] + sin_r*bbs[:,3]
        #rx = bbs[:,0] + cos_r*bbs[:,4] 
        #ry = bbs[:,1] - sin_r*bbs[:,3]
        #tx = bbs[:,0] - cos_r*bbs[:,4] 
        #ty = bbs[:,1] - sin_r*bbs[:,3]
        #bx = bbs[:,0] + cos_r*bbs[:,4] 
        #by = bbs[:,1] + sin_r*bbs[:,3]
        brX = bbs[:,4]*cos_r-bbs[:,3]*sin_r + bbs[:,0] 
        brY = bbs[:,4]*sin_r+bbs[:,3]*cos_r + bbs[:,1] 
        blX = -bbs[:,4]*cos_r-bbs[:,3]*sin_r + bbs[:,0]
        blY= -bbs[:,4]*sin_r+bbs[:,3]*cos_r + bbs[:,1] 
        trX = bbs[:,4]*cos_r+bbs[:,3]*sin_r + bbs[:,0] 
        trY = bbs[:,4]*sin_r-bbs[:,3]*cos_r + bbs[:,1] 
        tlX = -bbs[:,4]*cos_r+bbs[:,3]*sin_r + bbs[:,0]
        tlY = -bbs[:,4]*sin_r-bbs[:,3]*cos_r + bbs[:,1] 

        minX = min( torch.min(trX), torch.min(tlX), torch.min(blX), torch.min(brX) )
        minY = min( torch.min(trY), torch.min(tlY), torch.min(blY), torch.min(brY) )
        maxX = max( torch.max(trX), torch.max(tlX), torch.max(blX), torch.max(brX) )
        maxY = max( torch.max(trY), torch.max(tlY), torch.max(blY), torch.max(brY) )
        #if (math.isinf(minX) or math.isinf(minY) or math.isinf(maxX) or math.isinf(maxY) ):
        #    import pdb;pdb.set_trace()

        minX = min(max(minX.item(),0),imageWidth)
        minY = min(max(minY.item(),0),imageHeight)
        maxX = min(max(maxX.item(),0),imageWidth)
        maxY = min(max(maxY.item(),0),imageHeight)
        if minX>=maxX or minY>=maxY:
            return []

        #lx-=minX 
        #ly-=minY 
        #rx-=minX 
        #ry-=minY 
        #tx-=minX 
        #ty-=minY 
        #bx-=minX 
        #by-=minY 
        zeros = torch.zeros_like(trX)
        tImageWidth = torch.ones_like(trX)*imageWidth
        tImageHeight = torch.ones_like(trX)*imageHeight
        trX = torch.min(torch.max(trX,zeros),tImageWidth)
        trY = torch.min(torch.max(trY,zeros),tImageHeight)
        tlX = torch.min(torch.max(tlX,zeros),tImageWidth)
        tlY = torch.min(torch.max(tlY,zeros),tImageHeight)
        brX = torch.min(torch.max(brX,zeros),tImageWidth)
        brY = torch.min(torch.max(brY,zeros),tImageHeight)
        blX = torch.min(torch.max(blX,zeros),tImageWidth)
        blY = torch.min(torch.max(blY,zeros),tImageHeight)
        trX-=minX
        trY-=minY
        tlX-=minX
        tlY-=minY
        brX-=minX
        brY-=minY
        blX-=minX
        blY-=minY




        scaleCand = 0.5
        minX*=scaleCand
        minY*=scaleCand
        maxX*=scaleCand
        maxY*=scaleCand
        #lx  *=scaleCand
        #ly  *=scaleCand
        #rx  *=scaleCand
        #ry  *=scaleCand
        #tx  *=scaleCand
        #ty  *=scaleCand
        #bx  *=scaleCand
        #by  *=scaleCand
        trX *=scaleCand
        trY *=scaleCand
        tlX *=scaleCand
        tlY *=scaleCand
        brX *=scaleCand
        brY *=scaleCand
        blX *=scaleCand
        blY *=scaleCand
        h = bbs[:,3]*scaleCand
        w = bbs[:,4]*scaleCand
        r = bbs[:,2]

        distMul=1.0
        while distMul>0.03:

            boxesDrawn = np.zeros( (math.ceil(maxY-minY),math.ceil(maxX-minX)) ,dtype=int)#torch.IntTensor( (maxY-minY,maxX-minX) ).zero_()
            if boxesDrawn.shape[0]==0 or boxesDrawn.shape[1]==0:
                return []
            #import pdb;pdb.set_trace()
            numBoxes = bbs.size(0)
            for i in range(numBoxes):
                
                #img_f.line( boxesDrawn, (int(tlX[i]),int(tlY[i])),(int(trX[i]),int(trY[i])),i,1)
                #img_f.line( boxesDrawn, (int(trX[i]),int(trY[i])),(int(brX[i]),int(brY[i])),i,1)
                #img_f.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(brX[i]),int(brY[i])),i,1)
                #img_f.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(tlX[i]),int(tlY[i])),i,1)

                #These are to catch the wierd case of a (clipped) bb having 0 height or width
                #we just add a bit, this shouldn't greatly effect the heuristic pairing
                if int(tlY[i])==int(trY[i]) and int(tlY[i])==int(brY[i]) and int(tlY[i])==int(blY[i]):
                    if int(tlY[i])<2:
                        blY[i]+=1.1
                        brY[i]+=1.1
                    else:
                        tlY[i]-=1.1
                        trY[i]-=1.1
                if int(tlX[i])==int(trX[i]) and int(tlX[i])==int(brX[i]) and int(tlX[i])==int(blX[i]):
                    if int(tlX[i])<2:
                        trX[i]+=1.1
                        brX[i]+=1.1
                    else:
                        tlX[i]-=1.1
                        blX[i]-=1.1


                rr,cc = draw.polygon_perimeter([int(tlY[i]),int(trY[i]),int(brY[i]),int(blY[i])],[int(tlX[i]),int(trX[i]),int(brX[i]),int(blX[i])],boxesDrawn.shape,True)
                boxesDrawn[rr,cc]=i+1

            #how to walk?
            #walk until number found.
            # if in list, end
            # else add to list, continue
            #list is candidates
            maxDist = 600*scaleCand*distMul
            maxDistY = 200*scaleCand*distMul
            minWidth=30
            minHeight=20
            numFan=5
            
            def pathWalk(myId,startX,startY,angle,distStart=0,splitDist=100):
                hit=set()
                lineId = myId+numBoxes
                if angle<-180:
                    angle+=360
                if angle>180:
                    angle-=360
                if (angle>45 and angle<135) or (angle>-135 and angle<-45):
                    #compute slope based on y stepa
                    yStep=-1
                    #if angle==90 or angle==-90:

                    xStep=1/math.tan(math.pi*angle/180.0)
                else:
                    #compute slope based on x step
                    xStep=1
                    yStep=-math.tan(math.pi*angle/180.0)
                if angle>=135 or angle<-45:
                    xStep*=-1
                    yStep*=-1
                distSoFar=distStart
                prev=0
                numSteps=0
                y=startY
                while distSoFar<maxDist and abs(y-startY)<maxDistY:
                    x=int(round(startX + numSteps*xStep))
                    y=int(round(startY + numSteps*yStep))
                    numSteps+=1
                    if x<0 or y<0 or x>=boxesDrawn.shape[1] or y>=boxesDrawn.shape[0]:
                        break
                    here = boxesDrawn[y,x]
                    #print('{} {} {} : {}'.format(x,y,here,len(hit)))
                    if here>0 and here<=numBoxes and here!=myId:
                        if here in hit and prev!=here:
                            break
                        else:
                            hit.add(here)
                            #print('hit {} at {}, {}  ({})'.format(here,x,y,len(hit)))
                            #elif here == lineId or here == myId:
                            #break
                    else:
                        boxesDrawn[y,x]=lineId
                    prev=here
                    distSoFar= distStart+math.sqrt((x-startX)**2 + (y-startY)**2)

                    #if hitting and maxDist-distSoFar>splitMin and (distSoFar-distStart)>splitDist and len(toSplit)==0:
                    #    #split
                    #    toSplit.append((myId,x,y,angle+45,distSoFar,hit.copy(),splitDist*1.5))
                    #    toSplit.append((myId,x,y,angle-45,distSoFar,hit.copy(),splitDist*1.5))

                return hit

            def fan(boxId,x,y,angle,num,hit):
                deg = 90/(num+1)
                curDeg = angle-45+deg
                for i in range(num):
                    hit.update( pathWalk(boxId,x,y,curDeg) )
                    curDeg+=deg

            def drawIt():
                x = bbs[:,0]*scaleCand - minX
                y = bbs[:,1]*scaleCand - minY
                drawn = np.zeros( (math.ceil(maxY-minY),math.ceil(maxX-minX),3))#torch.IntTensor( (maxY-minY,maxX-minX) ).zero_()
                numBoxes = bbs.size(0)
                for a,b in candidates:
                    img_f.line( drawn, (int(x[a]),int(y[a])),(int(x[b]),int(y[b])),(random.random()*0.5,random.random()*0.5,random.random()*0.5),1)
                for i in range(numBoxes):
                    
                    #img_f.line( boxesDrawn, (int(tlX[i]),int(tlY[i])),(int(trX[i]),int(trY[i])),i,1)
                    #img_f.line( boxesDrawn, (int(trX[i]),int(trY[i])),(int(brX[i]),int(brY[i])),i,1)
                    #img_f.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(brX[i]),int(brY[i])),i,1)
                    #img_f.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(tlX[i]),int(tlY[i])),i,1)

                    rr,cc = draw.polygon_perimeter([int(tlY[i]),int(trY[i]),int(brY[i]),int(blY[i])],[int(tlX[i]),int(trX[i]),int(brX[i]),int(blX[i])])
                    drawn[rr,cc]=(random.random()*0.8+.2,random.random()*0.8+.2,random.random()*0.8+.2)
                img_f.imshow('res',drawn)
                #img_f.waitKey()

                rows,cols=boxesDrawn.shape
                colorMap = [(0,0,0)]
                for i in range(numBoxes):
                    colorMap.append((random.random()*0.8+.2,random.random()*0.8+.2,random.random()*0.8+.2))
                for i in range(numBoxes):
                    colorMap.append( (colorMap[i+1][0]/3,colorMap[i+1][1]/3,colorMap[i+1][2]/3) )
                draw2 = np.zeros((rows,cols,3))
                for r in range(rows):
                    for c in range(cols):
                        draw2[r,c] = colorMap[int(round(boxesDrawn[r,c]))]
                        #draw[r,c] = (255,255,255) if boxesDrawn[r,c]>0 else (0,0,0)

                img_f.imshow('d',draw2)
                img_f.waitKey()


            candidates=set()
            for i in range(numBoxes):
                boxId=i+1
                toSplit=[]
                hit = set()

                horzDiv = 1+math.ceil(w[i]/minWidth)
                vertDiv = 1+math.ceil(h[i]/minHeight)

                if horzDiv==1:
                    leftW=0.5
                    rightW=0.5
                    hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()+90) )
                    hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()-90) )
                else:
                    for j in range(horzDiv):
                        leftW = 1-j/(horzDiv-1)
                        rightW = j/(horzDiv-1)
                        hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()+90) )
                        hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()-90) )

                if vertDiv==1:
                    topW=0.5
                    botW=0.5
                    hit.update( pathWalk(boxId, tlX[i].item()*topW+blX[i].item()*botW, tlY[i].item()*topW+blY[i].item()*botW,r[i].item()+180) )
                    hit.update( pathWalk(boxId, trX[i].item()*topW+brX[i].item()*botW, trY[i].item()*topW+brY[i].item()*botW,r[i].item()) )
                else:
                    for j in range(vertDiv):
                        topW = 1-j/(vertDiv-1)
                        botW = j/(vertDiv-1)
                        hit.update( pathWalk(boxId, tlX[i].item()*topW+blX[i].item()*botW, tlY[i].item()*topW+blY[i].item()*botW,r[i].item()+180) )
                        hit.update( pathWalk(boxId, trX[i].item()*topW+brX[i].item()*botW, trY[i].item()*topW+brY[i].item()*botW,r[i].item()) )
                fan(boxId,tlX[i].item(),tlY[i].item(),r[i].item()+135,numFan,hit)
                fan(boxId,trX[i].item(),trY[i].item(),r[i].item()+45,numFan,hit)
                fan(boxId,blX[i].item(),blY[i].item(),r[i].item()+225,numFan,hit)
                fan(boxId,brX[i].item(),brY[i].item(),r[i].item()+315,numFan,hit)

                for jId in hit:
                    candidates.add( (min(i,jId-1),max(i,jId-1)) )
            
            #print('candidates:{} ({})'.format(len(candidates),distMul))
            #if len(candidates)>1:
            #    drawIt()
            if (len(candidates)+numBoxes<MAX_GRAPH_SIZE and len(candidates)<MAX_CANDIDATES) or return_all:
                return list(candidates)
            else:
                if self.useOldDecay:
                    distMul*=0.75
                else:
                    distMul=distMul*0.8 - 0.05
        #This is a problem, we couldn't prune down enough
        print("ERROR: could not prune number of candidates down: {} (should be {})".format(len(candidates),MAX_GRAPH_SIZE-numBoxes))
        return list(candidates)[:MAX_GRAPH_SIZE-numBoxes]

    #def textRecWrap(task):
    #    i,image = task
    #    if image.size(3)>0:
    #        return i,self.text_rec

    def getTranscriptions(self,bbs,image):
        if self.useCurvedBBs:
            assert(image.size(0)==1) #single imag
            self.text_rec.eval()
            #build batch
            max_w=0
            grids=[]
            for bb in bbs:
                grid = bb.getGrid(self.hw_input_height,image.device)
                max_w = max(max_w,grid.size(1))
                grids.append(grid)

            #batch the grids together padding to same length
            if self.atr_batch_size>1:
                to_pad = [max_w-g.size(1) for g in grids]
                grids = [F.pad(g,(0,0,0,p)) for g,p in zip(grids,to_pad)]


            num_batch = math.ceil(len(grids)/self.atr_batch_size)
            if self.use_tesseract:
                assert self.atr_batch_size==1
                lines = []
                for b in range(num_batch):
                    start=b*self.atr_batch_size
                    end=min((b+1)*self.atr_batch_size,len(grids))
                    b_grids = torch.stack(grids[start:end],dim=0)#.to(image.device)
                    b_grids[:,:,:,1]=2*b_grids[:,:,:,1]/image.size(2) -1 #normalize y
                    b_grids[:,:,:,0]=2*b_grids[:,:,:,0]/image.size(3) -1 #normalize x
                    batch_lines = F.grid_sample(image.expand(b_grids.size(0),-1,-1,-1),b_grids)
                    lines.append((b,batch_lines.cpu()))

                process = lambda a: (a[0],self.text_rec(a[1])) if a[1].size(3)>0 else (a[0],[''])
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.trans_threads) as executor:
                    res = executor.map(process,lines)
                output_strings=[None]*num_batch
                for i,text in res:
                    output_strings[i]=text[0]
                

            else:
                output_strings=[]
                for b in range(num_batch):
                    start=b*self.atr_batch_size
                    end=min((b+1)*self.atr_batch_size,len(grids))
                    b_grids = torch.stack(grids[start:end],dim=0)#.to(image.device)
                    b_grids[:,:,:,1]=2*b_grids[:,:,:,1]/image.size(2) -1 #normalize y
                    b_grids[:,:,:,0]=2*b_grids[:,:,:,0]/image.size(3) -1 #normalize x
                    batch_lines = F.grid_sample(image.expand(b_grids.size(0),-1,-1,-1),b_grids)
        
                    ##DEBUG
                    #d_lines = (1-batch_lines)/2
                    #for i in range(batch_lines.size(0)):
                    #    img_f.imshow('hwr {}'.format(i),(255*(1-d_lines[i,0])/2).cpu().numpy())
                    #img_f.show()
                    ##DEBUG
                    if batch_lines.size(3)>0:
                        with torch.no_grad():
                            resBatch = self.text_rec(batch_lines)
                    else:
                        resBatch = ['']*batch_lines.size(0)
                    if type(resBatch) is list:
                        batch_strings = resBatch
                    else:
                        resBatch = resBatch.cpu().detach().numpy().transpose(1,0,2)
                        batch_strings, decoded_raw_hw = decode_handwriting(resBatch, self.idx_to_char)
                    ##debug
                    #out_im = batch_lines.cpu().numpy().transpose([0,2,3,1])
                    #out_im = 256*(2-out_im)/2
                    #for i in range(batch_lines.size(0)):
                    #    img_f.imwrite('out2/line{}-{}.png'.format(i+index,batch_strings[i]),out_im[i])
                    #    print('DEBUG saved hwr image: out2/line{}-{}.png'.format(i+start,batch_strings[i]))
                    ##
                    output_strings += batch_strings

        else:
            assert(self.include_bb_conf)
            #get corners from bb predictions
            x = bbs[:,1]
            y = bbs[:,2]
            r = bbs[:,3]
            h = bbs[:,4]
            w = bbs[:,5]
            cos_r = torch.cos(r)
            sin_r = torch.sin(r)
            tlX = -w*cos_r + -h*sin_r +x
            tlY =  w*sin_r + -h*cos_r +y
            trX =  w*cos_r + -h*sin_r +x
            trY = -w*sin_r + -h*cos_r +y
            brX =  w*cos_r + h*sin_r +x
            brY = -w*sin_r + h*cos_r +y
            blX = -w*cos_r + h*sin_r +x
            blY =  w*sin_r + h*cos_r +y

            tlX = tlX.cpu()
            tlY = tlY.cpu()
            trX = trX.cpu()
            trY = trY.cpu()
            blX = blX.cpu()
            blY = blY.cpu()
            brX = brX.cpu()
            brY = brY.cpu()

            x1 = torch.min(torch.min(tlX,trX),torch.min(brX,blX)).int()
            x2 = torch.max(torch.max(tlX,trX),torch.max(brX,blX)).int()
            y1 = torch.min(torch.min(tlY,trY),torch.min(brY,blY)).int()
            y2 = torch.max(torch.max(tlY,trY),torch.max(brY,blY)).int()

            x1-=self.padATRx
            x2+=self.padATRx
            y1-=self.padATRy
            y2+=self.padATRy

            x1 = torch.max(x1,torch.tensor(0).int())
            x2 = torch.max(torch.min(x2,torch.tensor(image.size(3)-1).int()),torch.tensor(0).int())
            y1 = torch.max(y1,torch.tensor(0).int())
            y2 = torch.max(torch.min(y2,torch.tensor(image.size(2)-1).int()),torch.tensor(0).int())

            #h *=2
            #w *=2

            h = (y2-y1).float()
            if self.pad_text_height:
                h = torch.where(h<self.hw_input_height,torch.empty_like(h).fill_(self.hw_input_height),h)
            scale = self.hw_input_height/h
            all_scaled_w = (((x2-x1).float()+1)*scale).cpu()#.astype(int)
            scale=None


            if self.use_tesseract:
                image=image.cpu()
                def process(i):
                    if self.rotation:
                        crop = rotate(image[0,:,y1[i]:y2[i]+1,x1[i]:x2[i]+1],r[i],(h[i],w[i]))
                    else:
                        crop = image[...,y1[i]:y2[i]+1,x1[i]:x2[i]+1]
                    if self.pad_text_height and crop.size(2)<self.hw_input_height:
                        diff = self.hw_input_height-crop.size(2)
                        crop = F.pad(crop,(0,0,diff//2,diff//2+diff%2),"constant",-1)
                    elif crop.size(2)<h[i]:
                        diff = int(h[i])-crop.size(2)
                        if y1[i]==0:
                            crop = F.pad(crop,(0,0,0,diff),"constant",-1)
                        elif y2[i]==image.size(2)-1:
                            crop = F.pad(crop,(0,0,diff,0),"constant",-1)
                        else:
                            assert(False and 'why is it short if not getting cropped by image boundary?')               
                    scale = self.hw_input_height/crop.size(2)
                    scaled_w = math.ceil(crop.size(3)*scale)
                    crop = F.interpolate(crop, size=(self.hw_input_height,scaled_w), mode='bilinear',align_corners=False)
                    return i,self.text_rec(crop)

                assert self.atr_batch_size==1
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.trans_threads) as executor:
                    res = executor.map(process,range(bbs.size(0)))
                output_strings=[None]*bbs.size(0)
                for i,text in res:
                    output_strings[i]=text[0] #text is a list (of len 1)

            else:
                output_strings=[]
                for index in range(0,bbs.size(0),self.atr_batch_size):
                    num = min(self.atr_batch_size,bbs.size(0)-index)
                    max_w = math.ceil(all_scaled_w[index:index+num].max().item())

                
                    lines = torch.FloatTensor(num,image.size(1),self.hw_input_height,max_w).fill_(-1).to(image.device)
                    #imm = [None]*num
                    for i in range(index,index+num):
                        
                        if self.rotation:
                            crop = rotate(image[0,:,y1[i]:y2[i]+1,x1[i]:x2[i]+1],r[i],(h[i],w[i]))
                        else:
                            crop = image[...,y1[i]:y2[i]+1,x1[i]:x2[i]+1]
                        if self.pad_text_height and crop.size(2)<self.hw_input_height:
                            diff = self.hw_input_height-crop.size(2)
                            crop = F.pad(crop,(0,0,diff//2,diff//2+diff%2),"constant",-1)
                        elif crop.size(2)<h[i]:
                            diff = int(h[i])-crop.size(2)
                            if y1[i]==0:
                                crop = F.pad(crop,(0,0,0,diff),"constant",-1)
                            elif y2[i]==image.size(2)-1:
                                crop = F.pad(crop,(0,0,diff,0),"constant",-1)
                            else:
                                assert(False and 'why is it short if not getting cropped by image boundary?')
                        scale = self.hw_input_height/crop.size(2)
                        scaled_w = math.ceil(crop.size(3)*scale)
                        lines[i-index,:,:,0:scaled_w] = F.interpolate(crop, size=(self.hw_input_height,scaled_w), mode='bilinear',align_corners=False)[0]#.to(crop.device)
                        #imm[i-index] = lines[i-index].cpu().numpy().transpose([1,2,0])
                        #imm[i-index] = 256*(2-imm[i-index])/2



                    if lines.size(1)==1 and self.hw_channels==3:
                        lines = lines.expand(-1,3,-1,-1)
                    
                    with torch.no_grad():
                        self.text_rec.eval()
                        resBatch = self.text_rec(lines)
                    if type(resBatch) is list:
                        batch_strings = resBatch
                    else:
                        resBatch = resBatch.cpu().detach().numpy().transpose(1,0,2)
                        batch_strings, decoded_raw_hw = decode_handwriting(resBatch, self.idx_to_char)
                    ###debug
                    #for i in range(num):
                    #    img_f.imwrite('out2/line{}-{}.png'.format(i+index,batch_strings[i]),imm[i])
                    ###
                    output_strings += batch_strings
                    #res.append(resBatch)
            #res = torch.cat(res,dim=1)

            ### Debug ###
            #resN=res.data.cpu().numpy()
            #output_strings, decoded_raw_hw = decode_handwriting(resN, self.idx_to_char)
                #img_f.imshow('line',imm)
                #img_f.waitKey()
            ###
        return output_strings


    def makeGraphMinDegree(self,edges,bbs):
        #all distances
        if self.useCurvedBBs:
            points = np.array([bb.getCenterPoint() for bb in bbs])
            points1 = points[None,:,:]
            points2 = points[:,None,:]
            #distances = np.power(points1-points2,2).sum(axis=2)
            distances = np.abs(points1-points2).sum(axis=2) #L1 distance
        else:
            raise NotImplementedError('GraphMinDegree needs non-TextLine implemented')
        #find bbs needing degree bumped up
        degree=[0]*len(bbs)
        for n1,n2 in edges:
            degree[n1]+=1
            degree[n2]+=1
        #for bi,d in enumerate(degrees):
        #    if d < self.graph_min_degree:
        np.fill_diagonal(distances,float('inf'))
        keep_edges=set()
        for bi in range(len(bbs)):
            for i in range(self.graph_min_degree):
                closest = distances[bi].argmin()
                edge = (min(bi,closest),max(bi,closest))
                try:
                    ei = edges.index(edge)
                    keep_edges.add(ei)
                except ValueError:
                    keep_edges.add(len(edges))
                    edges.append(edge)
                distances[bi][closest]=float('inf')
        return edges,keep_edges

    def makeGraphFourConnected(self,edges,bbs):
        if self.useCurvedBBs:
            points = np.array([bb.getCenterPoint() for bb in bbs])
            points1 = points[None,:,:]
            points2 = points[:,None,:]
            distances = np.abs(points1-points2).sum(axis=2)
        else:
            raise NotImplementedError('GraphMinDegree needs non-TextLine implemented')

        keep_edges=set()
        all_rect = np.array([bb.boundingRect() for bb in bbs])[:,None,:] #channel for broadcast
        pointsI = points2
        pointsJ = points1

        #find directional relationships [e.g. j is on top of x]
        topL = all_rect[:,:,0]-(all_rect[:,:,1]-pointsJ[:,:,1])
        topR = all_rect[:,:,2]+(all_rect[:,:,1]-pointsJ[:,:,1])
        top = (topL<=pointsJ[:,:,0]) * (pointsJ[:,:,0]<topR) * (pointsJ[:,:,1]<all_rect[:,:,1])

        botL = all_rect[:,:,0]-(-all_rect[:,:,3]+pointsJ[:,:,1])
        botR = all_rect[:,:,2]+(-all_rect[:,:,3]+pointsJ[:,:,1])
        bot = (botL<pointsJ[:,:,0]) * (pointsJ[:,:,0]<=botR) * (pointsJ[:,:,1]>all_rect[:,:,3])

        leftT = all_rect[:,:,1]-(all_rect[:,:,0]-pointsJ[:,:,0])
        leftB = all_rect[:,:,3]+(all_rect[:,:,0]-pointsJ[:,:,0])
        left = (leftT<pointsJ[:,:,1]) * (pointsJ[:,:,1]<=leftB) * (pointsJ[:,:,0]<all_rect[:,:,0])

        rightT = all_rect[:,:,1]-(-all_rect[:,:,2]+pointsJ[:,:,0])
        rightB = all_rect[:,:,3]+(-all_rect[:,:,2]+pointsJ[:,:,0])
        right = (rightT<=pointsJ[:,:,1]) * (pointsJ[:,:,1]<rightB) * (pointsJ[:,:,0]>all_rect[:,:,2])

        inside = (pointsJ[:,:,1]>=all_rect[:,:,1]) * (pointsJ[:,:,1]<=all_rect[:,:,3]) * (pointsJ[:,:,0]>=all_rect[:,:,0]) * (pointsJ[:,:,0]<=all_rect[:,:,2])

        np.fill_diagonal(top,False)
        np.fill_diagonal(bot,False)
        np.fill_diagonal(left,False)
        np.fill_diagonal(right,False)
        np.fill_diagonal(inside,False)
        
        #summ = top.astype(int)+left.astype(int)+bot.astype(int)+right.astype(int)+inside.astype(int)
        #np.fill_diagonal(summ,1)
        #if (summ!=1).any():
        #    print('min: {}, max:{}'.format(summ.min(),summ.max()))
        #    if summ.min()==0:
        #        print(top[summ==0])
        #        print(bot[summ==0])
        #        print(left[summ==0])
        #        print(right[summ==0])
        #        print(inside[summ==0])
        #    if summ.max()>1:
        #        print(top[summ>1])
        #        print(bot[summ>1])
        #        print(left[summ>1])
        #        print(right[summ>1])
        #        print(inside[summ>1])
        #assert((summ==1).all())

        #Find closest for each relationship
        topD = top*distances
        topD[topD==0]=float('inf')#no relationship, so inf distance
        topMins = topD.argmin(axis=1)
        botD = bot*distances
        botD[botD==0]=float('inf')
        botMins = botD.argmin(axis=1)
        leftD = left*distances
        leftD[leftD==0]=float('inf')
        leftMins = leftD.argmin(axis=1)
        rightD = right*distances
        rightD[rightD==0]=float('inf')
        rightMins = rightD.argmin(axis=1)
        insideD = inside*distances
        insideD[insideD==0]=float('inf')
        insideMins = insideD.argmin(axis=1)
        
        for bi,bj in enumerate(topMins):
            if not math.isinf(topD[bi,bj]):
                edge = (min(bi,bj),max(bi,bj))
                try:
                    ei = edges.index(edge)
                    keep_edges.add(ei)
                except ValueError:
                    keep_edges.add(len(edges))
                    edges.append(edge)
        for bi,bj in enumerate(botMins):
            if not math.isinf(botD[bi,bj]):
                edge = (min(bi,bj),max(bi,bj))
                try:
                    ei = edges.index(edge)
                    keep_edges.add(ei)
                except ValueError:
                    keep_edges.add(len(edges))
                    edges.append(edge)
        for bi,bj in enumerate(leftMins):
            if not math.isinf(leftD[bi,bj]):
                edge = (min(bi,bj),max(bi,bj))
                try:
                    ei = edges.index(edge)
                    keep_edges.add(ei)
                except ValueError:
                    keep_edges.add(len(edges))
                    edges.append(edge)
        for bi,bj in enumerate(rightMins):
            if not math.isinf(rightD[bi,bj]):
                edge = (min(bi,bj),max(bi,bj))
                try:
                    ei = edges.index(edge)
                    keep_edges.add(ei)
                except ValueError:
                    keep_edges.add(len(edges))
                    edges.append(edge)
        for bi,bj in enumerate(insideMins):
            if not math.isinf(insideD[bi,bj]):
                edge = (min(bi,bj),max(bi,bj))
                try:
                    ei = edges.index(edge)
                    keep_edges.add(ei)
                except ValueError:
                    keep_edges.add(len(edges))
                    edges.append(edge)

        return edges,keep_edges

    def setDEBUG(self):
        self.debug=True
        def save_layerConv0(module,input,output):
            self.debug_conv0=output.cpu()
        self.relFeaturizerConv[0].register_forward_hook(save_layerConv0)
        def save_layerConv1(module,input,output):
            self.debug_conv1=output.cpu()
        self.relFeaturizerConv[1].register_forward_hook(save_layerConv1)
        #def save_layerFC(module,input,output):
            #    self.debug_fc=output.cpu()
        #self.relFeaturizerConv[0].register_forward_hook(save_layerFC)


    def runGraph(self,gtGroups,gtTrans,image,useBBs,saved_features,saved_features2,bbTrans,embeddings,merge_first_only=False,zero_embeddings=False):

        if zero_embeddings:
            embeddings=embeddings.new_zeros(embeddings.size())
        
        groups=[[i] for i in range(len(useBBs))]
        if self.merge_first:
            assert gtGroups is None
            #t#tic=timeit.default_timer()#t#
            #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            #We don't build a full graph, just propose edges and extract the edge features
            edgeOuts,edgeIndexes,merge_prop_scores = self.createGraph(useBBs,saved_features,saved_features2,image.size(-2),image.size(-1),text_emb=embeddings,image=image,merge_only=True)
            #_,edgeIndexes, edgeFeatures,_ = graph
            #t#time = timeit.default_timer()-tic#t#
            #t#self.opt_history['m1st createGraph'].append(time)#t#
            if edgeOuts is not None:
                #print(edgeOuts.size())
                edgeOuts = self.mergepred(edgeOuts)
                edgeOuts = edgeOuts[:,None,:] #introduce repition dim (to match graph network)

            allOutputBoxes=[useBBs]
            allNodeOuts=[None]
            allEdgeOuts=[edgeOuts]
            allGroups=[groups]
            allEdgeIndexes=[edgeIndexes]

            #print('merge first.   num bbs:{}, num edges: {}'.format(len(useBBs),len(edgeIndexes)))
            #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            
            if edgeIndexes is not None:
                startBBs = len(useBBs)
                #perform predicted merges
                #t#tic2=timeit.default_timer()#t#
                useBBs,bbTrans=self.mergeAndGroup(
                        self.mergeThresh[0],
                        None,
                        None,
                        edgeIndexes,
                        edgeOuts,
                        groups,
                        None,
                        None,
                        None,
                        useBBs,
                        bbTrans,
                        embeddings,
                        image,
                        skip_rec=merge_first_only,
                        merge_only=True)
                #This mergeAndGroup performs first ATR
                #t#time = timeit.default_timer()-tic2#t#
                #t#self.opt_history['m1st mergeAndGroup'].append(time)#t#
                groups=[[i] for i in range(len(useBBs))]
                #print('merge first reduced graph by {} nodes ({}->{}). max edge pred:{}, mean:{}'.format(startBBs-len(useBBs),startBBs,len(useBBs),torch.sigmoid(edgeOuts.max()),torch.sigmoid(edgeOuts.mean())))
            #t#time = timeit.default_timer()-tic#t#
            #t#self.opt_history['m1st Full'].append(time)#t#
            if merge_first_only:
                #t#for name,times in self.opt_history.items():#t#
                    #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
                    #t#if len(times)>300: #t#
                        #t#times.pop(0)   #t#
                        #t#if len(times)>600:#t#
                            #t#times.pop(0)#t#
                if not self.useCurvedBBs and self.detector_predNumNeighbors:
                    #Discard NN prediction. We don't use it anymore
                    allOutputBoxes = [ torch.cat([outBs[:,:6],outBs[:,7:]],dim=1) for outBs in allOutputBoxes]
                return allOutputBoxes, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, None, merge_prop_scores, None

            if bbTrans is not None:
                if gtTrans is not None:
                    if self.include_bb_conf:
                        justBBs = useBBs[:,1:]
                    else:
                        justBBs = useBBs
                    bbTrans=correctTrans(bbTrans,justBBs,gtTrans,gtBBs)
                embeddings = self.embedding_model(bbTrans,saved_features.device)
                if zero_embeddings:
                    embeddings=embeddings.new_zeros(embeddings.size())
                if self.add_noise_to_word_embeddings:
                    embeddings += torch.randn_like(embeddings).to(embeddings.device)*self.add_noise_to_word_embeddings*embeddings.mean()
            else:
                embeddings=None
        else:
            merge_prop_scores=None
            allOutputBoxes=[]
            allNodeOuts=[]
            allEdgeOuts=[]
            allGroups=[]
            allEdgeIndexes=[]



        #t#tic=timeit.default_timer()#t#
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        graph,edgeIndexes,rel_prop_scores,last_node_visual_feats,last_edge_visual_feats,keep_edges = self.createGraph(useBBs,saved_features,saved_features2,image.size(-2),image.size(-1),text_emb=embeddings,image=image)
        #print('edgeIndexes: {}'.format(len(edgeIndexes)))

        #print('createGraph')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['createGraph'].append(time)#t#

        #undirected
        #edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
        if graph is None:
            if not self.useCurvedBBs and self.detector_predNumNeighbors:
                #Discard NN prediction. We don't use it anymore
                #bbPredictions = torch.cat([bbPredictions[:,:6],bbPredictions[:,7:]],dim=1)
                useBBs = torch.cat([useBBs[:,:6],useBBs[:,7:]],dim=1)
            return [useBBs], None, None, None, None, rel_prop_scores, merge_prop_scores, (useBBs if self.useCurvedBBs else useBBs.cpu().detach(),None,None,bbTrans)
            return allOutputBoxes, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final

        if self.reintroduce_features=='map':
            last_node_visual_feats = graph[0]
            last_edge_visual_feats = graph[2]

        #t#tic=timeit.default_timer()#t#

        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        #print('{} node feats mean:{:.3}, std:{:.3}, min:{:.2}, max:{:.2}'.format(0,graph[0].mean(),graph[0].std(),   graph[0].min(), graph[0].max()))
        #print('  edge feats mean:{:.3}, std:{:.3}, min:{:.2}, max:{:.2}'.format(graph[2].mean(), graph[2].std(),  graph[2].min(), graph[2].max()))
        nodeOuts, edgeOuts, nodeFeats, edgeFeats, uniFeats = self.graphnets[0](graph)
        assert(edgeOuts is None or not torch.isnan(edgeOuts).any())
        edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
        #edgeOuts = (edgeOuts[:edgeOuts.size(0)//2] + edgeOuts[edgeOuts.size(0)//2:])/2 #average two directions of edge
        #edgeFeats = (edgeFeats[:edgeFeats.size(0)//2] + edgeFeats[edgeFeats.size(0)//2:])/2 #average two directions of edge
        #update BBs with node predictions
        useBBs = self.updateBBs(useBBs,groups,nodeOuts)
        allOutputBoxes.append(useBBs if self.useCurvedBBs else useBBs.cpu()) 
        allNodeOuts.append(nodeOuts)
        allEdgeOuts.append(edgeOuts)
        allGroups.append(groups)
        allEdgeIndexes.append(edgeIndexes)

        #print('graph 0:   bbs:{}, nodes:{}, edges:{}'.format(useBBs.size(0),nodeOuts.size(0),edgeOuts.size(0)))
        #print('init num bbs:{}, num keep:{}')
        
        for gIter,graphnet in enumerate(self.graphnets[1:]):
            if self.merge_first:
                gIter+=1
            
            if self.fully_connected:
                good_edges=[]
            else:
                good_edges=None
            #print('!D! {} before edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(gIter,edgeFeats.size(),len(useBBs),nodeFeats.size(),len(edgeIndexes)))
            #print('      graph num edges: {}'.format(graph[1].size()))
            useBBs,graph,groups,edgeIndexes,bbTrans,embeddings,same_node_map,keep_edges=self.mergeAndGroup(
                    self.mergeThresh[gIter],
                    self.keepEdgeThresh[gIter],
                    self.groupThresh[gIter],
                    edgeIndexes,
                    edgeOuts,
                    groups,
                    nodeFeats,
                    edgeFeats,
                    uniFeats,
                    useBBs,
                    bbTrans,
                    embeddings,
                    image,
                    good_edges=good_edges,
                    keep_edges=keep_edges,
                    gt_groups=gtGroups if gIter==0 else ([[g] for g in range(len(groups))] if gtGroups is not None else None))
            if zero_embeddings:
                embeddings=embeddings.new_zeros(embeddings.size())


            if self.reintroduce_features:
                graph,last_node_visual_feats,last_edge_visual_feats = self.appendVisualFeatures(
                        gIter if self.merge_first else gIter+1,
                        useBBs,
                        graph,
                        groups,
                        edgeIndexes,
                        saved_features,
                        saved_features2,
                        embeddings,
                        image.size(-2),
                        image.size(-1),
                        same_node_map,
                        last_node_visual_feats,
                        last_edge_visual_feats,
                        allEdgeIndexes[-1],
                        debug_image=None,
                        good_edges=good_edges)
            #print('graph 1-:   bbs:{}, nodes:{}, edges:{}'.format(useBBs.size(0),len(groups),len(edgeIndexes)))
            if len(edgeIndexes)==0:
                break #we have no graph, so we can just end here
            #print('!D! after  edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(graph[2].size(),len(useBBs),graph[0].size(),len(edgeIndexes)))
            #print('      graph num edges: {}'.format(graph[1].size()))
            #print('{} node feats mean:{:.3}, std:{:.3}, min:{:.2}, max:{:.2}'.format(gIter,graph[0].mean(),graph[0].std(),   graph[0].min(), graph[0].max()))
            #print('  edge feats mean:{:.3}, std:{:.3}, min:{:.2}, max:{:.2}'.format(graph[2].mean(), graph[2].std(),  graph[2].min(), graph[2].max()))
            nodeOuts, edgeOuts, nodeFeats, edgeFeats, uniFeats = graphnet(graph)
            #edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
            useBBs = self.updateBBs(useBBs,groups,nodeOuts)
            allOutputBoxes.append(useBBs if self.useCurvedBBs else useBBs.cpu()) 
            allNodeOuts.append(nodeOuts)
            allEdgeOuts.append(edgeOuts)
            allGroups.append(groups)
            allEdgeIndexes.append(edgeIndexes)


        ##Final state of the graph
        #print('!D! F before edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(edgeFeats.size(),useBBs.size(),nodeFeats.size(),len(edgeIndexes)))
        useBBs,graph,groups,edgeIndexes,bbTrans,_,same_node_map,keep_edges=self.mergeAndGroup(
                self.mergeThresh[-1],
                self.keepEdgeThresh[-1],
                self.groupThresh[-1],
                edgeIndexes,
                edgeOuts.detach(),
                groups,
                None,#nodeFeats.detach(),
                None,#edgeFeats.detach(),
                None,#uniFeats.detach() if uniFeats is not None else None,
                useBBs,
                bbTrans,
                None,
                image,
                gt_groups=[[g] for g in range(len(groups))] if gtGroups is not None else None,
                final=True)
        #print('!D! after  edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(graph[2].size(),useBBs.size(),graph[0].size(),len(edgeIndexes)))
        if not self.useCurvedBBs and self.detector_predNumNeighbors:
            #Discard NN prediction. We don't use it anymore
            useBBs = torch.cat([useBBs[:,:6],useBBs[:,7:]],dim=1)
        final=(useBBs if self.useCurvedBBs else useBBs.cpu().detach(),groups,edgeIndexes,bbTrans)
        #print('all iters GCN')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['all graph iters'].append(time)#t#


        #adjacencyMatrix = torch.zeros((bbPredictions.size(1),bbPredictions.size(1)))
        #for rel in relOuts:
        #    i,j,a=graphToDetectionsMap(

        #t#for name,times in self.opt_history.items():#t#
            #t#print('time {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#

        if not self.useCurvedBBs and self.detector_predNumNeighbors:
            #Discard NN prediction. We don't use it anymore
            allOutputBoxes = [ torch.cat([outBs[:,:6],outBs[:,7:]],dim=1) for outBs in allOutputBoxes]
        return allOutputBoxes, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final



    def alignGTBBs(self,useGTBBs,gtBBs,gtGroups,bbPredictions):
            #We'll correct the box predictions using the GT BBs, but no class/other GT
            useBBs = []
            gtBBs=gtBBs[0]


            #perform greedy alignment of gt and predicted. Only keep aligned predictions
            if not bbPredictions.is_cuda:
                gtBBs=gtBBs.cpu()
            if 'word_bbs' in useGTBBs or self.useCurvedBBs:
                ious = allIO_clipU(gtBBs,bbPredictions[:,1:],x1y1x2y2=self.useCurvedBBs) #iou calculation, words are oversegmented lines
            else:
                ious = allIOU(gtBBs,bbPredictions[:,1:],x1y1x2y2=self.useCurvedBBs) #iou calculation
            #if self.useCurvedBBs:
            #    ious = allIO_clipU(
            #else:
            #    ious = allIOU(gtBBs,bbPredictions[:,1:])
            #ious[ious<0.3]=0
            ious=ious.cpu()
            bbPredictions=bbPredictions.cpu()
            gtBBs=gtBBs.cpu()
            #sort, do highest ious first
            gt_used = [False]*gtBBs.size(0)
            num_gt_used = 0
            pred_used = [False]*bbPredictions.size(0)
            num_pred_used = 0
            ious_list = [(ious[gt_i,p_i],gt_i,p_i) for gt_i,p_i in ious.triu(1).nonzero(as_tuple=False)]
            ious=None
            ious_list.sort(key=lambda a:a[0], reverse=True)
            gt_parts=defaultdict(list)
            gt_to_new = {}
            for iou,gt_i,p_i in ious_list:
                gt_i=gt_i.item()
                if self.useCurvedBBs:
                    gt_used[gt_i]=True
                    gt_parts[gt_i].append((iou,bbPredictions[p_i,0:1],bbPredictions[p_i,6:]))
                elif not gt_used[gt_i] and not pred_used[p_i]:
                    gt_to_new[gt_i]=len(useBBs)
                    useBBs.append(torch.cat((bbPredictions[p_i,0:1],gtBBs[gt_i,0:5],bbPredictions[p_i,6:]), dim=0))
                    num_gt_used+=1
                    if num_gt_used>=gtBBs.size(0):
                        break
                    gt_used[gt_i]=True

                    if not pred_used[p_i]:
                        num_pred_used+=1
                        if num_pred_used>=bbPredictions.size(0) and not self.useCurvedBBs:
                            break
                        pred_used[p_i]=True
            ious_list=None

            if self.useCurvedBBs:
                for gt_i,parts in gt_parts.items():
                    ious,confs,clses = zip(*parts)
                    ious = torch.FloatTensor(ious)
                    confs = torch.cat(confs)
                    clses = torch.stack(clses,dim=0)
                    total_iou = ious.sum()
                    conf = (confs*ious).sum()/total_iou
                    cls = (clses*ious[:,None]).sum(dim=0)/total_iou
                    gt_to_new[gt_i]=len(useBBs)
                    useBBs.append(torch.cat((conf[None],gtBBs[gt_i,0:5],cls),dim=0))


            #Add any undetected boxes.
            for gt_i,used in enumerate(gt_used):
                if not used:
                    conf = torch.FloatTensor([1])
                    cls = torch.FloatTensor(self.numBBTypes+(1 if self.detector_predNumNeighbors else 0)).fill_(0.5)
                    gt_to_new[gt_i]=len(useBBs)
                    useBBs.append(torch.cat((conf,gtBBs[gt_i,0:5],cls),dim=0))

            if gtGroups is not None:
                gtGroups = [[gt_to_new[gt_i] for gt_i in group] for group in gtGroups]
            if len(useBBs)>0:
                useBBs = torch.stack(useBBs,dim=0).to(gtBBs.device)
            assert self.training or useBBs.size(0) == gtBBs.size(0)
            assert self.include_bb_conf or self.useCurvedBBs
            #if self.useCurvedBBs and self.use_overseg_non_max_sup:
            #    useBBs = non_max_sup_overseg(useBBs)
            gtBBs=gtBBs[None,...]

            return useBBs, gtBBs, gtGroups, gt_to_new
            
