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
from model.word2vec_adapter import Word2VecAdapter, Word2VecAdapterShallow, BPEmbAdapter
from model.hand_code_emb import HandCodeEmb
from skimage import draw
from model.net_builder import make_layers, getGroupSize
from utils.yolo_tools import non_max_sup_iou, non_max_sup_dist
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

MAX_CANDIDATES=2000
MAX_GRAPH_SIZE=4000

def minAndMaxXY(boundingRects):
    min_X,min_Y,max_X,max_Y = np.array(boundingRects).transpose(1,0)
    return min_X.min(),max_X.max(),min_Y.min(),max_Y.max()
def combineShapeFeats(feats):
    if len(feats)==1:
        return torch.FloatTensor(feats[0])
    feats = feats.sort(key=feats[17]) #sort into read order
    feats = torch.FloatTensor(feats)
    easy_feats = feats[:,0:6],mean(dim=0)
    tl=feats[0,6:8]
    tr=feats[0,8:10]
    br=feats[-1,12:14]
    bl=feats[-1,14:16]
    easy2_feats=feats[:,16:].mean(dim=0)
    return torch.cat((easy_feats,tl,tr,br,bl,easy2_feats),dim=0)
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

        if 'detector_checkpoint' in config:
            if os.path.exists(config['detector_checkpoint']):
                checkpoint = torch.load(config['detector_checkpoint'], map_location=lambda storage, location: storage)
            else:
                checkpoint = None
                print('Warning: unable to load {}'.format(config['detector_checkpoint']))
            detector_config = json.load(open(config['detector_config']))['model'] if 'detector_config' in config else checkpoint['config']['model']
            if checkpoint is None:
                self.detector = eval(checkpoint['config']['arch'])(detector_config)
                for p in self.detector.parameters():
                    import pdb;pdb.set_trace()
                    p.something = float('nan') #ensure this gets changed
            elif 'state_dict' in checkpoint:
                self.detector = eval(checkpoint['config']['arch'])(detector_config)
                self.detector.load_state_dict(checkpoint['state_dict'])
            else:
                self.detector = checkpoint['model']
        else:
            detector_config = config['detector_config']
            self.detector = eval(detector_config['arch'])(detector_config)
        self.useCurvedBBs = 'OverSeg' in checkpoint['config']['arch']
        useBeginningOfLast = config['use_beg_det_feats'] if 'use_beg_det_feats' in config else False
        useFeatsLayer = config['use_detect_layer_feats'] if 'use_detect_layer_feats' in config else -1
        useFeatsScale = config['use_detect_scale_feats'] if 'use_detect_scale_feats' in config else -2
        useFLayer2 = config['use_2nd_detect_layer_feats'] if 'use_2nd_detect_layer_feats' in config else None
        useFScale2 = config['use_2nd_detect_scale_feats'] if 'use_2nd_detect_scale_feats' in config else None
        detectorSavedFeatSize = config['use_detect_feats_size'] if 'use_detect_feats_size' in config else self.detector.last_channels
        assert((useFeatsScale==-2) or ('use_detect_feats_size' in config))
        detectorSavedFeatSize2 = config['use_2nd_detect_feats_size'] if 'use_2nd_detect_feats_size' in config else None
        
        #splitScaleDiff = config['split_features_scale_diff'] if 'split_features_scale_diff' in config else None
        self.splitFeatures= config['split_features_scale'] if 'split_features_scale' in config else False

        self.use2ndFeatures = useFLayer2 is not None
        if self.use2ndFeatures and not self.splitFeatures:
            detectorSavedFeatSize += detectorSavedFeatSize2
            
        self.detector.setForGraphPairing(useBeginningOfLast,useFeatsLayer,useFeatsScale,useFLayer2,useFScale2)


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
        self.confThresh = config['conf_thresh'] if 'conf_thresh' in config else 0.5
        self.useHardConfThresh = config['use_hard_conf_thresh'] if 'use_hard_conf_thresh' in config else True
        self.predNN = config['pred_nn'] if 'pred_nn' in config else False
        self.predClass = config['pred_class'] if 'pred_class' in config else False

        self.merge_first = config['merge_first'] if 'merge_first' in config else False

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
        self.includeRelRelEdges= config['use_rel_rel_edges'] if 'use_rel_rel_edges' in config else True
        #rel_channels = config['graph_config']['rel_channels']
        self.pool_h = config['featurizer_start_h']
        self.pool_w = config['featurizer_start_w']
        self.poolBB_h = config['featurizer_bb_start_h'] if 'featurizer_bb_start_h' in config else 2
        self.poolBB_w = config['featurizer_bb_start_w'] if 'featurizer_bb_start_w' in config else 3

        self.pool2_h=self.pool_h
        self.pool2_w=self.pool_w
        self.poolBB2_h=self.poolBB_h
        self.poolBB2_w=self.poolBB_w

        self.merge_pool_h = self.merge_pool2_h = config['merge_featurizer_start_h']
        self.merge_pool_w = self.merge_pool2_w = config['merge_featurizer_start_w']

        self.reintroduce_visual_features = config['reintroduce_visual_features']


        if 'use_rel_shape_feats' in config:
             config['use_shape_feats'] =  config['use_rel_shape_feats']
        self.useShapeFeats= config['use_shape_feats'] if 'use_shape_feats' in config else False
        self.usePositionFeature = config['use_position_feats'] if 'use_position_feats' in config else False
        assert(not self.usePositionFeature or self.useShapeFeats)
        self.normalizeHorz=config['normalize_horz'] if 'normalize_horz' in config else 400
        self.normalizeVert=config['normalize_vert'] if 'normalize_vert' in config else 50
        self.normalizeDist=(self.normalizeHorz+self.normalizeVert)/2
        
        if type(self.detector.scale[0]) is int:
            assert(self.detector.scale[0]==self.detector.scale[1])
        else:
            for level_sc in self.detector.scale:
                assert(level_sc[0]==level_sc[1])
        if useBeginningOfLast:
            detect_save_scale = self.detector.scale[0]
        else:
            detect_save_scale = self.detector.save_scale
        if self.use2ndFeatures:
            detect_save2_scale = self.detector.save2_scale

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
           if self.detector.predNumNeighbors and not self.useCurvedBBs:
               self.numShapeFeats+=2
               self.numShapeFeatsBB+=1
           if self.usePositionFeature:
               if not self.useCurvedBBs:
                   self.numShapeFeats+=4
               self.numShapeFeatsBB+=2
        else:
           self.numShapeFeats=0
           self.numShapeFeatsBB=0


        if 'text_rec' in config:
            self.numTextFeats = config['text_rec']['num_feats']
        else:
            self.numTextFeats = 0

        if type(config['graph_config']) is list:
            for graphconfig in config['graph_config']:
                graphconfig['num_shape_feats']=self.numShapeFeats
        else:
            config['graph_config']['num_shape_feats']=self.numShapeFeats
        featurizer_fc = config['featurizer_fc'] if 'featurizer_fc' in config else []
        if self.useShapeFeats!='only':

            self.expandedRelContext = config['expand_rel_context'] if 'expand_rel_context' in config else None
            if self.merge_first:
                self.expandedMergeContextY,self.expandedMergeContextX = config['expand_merge_context']

            if self.expandedRelContext is not None:
                bbMasks=3
            else:
                bbMasks=2
            self.expandedBBContext = config['expand_bb_context'] if 'expand_bb_context' in config else None
            if self.expandedBBContext is not None:
                bbMasks_bb=2
            else:
                bbMasks_bb=0

        self.use_fixed_masks = config['use_fixed_masks'] if 'use_fixed_masks' in config else False
        assert(self.use_fixed_masks)
        self.splitFeatureRes = config['split_feature_res'] if 'split_feature_res' in config else False

        feat_norm = detector_config['norm_type'] if 'norm_type' in detector_config else None
        feat_norm_fc = detector_config['norm_type_fc'] if 'norm_type_fc' in detector_config else None
        featurizer_conv = config['featurizer_conv'] if 'featurizer_conv' in config else [512,'M',512]
        if self.splitFeatures:
            featurizer_conv2 = config['featurizer_conv_first'] if 'featurizer_conv_first' in config else None
            featurizer_conv2 = [detectorSavedFeatSize2+bbMasks] + featurizer_conv2 #bbMasks are appended
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

            featurizer_conv = [detectorSavedFeatSize+last_ch_relC] + featurizer_conv
        else:
            featurizer_conv = [detectorSavedFeatSize+bbMasks] + featurizer_conv #bbMasks are appended
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
        layers, last_ch_relC = make_layers(featurizer_conv,norm=feat_norm,dropout=True) 
        if featurizer_fc is None: #we don't have a FC layer, so channels need to be the same as graph model expects
            if last_ch_relC+self.numShapeFeats!=graph_in_channels:
                new_layer = [last_ch_relC,'k1-{}'.format(graph_in_channels-self.numShapeFeats)]
                print('WARNING: featurizer_conv did not line up with graph_in_channels, adding layer k1-{}'.format(graph_in_channels-self.numShapeFeats))
                new_layer, last_ch_relC = make_layers(new_layer,norm=feat_norm,dropout=True) 
                layers+=new_layer
        layers.append( nn.AvgPool2d((fsizeY,fsizeX)) )
        self.relFeaturizerConv = nn.Sequential(*layers)

        if self.merge_first:
            if self.splitFeatures:
                raise NotImplementedError('split feature embedding not implemented for merge_first model')
            merge_featurizer_conv = config['merge_featurizer_conv']
            merge_featurizer_conv = [detectorSavedFeatSize+bbMasks] + merge_featurizer_conv #bbMasks are appended
            layers, last_ch_relC = make_layers(merge_featurizer_conv,norm=feat_norm,dropout=True) 
            #if last_ch_relC+self.numShapeFeats!=graph_in_channels:
            #    new_layer = [last_ch_relC,'k1-{}'.format(graph_in_channels-self.numShapeFeats)]
            #    print('WARNING: merge_featurizer_conv did not line up with graph_in_channels, adding layer k1-{}'.format(graph_in_channels-self.numShapeFeats))
            #    new_layer, last_ch_relC = make_layers(new_layer,norm=feat_norm,dropout=True) 
            #    layers+=new_layer
            layers.append( nn.AvgPool2d((fsizeY,fsizeX)) )
            self.mergeFeaturizerConv = nn.Sequential(*layers)
            self.mergepred = nn.Sequential(
                    nn.ReLU(True),
                    nn.Linear(last_ch_relC+self.numShapeFeats,last_ch_relC+self.numShapeFeats),
                    nn.ReLU(True),
                    nn.Linear(last_ch_relC+self.numShapeFeats,1)
                    )

        #self.roi_align = RoIAlign(self.pool_h,self.pool_w,1.0/detect_save_scale) Facebook implementation
        self.roi_align = RoIAlign((self.pool_h,self.pool_w),1.0/detect_save_scale,-1)
        if self.use2ndFeatures:
            #self.roi_align2 = RoIAlign(self.pool2_h,self.pool2_w,1.0/detect_save2_scale)
            self.roi_align2 = RoIAlign((self.pool2_h,self.pool2_w),1.0/detect_save2_scale,-1)
        else:
            last_ch_relC=0
        if self.merge_first:
            self.merge_roi_align = RoIAlign((self.merge_pool_h,self.merge_pool_w),1.0/detect_save_scale,-1)
            if self.use2ndFeatures:
                self.merge_roi_align2 = RoIAlign((self.merge_pool2_h,self.merge_pool2_w),1.0/detect_save2_scale,-1)


        #if config['graph_config']['arch'][:10]=='BinaryPair' or self.useShapeFeats=='only':
        #    feat_norm_fc=None
        if featurizer_fc is not None:
            featurizer_fc = [last_ch_relC+self.numShapeFeats] + featurizer_fc + ['FCnR{}'.format(graph_in_channels)]
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
                if featurizer is None:
                    convlayers = [ nn.Conv2d(detectorSavedFeatSize+bbMasks_bb,convOut,kernel_size=(2,3)) ]
                    if featurizer_fc is not None:
                        convlayers+=[   nn.GroupNorm(getGroupSize(convOut),convOut),
                                        nn.Dropout2d(p=0.1,inplace=True),
                                        nn.ReLU(inplace=True)
                                    ]
                else:
                    if self.splitFeatures:
                        featurizer_conv2 = config['bb_featurizer_conv_first'] if 'bb_featurizer_conv_first' in config else None
                        featurizer_conv2 = [detectorSavedFeatSize2+bbMasks_bb] + featurizer_conv2 #bbMasks are appended
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

                        featurizer_conv = [detectorSavedFeatSize+last_ch_relC] + featurizer_conv
                    else:
                        featurizer_conv = [detectorSavedFeatSize+bbMasks_bb] + featurizer
                    if featurizer_fc is None:
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

                #self.roi_alignBB = RoIAlign(self.poolBB_h,self.poolBB_w,1.0/detect_save_scale)
                self.roi_alignBB = RoIAlign((self.poolBB_h,self.poolBB_w),1.0/detect_save_scale,-1)
                if self.use2ndFeatures:
                    #self.roi_alignBB2 = RoIAlign(self.poolBB2_h,self.poolBB2_w,1.0/detect_save2_scale)
                    self.roi_alignBB2 = RoIAlign((self.poolBB2_h,self.poolBB2_w),1.0/detect_save2_scale,-1)
            else:
                featurizer_fc = [self.numShapeFeatsBB+self.numTextFeats]+featurizer_fc
            if featurizer_fc is not None:
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
        self.include_bb_conf=False
        if self.relationshipProposal=='feature_nn':
            self.include_bb_conf=True
            #num_classes = config['num_class']
            num_bb_feat = self.numBBTypes + (1 if self.detector.predNumNeighbors else 0) #config['graph_config']['bb_out']
            prop_feats = 30+2*num_bb_feat
            if self.useCurvedBBs:
                prop_feats += 8
                if self.shape_feats_normal:
                    prop_feats += 1
            self.rel_prop_nn = nn.Sequential(
                                nn.Linear(prop_feats,64),
                                nn.Dropout(0.25),
                                nn.ReLU(True),
                                nn.Linear(64,1)
                                )
            if self.merge_first:
                
                self.merge_prop_nn = nn.Sequential(
                                    nn.Linear(prop_feats,64),
                                    nn.Dropout(0.25),
                                    nn.ReLU(True),
                                    nn.Linear(64,1)
                                    )
            self.percent_rel_to_keep = config['percent_rel_to_keep'] if 'percent_rel_to_keep' in config else 0.2
            self.max_rel_to_keep = config['max_rel_to_keep'] if 'max_rel_to_keep' in config else 3000
            self.max_merge_rel_to_keep = config['max_merge_rel_to_keep'] if 'max_merge_rel_to_keep' in config else 5000
            self.roi_batch_size = config['roi_batch_size'] if 'roi_batch_size' in config else 300

        #HWR stuff
        if 'text_rec' in config:
            self.padATRy=3
            self.padATRx=10
            if 'CRNN' in config['text_rec']['model']:
                self.hw_channels = config['text_rec']['num_channels'] if 'num_channels' in config['text_rec'] else 1
                norm = config['text_rec']['norm'] if 'norm' in config['text_rec'] else 'batch'
                use_softmax = config['text_rec']['use_softmax'] if 'use_softmax' in config['text_rec'] else True
                if 'Small' in config['text_rec']['model']:
                    self.text_rec = SmallCRNN(config['text_rec']['num_char'],self.hw_channels,norm=norm,use_softmax=use_softmax)
                else:
                    self.text_rec = CRNN(config['text_rec']['num_char'],self.hw_channels,norm=norm,use_softmax=use_softmax)
                    
                self.atr_batch_size = config['text_rec']['batch_size']
                self.pad_text_height = config['text_rec']['pad_text_height'] if 'pad_text_height' in config['text_rec'] else False
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
                else:
                    raise NotImplementedError('Unknown text embedding method: {}'.format(config['text_rec']['embedding']))
            else:
                self.embedding_model = lambda x: None 

            self.merge_embedding_layer = nn.Sequential(nn.ReLU(True),nn.Linear(graph_in_channels+self.numTextFeats,graph_in_channels))
        else:
            self.text_rec=None


        self.add_noise_to_word_embeddings = config['add_noise_to_word_embeddings'] if 'add_noise_to_word_embeddings' in config else 0

        self.blankRelFeats = config['blankRelFeats'] if 'blankRelFeats' in config else False

        if 'DEBUG' in config:
            self.detector.setDEBUG()
            self.setDEBUG()
            self.debug=True
        else:
            self.debug=False
        #t#self.opt_cand=[]#t#
        #t#self.opt_createG=[]#t#
        if type(self.pairer) is BinaryPairReal and type(self.pairer.shape_layers) is not nn.Sequential:
            print("Shape feats aligned to feat dataset.")


    def unfreeze(self): 
        if self.detector_frozen:
            for param in self.detector.parameters(): 
                param.requires_grad=param.will_use_grad 
            self.detector_frozen=False
            print('Unfroze detector')
        

    def forward(self, image, gtBBs=None, gtNNs=None, useGTBBs=False, otherThresh=None, otherThreshIntur=None, hard_detect_limit=300, debug=False,old_nn=False,gtTrans=None,dont_merge=False):
        tic=timeit.default_timer()#t#
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        bbPredictions, offsetPredictions, _,_,_,_ = self.detector(image)
        _=None
        #print('detector')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        saved_features=self.detector.saved_features
        self.detector.saved_features=None
        if self.use2ndFeatures:
            saved_features2=self.detector.saved_features2
        else:
            saved_features2=None
        print('   detector: {}'.format(timeit.default_timer()-tic))#t#

        if saved_features is None:
            print('ERROR:no saved features!')
            import pdb;pdb.set_trace()

        
        tic=timeit.default_timer()#t#
        if self.useHardConfThresh:
            self.used_threshConf = self.confThresh
        else:
            maxConf = bbPredictions[:,:,0].max().item()
            if otherThreshIntur is None:
                confThreshMul = self.confThresh
            else:
                confThreshMul = self.confThresh*(1-otherThreshIntur) + otherThresh*otherThreshIntur
            self.used_threshConf = max(maxConf*confThreshMul,0.5)

        if self.training:
            self.used_threshConf += np.random.normal(0,0.1) #we'll tweak the threshold around to make training more robust

        ###
        #print('THresh: {}'.format(self.used_threshConf))
        ###

        if self.rotation:
            #TODO make this actually check for overseg...
            threshed_bbPredictions = []
            #for b in range(batchSize):
            threshed_bbPredictions.append(bbPredictions[0,bbPredictions[0,:,0]>self.used_threshConf].cpu())
            bbPredictions = threshed_bbPredictions
            #assert(False) #pretty sure this is untested...
            #bbPredictions = non_max_sup_dist(bbPredictions.cpu(),self.used_threshConf,2.5,hard_detect_limit)
        else:
            bbPredictions = non_max_sup_iou(bbPredictions.cpu(),self.used_threshConf,0.4,hard_detect_limit)
        #I'm assuming batch size of one
        assert(len(bbPredictions)==1)
        bbPredictions=bbPredictions[0]
        if self.no_grad_feats:
            bbPredictions=bbPredictions.detach()
        print('   process boxes: {}'.format(timeit.default_timer()-tic))#t#
        #bbPredictions should be switched for GT for training? Then we can easily use BCE loss. 
        #Otherwise we have to to alignment first
        if not useGTBBs:
            if bbPredictions.size(0)==0:
                return [bbPredictions], offsetPredictions, None, None, None, None, None, None, (None,None,None,None)
            if self.include_bb_conf or self.useCurvedBBs: 
                useBBs = bbPredictions
            else:
                useBBs = bbPredictions[:,1:] #remove confidence score
        elif useGTBBs=='saved':
            if self.include_bb_conf or self.useCurvedBBs:
                useBBs = gtBBs
            else:
                useBBs = gtBBs[:,1:]
        else:
            #if gtBBs is None:
            #    if self.text_rec is not None:
            #        transcriptions = self.getTranscriptions(useBBs,image)
            #    else:
            #        transcriptions=None
            #    return [bbPredictions], offsetPredictions, None, None, None, None, None, (useBBs.cpu().detach(),None,None,transcriptions)
            if self.useCurvedBBs:
                useBBs = gtBBs[0,gtBBs[0,:,0]>0.5] #get rid of batch channel and select true boxes (by conf)
            else:
                useBBs = gtBBs[0,:,0:5]
            if self.useShapeFeats or self.relationshipProposal=='feature_nn':
                if self.useCurvedBBs:
                    classes = gtBBs[0,gtBBs[0,:,0]>0.5,6:]
                    classes += torch.rand_like(classes)*0.42 -0.2
                else:
                    classes = gtBBs[0,:,13:]
                    #pos = random.uniform(0.51,0.99)
                    #neg = random.uniform(0.01,0.49)
                    #classes = torch.where(classes==0,torch.tensor(neg).to(classes.device),torch.tensor(pos).to(classes.device))
                    pos = torch.rand_like(classes)/2 +0.5
                    neg = torch.rand_like(classes)/2
                    classes = torch.where(classes==0,neg,pos)
                    if self.detector.predNumNeighbors:
                        nns = gtNNs.float()[0,:,None]
                        #nns += torch.rand_like(nns)/1.5
                        nns += (2*torch.rand_like(nns)-1)
                        useBBs = torch.cat((useBBs,nns,classes),dim=1)
                    else:
                        useBBs = torch.cat((useBBs,classes),dim=1)
            if self.include_bb_conf:
                if self.useCurvedBBs:
                    useBBs[:,0]+=torch.rand(useBBs.size(0))*0.45 - 0.2
                else:
                    #fake some confifence values
                    conf = torch.rand(useBBs.size(0),1)*0.33 +0.66
                    useBBs = torch.cat((conf.to(useBBs.device),useBBs),dim=1)

        useBBs=useBBs.detach()
        #if useGTBBs and self.useCurvedBBs:
        #    useBBs = xyrwhToCurved(useBBs)
        #el
        if self.useCurvedBBs:
            useBBs = [TextLine(bb) for bb in useBBs] #self.x1y1x2y2rToCurved(useBBs)

        if self.text_rec is not None:
            if useGTBBs and gtTrans is not None: # and len(gtTrans)==useBBs.size[0]:
                transcriptions = gtTrans
            else:
                transcriptions = self.getTranscriptions(useBBs,image)
                if gtTrans is not None:
                    if self.include_bb_conf:
                        justBBs = useBBs[:,1:]
                    else:
                        justBBs = useBBs
                    transcriptions=correctTrans(transcriptions,justBBs,gtTrans,gtBBs)
        else:
            transcriptions=None


        if len(useBBs):#useBBs.size(0)>1:
            if self.text_rec is not None:
                embeddings = self.embedding_model(transcriptions)
                if self.add_noise_to_word_embeddings:
                    embeddings += torch.randn_like(embeddings).to(embeddings.device)*self.add_noise_to_word_embeddings*embeddings.mean()
            else:
                embeddings=None


            if self.useMetaGraph:

                groups=[[i] for i in range(len(useBBs))]
                bbTrans = transcriptions
                if self.merge_first:
                    tic=timeit.default_timer()#t#
                    #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
                    #We don't build a full graph, just propose edges and extract the edge features
                    edgeOuts,edgeIndexes,merge_prop_scores = self.createGraph(useBBs,saved_features,saved_features2,image.size(-2),image.size(-1),text_emb=embeddings,image=image,merge_only=True)
                    #_,edgeIndexes, edgeFeatures,_ = graph
                    print('      m1st createGraph time: {}'.format(timeit.default_timer()-tic))#t#
                    #if graph is not None:
                    #    edgeOuts = self.mergepred(graph[2]) #classifier on edge features
                    #    edgeOuts = edgeOuts[:,None,:]
                    #else:
                    #    edgeOuts = None
                    if edgeOuts is not None:
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
                        tic2=timeit.default_timer()#t#
                        useBBs,bbTrans=self.mergeAndGroup(
                                self.mergeThresh[0],None,None,
                                edgeIndexes,edgeOuts,groups,None,None,None,useBBs,bbTrans,image,dont_merge=False,merge_only=True)
                        print('      m1st mergeAndGroup time: {}'.format(timeit.default_timer()-tic2))#t#
                        groups=[[i] for i in range(len(useBBs))]
                        print('merge first reduced graph by {} nodes ({}->{}). max edge pred:{}, mean:{}'.format(startBBs-len(useBBs),startBBs,len(useBBs),torch.sigmoid(edgeOuts.max()),torch.sigmoid(edgeOuts.mean())))
                    print('   merge first time: {}'.format(timeit.default_timer()-tic))#t#
                    if dont_merge:
                        return allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, None, merge_prop_scores, None

                else:
                    merge_prop_scores=None
                    allOutputBoxes=[]
                    allNodeOuts=[]
                    allEdgeOuts=[]
                    allGroups=[]
                    allEdgeIndexes=[]

                tic=timeit.default_timer()#t#
                #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
                graph,edgeIndexes,rel_prop_scores = self.createGraph(useBBs,saved_features,saved_features2,image.size(-2),image.size(-1),text_emb=embeddings,image=image)

                #print('createGraph')
                #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
                print('   createGraph time: {}'.format(timeit.default_timer()-tic))#t#

                #undirected
                #edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
                if graph is None:
                    return [bbPredictions], offsetPredictions, None, None, None, None, rel_prop_scores, merge_prop_scores, (useBBs if self.useCurvedBBs else useBBs.cpu().detach(),None,None,transcriptions)

                last_node_visual_feats = graph[0]
                last_edge_visual_feats = graph[2]

                tic=timeit.default_timer()#t#

                #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
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
                
                for gIter,graphnet in enumerate(self.graphnets[1:]):
                    if self.merge_first:
                        gIter+=1

                    #print('!D! {} before edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(gIter,edgeFeats.size(),len(useBBs),nodeFeats.size(),len(edgeIndexes)))
                    #print('      graph num edges: {}'.format(graph[1].size()))
                    useBBs,graph,groups,edgeIndexes,bbTrans,same_node_map=self.mergeAndGroup(
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
                            image,
                            dont_merge)

                    if self.reintroduce_visual_features:
                        graph,last_node_visual_feats,last_edge_visual_feats = self.appendVisualFeatures(
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
                                debug_image=None)
                    #print('graph 1-:   bbs:{}, nodes:{}, edges:{}'.format(useBBs.size(0),len(groups),len(edgeIndexes)))
                    if len(edgeIndexes)==0:
                        break #we have no graph, so we can just end here
                    #print('!D! after  edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(graph[2].size(),len(useBBs),graph[0].size(),len(edgeIndexes)))
                    #print('      graph num edges: {}'.format(graph[1].size()))
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
                useBBs,graph,groups,edgeIndexes,bbTrans,same_node_map=self.mergeAndGroup(
                        self.mergeThresh[-1],self.keepEdgeThresh[-1],self.groupThresh[-1],
                        edgeIndexes,edgeOuts.detach(),groups,nodeFeats.detach(),edgeFeats.detach(),uniFeats.detach() if uniFeats is not None else None,useBBs,bbTrans,image,dont_merge)
                #print('!D! after  edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(graph[2].size(),useBBs.size(),graph[0].size(),len(edgeIndexes)))
                final=(useBBs if self.useCurvedBBs else useBBs.cpu().detach(),groups,edgeIndexes,bbTrans)
                #print('all iters GCN')
                #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

                print('   thru all iters: {}'.format(timeit.default_timer()-tic))#t#


            else:
                raise NotImplementedError('Simple pairing not implemented for new grouping stuff')
            #adjacencyMatrix = torch.zeros((bbPredictions.size(1),bbPredictions.size(1)))
            #for rel in relOuts:
            #    i,j,a=graphToDetectionsMap(

            return allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final
        else:
            return [bbPredictions], offsetPredictions, None, None, None, None, None, (useBBs if self.useCurvedBBs else useBBs.cpu().detach(),None,None,transcriptions)


    #This ROIAligns features and creates mask images for each edge and node, and runs the embedding convnet and [appends?] these features to the graph... This is only neccesary if a node has been updated...
    #perhaps we need a saved visual feature. If the node/edge is updated, it is recomputed. It is appended  to the graphs current features at each call of a GCN
    def appendVisualFeatures(self,
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
            flip=None):

        node_features, _edge_indexes, edge_features, universal_features = graph
        #same_node_map, maps the old node id (index) to the new one

        node_visual_feats = torch.FloatTensor(node_features.size(0),prev_node_visual_feats.size(1)).to(node_features.device)
        has_feat = [False]*node_features.size(0)
        for old_id,new_id in same_node_map.items():
            has_feat[new_id]=True
            node_visual_feats[new_id] = prev_node_visual_feats[old_id]

        if not all(has_feat):
            if text_emb is not None:    
                need_new_ids,need_groups,need_text_emb = zip(* [(i,g,t) for i,(has,g,t) in enumerate(zip(has_feat,groups,text_emb)) if not has])
            else:
                need_new_ids,need_groups = zip(* [(i,g) for i,(has,g) in enumerate(zip(has_feat,groups)) if not has])
                need_text_emb = None
            if len(need_new_ids)>0:
                need_new_ids=list(need_new_ids)
                need_new_ids=list(need_new_ids)
                allMasks=self.makeAllMasks(image_height,image_width,bbs)
                node_visual_feats[need_new_ids] = self.computeNodeVisualFeatures(features,features2,image_height,image_width,bbs,need_groups,need_text_emb,allMasks,merge_only,debug_image)

        new_to_old_ids = {v:k for k,v in same_node_map.items()}
        edge_visual_feats = torch.FloatTensor(len(edge_indexes),prev_edge_visual_feats.size(1)).to(edge_features.device)
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
                    need_edge_ids.append(ei)
                    need_edge_node_ids.append((n0,n1))
            else:
                need_edge_ids.append(ei)
                need_edge_node_ids.append((n0,n1))
        if len(need_edge_ids)>0:
            edge_visual_feats[need_edge_ids] = self.computeEdgeVisualFeatures(features,features2,image_height,image_width,bbs,groups,need_edge_node_ids,allMasks,flip,merge_only,debug_image)

        #for now, we'll just sum the features.
        #new_graph = (torch.cat((node_features,node_visual_feats),dim=1),edge_indexes,torch.cat((edge_features,edge_visual_feats),dim=1),universal_features)
        if edge_features.size(1)==0:
            new_graph = (node_features+node_visual_feats,_edge_indexes,edge_visual_feats,universal_features)
        elif edge_features.size(0)==edge_visual_feats.size(0)*2:
            new_graph = (node_features+node_visual_feats,_edge_indexes,edge_features+edge_visual_feats.repeat(2,1),universal_features)
        else:
            new_graph = (node_features+node_visual_feats,_edge_indexes,edge_features+edge_visual_feats,universal_features)
        #edge features get repeated for bidirectional graph
        return new_graph, node_visual_feats, edge_visual_feats

    #This rewrites the confidence and class predictions based on the (re)predictions from the graph network
    def updateBBs(self,bbs,groups,nodeOuts):
        if self.useCurvedBBs:
            nodeConfPred = torch.sigmoid(nodeOuts[:,-1,self.nodeIdxConf]).cpu().detach()
            startIndex = 5+self.nodeIdxClass
            endIndex = 5+self.nodeIdxClassEnd
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
                bbConfPred = torch.FloatTensor(bbs.size(0),1)
                for i,group in enumerate(groups):
                    bbConfPred[group] = nodeConfPred[i].detach()
                if self.include_bb_conf:
                    bbs[:,0:1] = bbConfPred
                else:
                    bbs = torch.cat((bbConfPred,bbs.cpu()),dim=1)
            elif bbs.size(0)==1 and not self.include_bb_conf:
                bbs = torch.cat((torch.FloatTensor(1,1).fill_(1).to(bbs.device),bbs),dim=2)

            if self.predNN:
                raise NotImplementedError('Have not implemented num neighbor pred for new graph method')
                
            if self.predClass:
                startIndex = 5+self.nodeIdxClass
                endIndex = 5+self.nodeIdxClassEnd
                #if not useGTBBs:
                nodeClassPred = torch.sigmoid(nodeOuts[:,-1,self.nodeIdxClass:self.nodeIdxClassEnd].detach()).cpu()
                bbClasPred = torch.FloatTensor(bbs.size(0),self.nodeIdxClassEnd-self.nodeIdxClass)
                for i,group in enumerate(groups):
                    bbClasPred[group] = nodeClassPred[i].detach()
                bbs[:,startIndex:endIndex] = bbClasPred
        return bbs

    #This merges two bounding box predictions, assuming they were oversegmented
    def mergeBB(self,bb0,bb1,image):
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
                classIdx=6
                conf = (bb0[0:1]+bb1[0:1])/2
            else:
                locIdx=0
                classIdx=5
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
    def mergeAndGroupCurved(self,mergeThresh,keepEdgeThresh,groupThresh,oldEdgeIndexes,edgePredictions,oldGroups,oldNodeFeats,oldEdgeFeats,oldUniversalFeats,oldBBs,oldBBTrans,image,dont_merge=False,merge_only=False):
        assert(len(oldBBs)==0 or type(oldBBs[0]) is TextLine)
        #changedNodeIds=set()
        bbs={i:v for i,v in enumerate(oldBBs)}
        if self.text_rec is not None:
            bbTrans={i:v for i,v  in enumerate(oldBBTrans)}
        oldToNewBBIndexes={i:i for i in range(len(oldBBs))}
        #newBBs_line={}
        newBBIdCounter=0
        #toMergeBBs={}
        if not merge_only:
            edgePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach()
            #relPreds = torch.sigmoid(edgePredictions[:,-1,1]).cpu().detach()
            mergePreds = torch.sigmoid(edgePredictions[:,-1,2]).cpu().detach()
            groupPreds = torch.sigmoid(edgePredictions[:,-1,3]).cpu().detach()
        else:
            mergePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach()
        ##Prevent all nodes from merging during first iterations (bad init):
        if not dont_merge:
            mergedTo=set()
            #check for merges, where we will combine two BBs into one
            for i,(n0,n1) in enumerate(oldEdgeIndexes):
                #mergePred = edgePreds[i,-1,1]
                
                if mergePreds[i]>mergeThresh: #TODO condition this on whether it is correct. and GT?:
                    if len(oldGroups[n0])==1 and len(oldGroups[n1])==1: #can only merge ungrouped nodes. This assumption is used later in the code WXS
                        #changedNodeIds.add(n0)
                        #changedNodeIds.add(n1)
                        bbId0 = oldGroups[n0][0]
                        bbId1 = oldGroups[n1][0]
                        newId0 = oldToNewBBIndexes[bbId0]
                        bb0ToMerge = bbs[newId0]

                        newId1 = oldToNewBBIndexes[bbId1]
                        bb1ToMerge = bbs[newId1]


                        if newId0!=newId1:
                            bb0ToMerge.merge(bb1ToMerge) # "
                            #merge two merged bbs
                            oldToNewBBIndexes = {k:(v if v!=newId1 else newId0) for k,v in oldToNewBBIndexes.items()}
                            del bbs[newId1]
                            if self.text_rec is not None:
                                del bbTrans[newId1]
                            mergedTo.add(newId0)


            oldBBIdToNew = oldToNewBBIndexes
                    

            if self.text_rec is not None and len(newBBs)>0:
                doTransIndexes = [idx for idx in mergedTo is idx in bbs]
                doBBs = [bbs[idx] for idx in doTransIndexes]
                newTrans = self.getTranscriptions(doBBs,image)
                for i,idx in enumerate(doTransIndexes):
                    bbTrans[idx] = newTrans[i]
            if merge_only:
                newBBs=[]
                newBBTrans=[] if self.text_rec is not None else None
                for bbId,bb in bbs.items():
                    newBBs.append(bb)
                    if self.text_rec is not None:
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
            for i in range(oldNodeFeats.size(0)):
                assert(i in oldGroupToNew or i in workGroups)

            #We'll adjust the edges to acount for merges as well as prune edges and get ready for grouping
            #temp = oldEdgeIndexes
            #oldEdgeIndexes = []

            #Prune and adjust the edges (to groups)
            groupEdges=[]

            #D_numOldEdges=len(oldEdgeIndexes)
            #D_numOldAboveThresh=(edgePreds>keepEdgeThresh).sum()
            for i,(n0,n1) in enumerate(oldEdgeIndexes):
                if edgePreds[i]>keepEdgeThresh:
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

            #print('!D! original edges:{}, above thresh:{}, kept edges:{}'.format(D_numOldEdges,D_numOldAboveThresh,len(edgeFeats)))
             
        else:
            #skipping merging
            groupEdges=[]
            for i,(n0,n1) in enumerate(oldEdgeIndexes):
                if edgePreds[i]>keepEdgeThresh:
                    groupEdges.append((groupPreds[i].item(),n0,n1))
            #oldEdgeIndexes=None



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
        for i in range(oldNodeFeats.size(0)):
            assert(i in oldGroupToNewGrouping or i in oldGroupToNew)

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
        newNodeFeats = torch.FloatTensor(len(workGroups),oldNodeFeats.size(1)).to(oldNodeFeats.device)
        oldToNewNodeIds_unchanged={}
        oldToNewIds_all={}
        newGroups=[]
        groupNodeTrans=[]
        for i,(idx,bbIds) in enumerate(workGroups.items()):
            newGroups.append([bbIdToPos[bbId] for bbId in bbIds])
            featsToCombine=[]
            for oldIdx in newGroupToOldGrouping[idx]:
                oldToNewIds_all[oldIdx]=i
                featsToCombine.append(oldNodeFeats[oldIdx])
                if oldIdx in newGroupToOldMerge:
                    for mergedIdx in newGroupToOldMerge[oldIdx]:
                        featsToCombine.append(oldNodeFeats[mergedIdx])
                        oldToNewIds_all[mergedIdx]=i
            if len(featsToCombine)==1:
                oldToNewNodeIds_unchanged[oldIdx]=i
                newNodeFeats[i]=featsToCombine[0]
            else:
                newNodeFeats[i]=self.groupNodeFunc(featsToCombine)

            if self.text_rec is not None:
                groupTrans = [(bbs[bbId].getReadPosition(),bbTrans[bbId]) for bbId in bbIds]
                groupTrans.sort(key=lambda a:a[0])
                groupNodeTrans.append(' '.join([t[1] for t in groupTrans]))
        #D#
        assert(all([x in oldToNewIds_all for x in range(oldNodeFeats.size(0))]))

        
        #find overlapped edges and combine
        #first change all node ids to their new ones
        D_newToOld = {v:k for k,v in oldToNewNodeIds_unchanged.items()}
        newEdges_map=defaultdict(list)
        for i,(n0,n1)  in  enumerate(oldEdgeIndexes):
            new_n0 = oldToNewIds_all[n0]
            new_n1 = oldToNewIds_all[n1]
            newEdges_map[(min(new_n0,new_n1),max(new_n0,new_n1))].append(i)

            #D#
            if new_n0 in D_newToOld and new_n1 in D_newToOld:
                o0 = D_newToOld[new_n0]
                o1 = D_newToOld[new_n1]
                assert( (min(o0,o1),max(o0,o1)) in oldEdgeIndexes )
        #This leaves some old edges pointing to the same new edge, so combine their features
        newEdges=[]
        newEdgeFeats=torch.FloatTensor(len(newEdges_map),oldEdgeFeats.size(1)).to(oldEdgeFeats.device)
        for edge,oldIds in newEdges_map.items():
            if len(oldIds)==1:
                newEdgeFeats[len(newEdges)]=oldEdgeFeats[oldIds[0]]
            else:
                newEdgeFeats[len(newEdges)]=self.groupEdgeFunc([oldEdgeFeats[oId] for oId in oldIds])
            newEdges.append(edge)



        if self.text_rec is not None:
            newNodeEmbeddings = self.embedding_model(groupNodeTrans)
            if self.add_noise_to_word_embeddings>0:
                newNodeEmbeddings += torch.randn_like(newNodeEmbeddings).to(newNodeEmbeddings.device)*self.add_noise_to_word_embeddings
            newNodeFeats = self.merge_embedding_layer(torch.cat((newNodeFeats,newNodeEmbeddings),dim=1))

        edges = newEdges
        newEdges = list(newEdges) + [(y,x) for x,y in newEdges] #add reverse edges so undirected/bidirectional
        if len(newEdges)>0:
            newEdgeIndexes = torch.LongTensor(newEdges).t().to(oldEdgeFeats.device)
        else:
            newEdgeIndexes = torch.LongTensor(0)
        newEdgeFeats = newEdgeFeats.repeat(2,1)

        newGraph = (newNodeFeats, newEdgeIndexes, newEdgeFeats, oldUniversalFeats)

        ###DEBUG###
        newToOld = {v:k for k,v in oldToNewNodeIds_unchanged.items()}
        for n0,n1 in edges:
            if n0 in newToOld and n1 in newToOld:
                o0 = newToOld[n0]
                o1 = newToOld[n1]
                assert( (min(o0,o1),max(o0,o1)) in oldEdgeIndexes )

        ##D###

        return newBBs, newGraph, newGroups, edges, newBBTrans if self.text_rec is not None else None,  oldToNewNodeIds_unchanged

    def mergeAndGroup(self,mergeThresh,keepEdgeThresh,groupThresh,oldEdgeIndexes,edgePredictions,oldGroups,oldNodeFeats,oldEdgeFeats,oldUniversalFeats,oldBBs,bbTrans,image,dont_merge=False,merge_only=False):
        if self.useCurvedBBs:
            return self.mergeAndGroupCurved(mergeThresh,keepEdgeThresh,groupThresh,oldEdgeIndexes,edgePredictions,oldGroups,oldNodeFeats,oldEdgeFeats,oldUniversalFeats,oldBBs,bbTrans,image,dont_merge,merge_only)
        changedNodeIds=set()
        newBBs={}
        #newBBs_line={}
        newBBIdCounter=0
        #toMergeBBs={}
        oldToNewBBIndexes={}
        if not merge_only:
            edgePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach()
            #relPreds = torch.sigmoid(edgePredictions[:,-1,1]).cpu().detach()
            mergePreds = torch.sigmoid(edgePredictions[:,-1,2]).cpu().detach()
            groupPreds = torch.sigmoid(edgePredictions[:,-1,3]).cpu().detach()
        else:
            mergePreds = torch.sigmoid(edgePredictions[:,-1,0]).cpu().detach()
        ##Prevent all nodes from merging during first iterations (bad init):
        if not dont_merge:
        
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

                        newBB= self.mergeBB(bb0,bb1,image)

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
                    newNodeFeat = self.groupNodeFunc( [oldNodeFeats[on] for on in oldNodes] )
                    newNodeFeats.append(newNodeFeat)
                    changedNodeIds.update(oldNodes)
                else:
                    newNodeFeats.append(oldNodeFeats[id])
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
                        edgeFeats.append([oldEdgeFeats[i]])
                    #else:
                    #    It disapears
            #oldEdgeIndexes=None

            #print('!D! original edges:{}, above thresh:{}, kept edges:{}'.format(D_numOldEdges,D_numOldAboveThresh,len(edgeFeats)))
             
        else:
            #skipping merging
            bbs=oldBBs
            groupEdges=[]
            edgeFeats = []
            for i,(n0,n1) in enumerate(oldEdgeIndexes):
                if edgePreds[i]>keepEdgeThresh:
                    groupEdges.append((groupPreds[i].item(),n0,n1))
                    edgeFeats.append([oldEdgeFeats[i]])
            #oldEdgeIndexes=None



        #Find nodes that should be grouped
        ##NEWER, just merge the groups with the highest score between them. when merging edges, sum the scores
        #groupEdges=[]
        #edgeFeats = [[oldEdgeFeats[i]] for i in range(oldEdgeFeats.size(0))]
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
                    mEdgeFeats[g1E] += edgeFeats[i]
                elif g1E==g1 or g1E==g0:
                    mEdges[g0E].append(scoreE)
                    mEdgeFeats[g0E] += edgeFeats[i]
                else:
                    newGroupEdges.append((scoreE,g0E,g1E))
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
            assert(len(newEdgeFeats)==len(groupEdges))


            newNodeFeats[g0] += newNodeFeats[g1] #self.groupNodeFunc(newNodeFeats[g0],newNodeFeats[g1])
            del newNodeFeats[g1]
        #print('!D! num edges after grouping {}'.format(len(groupEdges)))

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
        assert(len(newEdgeFeats)==len(newEdges))


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

        newNodeFeats = torch.stack(newNodeFeats,dim=0)
        if self.text_rec is not None:
            newNodeEmbeddings = self.embedding_model(newNodeTrans)
            if self.add_noise_to_word_embeddings>0:
                newNodeEmbeddings += torch.randn_like(newNodeEmbeddings).to(newNodeEmbeddings.device)*self.add_noise_to_word_embeddings
            newNodeFeats = self.merge_embedding_layer(torch.cat((newNodeFeats,newNodeEmbeddings),dim=1))

        if len(newEdgeFeats)>0:
            newEdgeFeats = torch.stack(newEdgeFeats,dim=0)
        else:
            newEdgeFeats = torch.FloatTensor(0)
        edges = newEdges
        newEdges = list(newEdges) + [(y,x) for x,y in newEdges] #add reverse edges so undirected/bidirectional
        if len(newEdges)>0:
            newEdgeIndexes = torch.LongTensor(newEdges).t().to(oldEdgeFeats.device)
        else:
            newEdgeIndexes = torch.LongTensor(0)
        newEdgeFeats = newEdgeFeats.repeat(2,1)

        newGraph = (newNodeFeats, newEdgeIndexes, newEdgeFeats, oldUniversalFeats)

        ###DEBUG###
        newToOld = {v:k for k,v in oldToNewNodeIds.items()}
        for n0,n1 in edges:
            if n0 in newToOld and n1 in newToOld:
                o0 = newToOld[n0]
                o1 = newToOld[n1]
                assert( (min(o0,o1),max(o0,o1)) in oldEdgeIndexes )

        ##D###

        return bbs, newGraph, newGroups, edges, bbTrans,  oldToNewNodeIds


                



    def createGraph(self,bbs,features,features2,imageHeight,imageWidth,text_emb=None,flip=None,debug_image=None,image=None,merge_only=False):
        tic=timeit.default_timer()#t#
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        if self.relationshipProposal == 'line_of_sight':
            assert(not merge_only)
            candidates = self.selectLineOfSightEdges(bbs,imageHeight,imageWidth)
            rel_prop_scores = None
        elif self.relationshipProposal == 'feature_nn':
            candidates, rel_prop_scores = self.selectFeatureNNEdges(bbs,imageHeight,imageWidth,image,features.device,merge_only=merge_only)
            if not self.useCurvedBBs:
                bbs=bbs[:,1:] #discard confidence, we kept it so the proposer could see them
        print('     candidate: {}'.format(timeit.default_timer()-tic))#t#
        if len(candidates)==0:
            if self.useMetaGraph:
                return None, None, None
            else:
                return None,None,None,None,None, None

        random.shuffle(candidates)
        #print('proposed relationships: {}  (bbs:{})'.format(len(candidates),len(bbs)))
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        #print('------prop------')

        allMasks=self.makeAllMasks(imageHeight,imageWidth,bbs)
        groups=[[i] for i in range(len(bbs))]
        relFeats = self.computeEdgeVisualFeatures(features,features2,imageHeight,imageWidth,bbs,groups,candidates,allMasks,flip,merge_only,debug_image)

        #if self.useShapeFeats=='sp
        #print('rel features built')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        #print('------rel------')
        if merge_only:
            return relFeats, candidates, rel_prop_scores #we won't build the graph
    
        #compute features for the bounding boxes by themselves
        #This will be replaced with/appended to some type of word embedding
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        bb_features = self.computeNodeVisualFeatures(features,features2,imageHeight,imageWidth,bbs,groups,text_emb,allMasks,merge_only,debug_image)
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
            edgeFeatures= relFeats

            edges=candidates
            edges += [(y,x) for x,y in edges] #add backward edges for undirected graph
            edgeIndexes = torch.LongTensor(edges).t().to(relFeats.device)
            #now we need to also replicate the edgeFeatures
            edgeFeatures = edgeFeatures.repeat(2,1)

            #features
            universalFeatures=None

            #t#time = timeit.default_timer()-tic#t#
            #t#print('   create graph: {}'.format(time)) #old 0.37, new 0.16
            #t#self.opt_createG.append(time)
            #t#if len(self.opt_createG)>17:#t#
            #t#    print('   create graph running mean: {}'.format(np.mean(self.opt_createG)))#t#
            #t#    if len(self.opt_createG)>30:#t#
            #t#        self.opt_createG = self.opt_createG[1:]#t#
            return (nodeFeatures, edgeIndexes, edgeFeatures, universalFeatures), relIndexes, rel_prop_scores
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

            #t#time = timeit.default_timer()-tic#t#
            print('   create graph: {}'.format(time)) #old 0.37
            self.opt_createG.append(time)
            if len(self.opt_createG)>17:
                print('   create graph running mean: {}'.format(np.mean(self.opt_createG)))
                if len(self.opt_createG)>30:
                    self.opt_createG = self.opt_createG[1:]
            #return bb_features, adjacencyMatrix, rel_features
            return bbAndRel_features, (adjacencyMatrix,numOfNeighbors), numBB, numRel, relIndexes, rel_prop_scores

    def makeAllMasks(self,imageHeight,imageWidth,bbs):
        #build all-mask image, may want to move this up and use for relationship proposals
        if self.expandedRelContext is not None:
            allMasks = torch.zeros(imageHeight,imageWidth)
            if self.use_fixed_masks:
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
        else:
            pool_h=self.pool_h
            pool_w=self.pool_w
            pool2_h=self.pool2_h
            pool2_w=self.pool2_w
        #t#tic=timeit.default_timer()#t#

        #stackedEdgeFeatWindows = torch.FloatTensor((len(edges),features.size(1)+2,self.relWindowSize,self.relWindowSize)).to(features.device())

        if not self.useCurvedBBs:
            assert(all([len(g)==1 for g in groups]) )#not implemented for groups
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

        if debug_image is not None:
            debug_images=[]
            debug_masks=[]



        if self.useShapeFeats!='only':
            #get axis aligned rectangle from corners
            rois = torch.zeros((len(edges),5)) #(batchIndex,x1,y1,x2,y2) as expected by ROI Align
            if self.useCurvedBBs:

                ##old
                bbs_index1 = [bbs[c[0]] for c in edges]
                bbs_index2 = [bbs[c[1]] for c in edges]
                t_tic = timeit.default_timer()
                min_X1,min_Y1,max_X1,max_Y1 = torch.IntTensor([bb.boundingRect() for bb in bbs_index1]).permute(1,0)
                min_X2,min_Y2,max_X2,max_Y2 = torch.IntTensor([bb.boundingRect() for bb in bbs_index2]).permute(1,0)
                min_X = torch.min(min_X1,min_X2)
                min_Y = torch.min(min_Y1,min_Y2)
                max_X = torch.max(max_X1,max_X2)
                max_Y = torch.max(max_Y1,max_Y2)

                t_old_time = timeit.default_timer()-t_tic

                ##new
                t_tic = timeit.default_timer()
                min_X,max_X,min_Y,max_Y=torch.IntTensor([minAndMaxXY(
                    [bbs[b].boundingRect() for b in groups[c[0]]] + [bbs[b].boundingRect() for b in groups[c[1]]]
                        ) for c in edges]).permute(1,0)
                t_new_time = timeit.default_timer()-t_tic
                print('minMax difference: {}, old: {}, new: {}'.format(t_old_time-t_new_time,t_old_time,t_new_time))

                groups_index1 = [ [bbs[b] for b in groups[c[0]]] for c in edges ]
                groups_index2 = [ [bbs[b] for b in groups[c[1]]] for c in edges ]


            else:
                assert(all([len(g)==1 for g in groups])) #not implemented for groups
                bbs_index1 = bbs[[c[0] for c in edges]]
                bbs_index2 = bbs[[c[1] for c in edges]]

                tlX_index1 = tlX[[c[0] for c in edges]]
                tlX_index2 = tlX[[c[1] for c in edges]]
                trX_index1 = trX[[c[0] for c in edges]]
                trX_index2 = trX[[c[1] for c in edges]]
                blX_index1 = blX[[c[0] for c in edges]]
                blX_index2 = blX[[c[1] for c in edges]]
                brX_index1 = brX[[c[0] for c in edges]]
                brX_index2 = brX[[c[1] for c in edges]]
                tlY_index1 = tlY[[c[0] for c in edges]]
                tlY_index2 = tlY[[c[1] for c in edges]]
                trY_index1 = trY[[c[0] for c in edges]]
                trY_index2 = trY[[c[1] for c in edges]]
                blY_index1 = blY[[c[0] for c in edges]]
                blY_index2 = blY[[c[1] for c in edges]]
                brY_index1 = brY[[c[0] for c in edges]]
                brY_index2 = brY[[c[1] for c in edges]]
                max_X,_ = torch.max(torch.stack([tlX_index1,tlX_index2,
                                            trX_index1,trX_index2,
                                            blX_index1,blX_index2,
                                            brX_index1,brX_index2],dim=0), dim=0 )
                min_X,_ = torch.min(torch.stack([tlX_index1,tlX_index2,
                                            trX_index1,trX_index2,
                                            blX_index1,blX_index2,
                                            brX_index1,brX_index2],dim=0), dim=0 )

                max_Y,_ = torch.max(torch.stack([tlY_index1,tlY_index2,
                                            trY_index1,trY_index2,
                                            blY_index1,blY_index2,
                                            brY_index1,brY_index2],dim=0), dim=0 )
                min_Y,_ = torch.min(torch.stack([tlY_index1,tlY_index2,
                                            trY_index1,trY_index2,
                                            blY_index1,blY_index2,
                                            brY_index1,brY_index2],dim=0), dim=0 )
            if merge_only:
                padX = self.expandedMergeContextX
                padY = self.expandedMergeContextY
            else:
                padX=padY=  self.expandedRelContext
            max_X = torch.min(max_X+padX,torch.IntTensor([imageWidth-1]))
            min_X = torch.max(min_X-padX,torch.IntTensor([0]))
            max_Y = torch.min(max_Y+padY,torch.IntTensor([imageHeight-1]))
            min_Y = torch.max(min_Y-padY,torch.IntTensor([0]))
            rois[:,1]=min_X
            rois[:,2]=min_Y
            rois[:,3]=max_X
            rois[:,4]=max_Y



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
        if self.expandedRelContext is not None:
            #We're going to add a third mask for all bbs
            numMasks=3
        else:
            numMasks=2

        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        relFeats=[] #where we'll store the feature of each batch
        innerbatches = [(s,min(s+self.roi_batch_size,len(edges))) for s in range(0,len(edges),self.roi_batch_size)]
        #crop from feats, ROI pool
        for ib,(b_start,b_end) in enumerate(innerbatches): #we can batch extracting computing the feature vector from rois to save memory
            if ib>0:
                torch.set_grad_enabled(False)
            b_rois = rois[b_start:b_end]
            b_edges = edges[b_start:b_end]
            b_groups_index1 = groups_index1[b_start:b_end]
            b_groups_index2 = groups_index2[b_start:b_end]
            #with profiler.profile(profile_memory=True, record_shapes=True) as prof:

            if merge_only:
                stackedEdgeFeatWindows = self.merge_roi_align(features,b_rois.to(features.device))
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
            #print('{} roi profile'.format('merge' if merge_only else 'full'))
            #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

            #create and add masks
            masks = torch.zeros(stackedEdgeFeatWindows.size(0),numMasks,pool2_h,pool2_w)
            if self.useShapeFeats:
                shapeFeats = torch.FloatTensor(stackedEdgeFeatWindows.size(0),self.numShapeFeats)
            if self.detector.predNumNeighbors:
                extraPred=1
            else:
                extraPred=0


            #make instance specific masks and make shape (spatial) features
            if self.useShapeFeats!='only':
                if (random.random()<0.5 and flip is None and  not self.debug) or flip:
                    pass
                    #TODO
                feature_w = b_rois[:,3]-b_rois[:,1] +1
                feature_h = b_rois[:,4]-b_rois[:,2] +1
                w_m = pool2_w/feature_w
                h_m = pool2_h/feature_h

                if not self.useCurvedBBs:
                    tlX1 = (tlX_index1-b_rois[:,1])*w_m
                    trX1 = (trX_index1-b_rois[:,1])*w_m
                    brX1 = (brX_index1-b_rois[:,1])*w_m
                    blX1 = (blX_index1-b_rois[:,1])*w_m
                    tlY1 = (tlY_index1-b_rois[:,2])*h_m
                    trY1 = (trY_index1-b_rois[:,2])*h_m
                    brY1 = (brY_index1-b_rois[:,2])*h_m
                    blY1 = (blY_index1-b_rois[:,2])*h_m
                    tlX2 = (tlX_index2-b_rois[:,1])*w_m
                    trX2 = (trX_index2-b_rois[:,1])*w_m
                    brX2 = (brX_index2-b_rois[:,1])*w_m
                    blX2 = (blX_index2-b_rois[:,1])*w_m
                    tlY2 = (tlY_index2-b_rois[:,2])*h_m
                    trY2 = (trY_index2-b_rois[:,2])*h_m
                    brY2 = (brY_index2-b_rois[:,2])*h_m
                    blY2 = (blY_index2-b_rois[:,2])*h_m

            for i,(index1, index2) in enumerate(b_edges):
                if self.useShapeFeats!='only':
                    if self.useCurvedBBs:
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
                        rr, cc = draw.polygon(
                                    [round(tlY1[i].item()),round(trY1[i].item()),
                                    round(brY1[i].item()),round(blY1[i].item())],
                                    [round(tlX1[i].item()),round(trX1[i].item()),
                                    round(brX1[i].item()),round(blX1[i].item())], 
                                    [pool2_h,pool2_w])
                        masks[i,0,rr,cc]=1

                        rr, cc = draw.polygon(
                                    [round(tlY2[i].item()),round(trY2[i].item()),
                                    round(brY2[i].item()),round(blY2[i].item())],
                                    [round(tlX2[i].item()),round(trX2[i].item()),
                                    round(brX2[i].item()),round(blX2[i].item())], 
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

        

            if self.useShapeFeats:
                if self.useCurvedBBs:

                    #conf, x,y,r,h,w,tlx,tly,trx,try,brx,bry,blx,bly,r_left,r_rightA,classFeats = bb.getFeatureInfo()
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
                    shapeFeats[:,20:20+self.numBBTypes] = allFeats1[:,18:]
                    shapeFeats[:,20+self.numBBTypes:20+2*self.numBBTypes] = allFeats2[:,18:]
                    assert(not torch.isnan(shapeFeats).any())



                else:
                    if type(self.pairer) is BinaryPairReal and type(self.pairer.shape_layers) is not nn.Sequential:
                        #The index specification is to allign with the format feat nets are trained with
                        ixs=[0,1,2,3,3+self.numBBTypes,3+self.numBBTypes,4+self.numBBTypes,5+self.numBBTypes,6+self.numBBTypes,6+2*self.numBBTypes,6+2*self.numBBTypes,7+2*self.numBBTypes]
                    else:
                        ixs=[4,6,2,8,8+self.numBBTypes,5,7,3,8+self.numBBTypes,8+self.numBBTypes+self.numBBTypes,0,1]
                    
                    shapeFeats[:,ixs[0]] = 2*b_bbs_index1[:,3]/self.normalizeVert #bb preds half height/width
                    shapeFeats[:,ixs[1]] = 2*b_bbs_index1[:,4]/self.normalizeHorz
                    shapeFeats[:,ixs[2]] = b_bbs_index1[:,2]/math.pi
                    shapeFeats[:,ixs[3]:ixs[4]] = b_bbs_index1[:,extraPred+5:]# torch.sigmoid(b_bbs_index1[:,extraPred+5:])

                    shapeFeats[:,ixs[5]] = 2*b_bbs_index2[:,3]/self.normalizeVert
                    shapeFeats[:,ixs[6]] = 2*b_bbs_index2[:,4]/self.normalizeHorz
                    shapeFeats[:,ixs[7]] = b_bbs_index2[:,2]/math.pi
                    shapeFeats[:,ixs[8]:ixs[9]] = b_bbs_index2[:,extraPred+5:]#torch.sigmoid(b_bbs_index2[:,extraPred+5:])

                    shapeFeats[:,ixs[10]] = (b_bbs_index1[:,0]-b_bbs_index2[:,0])/self.normalizeHorz
                    shapeFeats[:,ixs[11]] = (b_bbs_index1[:,1]-b_bbs_index2[:,1])/self.normalizeVert
                    if self.useShapeFeats!='old':
                        startCorners = 8+self.numBBTypes+self.numBBTypes
                        shapeFeats[:,startCorners +0] = torch.sqrt( (tlX_index1-tlX_index2)**2 + (tlY_index1-tlY_index2)**2 )/self.normalizeDist
                        shapeFeats[:,startCorners +1] = torch.sqrt( (trX_index1-trX_index2)**2 + (trY_index1-trY_index2)**2 )/self.normalizeDist
                        shapeFeats[:,startCorners +3] = torch.sqrt( (brX_index1-brX_index2)**2 + (brY_index1-brY_index2)**2 )/self.normalizeDist
                        shapeFeats[:,startCorners +2] = torch.sqrt( (blX_index1-blX_index2)**2 + (blY_index1-blY_index2)**2 )/self.normalizeDist
                        startNN =startCorners+4
                    else:
                        startNN = 8+self.numBBTypes+self.numBBTypes
                    if self.detector.predNumNeighbors:
                        shapeFeats[:,startNN +0] = b_bbs_index1[:,5]
                        shapeFeats[:,startNN +1] = b_bbs_index2[:,5]
                        startPos=startNN+2
                    else:
                        startPos=startNN
                    if self.usePositionFeature:
                        if self.usePositionFeature=='absolute':
                            shapeFeats[:,startPos +0] = (b_bbs_index1[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                            shapeFeats[:,startPos +1] = (b_bbs_index1[:,1]-imageHeight/2)/(10*self.normalizeVert)
                            shapeFeats[:,startPos +2] = (b_bbs_index2[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                            shapeFeats[:,startPos +3] = (b_bbs_index2[:,1]-imageHeight/2)/(10*self.normalizeVert)
                        else:
                            shapeFeats[:,startPos +0] = (b_bbs_index1[:,0]-imageWidth/2)/(imageWidth/2)
                            shapeFeats[:,startPos +1] = (b_bbs_index1[:,1]-imageHeight/2)/(imageHeight/2)
                            shapeFeats[:,startPos +2] = (b_bbs_index2[:,0]-imageWidth/2)/(imageWidth/2)
                            shapeFeats[:,startPos +3] = (b_bbs_index2[:,1]-imageHeight/2)/(imageHeight/2)

            ###DEBUG
            if debug_image is not None:
                for i in range(4):
                    img_f.imshow('b{}-{} crop rel {}'.format(b_start,b_end,i),debug_images[i])
                    img_f.imshow('b{}-{} masks rel {}'.format(b_start,b_end,i),masks[i].numpy().transpose([1,2,0]))
                    img_f.imshow('b{}-{} mask all rel {}'.format(b_start,b_end,i),debug_masks[i].numpy())
                img_f.waitKey()
                debug_images=[]


            #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
            if self.useShapeFeats!='only':
                if self.splitFeatures:
                    stackedEdgeFeatWindows2 = torch.cat((stackedEdgeFeatWindows2,masks.to(stackedEdgeFeatWindows2.device)),dim=1)
                    if merge_only:
                        b_relFeats = self.mergeFeaturizerConv2(stackedEdgeFeatWindows2)
                    else:
                        b_relFeats = self.relFeaturizerConv2(stackedEdgeFeatWindows2)
                    stackedEdgeFeatWindows = torch.cat((stackedEdgeFeatWindows,b_relFeats),dim=1)
                else:
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
            if self.useShapeFeats:
                if self.useShapeFeats=='only':
                    b_relFeats = shapeFeats.to(features.device)
                else:
                    b_relFeats = torch.cat((b_relFeats,shapeFeats.to(features.device)),dim=1)
            assert(not torch.isnan(b_relFeats).any())
            relFeats.append(b_relFeats)
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

            rois = torch.zeros((len(groups),5))
            if self.useCurvedBBs:
                min_X,max_X,min_Y,max_Y=torch.IntTensor([
                        minAndMaxXY([bbs[b].boundingRect() for b in group])
                        for group in groups]).permute(1,0)
            else:
                assert(all([len(g)==1 for g in groups])) #not implemented for groups
                min_Y,_ = torch.min(torch.stack([tlY,trY,blY,brY],dim=0),dim=0)
                max_Y,_ = torch.max(torch.stack([tlY,trY,blY,brY],dim=0),dim=0) 
                min_X,_ = torch.min(torch.stack([tlX,trX,blX,brX],dim=0),dim=0) 
                max_X,_ = torch.max(torch.stack([tlX,trX,blX,brX],dim=0),dim=0) 
            if self.expandedBBContext is not None:
                max_X = torch.min(max_X+self.expandedBBContext,torch.IntTensor([imageWidth-1]))
                min_X = torch.max(min_X-self.expandedBBContext,torch.IntTensor([0]))
                max_Y = torch.min(max_Y+self.expandedBBContext,torch.IntTensor([imageHeight-1]))
                min_Y = torch.max(min_Y-self.expandedBBContext,torch.IntTensor([0]))
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
                    node_shapeFeats[:,0]= (bbs[:,2]+math.pi)/(2*math.pi)
                    node_shapeFeats[:,1]=bbs[:,3]/self.normalizeVert
                    node_shapeFeats[:,2]=bbs[:,4]/self.normalizeHorz
                    if self.detector.predNumNeighbors:
                        node_shapeFeats[:,3]=bbs[:,5]
                    node_shapeFeats[:,3+extraPred:self.numBBTypes+3+extraPred]=torch.sigmoid(bbs[:,5+extraPred:self.numBBTypes+5+extraPred])
                    if self.usePositionFeature:
                        if self.usePositionFeature=='absolute':
                            node_shapeFeats[:,self.numBBTypes+3+extraPred] = (bbs[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                            node_shapeFeats[:,self.numBBTypes+4+extraPred] = (bbs[:,1]-imageHeight/2)/(10*self.normalizeVert)
                        else:
                            node_shapeFeats[:,self.numBBTypes+3+extraPred] = (bbs[:,0]-imageWidth/2)/(imageWidth/2)
                            node_shapeFeats[:,self.numBBTypes+4+extraPred] = (bbs[:,1]-imageHeight/2)/(imageHeight/2)
            if self.useShapeFeats != "only" and self.expandedBBContext:
                #Add detected BB masks
                #warp to roi space
                feature_w = rois[:,3]-rois[:,1] +1
                feature_h = rois[:,4]-rois[:,2] +1
                w_m = self.poolBB2_w/feature_w
                h_m = self.poolBB2_h/feature_h

                if not self.useCurvedBBs:
                    tlX1 = (tlX-rois[:,1])*w_m
                    trX1 = (trX-rois[:,1])*w_m
                    brX1 = (brX-rois[:,1])*w_m
                    blX1 = (blX-rois[:,1])*w_m
                    tlY1 = (tlY-rois[:,2])*h_m
                    trY1 = (trY-rois[:,2])*h_m
                    brY1 = (brY-rois[:,2])*h_m
                    blY1 = (blY-rois[:,2])*h_m

                for i in range(len(groups)):
                    if self.useCurvedBBs:
                        for bb_id in groups[i]:
                            rr, cc = draw.polygon(
                                    (bbs[bb_id].polyYs()-rois[i,2].item())*h_m[i].item(),
                                    (bbs[bb_id].polyXs()-rois[i,1].item())*w_m[i].item(), 
                                    [self.poolBB2_h,self.poolBB2_w])
                    else:
                        rr, cc = draw.polygon([round(tlY1[bb_id].item()),round(trY1[bb_id].item()),round(brY1[bb_id].item()),round(blY1[bb_id].item())],[round(tlX1[bb_id].item()),round(trX1[bb_id].item()),round(brX1[bb_id].item()),round(blX1[bb_id].item())], (self.poolBB2_h,self.poolBB2_w))
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
                if text_emb is not None:
                    node_features = torch.cat( (node_features,text_emb), dim=1 )
            else:
                assert(self.useShapeFeats)
                node_features = node_shapeFeats.to(features.device)

            assert(not torch.isnan(node_features).any())
            if self.bbFeaturizerFC is not None:
                node_features = self.bbFeaturizerFC(node_features) #if uncommented, change rot on node_shapeFeats, maybe not
            assert(not torch.isnan(node_features).any())
        elif text_emb is not None:
            node_features = text_emb
        else:
            node_features = None
        return node_features

    def selectFeatureNNEdges(self,bbs,imageHeight,imageWidth,image,device,merge_only=False):
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

            if merge_only:
                line_counts=0
            else:
                #t#tic2=timeit.default_timer()#t#
                line_counts = self.betweenPixels(bbs,image)
                #t#print('     candidates betweenPixels: {}'.format(timeit.default_timer()-tic2))#t#
            numClassFeat = bbs[0].getCls().shape[0]
            
            #conf, x,y,r,h,w,tl, tr, br, bl = torch.FloatTensor([bb.getFeatureInfo() for bb in bbs]).permute(1,0)
            #conf, x,y,r,h,w,tlx,tly,trx,try,brx,bry,blx,bly,r_left,r_rightA,classFeats = bb.getFeatureInfo()

            #t#tic2=timeit.default_timer()#t#
            allFeats = torch.FloatTensor([bb.getFeatureInfo() for bb in bbs])
            #t#print('     candidates gather allFeats: {}'.format(timeit.default_timer()-tic2))#t#
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
            classFeat = bbs[:,6:]
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

        num_feats = 30+numClassFeat*2
        if self.useCurvedBBs:
            num_feats+=9
            if not self.shape_feats_normal:
                num_feats-=1
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
        features[:,:,26] = sin_r1
        features[:,:,27] = sin_r2
        features[:,:,28] = cos_r1
        features[:,:,29] = cos_r2
        features[:,:,30:30+numClassFeat] = classFeat1
        features[:,:,30+numClassFeat:30+2*numClassFeat] = classFeat2
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
        features = features.view(len(bbs)**2,num_feats) #flatten
        #t#time = timeit.default_timer()-tic#t#
        #t#print('   candidates feats: {}'.format(time))#t#
        #t#self.opt_cand.append(time)
        #t#if len(self.opt_cand)>30:#t#
        #t#    print('   candidates feats running mean: {}'.format(np.mean(self.opt_cand)))#t#
        #t#    self.opt_cand = self.opt_cand[1:]#t#
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
        rel_pred2d = rel_pred.view(len(bbs),len(bbs)) #unflatten
        actual_rels = [(i,j) for i in range(len(bbs)) for j in range(i+1,len(bbs))]
        rels_ordered = [ ((rel_pred2d[rel[0],rel[1]].item()+rel_pred2d[rel[1],rel[0]].item())/2,rel) for rel in actual_rels ]
        rels = [(i,j) for i in range(len(bbs)) for j in range(len(bbs))]

        rels_ordered.sort(key=lambda x: x[0], reverse=True)

        keep = math.ceil(self.percent_rel_to_keep*len(rels_ordered))
        if merge_only:
            keep = min(keep,self.max_merge_rel_to_keep)
        else:
            keep = min(keep,self.max_rel_to_keep)
        #print('keeping {} of {}'.format(keep,len(rels_ordered)))
        keep_rels = [r[1] for r in rels_ordered[:keep]]
        if keep<len(rels_ordered):
            implicit_threshold = rels_ordered[keep][0]
        else:
            implicit_threshold = rels_ordered[-1][0]-0.1 #We're taking everything

        #t#print('   candidates net and thresh: {}'.format(timeit.default_timer()-tic))#t#
        return keep_rels, (rel_pred, rels, implicit_threshold)


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
        #t#print('    candidates, betweenPixels, getCenter:{}, draw.line:{}, sum pixels:{}'.format(np.mean(TIME_getCenter),np.mean(TIME_draw_line),np.mean(TIME_sum_pixels)))#t#
        return values


    def selectLineOfSightEdges(self,bbs,imageHeight,imageWidth, return_all=False):
        if bbs.size(0)<2:
            return []
        #return list of index pairs


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

    def getTranscriptions(self,bbs,image):
        if self.useCurvedBBs:
            assert(image.size(0)==1) #single imag
            self.text_rec.eval()
            #build batch
            max_w=0
            for b in range(batch_size):
                grid = bbs.getGrid()
                max_w = max(max_x,grid.size(1))
                girds.append(grids)

            #batch the grids together padding to same length
            to_pad = [g.size(1)-max_w for g in grids]
            grids = [F.pad(g,(0,p,0,0)) for g,p in zip(grids,to_pad)]

            output_strings=[]
            num_batch = math.ceil(len(grids)/self.atr_batch_size)
            for b in range(num_batch):
                start=b*self.atr_batch_size
                end=min((b+1)*self.atr_batch_size,len(grids))
                b_grids = torch.stack(grids[start:end],dim=0).to(image.device)
                batch_lines = F.grid_sample(image.expand(b_grids.size(0),-1,-1,-1),b_grids)

                with torch.no_grad():
                    resBatch = self.text_rec(batch_lines).cpu().detach().numpy().transpose(1,0,2)
                batch_strings, decoded_raw_hw = decode_handwriting(resBatch, self.idx_to_char)
                ##debug
                out_im = batch_lines.cpu().numpy().transpose([0,2,3,1])
                out_im = 256*(2-out_im)/2
                for i in range(batch_lines.size(0)):
                    img_f.imwrite('out2/line{}-{}.png'.format(i+index,batch_strings[i]),out_im[i])
                    print('DEBUG saved hwr image: out2/line{}-{}.png'.format(i+start,batch_strings[i]))
                ##
                output_strings += batch_strings

        else:
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
            all_scaled_w = (((x2-x1).float()+1)*scale).cpu()#.int()
            scale=None

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
                    resBatch = self.text_rec(lines).cpu().detach().numpy().transpose(1,0,2)
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

