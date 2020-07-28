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
#from utils.string_utils import correctTrans
import editdistance
import math, os
import random
import json
from collections import defaultdict

import timeit
import cv2

MAX_CANDIDATES=325 #450
MAX_GRAPH_SIZE=370
#max seen 428, so why'd it crash on 375?

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


        if 'use_rel_shape_feats' in config:
             config['use_shape_feats'] =  config['use_rel_shape_feats']
        self.useShapeFeats= config['use_shape_feats'] if 'use_shape_feats' in config else False
        self.usePositionFeature = config['use_position_feats'] if 'use_position_feats' in config else False
        assert(not self.usePositionFeature or self.useShapeFeats)
        self.normalizeHorz=config['normalize_horz'] if 'normalize_horz' in config else 400
        self.normalizeVert=config['normalize_vert'] if 'normalize_vert' in config else 50
        self.normalizeDist=(self.normalizeHorz+self.normalizeVert)/2

        assert(self.detector.scale[0]==self.detector.scale[1])
        if useBeginningOfLast:
            detect_save_scale = self.detector.scale[0]
        else:
            detect_save_scale = self.detector.save_scale
        if self.use2ndFeatures:
            detect_save2_scale = self.detector.save2_scale

        if self.useShapeFeats:
           self.numShapeFeats=8+2*self.numBBTypes #we'll append some extra feats
           self.numShapeFeatsBB=3+self.numBBTypes
           if self.useShapeFeats!='old':
               self.numShapeFeats+=4
           if self.detector.predNumNeighbors:
               self.numShapeFeats+=2
               self.numShapeFeatsBB+=1
           if self.usePositionFeature:
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

        #self.roi_align = RoIAlign(self.pool_h,self.pool_w,1.0/detect_save_scale) Facebook implementation
        self.roi_align = RoIAlign((self.pool_h,self.pool_w),1.0/detect_save_scale,-1)
        if self.use2ndFeatures:
            #self.roi_align2 = RoIAlign(self.pool2_h,self.pool2_w,1.0/detect_save2_scale)
            self.roi_align2 = RoIAlign((self.pool2_h,self.pool2_w),1.0/detect_save2_scale,-1)
        else:
            last_ch_relC=0

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
            self.mergeThresh=[]
            self.groupThresh=[]
            self.keepEdgeThresh=[]
            for graphconfig in config['graph_config']:
                self.graphnets.append( eval(graphconfig['arch'])(graphconfig) )
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
        self.fixBiDirection= config['fix_bi_dir'] if 'fix_bi_dir' in config else False
        if 'max_graph_size' in config:
            MAX_GRAPH_SIZE = config['max_graph_size']

        self.useOldDecay = config['use_old_len_decay'] if 'use_old_len_decay' in config else False

        self.relationshipProposal= config['relationship_proposal'] if 'relationship_proposal' in config else 'line_of_sight'
        self.include_bb_conf=False
        if self.relationshipProposal=='feature_nn':
            self.include_bb_conf=True
            #num_classes = config['num_class']
            num_bb_feat = self.numBBTypes + (1 if self.detector.predNumNeighbors else 0) #config['graph_config']['bb_out']
            self.rel_prop_nn = nn.Sequential(
                                nn.Linear(26+2*num_bb_feat,64),
                                nn.Dropout(0.25),
                                nn.ReLU(True),
                                nn.Linear(64,1)
                                )
            self.percent_rel_to_keep = config['percent_rel_to_keep'] if 'percent_rel_to_keep' in config else 0.2
            self.max_rel_to_keep = config['max_rel_to_keep'] if 'max_rel_to_keep' in config else 3000

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
        #t#self.opt_cand=[]
        #t#self.opt_createG=[]
        if type(self.pairer) is BinaryPairReal and type(self.pairer.shape_layers) is not nn.Sequential:
            print("Shape feats aligned to feat dataset.")


    def unfreeze(self): 
        if self.detector_frozen:
            for param in self.detector.parameters(): 
                param.requires_grad=param.will_use_grad 
            self.detector_frozen=False
            print('Unfroze detector')
        

    def forward(self, image, gtBBs=None, gtNNs=None, useGTBBs=False, otherThresh=None, otherThreshIntur=None, hard_detect_limit=300, debug=False,old_nn=False,gtTrans=None):
        #t#tic=timeit.default_timer()
        bbPredictions, offsetPredictions, _,_,_,_ = self.detector(image)
        _=None
        saved_features=self.detector.saved_features
        self.detector.saved_features=None
        if self.use2ndFeatures:
            saved_features2=self.detector.saved_features2
        else:
            saved_features2=None
        #t#print('   detector: {}'.format(timeit.default_timer()-tic))

        if saved_features is None:
            print('ERROR:no saved features!')
            import pdb;pdb.set_trace()

        
        #t#tic=timeit.default_timer()
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
            assert(False) #pretty sure this is untested...
            bbPredictions = non_max_sup_dist(bbPredictions.cpu(),self.used_threshConf,2.5,hard_detect_limit)
        else:
            bbPredictions = non_max_sup_iou(bbPredictions.cpu(),self.used_threshConf,0.4,hard_detect_limit)
        #I'm assuming batch size of one
        assert(len(bbPredictions)==1)
        bbPredictions=bbPredictions[0]
        if self.no_grad_feats:
            bbPredictions=bbPredictions.detach()
        #t#print('   process boxes: {}'.format(timeit.default_timer()-tic))
        #bbPredictions should be switched for GT for training? Then we can easily use BCE loss. 
        #Otherwise we have to to alignment first
        if not useGTBBs:
            if bbPredictions.size(0)==0:
                return [bbPredictions], offsetPredictions, None, None, None, None, None, (None,None,None,None)
            if self.include_bb_conf:
                useBBs = bbPredictions
            else:
                useBBs = bbPredictions[:,1:] #remove confidence score
        elif useGTBBs=='saved':
            if self.include_bb_conf:
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
            useBBs = gtBBs[0,:,0:5]
            if self.useShapeFeats or self.relationshipProposal=='feature_nn':
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
                #fake some confifence values
                conf = torch.rand(useBBs.size(0),1)*0.33 +0.66
                useBBs = torch.cat((conf.to(useBBs.device),useBBs),dim=1)
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
        if useBBs.size(0)>1:
            if self.text_rec is not None:
                embeddings = self.embedding_model(transcriptions)
                if self.add_noise_to_word_embeddings:
                    embeddings += torch.randn_like(embeddings).to(embeddings.device)*self.add_noise_to_word_embeddings*embeddings.mean()
            else:
                embeddings=None
            if self.useMetaGraph:
                allOutputBoxes=[]
                allRelIndexes=[]
                allNodeOuts=[]
                allEdgeOuts=[]
                allGroups=[]
                allEdgeIndexes=[]
                graph,edgeIndexes,rel_prop_scores = self.createGraph(useBBs,saved_features,saved_features2,image.size(-2),image.size(-1),text_emb=embeddings)
                groups=[[i] for i in range(useBBs.size(0))]
                bbTrans = transcriptions
                #undirected
                #edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
                if graph is None:
                    return [bbPredictions], offsetPredictions, None, None, None, None, rel_prop_scores, (useBBs.cpu().detach(),None,None,transcriptions)

                nodeOuts, edgeOuts, nodeFeats, edgeFeats, uniFeats = self.graphnets[0](graph)
                assert(edgeOuts is None or not torch.isnan(edgeOuts).any())
                edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
                #edgeOuts = (edgeOuts[:edgeOuts.size(0)//2] + edgeOuts[edgeOuts.size(0)//2:])/2 #average two directions of edge
                #edgeFeats = (edgeFeats[:edgeFeats.size(0)//2] + edgeFeats[edgeFeats.size(0)//2:])/2 #average two directions of edge
                #update BBs with node predictions
                useBBs = self.updateBBs(useBBs,groups,nodeOuts)
                allOutputBoxes.append(useBBs.cpu()) 
                allNodeOuts.append(nodeOuts)
                allEdgeOuts.append(edgeOuts)
                allGroups.append(groups)
                allEdgeIndexes.append(edgeIndexes)

                #print('graph 0:   bbs:{}, nodes:{}, edges:{}'.format(useBBs.size(0),nodeOuts.size(0),edgeOuts.size(0)))
                
                #t#tic=timeit.default_timer()
                for gIter,graphnet in enumerate(self.graphnets[1:]):

                    #print('!D! {} before edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(gIter,edgeFeats.size(),useBBs.size(),nodeFeats.size(),len(edgeIndexes)))
                    ##t#print('      graph num edges: {}'.format(graph[1].size()))
                    useBBs,graph,groups,edgeIndexes,bbTrans=self.mergeAndGroup(
                            self.mergeThresh[gIter],self.keepEdgeThresh[gIter],self.groupThresh[gIter],
                            edgeIndexes,edgeOuts,groups,nodeFeats,edgeFeats,uniFeats,useBBs,bbTrans,image)
                    #print('graph 1-:   bbs:{}, nodes:{}, edges:{}'.format(useBBs.size(0),len(groups),len(edgeIndexes)))
                    if len(edgeIndexes)==0:
                        break #we have no graph, so we can just end here
                    #print('!D! after  edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(graph[2].size(),useBBs.size(),graph[0].size(),len(edgeIndexes)))
                    ##t#print('      graph num edges: {}'.format(graph[1].size()))
                    nodeOuts, edgeOuts, nodeFeats, edgeFeats, uniFeats = graphnet(graph)
                    #edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
                    useBBs = self.updateBBs(useBBs,groups,nodeOuts)
                    allOutputBoxes.append(useBBs.cpu()) 
                    allNodeOuts.append(nodeOuts)
                    allEdgeOuts.append(edgeOuts)
                    allGroups.append(groups)
                    allEdgeIndexes.append(edgeIndexes)

                ##Final state of the graph
                #print('!D! F before edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(edgeFeats.size(),useBBs.size(),nodeFeats.size(),len(edgeIndexes)))
                useBBs,graph,groups,edgeIndexes,bbTrans=self.mergeAndGroup(
                        self.mergeThresh[-1],self.keepEdgeThresh[-1],self.groupThresh[-1],
                        edgeIndexes,edgeOuts.detach(),groups,nodeFeats.detach(),edgeFeats.detach(),uniFeats.detach() if uniFeats is not None else None,useBBs.detach(),bbTrans,image)
                #print('!D! after  edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(graph[2].size(),useBBs.size(),graph[0].size(),len(edgeIndexes)))
                final=(useBBs.cpu().detach(),groups,edgeIndexes,bbTrans)

                #t#print('   thru all iters: {}'.format(timeit.default_timer()-tic))


            else:
                raise NotImplementedError('Simple pairing not implemented for new grouping stuff')
            #adjacencyMatrix = torch.zeros((bbPredictions.size(1),bbPredictions.size(1)))
            #for rel in relOuts:
            #    i,j,a=graphToDetectionsMap(

            return allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores, final
        else:
            return [bbPredictions], offsetPredictions, None, None, None, None, None, (useBBs.cpu().detach(),None,None,transcriptions)

    #This rewrites the confidence and class predictions based on the (re)predictions from the graph network
    def updateBBs(self,bbs,groups,nodeOuts):
        if bbs.size(0)>1:
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
    def mergeAndGroup(self,mergeThresh,keepEdgeThresh,groupThresh,oldEdgeIndexes,edgePreds,oldGroups,oldNodeFeats,oldEdgeFeats,oldUniversalFeats,oldBBs,bbTrans,image):
        newBBs={}
        #newBBs_line={}
        newBBIdCounter=0
        oldToNewBBIndexes={}
        relPreds = torch.sigmoid(edgePreds[:,-1,0]).cpu().detach()
        mergePreds = torch.sigmoid(edgePreds[:,-1,1]).cpu().detach()
        groupPreds = torch.sigmoid(edgePreds[:,-1,2]).cpu().detach()
        ##Prevent all nodes from merging during first iterations (bad init):
        if not(mergePreds.mean()>mergeThresh*0.99 and edgePreds.size(0)>5):
        
            for i,(n0,n1) in enumerate(oldEdgeIndexes):
                #mergePred = edgePreds[i,-1,1]
                
                if mergePreds[i]>mergeThresh: #TODO condition this on whether it is correct. and GT?:
                    if len(oldGroups[n0])==1 and len(oldGroups[n1])==1:
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
                            #newBBs_line[newBBIdCounter]=line
                            newBBIdCounter+=1
                        elif mergeNewId0 is None:
                            oldToNewBBIndexes[bbId0]=mergeNewId1
                            newBBs[mergeNewId1]=newBB
                            #newBBs_line[mergeNewId1]=line
                        elif mergeNewId1 is None:
                            oldToNewBBIndexes[bbId1]=mergeNewId0
                            newBBs[mergeNewId0]=newBB
                            #newBBs_line[mergeNewId0]=line
                        elif mergeNewId0!=mergeNewId1:
                            #merge two merged bbs
                            oldToNewBBIndexes[bbId1]=mergeNewId0
                            for old,new in oldToNewBBIndexes.items():
                                if new == mergeNewId1:
                                    oldToNewBBIndexes[old]=mergeNewId0
                            newBBs[mergeNewId0]=newBB
                            #newBBs_line[mergeNewId0]=line
                            #print('merge {} and {} (d), because of {} and {}'.format(mergeNewId0,mergeNewId1,bbId0,bbId1))
                            del newBBs[mergeNewId1]
                            #del newBBs_line[mergeNewId1]




            #Actually rewrite bbs
            if len(newBBs)==0:
                bbs = oldBBs
                oldBBIdToNew=list(range(oldBBs.size(0)))
            else:
                device = oldBBs.device
                bbs=[]
                oldBBIdToNew={}
                if self.text_rec is not None:
                    bbTransTmp=[]
                for i in range(oldBBs.size(0)):
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
                    

            if self.text_rec is not None:
                newTrans = self.getTranscriptions(bbs[-len(newBBs):],image)
                #newEmbeddings = self.embedding_model(newTrans)
                #now we need to embed and append these and the old trans to node features
                bbTrans += newTrans


            #rewrite groups with merged instances
            assignedGroup={} #this will allow us to remove merged instances
            oldGroupToNew={}
            workGroups = {i:v for i,v in enumerate(oldGroups)}
            for id,bbIds in enumerate(oldGroups):
                newGroup = [oldBBIdToNew[oldId] for oldId in bbIds]
                if len(newGroup)==1 and newGroup[0] in assignedGroup:
                    oldGroupToNew[id]=assignedGroup[newGroup[0]]
                    del workGroups[id]
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
                else:
                    newNodeFeats.append(oldNodeFeats[id])
            oldNodeFeats = torch.stack(newNodeFeats,dim=0)

            #We'll adjust the edges to acount for merges as well as prune edges and get ready for grouping
            #temp = oldEdgeIndexes
            #oldEdgeIndexes = []

            #Prune and adjust the edges (to groups)
            groupEdges=[]
            edgeFeats = []

            D_numOldEdges=len(oldEdgeIndexes)
            D_numOldAboveThresh=(relPreds>keepEdgeThresh).sum()
            for i,(n0,n1) in enumerate(oldEdgeIndexes):
                if relPreds[i]>keepEdgeThresh:
                    if n0 in oldGroupToNew:
                        n0=newIdToPos[oldGroupToNew[n0]]
                    else:
                        n0 = newIdToPos[n0]
                    if n1 in oldGroupToNew:
                        n1=newIdToPos[oldGroupToNew[n1]]
                    else:
                        n1 = newIdToPos[n1]
                    assert(n0<bbs.size(0) and n1<bbs.size(0))
                    if n0!=n1:
                        #oldEdgeIndexes.append((n0,n1))
                        groupEdges.append((groupPreds[i].item(),n0,n1))
                        edgeFeats.append([oldEdgeFeats[i]])
                    #else:
                    #    It disapears
            oldEdgeIndexes=None

            #print('!D! original edges:{}, above thresh:{}, kept edges:{}'.format(D_numOldEdges,D_numOldAboveThresh,len(edgeFeats)))
             
        else:
            bbs=oldBBs
            groupEdges=[]
            edgeFeats = []
            for i,(n0,n1) in enumerate(oldEdgeIndexes):
                if relPreds[i]>keepEdgeThresh:
                    groupEdges.append((groupPreds[i].item(),n0,n1))
                    edgeFeats.append([oldEdgeFeats[i]])
            oldEdgeIndexes=None



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

            workingGroups[g0] += workingGroups[g1]
            del workingGroups[g1]


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
        for oldG,bbIds in workingGroups.items():
            oldToIdx[oldG]=len(newGroups)
            newGroups.append(bbIds)
            newNodeFeats.append( self.groupNodeFunc(newNodeFeatsD[oldG]) )
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

        return bbs, newGraph, newGroups, edges, bbTrans


                



    def createGraph(self,bbs,features,features2,imageHeight,imageWidth,text_emb=None,flip=None,debug_image=None):
        #t#tic=timeit.default_timer()
        if self.relationshipProposal == 'line_of_sight':
            candidates = self.selectLineOfSightEdges(bbs.detach(),imageHeight,imageWidth)
            rel_prop_scores = None
        elif self.relationshipProposal == 'feature_nn':
            candidates, rel_prop_scores = self.selectFeatureNNEdges(bbs.detach(),imageHeight,imageWidth,features.device)
            bbs=bbs[:,1:] #discard confidence, we kept it so the proposer could see them
        #t#print('   candidate: {}'.format(timeit.default_timer()-tic))
        if len(candidates)==0:
            if self.useMetaGraph:
                return None, None, None
            else:
                return None,None,None,None,None, None
        #t#tic=timeit.default_timer()

        #stackedEdgeFeatWindows = torch.FloatTensor((len(candidates),features.size(1)+2,self.relWindowSize,self.relWindowSize)).to(features.device())

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
            rois = torch.zeros((len(candidates),5)) #(batchIndex,x1,y1,x2,y2) as expected by ROI Align
            bbs_index1 = bbs[[c[0] for c in candidates]]
            bbs_index2 = bbs[[c[1] for c in candidates]]
            tlX_index1 = tlX[[c[0] for c in candidates]]
            tlX_index2 = tlX[[c[1] for c in candidates]]
            trX_index1 = trX[[c[0] for c in candidates]]
            trX_index2 = trX[[c[1] for c in candidates]]
            blX_index1 = blX[[c[0] for c in candidates]]
            blX_index2 = blX[[c[1] for c in candidates]]
            brX_index1 = brX[[c[0] for c in candidates]]
            brX_index2 = brX[[c[1] for c in candidates]]
            tlY_index1 = tlY[[c[0] for c in candidates]]
            tlY_index2 = tlY[[c[1] for c in candidates]]
            trY_index1 = trY[[c[0] for c in candidates]]
            trY_index2 = trY[[c[1] for c in candidates]]
            blY_index1 = blY[[c[0] for c in candidates]]
            blY_index2 = blY[[c[1] for c in candidates]]
            brY_index1 = brY[[c[0] for c in candidates]]
            brY_index2 = brY[[c[1] for c in candidates]]
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
            max_X = torch.min(max_X+self.expandedRelContext,torch.FloatTensor([imageWidth-1]))
            min_X = torch.max(min_X-self.expandedRelContext,torch.FloatTensor([0]))
            max_Y = torch.min(max_Y+self.expandedRelContext,torch.FloatTensor([imageHeight-1]))
            min_Y = torch.max(min_Y-self.expandedRelContext,torch.FloatTensor([0]))
            rois[:,1]=min_X
            rois[:,2]=min_Y
            rois[:,3]=max_X
            rois[:,4]=max_Y



            ###DEBUG
            if debug_image is not None and i<5:
                assert(self.rotation==False)
                #print('crop {}: ({},{}), ({},{})'.format(i,minX.item(),maxX.item(),minY.item(),maxY.item()))
                #print(bbs[index1])
                #print(bbs[index2])
                crop = debug_image[0,:,int(minY):int(maxY),int(minX):int(maxX)+1].cpu()
                crop = (2-crop)/2
                if crop.size(0)==1:
                    crop = crop.expand(3,crop.size(1),crop.size(2))
                crop[0,int(tlY[index1].item()-minY):int(brY[index1].item()-minY)+1,int(tlX[index1].item()-minX):int(brX[index1].item()-minX)+1]*=0.5
                crop[1,int(tlY[index2].item()-minY):int(brY[index2].item()-minY)+1,int(tlX[index2].item()-minX):int(brX[index2].item()-minX)+1]*=0.5
                crop = crop.numpy().transpose([1,2,0])
                #cv2.imshow('crop {}'.format(i),crop)
                debug_images.append(crop)
                #import pdb;pdb.set_trace()
            ###
        #if debug_image is not None:
        #    cv2.waitKey()


        #crop from feats, ROI pool
        stackedEdgeFeatWindows = self.roi_align(features,rois.to(features.device))
        if features2 is not None:
            stackedEdgeFeatWindows2 = self.roi_align2(features2,rois.to(features.device))
            if not self.splitFeatures:
                stackedEdgeFeatWindows = torch.cat( (stackedEdgeFeatWindows,stackedEdgeFeatWindows2), dim=1)
                stackedEdgeFeatWindows2=None

        #create and add masks
        if self.expandedRelContext is not None:
            #We're going to add a third mask for all bbs, which we'll precompute here
            numMasks=3
            allMasks = torch.zeros(imageHeight,imageWidth)
            if self.use_fixed_masks:
                for bbIdx in range(bbs.size(0)):
                    rr, cc = draw.polygon([tlY[bbIdx],trY[bbIdx],brY[bbIdx],blY[bbIdx]],[tlX[bbIdx],trX[bbIdx],brX[bbIdx],blX[bbIdx]], [imageHeight,imageWidth])
                    allMasks[rr,cc]=1
        else:
            numMasks=2
        masks = torch.zeros(stackedEdgeFeatWindows.size(0),numMasks,self.pool2_h,self.pool2_w)
        if self.useShapeFeats:
            shapeFeats = torch.FloatTensor(len(candidates),self.numShapeFeats)
        if self.detector.predNumNeighbors:
            extraPred=1
        else:
            extraPred=0


        #make instance specific masks and make shape (spatial) features
        if self.useShapeFeats!='only':
            if (random.random()<0.5 and flip is None and  not self.debug) or flip:
                pass
                #TODO
            feature_w = rois[:,3]-rois[:,1] +1
            feature_h = rois[:,4]-rois[:,2] +1
            w_m = self.pool2_w/feature_w
            h_m = self.pool2_h/feature_h

            tlX1 = (tlX_index1-rois[:,1])*w_m
            trX1 = (trX_index1-rois[:,1])*w_m
            brX1 = (brX_index1-rois[:,1])*w_m
            blX1 = (blX_index1-rois[:,1])*w_m
            tlY1 = (tlY_index1-rois[:,2])*h_m
            trY1 = (trY_index1-rois[:,2])*h_m
            brY1 = (brY_index1-rois[:,2])*h_m
            blY1 = (blY_index1-rois[:,2])*h_m
            tlX2 = (tlX_index2-rois[:,1])*w_m
            trX2 = (trX_index2-rois[:,1])*w_m
            brX2 = (brX_index2-rois[:,1])*w_m
            blX2 = (blX_index2-rois[:,1])*w_m
            tlY2 = (tlY_index2-rois[:,2])*h_m
            trY2 = (trY_index2-rois[:,2])*h_m
            brY2 = (brY_index2-rois[:,2])*h_m
            blY2 = (blY_index2-rois[:,2])*h_m

        for i,(index1, index2) in enumerate(candidates):
            if self.useShapeFeats!='only':
                rr, cc = draw.polygon(
                            [round(tlY1[i].item()),round(trY1[i].item()),
                            round(brY1[i].item()),round(blY1[i].item())],
                            [round(tlX1[i].item()),round(trX1[i].item()),
                            round(brX1[i].item()),round(blX1[i].item())], 
                            [self.pool2_h,self.pool2_w])
                masks[i,0,rr,cc]=1

                rr, cc = draw.polygon(
                            [round(tlY2[i].item()),round(trY2[i].item()),
                            round(brY2[i].item()),round(blY2[i].item())],
                            [round(tlX2[i].item()),round(trX2[i].item()),
                            round(brX2[i].item()),round(blX2[i].item())], 
                            [self.pool2_h,self.pool2_w])
                masks[i,1,rr,cc]=1

                if self.expandedRelContext is not None:
                    cropArea = allMasks[round(rois[i,2].item()):round(rois[i,4].item())+1,round(rois[i,1].item()):round(rois[i,3].item())+1]
                    if len(cropArea.shape)==0:
                        raise ValueError("RoI is bad: {}:{},{}:{} for size {}".format(round(rois[i,2].item()),round(rois[i,4].item())+1,round(rois[i,1].item()),round(rois[i,3].item())+1,allMasks.shape))
                    masks[i,2] = F.interpolate(cropArea[None,None,...], size=(self.pool2_h,self.pool2_w), mode='bilinear',align_corners=False)[0,0]
                    #masks[i,2] = cv2.resize(cropArea,(stackedEdgeFeatWindows.size(2),stackedEdgeFeatWindows.size(3)))
                    if debug_image is not None:
                        debug_masks.append(cropArea)

        if self.useShapeFeats:
            if type(self.pairer) is BinaryPairReal and type(self.pairer.shape_layers) is not nn.Sequential:
                #The index specification is to allign with the format feat nets are trained with
                ixs=[0,1,2,3,3+self.numBBTypes,3+self.numBBTypes,4+self.numBBTypes,5+self.numBBTypes,6+self.numBBTypes,6+2*self.numBBTypes,6+2*self.numBBTypes,7+2*self.numBBTypes]
            else:
                ixs=[4,6,2,8,8+self.numBBTypes,5,7,3,8+self.numBBTypes,8+self.numBBTypes+self.numBBTypes,0,1]
            
            shapeFeats[:,ixs[0]] = 2*bbs_index1[:,3]/self.normalizeVert #bb preds half height/width
            shapeFeats[:,ixs[1]] = 2*bbs_index1[:,4]/self.normalizeHorz
            shapeFeats[:,ixs[2]] = bbs_index1[:,2]/math.pi
            shapeFeats[:,ixs[3]:ixs[4]] = bbs_index1[:,extraPred+5:]# torch.sigmoid(bbs_index1[:,extraPred+5:])

            shapeFeats[:,ixs[5]] = 2*bbs_index2[:,3]/self.normalizeVert
            shapeFeats[:,ixs[6]] = 2*bbs_index2[:,4]/self.normalizeHorz
            shapeFeats[:,ixs[7]] = bbs_index2[:,2]/math.pi
            shapeFeats[:,ixs[8]:ixs[9]] = bbs_index2[:,extraPred+5:]#torch.sigmoid(bbs_index2[:,extraPred+5:])

            shapeFeats[:,ixs[10]] = (bbs_index1[:,0]-bbs_index2[:,0])/self.normalizeHorz
            shapeFeats[:,ixs[11]] = (bbs_index1[:,1]-bbs_index2[:,1])/self.normalizeVert
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
                shapeFeats[:,startNN +0] = bbs_index1[:,5]
                shapeFeats[:,startNN +1] = bbs_index2[:,5]
                startPos=startNN+2
            else:
                startPos=startNN
            if self.usePositionFeature:
                if self.usePositionFeature=='absolute':
                    shapeFeats[:,startPos +0] = (bbs_index1[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                    shapeFeats[:,startPos +1] = (bbs_index1[:,1]-imageHeight/2)/(10*self.normalizeVert)
                    shapeFeats[:,startPos +2] = (bbs_index2[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                    shapeFeats[:,startPos +3] = (bbs_index2[:,1]-imageHeight/2)/(10*self.normalizeVert)
                else:
                    shapeFeats[:,startPos +0] = (bbs_index1[:,0]-imageWidth/2)/(imageWidth/2)
                    shapeFeats[:,startPos +1] = (bbs_index1[:,1]-imageHeight/2)/(imageHeight/2)
                    shapeFeats[:,startPos +2] = (bbs_index2[:,0]-imageWidth/2)/(imageWidth/2)
                    shapeFeats[:,startPos +3] = (bbs_index2[:,1]-imageHeight/2)/(imageHeight/2)

        ###DEBUG
        if debug_image is not None:
            for i in range(4):
                cv2.imshow('crop rel {}'.format(i),debug_images[i])
                cv2.imshow('masks rel {}'.format(i),masks[i].numpy().transpose([1,2,0]))
                cv2.imshow('mask all rel {}'.format(i),debug_masks[i].numpy())
            cv2.waitKey()
            debug_images=[]


        if self.useShapeFeats!='only':
            if self.splitFeatures:
                stackedEdgeFeatWindows2 = torch.cat((stackedEdgeFeatWindows2,masks.to(stackedEdgeFeatWindows2.device)),dim=1)
                relFeats = self.relFeaturizerConv2(stackedEdgeFeatWindows2)
                stackedEdgeFeatWindows = torch.cat((stackedEdgeFeatWindows,relFeats),dim=1)
            else:
                stackedEdgeFeatWindows = torch.cat((stackedEdgeFeatWindows,masks.to(stackedEdgeFeatWindows.device)),dim=1)
                #import pdb; pdb.set_trace()
            relFeats = self.relFeaturizerConv(stackedEdgeFeatWindows) #preparing for graph feature size
            relFeats = relFeats.view(relFeats.size(0),relFeats.size(1))
        if self.useShapeFeats:
            if self.useShapeFeats=='only':
                relFeats = shapeFeats.to(features.device)
            else:
                relFeats = torch.cat((relFeats,shapeFeats.to(relFeats.device)),dim=1)
        if self.blankRelFeats:
            relFeats = relFeats.zero_()
        if self.relFeaturizerFC is not None:
            relFeats = self.relFeaturizerFC(relFeats)
        #if self.useShapeFeats=='sp
    
        #compute features for the bounding boxes by themselves
        #This will be replaced with some type of word embedding
        if self.useBBVisualFeats:
            assert(features.size(0)==1)
            if self.useShapeFeats:
                bb_shapeFeats=torch.FloatTensor(bbs.size(0),self.numShapeFeatsBB)
            if self.useShapeFeats != "only" and self.expandedBBContext:
                masks = torch.zeros(bbs.size(0),2,self.poolBB2_h,self.poolBB2_w)

            rois = torch.zeros((bbs.size(0),5))
            min_Y,_ = torch.min(torch.stack([tlY,trY,blY,brY],dim=0),dim=0)
            max_Y,_ = torch.max(torch.stack([tlY,trY,blY,brY],dim=0),dim=0) 
            min_X,_ = torch.min(torch.stack([tlX,trX,blX,brX],dim=0),dim=0) 
            max_X,_ = torch.max(torch.stack([tlX,trX,blX,brX],dim=0),dim=0) 
            if self.expandedBBContext is not None:
                max_X = torch.min(max_X+self.expandedBBContext,torch.FloatTensor([imageWidth-1]))
                min_X = torch.max(min_X-self.expandedBBContext,torch.FloatTensor([0]))
                max_Y = torch.min(max_Y+self.expandedBBContext,torch.FloatTensor([imageHeight-1]))
                min_Y = torch.max(min_Y-self.expandedBBContext,torch.FloatTensor([0]))
            rois[:,1]=min_X
            rois[:,2]=min_Y
            rois[:,3]=max_X
            rois[:,4]=max_Y

            if self.useShapeFeats:
                bb_shapeFeats[:,0]= (bbs[:,2]+math.pi)/(2*math.pi)
                bb_shapeFeats[:,1]=bbs[:,3]/self.normalizeVert
                bb_shapeFeats[:,2]=bbs[:,4]/self.normalizeHorz
                if self.detector.predNumNeighbors:
                    bb_shapeFeats[:,3]=bbs[:,5]
                bb_shapeFeats[:,3+extraPred:self.numBBTypes+3+extraPred]=torch.sigmoid(bbs[:,5+extraPred:self.numBBTypes+5+extraPred])
                if self.usePositionFeature:
                    if self.usePositionFeature=='absolute':
                        bb_shapeFeats[:,self.numBBTypes+3+extraPred] = (bbs[:,0]-imageWidth/2)/(5*self.normalizeHorz)
                        bb_shapeFeats[:,self.numBBTypes+4+extraPred] = (bbs[:,1]-imageHeight/2)/(10*self.normalizeVert)
                    else:
                        bb_shapeFeats[:,self.numBBTypes+3+extraPred] = (bbs[:,0]-imageWidth/2)/(imageWidth/2)
                        bb_shapeFeats[:,self.numBBTypes+4+extraPred] = (bbs[:,1]-imageHeight/2)/(imageHeight/2)
            if self.useShapeFeats != "only" and self.expandedBBContext:
                #Add detected BB masks
                #warp to roi space
                feature_w = rois[:,3]-rois[:,1] +1
                feature_h = rois[:,4]-rois[:,2] +1
                w_m = self.poolBB2_w/feature_w
                h_m = self.poolBB2_h/feature_h

                tlX1 = (tlX-rois[:,1])*w_m
                trX1 = (trX-rois[:,1])*w_m
                brX1 = (brX-rois[:,1])*w_m
                blX1 = (blX-rois[:,1])*w_m
                tlY1 = (tlY-rois[:,2])*h_m
                trY1 = (trY-rois[:,2])*h_m
                brY1 = (brY-rois[:,2])*h_m
                blY1 = (blY-rois[:,2])*h_m

                for i in range(bbs.size(0)):
                    rr, cc = draw.polygon([round(tlY1[i].item()),round(trY1[i].item()),round(brY1[i].item()),round(blY1[i].item())],[round(tlX1[i].item()),round(trX1[i].item()),round(brX1[i].item()),round(blX1[i].item())], (self.poolBB2_h,self.poolBB2_w))
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
                    cv2.imshow('crop bb {}'.format(i),crop)
                    cv2.imshow('masks bb {}'.format(i),torch.cat((masks[i],torch.zeros(1,self.poolBB2_h,self.poolBB2_w)),dim=0).numpy().transpose([1,2,0]))
                    #debug_images.append(crop)

            if debug_image is not None:
                cv2.waitKey()
            if self.useShapeFeats != "only":
                #bb_features[i]= F.avg_pool2d(features[0,:,minY:maxY+1,minX:maxX+1], (1+maxY-minY,1+maxX-minX)).view(-1)
                bb_features = self.roi_alignBB(features,rois.to(features.device))
                assert(not torch.isnan(bb_features).any())
                if features2 is not None:
                    bb_features2 = self.roi_alignBB2(features2,rois.to(features.device))
                    if not self.splitFeatures:
                        bb_features = torch.cat( (bb_features,bb_features2), dim=1)
                if self.expandedBBContext:
                    if self.splitFeatures:
                        bb_features2 = torch.cat( (bb_features2,masks.to(bb_features2.device)) ,dim=1)
                        bb_features2 = self.bbFeaturizerConv2(bb_features2)
                        bb_features = torch.cat( (bb_features,bb_features2), dim=1)
                    else:
                        bb_features = torch.cat( (bb_features,masks.to(bb_features.device)) ,dim=1)
                bb_features = self.bbFeaturizerConv(bb_features)
                bb_features = bb_features.view(bb_features.size(0),bb_features.size(1))
                if self.useShapeFeats:
                    bb_features = torch.cat( (bb_features,bb_shapeFeats.to(bb_features.device)), dim=1 )
                if text_emb is not None:
                    bb_features = torch.cat( (bb_features,text_emb), dim=1 )
            else:
                assert(self.useShapeFeats)
                bb_features = bb_shapeFeats.to(features.device)

            assert(not torch.isnan(bb_features).any())
            if self.bbFeaturizerFC is not None:
                bb_features = self.bbFeaturizerFC(bb_features) #if uncommented, change rot on bb_shapeFeats, maybe not
            assert(not torch.isnan(bb_features).any())
        elif text_emb is not None:
            bb_features = text_emb
        else:
            bb_features = None
        
        #We're not adding diagonal (self-rels) here!
        #Expecting special handeling during graph conv
        #candidateLocs = torch.LongTensor(candidates).t().to(relFeats.device)
        #ones = torch.ones(len(candidates)).to(relFeats.device)
        #adjacencyMatrix = torch.sparse.FloatTensor(candidateLocs,ones,torch.Size([bbs.size(0),bbs.size(0)]))

        #assert(relFeats.requries_grad)
        #rel_features = torch.sparse.FloatTensor(candidateLocs,relFeats,torch.Size([bbs.size(0),bbs.size(0),relFeats.size(1)]))
        #assert(rel_features.requries_grad)
        relIndexes=candidates
        numBB = bbs.size(0)
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

            #t#time = timeit.default_timer()-tic
            #t#print('   create graph: {}'.format(time)) #old 0.37, new 0.16
            #t#self.opt_createG.append(time)
            #t#if len(self.opt_createG)>17:
            #t#    print('   create graph running mean: {}'.format(np.mean(self.opt_createG)))
            #t#    if len(self.opt_createG)>30:
            #t#        self.opt_createG = self.opt_createG[1:]
            return (nodeFeatures, edgeIndexes, edgeFeatures, universalFeatures), relIndexes, rel_prop_scores
        else:
            if bb_features is None:
                numBB=0
                bbAndRel_features=relFeats
                adjacencyMatrix = None
                numOfNeighbors = None
            else:
                bbAndRel_features = torch.cat((bb_features,relFeats),dim=0)
                numOfNeighbors = torch.ones(bbs.size(0)+len(candidates)) #starts at one for yourself
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

            #t#time = timeit.default_timer()-tic
            #t#print('   create graph: {}'.format(time)) #old 0.37
            #t#self.opt_createG.append(time)
            #t#if len(self.opt_createG)>17:
            #t#    print('   create graph running mean: {}'.format(np.mean(self.opt_createG)))
            #t#    if len(self.opt_createG)>30:
            #t#        self.opt_createG = self.opt_createG[1:]
            #return bb_features, adjacencyMatrix, rel_features
            return bbAndRel_features, (adjacencyMatrix,numOfNeighbors), numBB, numRel, relIndexes, rel_prop_scores




    def selectFeatureNNEdges(self,bbs,imageHeight,imageWidth,device):
        if bbs.size(0)<2:
            return []

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
        #26-n: classpred1
        #n+1-m: classpred2

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

        #t#tic=timeit.default_timer()
        line_of_sight = self.selectLineOfSightEdges(bbs,imageHeight,imageWidth,return_all=True)
        #t#print('   candidates line-of-sight: {}'.format(timeit.default_timer()-tic))
        #t#tic=timeit.default_timer()
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
        features = torch.FloatTensor(bbs.size(0),bbs.size(0), 26+numClassFeat*2)
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
        features[:,:,23].zero_()
        for index1,index2 in line_of_sight:
            features[index1,index2,23]=1
            features[index2,index1,23]=1
        features[:,:,24] = conf1
        features[:,:,25] = conf2
        features[:,:,26:26+numClassFeat] = classFeat1
        features[:,:,26+numClassFeat:] = classFeat2

        features[:,:,0:7]/=self.normalizeHorz
        features[:,:,7:14]/=self.normalizeVert
        features[:,:,14:19]/=(self.normalizeVert+self.normalizeHorz)/2
        features = features.view(bbs.size(0)**2,26+numClassFeat*2) #flatten
        #t#time = timeit.default_timer()-tic
        #t#print('   candidates feats: {}'.format(time))
        #t#self.opt_cand.append(time)
        #t#if len(self.opt_cand)>30:
        #t#    print('   candidates feats running mean: {}'.format(np.mean(self.opt_cand)))
        #t#    self.opt_cand = self.opt_cand[1:]
        #t#tic=timeit.default_timer()
        rel_pred = self.rel_prop_nn(features.to(device))
        rel_pred2d = rel_pred.view(bbs.size(0),bbs.size(0)) #unflatten
        actual_rels = [(i,j) for i in range(bbs.size(0)) for j in range(i+1,bbs.size(0))]
        rels_ordered = [ ((rel_pred2d[rel[0],rel[1]].item()+rel_pred2d[rel[1],rel[0]].item())/2,rel) for rel in actual_rels ]
        rels = [(i,j) for i in range(bbs.size(0)) for j in range(bbs.size(0))]

        rels_ordered.sort(key=lambda x: x[0], reverse=True)

        keep = math.ceil(self.percent_rel_to_keep*len(rels_ordered))
        keep = min(keep,self.max_rel_to_keep)
        #print('keeping {} of {}'.format(keep,len(rels_ordered)))
        keep_rels = [r[1] for r in rels_ordered[:keep]]
        if keep<len(rels_ordered):
            implicit_threshold = rels_ordered[keep][0]
        else:
            implicit_threshold = rels_ordered[-1][0]-0.1 #We're taking everything

        #t#print('   candidates net and thresh: {}'.format(timeit.default_timer()-tic))
        return keep_rels, (rel_pred, rels, implicit_threshold)



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
                
                #cv2.line( boxesDrawn, (int(tlX[i]),int(tlY[i])),(int(trX[i]),int(trY[i])),i,1)
                #cv2.line( boxesDrawn, (int(trX[i]),int(trY[i])),(int(brX[i]),int(brY[i])),i,1)
                #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(brX[i]),int(brY[i])),i,1)
                #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(tlX[i]),int(tlY[i])),i,1)

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
                    cv2.line( drawn, (int(x[a]),int(y[a])),(int(x[b]),int(y[b])),(random.random()*0.5,random.random()*0.5,random.random()*0.5),1)
                for i in range(numBoxes):
                    
                    #cv2.line( boxesDrawn, (int(tlX[i]),int(tlY[i])),(int(trX[i]),int(trY[i])),i,1)
                    #cv2.line( boxesDrawn, (int(trX[i]),int(trY[i])),(int(brX[i]),int(brY[i])),i,1)
                    #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(brX[i]),int(brY[i])),i,1)
                    #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(tlX[i]),int(tlY[i])),i,1)

                    rr,cc = draw.polygon_perimeter([int(tlY[i]),int(trY[i]),int(brY[i]),int(blY[i])],[int(tlX[i]),int(trX[i]),int(brX[i]),int(blX[i])])
                    drawn[rr,cc]=(random.random()*0.8+.2,random.random()*0.8+.2,random.random()*0.8+.2)
                cv2.imshow('res',drawn)
                #cv2.waitKey()

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

                cv2.imshow('d',draw2)
                cv2.waitKey()


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
            #    cv2.imwrite('out2/line{}-{}.png'.format(i+index,batch_strings[i]),imm[i])
            ###
            output_strings += batch_strings
            #res.append(resBatch)
        #res = torch.cat(res,dim=1)

        ### Debug ###
        #resN=res.data.cpu().numpy()
        #output_strings, decoded_raw_hw = decode_handwriting(resN, self.idx_to_char)
            #cv2.imshow('line',imm)
            #cv2.waitKey()
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
