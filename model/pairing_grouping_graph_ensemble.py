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

class PairingGroupingGraphEnsemble(PairingGroupingGraph):
    def __init__(self, config):
        super(PairingGroupingGraph, self).__init__(config)

        self.models=[]
        for checkpoint_path in config['models']:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
            checkpoint['config']['model']['arch'] = checkpoint['config']['arch']
            config = checkpoint['config']['model']
            model = eval(config['arch'])(config)
            if 'swa_state_dict' in checkpoint and checkpoint['iteration']>checkpoint['config']['trainer']['swa_start']:
                state_dict = {key[7:]:value for key,value in checkpoint['swa_state_dict'].items() if key.startswith('module.')}
            else:
                state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            self.models.append(model)

        self.useCurvedBBs = self.models[0].useCurvedBBs
        assert all(self.useCurvedBBs==model.useCurvedBBs for model in self.models[1:])
        self.detector_predNumNeighbors = self.models[0].detector_predNumNeighbors
        assert all(self.detector_predNumNeighbors==model.detector_predNumNeighbors for model in self.models[1:])
        self.relationshipProposal = self.models[0].relationshipProposal
        assert all(self.relationshipProposal==model.relationshipProposal for model in self.models[1:])
        self.graph_min_degree = self.models[0].graph_min_degree
        assert all(self.graph_min_degree==model.graph_min_degree for model in self.models[1:])
        self.graph_four_connected = self.models[0].graph_four_connected
        assert all(self.graph_four_connected==model.graph_four_connected for model in self.models[1:])
        self.fully_connected = self.models[0].fully_connected
        assert all(self.fully_connected==model.fully_connected for model in self.models[1:])
        self.no_betweenPixels = self.models[0].no_betweenPixels
        assert all(self.no_betweenPixels==model.no_betweenPixels for model in self.models[1:])
        self.rel_hard_thresh = self.models[0].rel_hard_thresh
        assert all(self.rel_hard_thresh==model.rel_hard_thresh for model in self.models[1:])
        self.max_rel_to_keep = self.models[0].max_rel_to_keep
        assert all(self.max_rel_to_keep==model.max_rel_to_keep for model in self.models[1:])
        self.numBBTypes = self.models[0].numBBTypes
        assert all(self.numBBTypes==model.numBBTypes for model in self.models[1:])
        self.predClass = self.models[0].predClass
        assert all(self.predClass==model.predClass for model in self.models[1:])
        self.rotation = self.models[0].rotation
        assert all(self.rotation==model.rotation for model in self.models[1:])
        self.scale = self.models[0].scale
        assert all(self.scale==model.scale for model in self.models[1:])
        self.predNN = self.models[0].predNN
        assert all(self.predNN==model.predNN for model in self.models[1:])
        self.include_bb_conf = self.models[0].include_bb_conf
        assert all(self.include_bb_conf==model.include_bb_conf for model in self.models[1:])
        self.merge_first = self.models[0].merge_first
        assert all(self.merge_first==model.merge_first for model in self.models[1:])
        self.legacy = self.models[0].legacy
        assert all(self.legacy==model.legacy for model in self.models[1:])
        self.normalizeHorz = self.models[0].normalizeHorz
        assert all(self.normalizeHorz==model.normalizeHorz for model in self.models[1:])
        self.normalizeVert = self.models[0].normalizeVert
        assert all(self.normalizeVert==model.normalizeVert for model in self.models[1:])
        self.normalizeDist = self.models[0].normalizeDist
        assert all(self.normalizeDist==model.normalizeDist for model in self.models[1:])
        self.percent_rel_to_keep = self.models[0].percent_rel_to_keep
        assert all(self.percent_rel_to_keep==model.percent_rel_to_keep for model in self.models[1:])

        self.num_graphnets = len(self.models[0].graphnets)
        assert all(self.num_graphnets==len(model.graphnets) for model in self.models[1:])

        self.useMetaGraph=True
        self.detector_frozen=True


        self.PRIMARY_MODEL=0
        self.anchors = self.models[self.PRIMARY_MODEL].anchors
        self.nodeIdxConf = self.models[self.PRIMARY_MODEL].nodeIdxConf 
        self.nodeIdxClass = self.models[self.PRIMARY_MODEL].nodeIdxClass
        self.nodeIdxClassEnd = self.models[self.PRIMARY_MODEL].nodeIdxClassEnd

        self.mergeThresh = self.models[self.PRIMARY_MODEL].mergeThresh
        self.keepEdgeThresh = self.models[self.PRIMARY_MODEL].keepEdgeThresh
        self.groupThresh = self.models[self.PRIMARY_MODEL].groupThresh


    def forward(self, image, gtBBs=None, gtNNs=None, useGTBBs=False, otherThresh=None, otherThreshIntur=None, hard_detect_limit=5000, debug=False,old_nn=False,gtTrans=None,merge_first_only=False, gtGroups=None):
        assert(image.size(0)==1) #implementation designed for batch size of 1. Should work to do data parallelism, since each copy of the model will get a batch size of 1
        self.merges_performed=0 #just tracking to see if it's working
        #t###tic=timeit.default_timer()#t#
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        models_saved_features=[]
        models_saved_features2=[]
        models_bbPredictions=[]
        for model in self.models:
            if not model.detector.forGraphPairing:
                model.detector.setForGraphPairing(*model.set_detect_params)

            bbPredictions, offsetPredictions, _,_,_,_ = model.detector(image)
            _=None
            if model.detector.saved_features is None:
                model.detector.setForGraphPairing(*model.set_detect_params)
                bbPredictions, offsetPredictions, _,_,_,_ = model.detector(image)
            saved_features=model.detector.saved_features
            model.detector.saved_features=None

            if model.use2ndFeatures:
                saved_features2=model.detector.saved_features2
            else:
                saved_features2=None
            #t###print('   detector: {}'.format(timeit.default_timer()-tic))#t#

            if saved_features is None:
                print('ERROR:no saved features!')
                import pdb;pdb.set_trace()

            
            #t###tic=timeit.default_timer()#t#
            if model.useHardConfThresh:
                model.used_threshConf = model.detect_conf_thresh
            else:
                maxConf = bbPredictions[:,:,0].max().item()
                if otherThreshIntur is None:
                    confThreshMul = model.detect_conf_thresh
                else:
                    confThreshMul = model.detect_conf_thresh*(1-otherThreshIntur) + otherThresh*otherThreshIntur
                model.used_threshConf = max(maxConf*confThreshMul,0.5)

            if model.training:
                model.used_threshConf += np.random.normal(0,0.1) #we'll tweak the threshold around to make training more robust

            if model.useCurvedBBs:
                threshed_bbPredictions = [bbPredictions[0,bbPredictions[0,:,0]>model.used_threshConf].cpu()]
                if model.use_overseg_non_max_sup:
                    threshed_bbPredictions[0] = non_max_sup_overseg(threshed_bbPredictions[0])
                bbPredictions = threshed_bbPredictions
            else:
                bbPredictions = non_max_sup_iou(bbPredictions.cpu(),model.used_threshConf,0.4,hard_detect_limit)
            #print(bbPredictions[0].size())

            #I'm assuming batch size of one
            assert(len(bbPredictions)==1)
            bbPredictions=bbPredictions[0]
            if model.no_grad_feats:
                bbPredictions=bbPredictions.detach()

            models_saved_features.append(saved_features)
            models_saved_features2.append(saved_features2)
            models_bbPredictions.append(bbPredictions)

        ##TODO ##Merge bbs
        bbPredictions = models_bbPredictions[self.PRIMARY_MODEL] #for now just pick one...

        if useGTBBs and  gtBBs is not None:
            useBBs, gtBBs, gtGroups = self.alignGTBBs(useGTBBs,gtBBs,gtGroups,bbPredictions)
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

        models_transcriptions=[]
        for model in self.models:
            if model.text_rec is not None:
                if useGTBBs and gtTrans is not None: # and len(gtTrans)==useBBs.size[0]:
                    assert 'word_bbs' not in useGTBBs and not model.useCurvedBBs
                    #transcriptions = gtTrans
                    transcriptions = ['']*useBBs.size(0)
                    for i,trans in enumerate(gtTrans):
                        transcriptions[gt_to_new[i]]=trans
                    #transcriptions = [gtTrans[new_to_gt[newi]] if newi in new_to_gt else '' for newi in range(useBBs.size(0))] 
                elif not model.merge_first: #skip if oversegmented, for speed
                    transcriptions = model.getTranscriptions(useBBs,image)
                    if gtTrans is not None:
                        if model.include_bb_conf:
                            justBBs = useBBs[:,1:]
                        else:
                            justBBs = useBBs
                        transcriptions=correctTrans(transcriptions,justBBs,gtTrans,gtBBs)
                else:
                    transcriptions = None
            else:
                transcriptions=None
            models_transcriptions.append(transcriptions)


        if len(useBBs):#useBBs.size(0)>1:
            models_embeddings=[]
            for model,transcriptions in zip(self.models,models_transcriptions):
                if transcriptions is not None:
                    embeddings = model.embedding_model(transcriptions,saved_features.device)
                    if model.add_noise_to_word_embeddings:
                        embeddings += torch.randn_like(embeddings).to(embeddings.device)*model.add_noise_to_word_embeddings*embeddings.mean()
                else:
                    embeddings=None
                models_embeddings.append(embeddings)


            if not self.useMetaGraph:
                raise NotImplementedError('Simple pairing not implemented for new grouping stuff')

            models_bbTrans = models_transcriptions

            allOutputBoxes, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final = self.runGraph_models(
                    gtGroups,
                    gtTrans,
                    image,
                    useBBs,
                    models_saved_features,
                    models_saved_features2,
                    models_bbTrans,
                    models_embeddings,
                    merge_first_only)

            return allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final

        else:
            if not self.useCurvedBBs and self.detector_predNumNeighbors:
                #Discard NN prediction. We don't use it anymore
                bbPredictions = torch.cat([bbPredictions[:,:6],bbPredictions[:,7:]],dim=1)
                useBBs = torch.cat([useBBs[:,:6],useBBs[:,7:]],dim=1)
            return [bbPredictions], offsetPredictions, None, None, None, None, None, None, (useBBs if self.useCurvedBBs else useBBs.cpu().detach(),None,None,transcriptions)


    #Use the graph network's predictions to merge oversegmented detections and group nodes into a single node
    def mergeAndGroup_models(self,
            mergeThresh,
            keepEdgeThresh,
            groupThresh,
            oldEdgeIndexes,
            edgePredictions,
            oldGroups,
            models_oldNodeFeats,
            models_oldEdgeFeats,
            models_oldUniversalFeats,
            oldBBs,
            models_oldBBTrans,
            models_old_text_emb,
            image,
            skip_rec=False,
            merge_only=False,
            good_edges=None,
            keep_edges=None,
            gt_groups=None,
            final=False):
        #assert(len(oldBBs)==0 or type(oldBBs[0]) is TextLine)
        oldNumGroups=len(oldGroups)
        #changedNodeIds=set()
        if self.useCurvedBBs:
            bbs={i:TextLine(clone=v) for i,v in enumerate(oldBBs)}
        else:
            oldBBs=oldBBs.cpu()
            bbs={i:v for i,v in enumerate(oldBBs)}

        models_bbTrans = [None]*len(self.models)
        for mi,(model,oldBBTrans) in enumerate(zip(self.models,models_oldBBTrans)):
            if model.text_rec is not None and oldBBTrans is not None:
                models_bbTrans[mi]={i:v for i,v  in enumerate(oldBBTrans)}
            else:
                models_bbTrans[mi]=None
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
                
        for mi,model in enumerate(self.models):
            if model.text_rec is not None and len(bbs)>0 and not skip_rec:
                if merge_only :
                    doTransIndexes = [idx for idx in bbs] #everything, since we skip recognition for speed
                else:
                    doTransIndexes = [idx for idx in mergedTo if idx in bbs]
                if len(doTransIndexes)>0:
                    doBBs = [bbs[idx] for idx in doTransIndexes]
                    if not self.useCurvedBBs:
                        doBBs = torch.stack(doBBs,dim=0)
                    newTrans = model.getTranscriptions(doBBs,image)
                    for i,idx in enumerate(doTransIndexes):
                        models_bbTrans[mi][idx] = newTrans[i]
        if merge_only:
            newBBs=[]
            models_newBBTrans=[None]*len(self.models)
            for mi,model in enumerate(self.models):
                models_newBBTrans[mi]=[] if model.text_rec is not None else None
            for bbId,bb in bbs.items():
                newBBs.append(bb)
                for mi,(model,bbTrans) in enumerate(zip(self.models,models_bbTrans)):
                    if model.text_rec is not None and not skip_rec:
                        models_newBBTrans[mi].append(bbTrans[bbId])
            return newBBs, models_newBBTrans

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
        models_newBBTrans=[[] for ii in range(len(self.models))]
        for i,(bbId,bb) in enumerate(bbs.items()):
            bbIdToPos[bbId]=i
            newBBs.append(bb)
            for mi,(model,bbTrans) in enumerate(zip(self.models,models_bbTrans)):
                if model.text_rec is not None:
                    models_newBBTrans[mi].append(bbTrans[bbId])

        ##pull the features together for nodes
        #Actually change workGroups to list
        newGroupToOldGrouping=defaultdict(list) #tracks what has been merged
        for k,v in oldGroupToNewGrouping.items():
            newGroupToOldGrouping[v].append(k)
        models_newNodeFeats=[None]*len(self.models)
        models_new_text_emb=[None]*len(self.models)
        for mi,(model,oldNodeFeats,old_text_emb) in enumerate(zip(self.models,models_oldNodeFeats,models_old_text_emb)):
            if oldNodeFeats is not None:
                newNodeFeats = torch.FloatTensor(len(workGroups),oldNodeFeats.size(1)).to(oldNodeFeats.device)
            else:
                newNodeFeats = None
            if old_text_emb is not None and model.text_rec is None:
                new_text_emb = torch.FloatTensor(len(workGroups),old_text_emb.size(1)).to(old_text_emb.device)
            else:
                new_text_emb = None
            models_newNodeFeats[mi]=newNodeFeats
            models_new_text_emb[mi]=new_text_emb

        oldToNewNodeIds_unchanged={}
        oldToNewIds_all={}
        newGroups=[]
        models_groupNodeTrans=[[] for ii in range(len(self.models))]
        for i,(idx,bbIds) in enumerate(workGroups.items()):
            newGroups.append([bbIdToPos[bbId] for bbId in bbIds])
            models_featsToCombine=[[] for ii in range(len(self.models))]
            models_embeddings_to_combine=[[] for ii in range(len(self.models))]

            for oldNodeIdx in newGroupToOldGrouping[idx]:
                oldToNewIds_all[oldNodeIdx]=i
                for mi,(oldNodeFeats,old_text_emb) in enumerate(zip(models_oldNodeFeats,models_old_text_emb)):
                    models_featsToCombine[mi].append(oldNodeFeats[oldNodeIdx] if oldNodeFeats is not None else None)
                    models_embeddings_to_combine[mi].append(old_text_emb[oldNodeIdx] if old_text_emb is not None else None)
                if oldNodeIdx in newGroupToOldMerge:
                    for mergedIdx in newGroupToOldMerge[oldNodeIdx]:
                        for mi,(oldNodeFeats,old_text_emb) in enumerate(zip(models_oldNodeFeats,models_old_text_emb)):
                            models_featsToCombine[mi].append(oldNodeFeats[mergedIdx] if oldNodeFeats is not None else None)
                            models_embeddings_to_combine[mi].append(old_text_emb[mergedIdx] if old_text_emb is not None else None)
                        oldToNewIds_all[mergedIdx]=i

            if len(models_featsToCombine[0])==1:
                oldToNewNodeIds_unchanged[oldNodeIdx]=i
                for mi,(featsToCombine,embeddings_to_combine,oldNodeFeats,new_text_emb,model) in enumerate(zip(models_featsToCombine,models_embeddings_to_combine,models_oldNodeFeats,models_new_text_emb,self.models)):
                    if oldNodeFeats is not None:
                        models_newNodeFeats[mi][i]=featsToCombine[0]
                    if new_text_emb is not None and model.text_rec is None:
                        models_new_text_emb[mi][i]=embeddings_to_combine[0]
            else:
                for mi,(featsToCombine,embeddings_to_combine,oldNodeFeats,new_text_emb,model) in enumerate(zip(models_featsToCombine,models_embeddings_to_combine,models_oldNodeFeats,models_new_text_emb,self.models)):
                    if oldNodeFeats is not None:
                        models_newNodeFeats[mi][i]=model.groupNodeFunc(featsToCombine)
                    if new_text_emb is not None and model.text_rec is None:
                        models_new_text_emb[mi][i]=torch.stack(embeddings_to_combine,dim=0).mean(dim=0)


            for mi,(model,bbTrans) in enumerate(zip(self.models,models_bbTrans)):
                if model.text_rec is not None:
                    if self.useCurvedBBs:
                        groupTrans = [(bbs[bbId].getReadPosition(),bbTrans[bbId]) for bbId in bbIds]
                    else:
                        groupTrans = [(bbs[bbId][2],bbTrans[bbId]) for bbId in bbIds] #by center y
                    groupTrans.sort(key=lambda a:a[0])
                    models_groupNodeTrans[mi].append(' '.join([t[1] for t in groupTrans]))
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
        models_newEdgeFeats=[None]*len(self.models)
        for mi,oldEdgeFeats in enumerate(models_oldEdgeFeats):
            if oldEdgeFeats is not None:
                newEdgeFeats=torch.FloatTensor(len(newEdges_map),oldEdgeFeats.size(1)).to(oldEdgeFeats.device)
            else:
                newEdgeFeats = None
            models_newEdgeFeats[mi]=newEdgeFeats

        if self.fully_connected:
            good_edges_copy = good_edges.copy()
            good_edges.clear() #yes, I'm returning data by chaning the internal state of an argument object. Sorry.
        if keep_edges is not None:
            old_keep_edges=keep_edges
            keep_edges=set()
        for edge,oldIds in newEdges_map.items():
            for mi,(model,oldEdgeFeats) in enumerate(zip(self.models,models_oldEdgeFeats)):
                if oldEdgeFeats is not None:
                    if len(oldIds)==1:
                        models_newEdgeFeats[mi][len(newEdges)]=oldEdgeFeats[oldIds[0]]
                    else:
                        models_newEdgeFeats[mi][len(newEdges)]=model.groupEdgeFunc([oldEdgeFeats[oId] for oId in oldIds])
            if keep_edges is not None:
                if any([oId in old_keep_edges for oId in oldIds]):
                    keep_edges.add(len(newEdges))
            newEdges.append(edge)
            if self.fully_connected:
                good_edges.append(any([good_edges_copy[oId] for oId in oldIds]))


        for mi,(model,oldNodeFeats,groupNodeTrans) in enumerate(zip(self.models,models_oldNodeFeats,models_groupNodeTrans)):
            if model.text_rec is not None and oldNodeFeats is not None:
                newNodeEmbeddings = model.embedding_model(groupNodeTrans,oldNodeFeats.device)
                if model.add_noise_to_word_embeddings>0:
                    newNodeEmbeddings += torch.randn_like(newNodeEmbeddings).to(newNodeEmbeddings.device)*model.add_noise_to_word_text_emb
                if model.legacy_read:
                    models_newNodeFeats[mi] = model.merge_embedding_layer(torch.cat((models_newNodeFeats[mi],newNodeEmbeddings),dim=1))
                    models_new_text_emb[mi]=models_old_text_emb[mi]
                else:
                    models_new_text_emb[mi] = newNodeEmbeddings
                #models_newNodeEmbeddings[mi] = newNodeEmbeddings

        edges = newEdges
        newEdges = list(newEdges) + [(y,x) for x,y in newEdges] #add reverse edges so undirected/bidirectional
        if len(newEdges)>0:
            newEdgeIndexes = torch.LongTensor(newEdges).t()
            if models_oldEdgeFeats[0] is not None:
                newEdgeIndexes = newEdgeIndexes.to(models_oldEdgeFeats[0].device)
        else:
            newEdgeIndexes = torch.LongTensor(0)
        for mi,oldEdgeFeats in enumerate(models_oldEdgeFeats):
            if oldEdgeFeats is not None:
                models_newEdgeFeats[mi] = models_newEdgeFeats[mi].repeat(2,1)

        models_newGraph = [None]*len(self.models)
        for mi,(newNodeFeats, newEdgeFeats, oldUniversalFeats) in enumerate(zip(models_newNodeFeats, models_newEdgeFeats, models_oldUniversalFeats)):
            models_newGraph[mi] = (newNodeFeats, newEdgeIndexes, newEdgeFeats, oldUniversalFeats)

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

        return newBBs, models_newGraph, newGroups, edges, models_newBBTrans, models_new_text_emb,  oldToNewNodeIds_unchanged, keep_edges



                



    def createGraph_models(self,bbs,models_features,models_features2,imageHeight,imageWidth,models_text_emb=None,flip=None,debug_image=None,image=None,merge_only=False):
        #t#tic=timeit.default_timer()#t#
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        if self.relationshipProposal == 'line_of_sight':
            raise NotImplementedError('only nn prop implemented')
            assert(not merge_only)
            candidates = self.selectLineOfSightEdges(bbs,imageHeight,imageWidth)
            rel_prop_scores = None
        elif self.relationshipProposal == 'feature_nn':
            candidates, rel_prop_scores = self.selectFeatureNNEdges_models(bbs,imageHeight,imageWidth,image,models_features[0].device,merge_only=merge_only,models_text_emb=models_text_emb)
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

        models_allMasks=[]
        for model in self.models:
            if (not merge_only and  model.useShapeFeats!='only') or model.merge_use_mask:
                allMasks=model.makeAllMasks(imageHeight,imageWidth,bbs,merge_only)
            else:
                allMasks=None
            models_allMasks.append(allMasks)
        groups=[[i] for i in range(len(bbs))]
        models_edge_vis_features = [None]*len(self.models)
        for mi,(model,features,features2,allMasks) in enumerate(zip(self.models,models_features,models_features2,models_allMasks)):
            models_edge_vis_features[mi] = model.computeEdgeVisualFeatures(features,features2,imageHeight,imageWidth,bbs,groups,candidates,allMasks,flip,merge_only,debug_image)

        #if self.useShapeFeats=='sp
        #print('rel features built')
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        #print('------rel------')
        if merge_only:
            return models_edge_vis_features, candidates, rel_prop_scores #we won't build the graph
        
        models_rel_features = [None]*len(self.models)
        for mi,(model,edge_vis_features) in enumerate(zip(self.models,models_edge_vis_features)):
            if model.reintroduce_edge_visual_maps is not None:
                models_rel_features[mi] = model.reintroduce_edge_visual_maps[0](edge_vis_features) #this is an extra linear layer to prep the features for the graph (which expects non-activated values)
            else:
                models_rel_features[mi] = edge_vis_features
    
        #compute features for the bounding boxes by themselves
        #This will be replaced with/appended to some type of word embedding
        #with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        models_node_vis_features = [None]*len(self.models)
        models_bb_features = [None]*len(self.models)
        for mi,(model,features,features2,allMasks,text_emb) in enumerate(zip(self.models,models_features,models_features2,models_allMasks,models_text_emb)):
            node_vis_features = model.computeNodeVisualFeatures(features,features2,imageHeight,imageWidth,bbs,groups,text_emb,allMasks,merge_only,debug_image)
            if model.reintroduce_node_visual_maps is not None:
                #print('node_vis_features: {}'.format(node_vis_features.size()))
                if node_vis_features.size(0)==0:
                    print(node_vis_features.size())
                try:
                    bb_features = model.reintroduce_node_visual_maps[0](node_vis_features) #this is an extra linear layer to prep the features for the graph (which expects non-activated values)
                except RuntimeError as e:
                    print('text_emb = {}'.format(text_emb))
                    print('node_vis_features: {}, layer: {}'.format(node_vis_features.size(),model.reintroduce_node_visual_maps[0]))
                    raise e
            else:
                bb_features = node_vis_features
            models_node_vis_features[mi] = node_vis_features
            models_bb_features[mi] = bb_features
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
            models_nodeFeatures= models_bb_features
            if self.fully_connected and not merge_only:
                edges = [(a,b) for b in range(numBB) for a in range(b)]

                models_edgeFeatures= [torch.FloatTensor(len(edges),rel_features.size(1)).zero_().to(rel_features.device)]*len(self.models)
                #TODO this could be optimized
                for i,e in enumerate(candidates):
                    j = edges.index(e)
                    for mi,rel_features in enumerate(models_rel_features):
                        models_edgeFeatures[mi][j]=rel_features[i]
            else:
                models_edgeFeatures= models_rel_features
                edges=candidates

            edges += [(y,x) for x,y in edges] #add backward edges for undirected graph
            edgeIndexes = torch.LongTensor(edges).t().to(models_rel_features[0].device)
            #now we need to also replicate the edgeFeatures
            for mi in range(len(self.models)):
                models_edgeFeatures[mi] = models_edgeFeatures[mi].repeat(2,1)

            #features
            models_universalFeatures=[None]*len(self.models)

            #t##time = timeit.default_timer()-tic#t#
            #print('   create graph: {}'.format(time)) #old 0.37, new 0.16
            ##self.opt_createG.append(time)
            #t##if len(self.opt_createG)>17:#t#
            #t##    print('   create graph running mean: {}'.format(np.mean(self.opt_createG)))#t#
            #t##    if len(self.opt_createG)>30:#t#
            #t##        self.opt_createG = self.opt_createG[1:]#t#
            models_graph = [(nodeFeatures,edgeFeatures,universalFeatures) for nodeFeatures,edgeFeatures,universalFeatures in zip(models_nodeFeatures,models_edgeFeatures,models_universalFeatures)]
            return models_graph, relIndexes, rel_prop_scores, models_node_vis_features,models_edge_vis_features, keep_edges
        else:
            assert False
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






    def selectFeatureNNEdges_models(self,bbs,imageHeight,imageWidth,image,device,merge_only=False,models_text_emb=False):
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
        
        prop_with_emb=[model.prop_with_text_emb for model in self.models]
        if not any(prop_with_emb):

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

        else:
            raise NotImplementedError('did not for prop with text.')
            #need seperate features for each :(

            if self.prop_with_text_emb:
                num_feats += 2*self.numTextFeats
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
            raise NotImplementedError('yep')
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
            rel_pred=0
            for model in self.models:
                rel_pred += model.rel_prop_nn(features.to(device))
            rel_pred /= len(self.models)

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





    #this is modified to do late fusion at each graph edit
    def runGraph_models(self,gtGroups,gtTrans,image,useBBs,models_saved_features,models_saved_features2,models_bbTrans,models_embeddings,merge_first_only=False):
        
        groups=[[i] for i in range(len(useBBs))]
        if self.merge_first:
            raise NotImplementedError("Merge first not implemented for emsemble late fusion")
            assert gtGroups is None
            models_edgeOuts=[]
            models_edgeIndexes=[]
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
                return allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, None, merge_prop_scores, None

            if bbTrans is not None:
                if gtTrans is not None:
                    if self.include_bb_conf:
                        justBBs = useBBs[:,1:]
                    else:
                        justBBs = useBBs
                    bbTrans=correctTrans(bbTrans,justBBs,gtTrans,gtBBs)
                embeddings = self.embedding_model(bbTrans,saved_features.device)
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


        rel_prop_scores=None
        models_graph,edgeIndexes,_,models_last_node_visual_feats,models_last_edge_visual_feats,keep_edges = self.createGraph_models(useBBs,models_saved_features,models_saved_features2,image.size(-2),image.size(-1),models_text_emb=models_embeddings,image=image)
        models_graph=[]
        models_edgeIndexes=[]
        models_last_node_visual_feats=[]
        models_last_edge_visual_feats=[]
        models_keep_edges=[]
        models_nodeOuts=[]
        models_edgeOuts=[]
        models_nodeFeats=[]
        models_edgeFeats=[]
        models_uniFeats=[]

        for model,saved_features,saved_features2,embeddings in zip(self.models,models_saved_features,models_saved_features2,models_embeddings):
            graph,edgeIndexes,_,last_node_visual_feats,last_edge_visual_feats,keep_edges = model.createGraph(useBBs,saved_features,saved_features2,image.size(-2),image.size(-1),text_emb=embeddings,image=image)
            assert graph is not None #I don't think this happens in eval
            if model.reintroduce_features=='map':
                last_node_visual_feats = graph[0]
                last_edge_visual_feats = graph[2]



            nodeOuts, edgeOuts, nodeFeats, edgeFeats, uniFeats = model.graphnets[0](graph)
            assert(edgeOuts is None or not torch.isnan(edgeOuts).any())
            edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]

            models_graph.append(graph)
            models_edgeIndexes.append(edgeIndexes)
            models_last_node_visual_feats.append(last_node_visual_feats)
            models_last_edge_visual_feats.append(last_edge_visual_feats)
            models_keep_edges.append(keep_edges)

            models_nodeOuts.append(nodeOuts)
            models_edgeOuts.append(edgeOuts)
            models_nodeFeats.append(nodeFeats)
            models_edgeFeats.append(edgeFeats)
            models_uniFeats.append(uniFeats)

        nodeOuts = torch.stack(models_nodeOuts,dim=0).mean(dim=0)
        edgeOuts = torch.stack(models_edgeOuts,dim=0).mean(dim=0)
        #cat_nodeFeats = torch.cat(nodeFeats,dim=1)
        #cat_edgeFeats = torch.cat(edgeFeats,dim=1)
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
        
        #for gIter,graphnet in enumerate(self.graphnets[1:]):
        for gIter in range(self.num_graphnets-1):
            if self.merge_first:
                raise NotImplementedError('yep')
                gIter+=1
            
            if self.fully_connected:
                good_edges=[]
            else:
                good_edges=None
            #print('!D! {} before edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(gIter,edgeFeats.size(),len(useBBs),nodeFeats.size(),len(edgeIndexes)))
            #print('      graph num edges: {}'.format(graph[1].size()))
            useBBs,models_graph,groups,edgeIndexes,models_bbTrans,models_embeddings,same_node_map,keep_edges=self.mergeAndGroup_models(
                    self.mergeThresh[gIter],
                    self.keepEdgeThresh[gIter],
                    self.groupThresh[gIter],
                    edgeIndexes,
                    edgeOuts,
                    groups,
                    models_nodeFeats,
                    models_edgeFeats,
                    models_uniFeats,
                    useBBs,
                    models_bbTrans,
                    models_embeddings,
                    image,
                    good_edges=good_edges,
                    keep_edges=keep_edges,
                    gt_groups=gtGroups if gIter==0 else ([[g] for g in range(len(groups))] if gtGroups is not None else None))


            for mi,model in enumerate(self.models):
                if model.reintroduce_features:
                    graph,last_node_visual_feats,last_edge_visual_feats = model.appendVisualFeatures(
                            gIter if self.merge_first else gIter+1,
                            useBBs,
                            models_graph[mi],
                            groups,
                            edgeIndexes,
                            models_saved_features[mi],
                            models_saved_features2[mi],
                            models_embeddings[mi],
                            image.size(-2),
                            image.size(-1),
                            same_node_map,
                            models_last_node_visual_feats[mi],
                            models_last_edge_visual_feats[mi],
                            allEdgeIndexes[-1],
                            debug_image=None,
                            good_edges=good_edges)

                    models_graph[mi]=graph
                    models_last_node_visual_feats[mi]=last_node_visual_feats
                    models_last_edge_visual_feats[mi]=last_edge_visual_feats
            #print('graph 1-:   bbs:{}, nodes:{}, edges:{}'.format(useBBs.size(0),len(groups),len(edgeIndexes)))
            if len(edgeIndexes)==0:
                break #we have no graph, so we can just end here

            for mi,model in enumerate(self.models):
                graphnet = model.graphnets[1+gIter]
                nodeOuts, edgeOuts, nodeFeats, edgeFeats, uniFeats = graphnet(models_graph[mi])
                models_nodeOuts[mi]=nodeOuts
                models_edgeOuts[mi]=edgeOuts
                models_nodeFeats[mi]=nodeFeats
                models_edgeFeats[mi]=edgeFeats
                models_uniFeats[mi]=uniFeats

            nodeOuts = torch.stack(models_nodeOuts,dim=0).mean(dim=0)
            edgeOuts = torch.stack(models_edgeOuts,dim=0).mean(dim=0)
            #edgeIndexes = edgeIndexes[:len(edgeIndexes)//2]
            useBBs = self.updateBBs(useBBs,groups,nodeOuts)
            allOutputBoxes.append(useBBs if self.useCurvedBBs else useBBs.cpu()) 
            allNodeOuts.append(nodeOuts)
            allEdgeOuts.append(edgeOuts)
            allGroups.append(groups)
            allEdgeIndexes.append(edgeIndexes)


        ##Final state of the graph
        #print('!D! F before edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(edgeFeats.size(),useBBs.size(),nodeFeats.size(),len(edgeIndexes)))
        useBBs,graph,groups,edgeIndexes,bbTrans,_,same_node_map,keep_edges=self.models[self.PRIMARY_MODEL].mergeAndGroup(
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
                models_bbTrans[self.PRIMARY_MODEL],
                None,
                image,
                gt_groups=[[g] for g in range(len(groups))] if gtGroups is not None else None,
                final=True)
        #print('!D! after  edge size: {}, bbs: {}, node size: {}, edge I size: {}'.format(graph[2].size(),useBBs.size(),graph[0].size(),len(edgeIndexes)))
        if not self.useCurvedBBs and self.detector_predNumNeighbors:
            #Discard NN prediction. We don't use it anymore
            useBBs = torch.cat([useBBs[:,:6],useBBs[:,7:]],dim=1)
        final=(useBBs if self.useCurvedBBs else useBBs.cpu().detach(),groups,edgeIndexes,bbTrans)

        if not self.useCurvedBBs and self.detector_predNumNeighbors:
            #Discard NN prediction. We don't use it anymore
            allOutputBoxes = [ torch.cat([outBs[:,:6],outBs[:,7:]],dim=1) for outBs in allOutputBoxes]
        return allOutputBoxes, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final
