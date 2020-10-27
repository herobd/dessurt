from skimage import color, io
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.img_f as img_f
from utils import util
from utils.util import plotRect
from model.alignment_loss import alignment_loss
import math
from model.loss import *
from collections import defaultdict
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from model.optimize import optimizeRelationships, optimizeRelationshipsSoft
import json
from utils.forms_annotations import fixAnnotations, getBBInfo
from evaluators.draw_graph import draw_graph



def FUNSDGraphPair_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    if toEval is None:
        toEval = ['allEdgePred','allEdgeIndexes','allNodePred','allOutputBoxes', 'allPredGroups', 'allEdgePredTypes','final']

    rel_thresholds = [config['THRESH']] if 'THRESH' in config else [0.5]
    if ('sweep_threshold' in config and config['sweep_threshold']) or ('sweep_thresholds' in config and config['sweep_thresholds']):
        rel_thresholds = np.arange(0.1,1.0,0.05)
    if ('sweep_threshold_big' in config and config['sweep_threshold_big']) or ('sweep_thresholds_big' in config and config['sweep_thresholds_big']):
        rel_thresholds = np.arange(0,20.0,1)
    if ('sweep_threshold_small' in config and config['sweep_threshold_small']) or ('sweep_thresholds_small' in config and config['sweep_thresholds_small']):
        rel_thresholds = np.arange(0,0.1,0.01)
    draw_rel_thresh_over = config['draw_thresh'] if 'draw_thresh' in config else rel_thresholds[0]
    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetBB, targetBBSizes = instance
    model = trainer.model
    data = instance['img']
    batchSize = data.shape[0]
    assert(batchSize==1)
    targetBoxes = instance['bb_gt']
    #adjacency = instance['adj']
    #adjacency = list(adjacency)
    imageName = instance['imgName']
    scale = instance['scale']
    target_num_neighbors = instance['num_neighbors']
    if not trainer.model.detector.predNumNeighbors:
        instance['num_neighbors']=None

    trackAtt = config['showAtt'] if 'showAtt' in config else False
    if trackAtt:
        if model.pairer is None:
            for gn in mode.graphnets:
                gn.trackAtt=True
        else:
            trainer.model.pairer.trackAtt=True
    if 'repetitions' in config:
        trainer.model.pairer.repetitions=config['repetitions']
    pretty = config['pretty'] if 'pretty' in config else False
    if 'repetitions' in config:
        trainer.model.pairer.repetitions=config['repetitions']
    useDetections = config['useDetections'] if 'useDetections' in config else False
    if 'useDetect' in config:
        useDetections = config['useDetect']
    confThresh = config['conf_thresh'] if 'conf_thresh' in config else None


    numClasses=len(trainer.classMap)

    resultsDirName='results'
    #if outDir is not None and resultsDirName is not None:
        #rPath = os.path.join(outDir,resultsDirName)
        #if not os.path.exists(rPath):
        #    os.mkdir(rPath)
        #for name in targetBoxes:
        #    nPath = os.path.join(rPath,name)
        #    if not os.path.exists(nPath):
        #        os.mkdir(nPath)

    #dataT = __to_tensor(data,gpu)
    print('{}: {} x {}'.format(imageName,data.shape[2],data.shape[3]))
    if useDetections=='gt':
        losses, log, out = trainer.newRun(instance,True,get=toEval)
    elif type(useDetections) is str:
        raise NotImplementedError('using saved detections not adjusted for new eval')
        dataset=config['DATASET']
        jsonPath = os.path.join(useDetections,imageName+'.json')
        with open(os.path.join(jsonPath)) as f:
            annotations = json.loads(f.read())
        fixAnnotations(dataset,annotations)
        savedBoxes = torch.FloatTensor(len(annotations['byId']),6+model.detector.predNumNeighbors+numClasses)
        for i,(id,bb) in enumerate(annotations['byId'].items()):
            qX, qY, qH, qW, qR, qIsText, qIsField, qIsBlank, qNN = getBBInfo(bb,dataset.rotate,useBlankClass=not dataset.no_blanks)
            savedBoxes[i,0]=1 #conf
            savedBoxes[i,1]=qX*scale #x-center, already scaled
            savedBoxes[i,2]=qY*scale #y-center
            savedBoxes[i,3]=qR #rotation
            savedBoxes[i,4]=qH*scale/2
            savedBoxes[i,5]=qW*scale/2
            if model.detector.predNumNeighbors:
                extra=1
                savedBoxes[i,6]=qNN
            else:
                extra=0
            savedBoxes[i,6+extra]=qIsText
            savedBoxes[i,7+extra]=qIsField
            
        if gpu is not None:
            savedBoxes=savedBoxes.to(gpu)
        outputBoxes, outputOffsets, relPred, relIndexes, bbPred = model(dataT,savedBoxes,None,"saved",
                otherThresh=confThresh,
                otherThreshIntur=1 if confThresh is not None else None,
                hard_detect_limit=600,
                old_nn=True)
        outputBoxes=savedBoxes.cpu()
    elif useDetections:
        print('Unknown detection flag: '+useDetections)
        exit()
    else:
        losses, log, out = trainer.newRun(instance,False,get=toEval)

    if trackAtt:
        if model.pairer is None:
            #liist of graph nets, get all the attention!
            allAttList = [gn.attn for gn in model.graphnets]
        else:
            attList = model.pairer.attn
    #relPredFull = relPred
    allEdgePred = out['allEdgePred']
    allEdgeIndexes = out['allEdgeIndexes']
    allNodePred = out['allNodePred']
    allOutputBoxes = out['allOutputBoxes']
    allPredGroups = out['allPredGroups']
    allEdgePredTypes = out['allEdgePredTypes']

    if targetBoxes is not None:
        targetSize=targetBoxes.size(1)
    else:
        targetSize=0

    toRet={}#log
    if allEdgePred is not None:
        for gIter,(edgePred, relIndexes, bbPred, outputBoxes, predGroups, edgePredTypes) in enumerate(zip(allEdgePred,allEdgeIndexes,allNodePred,allOutputBoxes,allPredGroups,allEdgePredTypes)):



            if trackAtt and (not model.merge_first or gIter>0):
                attList = allAttList[gIter-1 if model.merge_first else gIter]
                data = data.numpy()
                imageO = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0))
                bbs = outputBoxes.numpy()
                for attL,attn in enumerate(attList):
                    image = imageO.copy()
                    if image.shape[2]==1:
                        image = img_f.cvtColor(image,cv2.COLOR_GRAY2RGB)
                    for i in range(len(relIndexes)):
                        
                        ind1 = relIndexes[i][0]
                        ind2 = relIndexes[i][1]
                        x1 = int(round(bbs[ind1,1]))
                        y1 = int(round(bbs[ind1,2]))
                        x2 = int(round(bbs[ind2,1]))
                        y2 = int(round(bbs[ind2,2]))
                        xh = (x1+x2)//2
                        yh = (y1+y2)//2

                        #a1 = attn[0,:,ind1,i].max().item()
                        #a2 = attn[0,:,ind2,i].max().item()
                        #color1 = (a1,0,0.5-abs(a1-0.5))
                        #color2 = (a2,0,0.5-abs(a2-0.5))
                        a1 = attn[0,:,ind1,i]
                        a2 = attn[0,:,ind2,i]
                        color1 = (a1[0].item(),a1[1].item(),a1[2].item())
                        color2 = (a2[0].item(),a2[1].item(),a2[2].item())

                        img_f.line(image,(x1,y1),(xh,yh),color1,1)
                        img_f.line(image,(x2,y2),(xh,yh),color2,1)
                    #img_f.imshow('attention',image)
                    #img_f.waitKey()
                    saveName='{}_Att_gI:{}_L:{}.png'.format(imageName,gIter,attL)
                    io.imsave(os.path.join(outDir,saveName),image)




            if outDir is not None:
                if gIter==0 and trainer.model.merge_first:
                    saveName = '{}_gI{}_mergeFirst_recall:{:.2f}_prec:{:.2f}_Fm:{:.2f}'.format(imageName,gIter,log['recallMergeFirst_0'],log['precMergeFirst_0'],log['FmMergeFirst_0'])
                else:
                    saveName = '{}_gI{}_Fms_edge:{:.2f}_rel:{:.2f}_merge:{:.2f}_group:{:.2f}'.format(imageName,gIter,log['FmEdge_{}'.format(gIter)],log['FmRel_{}'.format(gIter)],log['FmOverSeg_{}'.format(gIter)],log['FmGroup_{}'.format(gIter)])
                    #for j in range(metricsOut.shape[1]):
                #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
                path = os.path.join(outDir,saveName+'.png')
                draw_graph(outputBoxes,trainer.model.used_threshConf,bbPred.cpu().detach() if bbPred is not None else None,torch.sigmoid(edgePred).cpu().detach(),relIndexes,predGroups,data,edgePredTypes,targetBoxes,trainer.model,path,useTextLines=trainer.model.useCurvedBBs,targetGroups=instance['gt_groups'],targetPairs=instance['gt_groups_adj'])
                #io.imsave(os.path.join(outDir,saveName),image)
                #print('saved: '+os.path.join(outDir,saveName))

            #what is bbAlignment?
            #if model.detector.predNumNeighbors and not useDetections:
            #    predNN_d = outputBoxes[:,6]
            #    diffs=torch.abs(predNN_d-target_num_neighbors[0][bbAlignment].float())
            #    nn_acc_d = (diffs<0.5).sum().item()
            #    nn_acc_d /= predNN.size(0)
            #    toRet['{}: nn_acc_detector'.format(gIter)] = nn_acc_d

        #print('\n{} ap:{}\tnumMissedByDetect:{}\tmissedByHuer:{}'.format(imageName,rel_ap,numMissedByDetect,numMissedByHeur))

    if outDir is not None:
        path = os.path.join(outDir,'{}_final_relFm:{}_r+p:{}+{}_bbFm:{}_r+p:{}+{}.png'.format(imageName,log['final_rel_Fm'],log['final_rel_recall'],log['final_rel_prec'],log['final_bb_allFm'],log['final_bb_allRecall'],log['final_bb_allPrec']))
        finalOutputBoxes, finalPredGroups, finalEdgeIndexes, finalBBTrans = out['final']
        draw_graph(finalOutputBoxes,trainer.model.used_threshConf,None,None,finalEdgeIndexes,finalPredGroups,data,None,targetBoxes,trainer.model,path,bbTrans=finalBBTrans,useTextLines=trainer.model.useCurvedBBs,targetGroups=instance['gt_groups'],targetPairs=instance['gt_groups_adj'])

    for key in losses.keys():
        losses[key] = losses[key].item()
    retData= { 
               **toRet,
               **losses,

             }
    for key,value in log.items():
        #if key.startswith('final'):
        if type(value) is np.ndarray:
            retData[key]={i:[value[i]] for i in range(value.shape[0])}
        else:
            retData[key]=[value]
    #if rel_ap is not None: #none ap if no relationships
    #    retData['rel_AP']=rel_ap
    #    retData['no_targs']=0
    #    #calculate rel_ap differences for timesteps
    #    for t in range(1,len(rel_ap_otherTimes)):
    #        diff = rel_ap_otherTimes[t]-rel_ap_otherTimes[t-1]
    #        retData['rel_AP_gain{}_{}'.format(t-1,t)] = diff
    #    if len(rel_ap_otherTimes)>0:
    #        diff = rel_ap-rel_ap_otherTimes[-1]
    #        retData['rel_AP_gain{}_{}'.format(len(rel_ap_otherTimes)-1,len(rel_ap_otherTimes))] = diff
    #else:
    #    retData['no_targs']=1
    return (
             retData,
             None
            )


