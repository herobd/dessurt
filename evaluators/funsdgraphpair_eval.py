from skimage import color, io
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from utils import util
from model.alignment_loss import alignment_loss
import math
from model.loss import *
from collections import defaultdict
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from model.optimize import optimizeRelationships, optimizeRelationshipsSoft
import json
from utils.forms_annotations import fixAnnotations, getBBInfo
from evaluators.draw_graph import draw_graph


def plotRect(img,color,xyrhw,lineWidth=1):
    xc=xyrhw[0].item()
    yc=xyrhw[1].item()
    rot=xyrhw[2].item()
    h=xyrhw[3].item()
    w=xyrhw[4].item()
    h = min(30000,h)
    w = min(30000,w)
    tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(w*math.sin(rot)+h*math.cos(rot) + yc) )
    tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
    br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(w*math.sin(rot)-h*math.cos(rot) + yc) )
    bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(-w*math.sin(rot)-h*math.cos(rot) + yc) )

    cv2.line(img,tl,tr,color,lineWidth)
    cv2.line(img,tr,br,color,lineWidth)
    cv2.line(img,br,bl,color,lineWidth)
    cv2.line(img,bl,tl,color,lineWidth)

def FUNSDGraphPair_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    if toEval is None:
        toEval = ['allEdgePred','allEdgeIndexes','allNodePred','allOutputBoxes', 'allPredGroups', 'allEdgePredTypes']

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
    adjacency = instance['adj']
    adjacency = list(adjacency)
    imageName = instance['imgName']
    scale = instance['scale']
    target_num_neighbors = instance['num_neighbors']
    if not trainer.model.detector.predNumNeighbors:
        instance['num_neighbors']=None

    trackAtt = config['showAtt'] if 'showAtt' in config else False
    if trackAtt:
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
    #print('{}: {} x {}'.format(imageName,data.shape[2],data.shape[3]))
    if useDetections=='gt':
        losses, log, out = trainer.run(instance,True,get=toEval)
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

    toRet={}
    for gIter,(edgePred, relIndexes, bbPred, outputBoxes, predGroups, edgePredTypes) in enumerate(zip(allEdgePred,allEdgeIndexes,allNodePred,allOutputBoxes,allPredGroups,allEdgePredTypes)):


        relPred_otherTimes = edgePred[:,:-1]
        relPred = edgePred[:,-1] #remove time
        if model.predNN and bbPred is not None:
            predNN = bbPred[:,-1,0]
        else:
            predNN=None
        if  model.detector.predNumNeighbors and not useDetections:
            #useOutputBBs=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
            extraPreds=1
            if not model.predNN:
                predNN = outputBoxes[:,6]
        else:
            extraPreds=0
            if not model.predNN:
                predNN = None
            #useOutputBBs=outputBoxes


        if 'rule' in config:
            if config['rule']=='closest':
                if relPred.size(1)>1:
                    relPred = relPred[:,0:1]
                dists = torch.FloatTensor(relPred.size())
                differentClass = torch.FloatTensor(relPred.size())
                predClasses = torch.argmax(outputBoxes[:,extraPreds+6:extraPreds+6+numClasses],dim=1)
                for i,(bb1,bb2) in enumerate(relIndexes):
                    dists[i] = math.sqrt((outputBoxes[bb1,1]-outputBoxes[bb2,1])**2 + (outputBoxes[bb1,2]-outputBoxes[bb2,2])**2)
                    differentClass[i] = predClasses[bb1]!=predClasses[bb2]
                maxDist = torch.max(dists)
                minDist = torch.min(dists)
                relPred = 1-(dists-minDist)/(maxDist-minDist)
                relPred *= differentClass
            elif config['rule']=='icdar':
                if relPred.size(1)>1:
                    relPred = relPred[:,0:1]
                height = torch.FloatTensor(relPred.size())
                dists = torch.FloatTensor(relPred.size())
                right = torch.FloatTensor(relPred.size())
                sameClass = torch.FloatTensor(relPred.size())
                predClasses = torch.argmax(outputBoxes[:,extraPreds+6:extraPreds+6+numClasses],dim=1)
                for i,(bb1,bb2) in enumerate(relIndexes):
                    sameClass[i] = predClasses[bb1]==predClasses[bb2]
                    
                    #g4 of the paper
                    height[i] = max(outputBoxes[bb1,4],outputBoxes[bb2,4])/min(outputBoxes[bb1,4],outputBoxes[bb2,4])

                    #g5 of the paper
                    if predClasses[bb1]==0:
                        widthLabel = outputBoxes[bb1,5]*2 #we predict half width
                        widthValue = outputBoxes[bb2,5]*2
                        dists[i] = math.sqrt(((outputBoxes[bb1,1]+widthLabel)-(outputBoxes[bb2,1]-widthValue))**2 + (outputBoxes[bb1,2]-outputBoxes[bb2,2])**2)
                    else:
                        widthLabel = outputBoxes[bb2,5]*2 #we predict half width
                        widthValue = outputBoxes[bb1,5]*2
                        dists[i] = math.sqrt(((outputBoxes[bb1,1]-widthValue)-(outputBoxes[bb2,1]+widthLabel))**2 + (outputBoxes[bb1,2]-outputBoxes[bb2,2])**2)
                    if dists[i]>2*widthLabel:
                        dists[i]/=widthLabel
                    else: #undefined
                        dists[i] = min(1,dists[i]/widthLabel)
                
                    #g6 of the paper
                    if predClasses[bb1]==0:
                        widthValue = outputBoxes[bb2,5]*2
                        hDist = outputBoxes[bb1,1]-outputBoxes[bb2,1]
                    else:
                        widthValue = outputBoxes[bb1,5]*2
                        hDist = outputBoxes[bb2,1]-outputBoxes[bb1,1]
                    right[i] = hDist/widthValue
                relPred = 1-(height+dists+right + 10000*sameClass)
            else:
                print('ERROR, unknown rule {}'.format(config['rule']))
                exit()
            relPred_otherTimes=torch.FloatTensor()
        elif relPred is not None:
            relPred = torch.sigmoid(relPred)[:,0]
            relPred_otherTimes=torch.sigmoid(relPred_otherTimes)[:,:,0]




        relCand = relIndexes
        if relCand is None:
            relCand=[]

        if model.rotation:
            bbAlignment, bbFullHit = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,0.9,numClasses,extraPreds,hard_thresh=False)
        else:
            bbAlignment, bbFullHit = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.5,numClasses,extraPreds,hard_thresh=False)
        if targetBoxes is not None:
            target_for_b = targetBoxes[0,:,:]
        else:
            target_for_b = torch.empty(0)

        if outputBoxes.size(0)>0:
            maxConf = outputBoxes[:,0].max().item()
            minConf = outputBoxes[:,0].min().item()
            if useDetections:
                minConf=0
        #threshConf = max(maxConf*THRESH,0.5)
        #if model.rotation:
        #    outputBoxes = non_max_sup_dist(outputBoxes.cpu(),threshConf,3)
        #else:
        #    outputBoxes = non_max_sup_iou(outputBoxes.cpu(),threshConf,0.4)
        if model.rotation:
            ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,model.numBBTypes,beforeCls=extraPreds)
        else:
            ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes,0.5,model.numBBTypes,beforeCls=extraPreds)

        #precisionHistory={}
        #precision=-1
        #minStepSize=0.025
        #targetPrecisions=[None]
        #for targetPrecision in targetPrecisions:
        #    if len(precisionHistory)>0:
        #        closestPrec=9999
        #        for prec in precisionHistory:
        #            if abs(targetPrecision-prec)<abs(closestPrec-targetPrecision):
        #                closestPrec=prec
        #        precision=prec
        #        stepSize=precisionHistory[prec][0]
        #    else:
        #        stepSize=0.1
        #
        #    while True: #abs(precision-targetPrecision)>0.001:
        for rel_threshold in rel_thresholds:

                if 'optimize' in config and config['optimize']:
                    relPred_otherTimes=torch.FloatTensor()
                    if 'penalty' in config:
                        penalty = config['penalty']
                    else:
                        penalty = 0.25
                    print('optimizing with penalty {}'.format(penalty))
                    thresh=0.15
                    while thresh<0.45:
                        keep = relPred>thresh
                        newRelPred = relPred[keep]
                        if newRelPred.size(0)<700:
                            break
                    if newRelPred.size(0)>0:
                        #newRelCand = [ cand for i,cand in enumerate(relCand) if keep[i] ]
                        usePredNN= predNN is not None and config['optimize']!='gt'
                        idMap={}
                        newId=0
                        newRelCand=[]
                        numNeighbors=[]
                        for index,(id1,id2) in enumerate(relCand):
                            if keep[index]:
                                if id1 not in idMap:
                                    idMap[id1]=newId
                                    if not usePredNN:
                                        numNeighbors.append(target_num_neighbors[0,bbAlignment[id1]].item())
                                    else:
                                        numNeighbors.append(predNN[id1].item())
                                    newId+=1
                                if id2 not in idMap:
                                    idMap[id2]=newId
                                    if not usePredNN:
                                        numNeighbors.append(target_num_neighbors[0,bbAlignment[id2]].item())
                                    else:
                                        numNeighbors.append(predNN[id2].item())
                                    newId+=1
                                newRelCand.append( [idMap[id1],idMap[id2]] )            


                        #if not usePredNN:
                            #    decision = optimizeRelationships(newRelPred,newRelCand,numNeighbors,penalty)
                        #else:
                        decision= optimizeRelationshipsSoft(newRelPred.cpu(),newRelCand,numNeighbors,penalty, rel_threshold)
                        decision= torch.from_numpy( np.round_(decision).astype(int) )
                        decision=decision.to(relPred.device)
                        relPred[keep] = torch.where(0==decision,relPred[keep]-1,relPred[keep])
                        relPred[1-keep] -=1
                        rel_threshold_use=0#-0.5
                    else:
                        rel_threshold_use=rel_threshold
                else:
                    rel_threshold_use=rel_threshold

                #threshed in model
                #if len(precisionHistory)==0:

                #class_acc=0
                useOutputBBs=None

                truePred=falsePred=badPred=0
                scores=[]
                if len(relPred_otherTimes)>1:
                    scores_otherTimes=[[] for i in range(relPred_otherTimes.size(1))]
                matches=0
                i=0
                numMissedByHeur=0
                targGotHit=set()
                truePred_len=[]
                falsePred_len=[]
                falseRej_len=[]
                trueRej_len=[]
                #print('debug relPred: {}'.format(relPred.shape))
                for i,(n0,n1) in enumerate(relCand):
                    x1 = outputBoxes[n0,1].item()
                    y1 = outputBoxes[n0,2].item()
                    x2 = outputBoxes[n1,1].item()
                    y2 = outputBoxes[n1,2].item()
                    rel_length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    t0 = bbAlignment[n0].item()
                    t1 = bbAlignment[n1].item()
                    if t0>=0 and bbFullHit[n0]:
                        targGotHit.add(t0)
                    if t1>=0 and bbFullHit[n1]:
                        targGotHit.add(t1)
                    if t0>=0 and t1>=0 and bbFullHit[n0] and bbFullHit[n1]:
                        if (min(t0,t1),max(t0,t1)) in adjacency:
                            matches+=1
                            gtRel=True
                            if relPred[i]>rel_threshold_use:
                                truePred+=1
                                truePred_len.append(rel_length)
                            else:
                                falseRej_len.append(rel_length)
                        else:
                            gtRel=False
                            if relPred[i]>rel_threshold_use:
                                falsePred+=1
                                falsePred_len.append(rel_length)
                            else:
                                trueRej_len.append(rel_length)
                    else:
                        gtRel=False
                        if relPred[i]>rel_threshold_use:
                            badPred+=1
                            falsePred_len.append(rel_length)
                        else:
                            trueRej_len.append(rel_length)
                    scores.append( (relPred[i],gtRel) )
                    if len(relPred_otherTimes.size())>1:
                        for t in range(relPred_otherTimes.size(1)):
                            scores_otherTimes[t].append( (relPred_otherTimes[i,t],gtRel) )
                for i in range(len(adjacency)-matches):
                    numMissedByHeur+=1
                    scores.append( (float('nan'),True) )
                    if len(relPred_otherTimes.size())>1:
                        for t in range(relPred_otherTimes.size(1)):
                            scores_otherTimes[t].append( (float('nan'),True) )
                rel_ap=computeAP(scores)
                rel_ap_otherTimes=[]
                if len(relPred_otherTimes.size())>1:
                    for t in range(relPred_otherTimes.size(1)):
                        rel_ap_otherTimes.append( computeAP(scores_otherTimes[t]) )

                numMissedByDetect=0
                for t0,t1 in adjacency:
                    if t0 not in targGotHit or t1 not in targGotHit:
                        numMissedByHeur-=1
                        numMissedByDetect+=1
                if len(adjacency)>0:
                    relRecall = truePred/len(adjacency)
                    heurRecall = (len(adjacency)-numMissedByHeur)/len(adjacency)
                    detectRecall = (len(adjacency)-numMissedByDetect)/len(adjacency)
                else:
                    relRecall = 1
                    heurRecall = 1
                    detectRecall = 1
                #if falsePred>0:
                #    relPrec = truePred/(truePred+falsePred)
                #else:
                #    relPrec = 1
                if falsePred+badPred>0:
                    precision = truePred/(truePred+falsePred+badPred)
                else:
                    precision = 1
        

                toRet['{}: prec@{}'.format(gIter,rel_threshold)]=precision
                toRet['{}: recall@{}'.format(gIter,rel_threshold)]=relRecall
                if relRecall+precision>0:
                    toRet['{}: F-M@{}'.format(gIter,rel_threshold)]=2*relRecall*precision/(relRecall+precision)
                else:
                    toRet['{}: F-M@{}'.format(gIter,rel_threshold)]=0
                toRet['{}: rel_AP@{}'.format(gIter,rel_threshold)]=rel_ap
                toRet['{}: rel_truePred_len@{}'.format(gIter,rel_threshold)]=truePred_len
                toRet['{}: rel_trueRej_len@{}'.format(gIter,rel_threshold)]=trueRej_len
                toRet['{}: rel_falseRej_len@{}'.format(gIter,rel_threshold)]=falseRej_len
                toRet['{}: rel_falsePred_len@{}'.format(gIter,rel_threshold)]=falsePred_len
                #precisionHistory[precision]=(draw_rel_thresh,stepSize)
                #if targetPrecision is not None:
                #    if abs(precision-targetPrecision)<0.001:
                #        break
                #    elif stepSize<minStepSize:
                #        if precision<targetPrecision:
                #            draw_rel_thresh += stepSize*2
                #            continue
                #        else:
                #            break
                #    elif precision<targetPrecision:
                #        draw_rel_thresh += stepSize
                #        if not wasTooSmall:
                #            reverse=True
                #            wasTooSmall=True
                #        else:
                #            reverse=False
                #    else:
                #        draw_rel_thresh -= stepSize
                #        if wasTooSmall:
                #            reverse=True
                #            wasTooSmall=False
                #        else:
                #            reverse=False
                #    if reverse:
                #        stepSize *= 0.5
                #else:
                #    break


                #import pdb;pdb.set_trace()

                #for b in range(len(outputBoxes)):
                
                
                dists=defaultdict(list)
                dists_x=defaultdict(list)
                dists_y=defaultdict(list)
                scaleDiffs=defaultdict(list)
                rotDiffs=defaultdict(list)
                b=0
                #print('image {} has {} {}'.format(startIndex+b,targetBoxesSizes[name][b],name))
                #bbImage = np.ones_like(image):w

        if trackAtt:
            data = data.numpy()
            imageO = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0))
            bbs = outputBoxes.numpy()
            for attL,attn in enumerate(attList):
                image = imageO.copy()
                if image.shape[2]==1:
                    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                for i in range(len(relCand)):
                    
                    ind1 = relCand[i][0]
                    ind2 = relCand[i][1]
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

                    cv2.line(image,(x1,y1),(xh,yh),color1,1)
                    cv2.line(image,(x2,y2),(xh,yh),color2,1)
                #cv2.imshow('attention',image)
                #cv2.waitKey()
                saveName='{}_gI{}_att{}.png'.format(imageName,gIter,attL)
                io.imsave(os.path.join(outDir,saveName),image)




        elif outDir is not None:

            saveName = '{}_gI{}_boxes_prec:{:.2f},{:.2f}_recall:{:.2f},{:.2f}_rels_AP:{:.3f}'.format(imageName,gIter,prec_5[0],prec_5[1],recall_5[0],recall_5[1],rel_ap if rel_ap is not None else -1)
            #for j in range(metricsOut.shape[1]):
            #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
            path = os.path.join(outDir,saveName+'.png')
            draw_graph(outputBoxes,trainer.model.used_threshConf,torch.sigmoid(bbPred).cpu().detach(),torch.sigmoid(edgePred).cpu().detach(),relIndexes,predGroups,data,edgePredTypes,targetBoxes,trainer.model,path)
            #io.imsave(os.path.join(outDir,saveName),image)
            #print('saved: '+os.path.join(outDir,saveName))

        if model.detector.predNumNeighbors and not useDetections:
            predNN_d = outputBoxes[:,6]
            diffs=torch.abs(predNN_d-target_num_neighbors[0][bbAlignment].float())
            nn_acc_d = (diffs<0.5).sum().item()
            nn_acc_d /= predNN.size(0)
            toRet['{}: nn_acc_detector'.format(gIter)] = nn_acc_d

        print('\n{} ap:{}\tnumMissedByDetect:{}\tmissedByHuer:{}'.format(imageName,rel_ap,numMissedByDetect,numMissedByHeur))
    for key in losses.keys():
        losses[key] = losses[key].item()
    retData= { 'bb_ap':[ap_5],
               'bb_recall':[recall_5],
               'bb_prec':[prec_5],
               'bb_Fm': -1,#(recall_5[0]+recall_5[1]+prec_5[0]+prec_5[1])/4,
               'rel_recall':relRecall,
               'rel_precision':precision,
               'rel_Fm':2*relRecall*precision/(relRecall+precision) if relRecall+precision>0 else 0,
               'relMissedByHeur':numMissedByHeur,
               'relMissedByDetect':numMissedByDetect,
               'heurRecall': heurRecall,
               'detectRecall': detectRecall,
               **toRet,
               **losses,

             }
    for key,value in log.items():
        if key.startswith('final'):
            retData[key]=value
    if rel_ap is not None: #none ap if no relationships
        retData['rel_AP']=rel_ap
        retData['no_targs']=0
        #calculate rel_ap differences for timesteps
        for t in range(1,len(rel_ap_otherTimes)):
            diff = rel_ap_otherTimes[t]-rel_ap_otherTimes[t-1]
            retData['rel_AP_gain{}_{}'.format(t-1,t)] = diff
        if len(rel_ap_otherTimes)>0:
            diff = rel_ap-rel_ap_otherTimes[-1]
            retData['rel_AP_gain{}_{}'.format(len(rel_ap_otherTimes)-1,len(rel_ap_otherTimes))] = diff
    else:
        retData['no_targs']=1
    return (
             retData,
             None
            )


