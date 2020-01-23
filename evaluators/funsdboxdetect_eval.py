from skimage import color, io
import os
import numpy as np
import torch
import cv2
from utils import util
from model.alignment_loss import alignment_loss
import math
from model.loss import *
from collections import defaultdict
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist
import json

#THRESH=0.5

def getCorners(xyrhw):
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
    return tl,tr,br,bl
def plotRect(img,color,xyrhw,lineW=1):
    tl,tr,br,bl = getCorners(xyrhw)

    cv2.line(img,tl,tr,color,lineW)
    cv2.line(img,tr,br,color,lineW)
    cv2.line(img,br,bl,color,lineW)
    cv2.line(img,bl,tl,color,lineW)

def FUNSDBoxDetect_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics


    THRESH = config['THRESH'] if 'THRESH' in config else 0.92
    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetBB, targetBBSizes = instance
    data = instance['img']
    batchSize = data.shape[0]
    targetBBs = instance['bb_gt']
    targetBBsSizes = instance['bb_sizes']
    targetPoints = instance['point_gt']
    targetPixels = instance['pixel_gt']
    imageName = instance['imgName']
    scale = instance['scale']
    target_num_neighbors = instance['num_neighbors']

    pretty = config['pretty'] if 'pretty' in config else False

    resultsDirName='results'
    #if outDir is not None and resultsDirName is not None:
        #rPath = os.path.join(outDir,resultsDirName)
        #if not os.path.exists(rPath):
        #    os.mkdir(rPath)
        #for name in targetBBs:
        #    nPath = os.path.join(rPath,name)
        #    if not os.path.exists(nPath):
        #        os.mkdir(nPath)

    #dataT = __to_tensor(data,gpu)
    #print('{}: {} x {}'.format(imageName,data.shape[2],data.shape[3]))
    #outputBBs, outputOffsets, outputLines, outputOffsetLines, outputPoints, outputPixels = model(dataT)
    losses,log,out = trainer.run(instance,get=toEval)
    #if outputPixels is not None:
    #    outputPixels = torch.sigmoid(outputPixels)
    index=0
    loss=0
    index=0
    ttt_hit=True
    #if 22>=startIndex and 22<startIndex+batchSize:
    #    ttt_hit=22-startIndex
    #else:
    #    return 0
    if 'points' in out:
        alignmentPointsPred={}
        alignmentPointsTarg={}
        index=0
        for name,targ in targetPointsT.items():
            #print(outputPoints[0].shape)
            #print(targetPointsSizes)
            #print('{} {}'.format(index, name))
            lossThis, predIndexes, targetPointsIndexes = alignment_loss(outputPoints[index],targ,targetPointsSizes[name],**config['loss_params']['point'],return_alignment=True, debug=ttt_hit, points=True)
            alignmentPointsPred[name]=predIndexes
            alignmentPointsTarg[name]=targetPointsIndexes
            index+=1

    data = data.cpu().data.numpy()

    if 'bbs' in out:
        outputBBs = out['bbs']
        maxConf = outputBBs[:,:,0].max().item()
        threshConf = max(maxConf*THRESH,0.5)
        if trainer.model.rotation:
            outputBBs = non_max_sup_dist(outputBBs.cpu(),threshConf,3)
        else:
            outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)

    numClasses = trainer.model.numBBTypes
    
    if 'points' in out:
        outputPointsOld = outputPoints
        targetPointsOld = targetPoints
        outputPoints={}
        targetPoints={}
        i=0
        for name,targ in targetPointsOld.items():
            if targ is not None:
                targetPoints[name] = targ.data.numpy()
            else:
                targetPoints[name]=None
            outputPoints[name] = outputPointsOld[i].cpu().data.numpy()
            i+=1
    if 'pixels' in out and outputPixels is not None:
        outputPixels = outputPixels.cpu().data.numpy()
    #metricsOut = __eval_metrics(outputBBs,targetBBs)
    #metricsOut = 0
    #if outDir is None:
    #    return metricsOut
    
    dists=defaultdict(list)
    dists_x=defaultdict(list)
    dists_y=defaultdict(list)
    scaleDiffs=defaultdict(list)
    rotDiffs=defaultdict(list)

    if 'bbs' in out:
        allPredNNs=[]
        for b in range(batchSize):
            #outputBBs[b] = outputBBs[b].data.numpy()
            #print('image {} has {} {}'.format(startIndex+b,targetBBsSizes[name][b],name))
            #bbImage = np.ones_like(image)
            bbs = outputBBs[b].data.numpy()
            if bbs.shape[0]>0:
                if trainer.model.predNumNeighbors:
                    predNN= bbs[:,6]
                    allPredNNs+=predNN.tolist()
                    predClass= bbs[:,7:]
                else:
                    predClass= bbs[:,6:]
            else:
                predNN=bbs #i.e. a zero size tensor
                predClass=bbs

            if 'save_json' in config:
                assert(batchSize==1)
                scale=scale[0]
                if targetBBs is not None:
                    if trainer.model.rotation:
                        targIndex, predWithNoIntersection = getTargIndexForPreds_dist(targetBBs[b],torch.from_numpy(bbs),1.1,numClasses,extraPreds)
                    else:
                        targIndex, predWithNoIntersection = getTargIndexForPreds_iou(targetBBs[b],torch.from_numpy(bbs),0.4,numClasses,extraPreds)
                    newId=targetBBs[b].size(0)
                else:
                    targIndex = -1*torch.ones(bbs.shape[0])
                    newId=1
                bbsData=[]
                for j in range(bbs.shape[0]):
                    tl,tr,br,bl = getCorners(bbs[j,1:])
                    id = targIndex[j].item()
                    if id<0:
                        id='u{}'.format(newId)
                        newId+=1
                    else:
                        id='m{}'.format(id)
                    bb = {
                            'id': id,
                            'poly_points': [ [float(tl[0]/scale),float(tl[1]/scale)], 
                                             [float(tr[0]/scale),float(tr[1]/scale)], 
                                             [float(br[0]/scale),float(br[1]/scale)], 
                                             [float(bl[0]/scale),float(bl[1]/scale)] ],
                            'type':'detectorPrediction',
                            'textPred': float(predClass[j,0]),
                            'fieldPred': float(predClass[j,1])
                    }
                    if numClasses==2 and trainer.model.numBBTypes==3:
                        bb['blankPred']=float(predClass[j,2])
                    if trainer.model.predNumNeighbors:
                        bb['nnPred']=float(predNN[j])
                    bbsData.append(bb)

                if instance['pairs'] is None:
                    #import pdb; pdb.set_trace()
                    instance['pairs']=[]
                pairsData=[ ('m{}'.format(i1),'m{}'.format(i2)) for i1,i2 in instance['pairs'] ]

                saveJSON = os.path.join(config['save_json'],imageName[b]+'.json')
                allData = {
                        'textBBs': bbsData,
                        'fieldBBs': [],
                        'pairs': pairsData
                }
                with open(saveJSON,'w') as f:
                    json.dump(allData,f)
                    print('wrote {}'.format(saveJSON))


            if outDir is not None:
                #Write the results so we can train LF with them
                #saveFile = os.path.join(outDir,resultsDirName,name,'{}'.format(imageName[b]))
                #we must rescale the output to be according to the original image
                #rescaled_outputBBs_xyrs = outputBBs_xyrs[name][b]
                #rescaled_outputBBs_xyrs[:,1] /= scale[b]
                #rescaled_outputBBs_xyrs[:,2] /= scale[b]
                #rescaled_outputBBs_xyrs[:,4] /= scale[b]

                #np.save(saveFile,rescaled_outputBBs_xyrs)
                image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
                if image.shape[2]==1:
                    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                #if name=='text_start_gt':

                if not pretty:
                    for j in range(targetBBsSizes[b]):
                        plotRect(image,(1,0.5,0),targetBBs[b,j,0:5])
                        #if trainer.model.predNumNeighbors:
                        #    x=int(targetBBs[b,j,0])
                        #    y=int(targetBBs[b,j,1]+targetBBs[b,j,3])
                        #    cv2.putText(image,'{:.2f}'.format(gtNumNeighbors[b,j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0.6,0.3,0),2,cv2.LINE_AA)
                    #if alignmentBBs[b] is not None:
                    #    aj=alignmentBBs[b][j]
                    #    xc_gt = targetBBs[b,j,0]
                    #    yc_gt = targetBBs[b,j,1]
                    #    xc=outputBBs[b,aj,1]
                    #    yc=outputBBs[b,aj,2]
                    #    cv2.line(image,(xc,yc),(xc_gt,yc_gt),(0,1,0),1)
                    #    shade = 0.0+(outputBBs[b,aj,0]-threshConf)/(maxConf-threshConf)
                    #    shade = max(0,shade)
                    #    if outputBBs[b,aj,6] > outputBBs[b,aj,7]:
                    #        color=(0,shade,shade) #text
                    #    else:
                    #        color=(shade,shade,0) #field
                    #    plotRect(image,color,outputBBs[b,aj,1:6])

                #bbs=[]
                #pred_points=[]
                #maxConf = outputBBs[b,:,0].max()
                #threshConf = 0.5 
                #threshConf = max(maxConf*0.9,0.5)
                #print("threshConf:{}".format(threshConf))
                #for j in range(outputBBs.shape[1]):
                #    conf = outputBBs[b,j,0]
                #    if conf>threshConf:
                #        bbs.append((conf,j))
                #    #pred_points.append(
                #bbs.sort(key=lambda a: a[0]) #so most confident bbs are draw last (on top)
                #import pdb; pdb.set_trace()
                for j in range(bbs.shape[0]):
                    #circle aligned predictions
                    conf = bbs[j,0]
                    if outDir is not None:
                        shade = 0.0+(conf-threshConf)/(maxConf-threshConf)
                        #print(shade)
                        #if name=='text_start_gt' or name=='field_end_gt':
                        #    cv2.bb(bbImage[:,:,1],p1,p2,shade,2)
                        #if name=='text_end_gt':
                        #    cv2.bb(bbImage[:,:,2],p1,p2,shade,2)
                        #elif name=='field_end_gt' or name=='field_start_gt':
                        #    cv2.bb(bbImage[:,:,0],p1,p2,shade,2)
                        maxClassIndex = np.argmax(predClass[j])
                        if maxClassIndex==0: #header
                            color=(0,0,shade)
                        elif maxClassIndex==1: #question
                            color=(0,shade,shade)
                        elif maxClassIndex==2: #answer
                            color=(shade,shade,0)
                        elif maxClassIndex==3: #other
                            color=(shade,0,shade)
                        else:
                            assert(False)

                        
                        if pretty:
                            lineW=2
                        else:
                            lineW=1
                        plotRect(image,color,bbs[j,1:6],lineW)
                        if trainer.model.predNumNeighbors and not pretty:
                            x=int(bbs[j,1])
                            y=int(bbs[j,2]-bbs[j,4])
                            #color = int(min(abs(predNN[j]-target_num_neighbors[j]),2)*127)
                            #cv2.putText(image,'{}/{}'.format(predNN[j],target_num_neighbors[j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3,(color,0,0),2,cv2.LINE_AA)
                            cv2.putText(image,'{:.2f}'.format(predNN[j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color,2,cv2.LINE_AA)


                #for j in alignmentBBsTarg[name][b]:
                #    p1 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
                #    p2 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
                #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                #    #print(mid)
                #    #print(rad)
                #    cv2.circle(image,mid,rad,(1,0,1),1)

                #saveName = '{}_boxes_AP:{:.2f}'.format(imageName[b],aps_5all[b])
                saveName = '{}_boxes'.format(imageName[b])


                #for j in range(metricsOut.shape[1]):
                #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
                saveName+='.png'
                io.imsave(os.path.join(outDir,saveName),image)
                print('saved: '+os.path.join(outDir,saveName))
            if 'points' in out:
                for name, out in outputPoints.items():
                    image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
                    #if name=='text_start_gt':
                    for j in range(targetPointsSizes[name][b]):
                        p1 = (targetPoints[name][b,j,0], targetPoints[name][b,j,1])
                        cv2.circle(image,p1,2,(1,0.5,0),-1)
                    points=[]
                    maxConf = max(out[b,:,0].max(),1.0)
                    threshConf = maxConf*0.1
                    for j in range(out.shape[1]):
                        conf = out[b,j,0]
                        if conf>threshConf:
                            p1 = (out[b,j,1],out[b,j,2])
                            points.append((conf,p1,j))
                    points.sort(key=lambda a: a[0]) #so most confident bbs are draw last (on top)
                    for conf, p1, j in points:
                        shade = 0.0+conf/maxConf
                        if name=='table_points':
                            color=(0,0,shade)
                        else:
                            color=(shade,0,0)
                        cv2.circle(image,p1,2,color,-1)
                        if alignmentPointsPred[name] is not None and j in alignmentPointsPred[name][b]:
                            mid = p1 #( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                            rad = 4 #round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                            #print(mid)
                            #print(rad)
                            #cv2.circle(image,mid,rad,(0,1,1),1)
                    #for j in alignmentBBsTarg[name][b]:
                    #    p1 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
                    #    p2 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
                    #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                    #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                    #    #print(mid)
                    #    #print(rad)
                    #    cv2.circle(image,mid,rad,(1,0,1),1)

                    saveName = '{:06}_{}'.format(startIndex+b,name)
                    #for j in range(metricsOut.shape[1]):
                    #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
                    saveName+='.png'
                    io.imsave(os.path.join(outDir,saveName),image)

            if 'pixels' in out:
                image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
                if outputPixels is not None:
                    for ch in range(outputPixels.shape[1]):
                        image[:,:,ch] = 1-outputPixels[b,ch,:,:]
                    saveName = '{:06}_pixels.png'.format(startIndex+b,name)
                    io.imsave(os.path.join(outDir,saveName),image)
            #print('finished writing {}'.format(startIndex+b))
        
    #return metricsOut
    toRet = log
    #toRet=   { 'ap_5':aps_5,
    #             #'class_aps': class_aps,
    #             #'ap_3':aps_3,
    #             #'ap_7':aps_7,
    #           'recall':recalls_5,
    #           'prec':precs_5,
    #           'nn_loss': nn_loss,
    #         }
    #for i in range(numClasses):
    #    toRet['class{}_ap'.format(i)]=class_aps[i]

    return (
             toRet,
             None
            )

