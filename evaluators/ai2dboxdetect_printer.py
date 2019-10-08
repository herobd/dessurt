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
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist

#THRESH=0.5

def plotRect(img,color,xyrhw):
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

    cv2.line(img,tl,tr,color,1)
    cv2.line(img,tr,br,color,1)
    cv2.line(img,br,bl,color,1)
    cv2.line(img,bl,tl,color,1)

def AI2DBoxDetect_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    def __to_tensor_old(data, gpu):
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)
        if gpu is not None:
            data = data.to(gpu)
        return data
    def __to_tensor(instance,gpu):
        data = instance['img']
        if 'bb_gt' in instance:
            targetBBs = instance['bb_gt']
            targetBBs_sizes = instance['bb_sizes']
        else:
            targetBBs = {}
            targetBBs_sizes = {}
        if 'point_gt' in instance:
            targetPoints = instance['point_gt']
            targetPoints_sizes = instance['point_label_sizes']
        else:       
            targetPoints = {}
            targetPoints_sizes = {}
        if 'pixel_gt' in instance:
            targetPixels = instance['pixel_gt']
        else:
            targetPixels = None
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)
                    
        def sendToGPU(targets):
            new_targets={}
            for name, target in targets.items():
                if target is not None:
                    new_targets[name] = target.to(gpu)
                else:
                    new_targets[name] = None
            return new_targets
            
        if gpu is not None:
            data = data.to(gpu)
            if targetBBs is not None:
                targetBBs=targetBBs.to(gpu)
            targetPoints=sendToGPU(targetPoints)
            if targetPixels is not None:
                targetPixels=targetPixels.to(gpu)
        return data, targetBBs, targetBBs_sizes, targetPoints, targetPoints_sizes, targetPixels

    THRESH = config['THRESH'] if 'THRESH' in config else 0.92
    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetBB, targetBBSizes = instance
    if lossFunc is None:
        yolo_loss = YoloLoss(model.numBBTypes,model.rotation,model.scale,model.anchors,**config['loss_params']['box'])
    else:
        yolo_loss = lossFunc
    data = instance['img']
    batchSize = data.shape[0]
    targetBBs = instance['bb_gt']
    targetPoints = instance['point_gt']
    targetPixels = instance['pixel_gt']
    imageName = instance['imgName']
    scale = instance['scale']
    gtNumNeighbors = instance['num_neighbors']
    dataT, targetBBsT, targetBBsSizes, targetPointsT, targetPointsSizes, targetPixelsT = __to_tensor(instance,gpu)

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
    outputBBs, outputOffsets, outputLines, outputOffsetLines, outputPoints, outputPixels = model(dataT)
    if outputPixels is not None:
        outputPixels = torch.sigmoid(outputPixels)
    index=0
    loss=0
    index=0
    ttt_hit=True
    #if 22>=startIndex and 22<startIndex+batchSize:
    #    ttt_hit=22-startIndex
    #else:
    #    return 0
    lossThis, position_loss, conf_loss, class_loss, nn_loss, recall, precision = yolo_loss(outputOffsets,targetBBsT,targetBBsSizes)
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
    maxConf = outputBBs[:,:,0].max().item()
    threshConf = max(maxConf*THRESH,0.5)

    if model.rotation:
        outputBBs = non_max_sup_dist(outputBBs.cpu(),threshConf,3)
    else:
        outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)

    aps_3=[]
    aps_5=[]
    aps_7=[]
    recalls_5=[]
    precs_5=[]
    numClasses = model.numBBTypes
    if 'no_blanks' in config['data_loader'] and not config['data_loader']['no_blanks']:
        numClasses-=1
    
    for b in range(batchSize):
        if targetBBs is not None:
            target_for_b = targetBBs[b,:targetBBsSizes[b],:]
        else:
            target_for_b = torch.empty(0)
        if model.rotation:
            ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBBs[b],0.9,numClasses)
            ap_3, prec_3, recall_3 =AP_dist(target_for_b,outputBBs[b],1.3,numClasses)
            ap_7, prec_7, recall_7 =AP_dist(target_for_b,outputBBs[b],0.5,numClasses)
        else:
            ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBBs[b],0.5,numClasses)
            ap_3, prec_3, recall_3 =AP_iou(target_for_b,outputBBs[b],0.3,numClasses)
            ap_7, prec_7, recall_7 =AP_iou(target_for_b,outputBBs[b],0.7,numClasses)


        aps_5.append(ap_5 )
        aps_3.append(ap_3 )
        aps_7.append(ap_7 )
        recalls_5.append(recall_5)
        precs_5.append(prec_5)
        #for b in range(len(outputBBs)):
        outputBBs[b] = outputBBs[b].data.numpy()
    
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
    if outputPixels is not None:
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
    for b in range(batchSize):
        #print('image {} has {} {}'.format(startIndex+b,targetBBsSizes[name][b],name))
        #bbImage = np.ones_like(image)
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

            for j in range(targetBBsSizes[b]):
                plotRect(image,(1,0.5,0),targetBBs[b,j,0:5])
                x=int(targetBBs[b,j,0])
                x=int(targetBBs[b,j,1])
                cv2.putText(image,'{}'.format(gtNumNeighbors[j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3,(0.6,0.3,0),2,cv2.LINE_AA)
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
            bbs = outputBBs[b]
            if model.predNumNeighbors:
                predNN= bbs[:,6]
                predClass= bbs[:,7:]
            else:
                predClass= bbs[:,6:]
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
                    pc = torch.argmax(predClass[j])
                    if pc==0:
                        color=[0,0,shade] #blob
                    elif pc==1:
                        color=[shade,0,shade*0.7] #arrow
                    elif pc==2:
                        color=[shade,shade*0.7,0] #arrow
                    elif pc==3:
                        color=[0,shade,0] #arrow
                    else:
                        print('unknown class number {} ({})'.format(pc,predClass[j]))
                        assert(False)
                    plotRect(image,color,bbs[j,1:6])
                if model.predNumNeighbors:
                    x=int(bbs[j,1])
                    y=int(bbs[j,2])
                    #color = int(min(abs(predNN[j]-gtNumNeighbors[j]),2)*127)
                    #cv2.putText(image,'{}/{}'.format(predNN[j],gtNumNeighbors[j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3,(color,0,0),2,cv2.LINE_AA)
                    cv2.putText(image,'{}'.format(predNN[j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3,color,2,cv2.LINE_AA)


            #for j in alignmentBBsTarg[name][b]:
            #    p1 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
            #    p2 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
            #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
            #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
            #    #print(mid)
            #    #print(rad)
            #    cv2.circle(image,mid,rad,(1,0,1),1)

            saveName = '{}_boxes_t:{:.2f}_f:{:.2f}'.format(imageName[b],aps_5[b][0],aps_5[b][1])
            #for j in range(metricsOut.shape[1]):
            #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
            saveName+='.png'
            io.imsave(os.path.join(outDir,saveName),image)
            #print('saved: '+os.path.join(outDir,saveName))

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

            image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
            if outputPixels is not None:
                for ch in range(outputPixels.shape[1]):
                    image[:,:,ch] = 1-outputPixels[b,ch,:,:]
                saveName = '{:06}_pixels.png'.format(startIndex+b,name)
                io.imsave(os.path.join(outDir,saveName),image)
            #print('finished writing {}'.format(startIndex+b))
        
    #return metricsOut
    return (
             #{ 'ap_5':np.array(aps_5).sum(axis=0),
             #  'ap_3':np.array(aps_3).sum(axis=0),
             #  'ap_7':np.array(aps_7).sum(axis=0),
             #  'recall':np.array(recalls_5).sum(axis=0),
             #  'prec':np.array(precs_5).sum(axis=0),
             #}, 
             { 'ap_5':aps_5,
               'ap_3':aps_3,
               'ap_7':aps_7,
               'recall':recalls_5,
               'prec':precs_5,
               'nn_loss': nn_loss,
             }, 
             (lossThis, position_loss, conf_loss, class_loss, nn_loss, recall, precision)
            )


