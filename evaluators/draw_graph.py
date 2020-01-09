import numpy as np
import math, os, random
import cv2
from skimage import color, io

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
    return tr,tl,br,bl

def plotRect(img,color,xyrhw,lineWidth=1):
    tr,tl,br,bl=getCorners(xyrhw)

    cv2.line(img,tl,tr,color,lineWidth)
    cv2.line(img,tr,br,color,lineWidth)
    cv2.line(img,br,bl,color,lineWidth)
    cv2.line(img,bl,tl,color,lineWidth)

def draw_graph(outputBoxes,bb_thresh,nodePred,edgePred,edgeIndexes,predGroups,image,predTypes,targetBoxes,model,path,verbosity=2):
    #for graphIteration,(outputBoxes,nodePred,edgePred,edgeIndexes,predGroups) in zip(allOutputBoxes,allNodePred,allEdgePred,allEdgeIndexes,allPredGroups):
        outputBoxes = outputBoxes.data.numpy()
        data = image.cpu().numpy()
        b=0
        image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
        if image.shape[2]==1:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        #if name=='text_start_gt':

        if verbosity>1:
            #Draw GT bbs
            for j in range(targetBoxes.size(1)):
                plotRect(image,(1,0.5,0),targetBoxes[0,j,0:5])

        if verbosity>1:
            #Draw pred bbs
            bbs = outputBoxes
            for j in range(bbs.shape[0]):
                #circle aligned predictions
                conf = bbs[j,0]
                shade = (conf-bb_thresh)/(1-bb_thresh)
                #print(shade)
                #if name=='text_start_gt' or name=='field_end_gt':
                #    cv2.bb(bbImage[:,:,1],p1,p2,shade,2)
                #if name=='text_end_gt':
                #    cv2.bb(bbImage[:,:,2],p1,p2,shade,2)
                #elif name=='field_end_gt' or name=='field_start_gt':
                #    cv2.bb(bbImage[:,:,0],p1,p2,shade,2)
                maxIndex =np.argmax(bbs[j,model.nodeIdxClass:model.nodeIdxClassEnd])
                if maxIndex==0:
                    color=(0,0,shade) #header
                elif maxIndex==1:
                    color=(0,shade,shade) #question
                elif maxIndex==2:
                    color=(shade,shade,0) #answer
                elif maxIndex==3:
                    color=(shade,0,shade) #other
                else:
                    raise NotImplementedError('Only 4 colors/classes implemented for drawing')
                lineWidth=1
                plotRect(image,color,bbs[j,1:6],lineWidth)

                if verbosity>3 and predNN is not None:
                    x=int(bbs[j,1])
                    y=int(bbs[j,2])#-bbs[j,4])
                    targ_j = bbAlignment[j].item()
                    if targ_j>=0:
                        gtNN = target_num_neighbors[0,targ_j].item()
                    else:
                        gtNN = 0
                    pred_nn = predNN[j].item()
                    color = min(abs(pred_nn-gtNN),1)#*0.5
                    cv2.putText(image,'{:.2}/{}'.format(pred_nn,gtNN),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(color,0,0),2,cv2.LINE_AA)

        #Draw pred groups (based on bb pred)
        groupCenters=[]
        for group in predGroups:
            maxX=maxY=0
            minY=minX=99999999
            idColor = [random.random()/2+0.5 for i in range(3)]
            for j in group:
                tr,tl,br,bl=getCorners(outputBoxes[j,1:6])
                image[tl[1]:tl[1]+2,tl[0]:tl[0]+2]=idColor
                image[tr[1]:tr[1]+1,tr[0]:tr[0]+1]=idColor
                image[bl[1]:bl[1]+1,bl[0]:bl[0]+1]=idColor
                image[br[1]:br[1]+1,br[0]:br[0]+1]=idColor
                maxX=max(maxX,tr[0],tl[0],br[0],bl[0])
                minX=min(minX,tr[0],tl[0],br[0],bl[0])
                maxY=max(maxY,tr[1],tl[1],br[1],bl[1])
                minY=min(minY,tr[1],tl[1],br[1],bl[1])
            minX-=2
            minY-=2
            maxX+=2
            maxY+=2
            lineWidth=2
            color=(shade/2,0,shade)
            if len(group)>1:
                cv2.line(image,(minX,minY),(maxX,minY),color,lineWidth)
                cv2.line(image,(maxX,minY),(maxX,maxY),color,lineWidth)
                cv2.line(image,(maxX,maxY),(minX,maxY),color,lineWidth)
                cv2.line(image,(minX,maxY),(minX,minY),color,lineWidth)
                image[minY:minY+3,minX:minX+3]=idColor
            image[maxY:maxY+1,minX:minX+1]=idColor
            image[maxY:maxY+1,maxX:maxX+1]=idColor
            image[minY:minY+1,maxX:maxX+1]=idColor
            groupCenters.append(((minX+maxX)//2,(minY+maxY)//2))



        #Draw pred pairings
        #draw_rel_thresh = relPred.max() * draw_rel_thresh_over
        numrelpred=0
        #hits = [False]*len(adjacency)
        edgesToDraw=[]
        for i,(g1,g2) in enumerate(edgeIndexes):
            
            #if score>draw_rel_thresh:
            x1,y1 = groupCenters[g1]
            x2,y2 = groupCenters[g2]
            if ( (predTypes[0][i]=='TN' or predTypes[0][i]=='UN') and
                 (predTypes[1][i]=='TN' or predTypes[1][i]=='UN') and
                 (predTypes[2][i]=='TN' or predTypes[2][i]=='UN') ):
                lineColor = (0,0,edgePred[i,-1,0].item()) #BLUE
                cv2.line(image,(x1,y1),(x2,y2),lineColor,1)
            else:
                edgesToDraw.append((i,x1,y1,x2,y2))

        edgeClassification = [(predTypes[i],edgePred[:,-1,i]) for i in range(len(predTypes))]

        for i,x1,y1,x2,y2 in edgesToDraw:
                lineColor = (0,edgePred[i,-1,0].item(),0)
                boxColors=[]
                for predType,pred in edgeClassification:
                    if predType[i]=='TP':
                        color = (0,pred[i].item(),0) #Green
                    elif predType[i]=='FP':
                        color = (pred[i].item(),pred[i].item()*0.5,0) #Orange
                    elif predType[i]=='TN':
                        color = (0,0,1-pred[i].item()) #Blue
                    elif predType[i]=='TN':
                        color = (1-pred[i].item(),0,0) #Red
                    else: #We don;t know the GT
                        color = (pred[i].item(),pred[i].item(),pred[i].item())
                    boxColors.append(color)
                cv2.line(image,(x1,y1),(x2,y2),lineColor,2)
                cX = (x1+x2)//2
                cY = (y1+y2)//2
                #print('{} {},  {} {},  > {} {}'.format(x1,y1,x2,y2,cX,cY))
                for i,(offsetX,offsetY,s) in enumerate([(-2,-2,3),(1,-2,3),(1,1,3),(-2,1,3)]):
                    if i>=len(boxColors):
                        break
                    tX=cX+offsetX
                    tY=cY+offsetY
                    image[tY:tY+s,tX:tX+s]=boxColors[i]


        #Draw alginment between gt and pred bbs
        if verbosity>3:
            raise NotImplementedError('alginment lines not implemented')
            for predI in range(bbs.shape[0]):
                targI=bbAlignment[predI].item()
                x1 = int(round(bbs[predI,1]))
                y1 = int(round(bbs[predI,2]))
                if targI>0:

                    x2 = round(targetBoxes[0,targI,0].item())
                    y2 = round(targetBoxes[0,targI,1].item())
                    cv2.line(image,(x1,y1),(x2,y2),(1,0,1),1)
                else:
                    #draw 'x', indicating not match
                    cv2.line(image,(x1-5,y1-5),(x1+5,y1+5),(.1,0,.1),1)
                    cv2.line(image,(x1+5,y1-5),(x1-5,y1+5),(.1,0,.1),1)
        #Draw GT pairings
        #TODO
        #if not pretty:
        #    gtcolor=(0.5,0,0.5)
        #    wth=3
        #else:
        #    #gtcolor=(1,0,0.6)
        #    gtcolor=(1,0.6,0)
        #    wth=3
        #for aId,(i,j) in enumerate(adjacency):
        #    if not pretty or not hits[aId]:
        #        x1 = round(targetBoxes[0,i,0].item())
        #        y1 = round(targetBoxes[0,i,1].item())
        #        x2 = round(targetBoxes[0,j,0].item())
        #        y2 = round(targetBoxes[0,j,1].item())
        #        cv2.line(image,(x1,y1),(x2,y2),gtcolor,wth)

    

        io.imsave(path,image)