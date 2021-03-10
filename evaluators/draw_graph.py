import numpy as np
import math, os, random
import utils.img_f as img_f
from skimage import color, io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from utils.forms_annotations import calcCorners
import torch

def getCorners(xyrhw):
    xc=xyrhw[0].item()
    yc=xyrhw[1].item()
    rot=xyrhw[2].item()
    h=xyrhw[3].item()
    w=xyrhw[4].item()
    h = min(30000,h)
    w = min(30000,w)
    #tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(w*math.sin(rot)+h*math.cos(rot) + yc) )
    #tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
    #br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(w*math.sin(rot)-h*math.cos(rot) + yc) )
    #bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(-w*math.sin(rot)-h*math.cos(rot) + yc) )
    #return tr,tl,br,bl
    tl,tr,br,bl= calcCorners(xc,yc,rot,h,w)
    return [int(x) for x in tl],[int(x) for x in tr],[int(x) for x in br],[int(x) for x in bl]

def plotRect(img,color,xyrhw,lineWidth=1):
    tl,tr,br,bl=getCorners(xyrhw)

    img_f.line(img,tl,tr,color,lineWidth)
    img_f.line(img,tr,br,color,lineWidth)
    img_f.line(img,br,bl,color,lineWidth)
    img_f.line(img,bl,tl,color,lineWidth)


def draw_graph(outputBoxes,bb_thresh,nodePred,edgePred,edgeIndexes,predGroups,image,predTypes,missedRels,missedGroups,targetBoxes,classMap,path,verbosity=2,bbTrans=None,useTextLines=False,targetGroups=None,targetPairs=None,bbAlignment=None):
    #for graphIteration,(outputBoxes,nodePred,edgePred,edgeIndexes,predGroups) in zip(allOutputBoxes,allNodePred,allEdgePred,allEdgeIndexes,allPredGroups):
        if bbTrans is not None:
            transPath = path[:-3]+'txt'
            transOut = open(transPath,'w')
        if not useTextLines and outputBoxes is not None and not useTextLines:
            outputBoxes = outputBoxes.data.numpy()
        image = image.cpu().numpy()
        b=0
        image = (1-((1+np.transpose(image[b][:,:,:],(1,2,0)))/2.0))
        if image.shape[2]==1:
            image = img_f.gray2rgb(image)
            #image = img_f.gray2rgb(image*255)/255
        #if name=='text_start_gt':

        if verbosity>2 and targetBoxes is not None:
            #Draw GT bbs
            for j in range(targetBoxes.size(1)):
                plotRect(image,(1,0.5,0),targetBoxes[0,j,0:5])
        if verbosity>0 and targetGroups is not None:
            color=(0.99,0,0.3)
            lineWidth=1
            groupCenters=[]
            for group in targetGroups:
                xs=[]
                ys=[]
                for bbId in group:
                    corners = getCorners(targetBoxes[0,bbId,0:5])
                    xs+=[c[0] for c in corners]
                    ys+=[c[1] for c in corners]
                maxX = max(xs)+1
                minX = min(xs)-1
                maxY = max(ys)+1
                minY = min(ys)-1
                if len(group)>1 and missedGroups is None:
                    img_f.line(image,(minX,minY),(maxX,minY),color,lineWidth)
                    img_f.line(image,(maxX,maxY),(maxX,minY),color,lineWidth)
                    img_f.line(image,(minX,maxY),(maxX,maxY),color,lineWidth)
                    img_f.line(image,(minX,minY),(minX,maxY),color,lineWidth)
                groupCenters.append((round((minX+maxX)/2),round((minY+maxY)/2)))

            #now to pairs
            #for pair in targetPairs:
            #if len(predTypes)==1:
            #    print('num missing: {}'.format(len(missedRels)))
        if verbosity>0:
            for pair in missedRels:
                img_f.line(image,groupCenters[pair[0]],groupCenters[pair[1]],(1,0,0.1),3,draw='mult')
                    #if len(predTypes)==1:
                    #    print('{} -- {}'.format(groupCenters[pair[0]],groupCenters[pair[1]]))

        to_write_text=[]
        bbs = outputBoxes
        numClasses=len(classMap)
        if 'blank' in classMap:
            blank=True
            numClasses-=1
            for cls,idx in classMap.items():
                if cls!='blank':
                    assert(idx<classMap['blank'])
        else:
            blank=False
        #if verbosity>0 and outputBoxes is not None:
        #    #Draw pred bbs
        #    for j in range(len(bbs)):
        #        #circle aligned predictions
        #        if useTextLines:
        #            conf = bbs[j].getConf()
        #            maxIndex = np.argmax(bbs[j].getCls()[:numClasses])
        #            if 'gI0' in path:
        #                assert(len(bbs[j].all_primitive_rects)==1)
        #            if blank:
        #                is_blank = bbs[j].getCls()[-1]>0.5
        #        else:
        #            conf = bbs[j,0]
        #            maxIndex =np.argmax(bbs[j,6:6+numClasses])
        #            if blank:
        #                is_blank = bbs[j,-1]>0.5
        #        shade = conf#(conf-bb_thresh)/(1-bb_thresh)
        #        #print(shade)
        #        #if name=='text_start_gt' or name=='field_end_gt':
        #        #    img_f.bb(bbImage[:,:,1],p1,p2,shade,2)
        #        #if name=='text_end_gt':
        #        #    img_f.bb(bbImage[:,:,2],p1,p2,shade,2)
        #        #elif name=='field_end_gt' or name=='field_start_gt':
        #        #    img_f.bb(bbImage[:,:,0],p1,p2,shade,2)
        #        if maxIndex==0:
        #            color=(0,0,shade) #header
        #        elif maxIndex==1:
        #            color=(0,shade,shade) #question
        #        elif maxIndex==2:
        #            color=(shade,shade,0) #answer
        #        elif maxIndex==3:
        #            color=(shade,0,shade) #other
        #        else:
        #            raise NotImplementedError('Only 4 colors/classes implemented for drawing')
        #        lineWidth=1
        #        
        #        if useTextLines:
        #            pts = bbs[j].polyPoints()
        #            pts = pts.reshape((-1,1,2))
        #            if verbosity<3 or bbAlignment[j].item()!=-1:
        #                fill = 'transparent'
        #            else:
        #                fill = False
        #            img_f.polylines(image,pts.astype(np.int),fill,color,lineWidth)
        #            x,y = bbs[j].getCenterPoint()
        #            x=int(x)
        #            y=int(y)
        #        else:
        #            plotRect(image,color,bbs[j,1:6],lineWidth)
        #            x=int(bbs[j,1])
        #            y=int(bbs[j,2])

        #        if blank and is_blank:
        #            #draw a B at center of box
        #            if x-4<0:
        #                x=4
        #            if y-4<0:
        #                y=4
        #            if x+4>=image.shape[1]:
        #                x=image.shape[1]-5
        #            if y+4>=image.shape[0]:
        #                y=image.shape[0]-5
        #            image[y-2:y+3,x-1]=color
        #            image[y-2,x]=color
        #            image[y-1,x+1]=color
        #            image[y,x]=color
        #            image[y+1,x+1]=color
        #            image[y+2,x]=color

        #            image[y-4:y+5,x-4]=color
        #            image[y-4:y+5,x+4]=color
        #            image[y-4,x-4:x+5]=color
        #            image[y+4,x-4:x+5]=color


        #        #if verbosity>3 and predNN is not None:
        #        #    targ_j = bbAlignment[j].item()
        #        #    if targ_j>=0:
        #        #        gtNN = target_num_neighbors[0,targ_j].item()
        #        #    else:
        #        #        gtNN = 0
        #        #    pred_nn = predNN[j].item()
        #        #    color2 = min(abs(pred_nn-gtNN),1)#*0.5
        #        #    img_f.putText(image,'{:.2}/{}'.format(pred_nn,gtNN),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(color2,0,0),2,cv2.LINE_AA)
        #        #if bbTrans is not None:
        #        #    to_write_text.append(('{}'.format(j),(int(x),int(y)),(int(round(color[0]*255)),int(round(color[1]*255)),int(round(color[2]*255)))))
        #        #    #img_f.putText(image,'{}'.format(j),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color,2,cv2.LINE_AA)
        #        #    transOut.write('{}: {}\n'.format(j,bbTrans[j]))
        if bbTrans is not None:
            transOut.close()
            if len(to_write_text)>0:
                pil_image = Image.fromarray((image*255).astype(np.uint8))
                pil_draw = ImageDraw.Draw(pil_image)
                try:
                    font = ImageFont.truetype("UbuntuMono-R.ttf", 9)
                    for text,loc,color in to_write_text:
                        pil_draw.text(loc,text,color,font=font)
                except OSError:
                    try:
                        font = ImageFont.truetype("google-roboto", 9)
                        for text,loc,color in to_write_text:
                            pil_draw.text(loc,text,color,font=font)
                    except OSError:
                        pass

                image = np.array(pil_image).astype(np.float32)/255


        #Draw pred groups (based on bb pred)
        groupCenters=[]
        if predGroups is None and bbs is not None:
            predGroups = [[i] for i in range(len(bbs))]
        elif predGroups is None:
            predGroups = []

        for group in predGroups:
            maxX=maxY=0
            minY=minX=99999999
            idColor = [random.random()/2+0.5 for i in range(3)]
            for j in group:
                if useTextLines:
                    conf = bbs[j].getConf()
                    maxIndex = np.argmax(bbs[j].getCls()[:numClasses])
                    if 'gI0' in path:
                        assert(len(bbs[j].all_primitive_rects)==1)
                    if blank:
                        is_blank = bbs[j].getCls()[-1]>0.5
                else:
                    conf = bbs[j,0]
                    maxIndex =np.argmax(bbs[j,6:6+numClasses])
                    if blank:
                        is_blank = bbs[j,-1]>0.5
                shade = conf#(conf-bb_thresh)/(1-bb_thresh)
                #print(shade)
                #if name=='text_start_gt' or name=='field_end_gt':
                #    img_f.bb(bbImage[:,:,1],p1,p2,shade,2)
                #if name=='text_end_gt':
                #    img_f.bb(bbImage[:,:,2],p1,p2,shade,2)
                #elif name=='field_end_gt' or name=='field_start_gt':
                #    img_f.bb(bbImage[:,:,0],p1,p2,shade,2)
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
                
                if verbosity>1 or len(group)==1:
                    if useTextLines:
                        pts = bbs[j].polyPoints()
                        pts = pts.reshape((-1,1,2))
                        if verbosity<3 or bbAlignment[j].item()!=-1:
                            fill = 'transparent'
                        else:
                            fill = False
                        img_f.polylines(image,pts.astype(np.int),fill,color,lineWidth)
                        x,y = bbs[j].getCenterPoint()
                        x=int(x)
                        y=int(y)
                    else:
                        plotRect(image,color,bbs[j,1:6],lineWidth)
                        x=int(bbs[j,1])
                        y=int(bbs[j,2])

                    if blank and is_blank:
                        #draw a B at center of box
                        if x-4<0:
                            x=4
                        if y-4<0:
                            y=4
                        if x+4>=image.shape[1]:
                            x=image.shape[1]-5
                        if y+4>=image.shape[0]:
                            y=image.shape[0]-5
                        image[y-2:y+3,x-1]=color
                        image[y-2,x]=color
                        image[y-1,x+1]=color
                        image[y,x]=color
                        image[y+1,x+1]=color
                        image[y+2,x]=color

                        image[y-4:y+5,x-4]=color
                        image[y-4:y+5,x+4]=color
                        image[y-4,x-4:x+5]=color
                        image[y+4,x-4:x+5]=color
                if useTextLines:
                    pts = outputBoxes[j].polyPoints()
                    for pt in pts:
                        image[int(pt[1]):int(pt[1])+2,int(pt[0]):int(pt[0])+2]=idColor
                    maxX = max(maxX,*outputBoxes[j].polyXs())
                    minX = min(minX,*outputBoxes[j].polyXs())
                    maxY = max(maxY,*outputBoxes[j].polyYs())
                    minY = min(minY,*outputBoxes[j].polyYs())
                else:
                    tr,tl,br,bl=getCorners(outputBoxes[j,1:6])
                    if verbosity>1:
                        image[tl[1]:tl[1]+2,tl[0]:tl[0]+2]=idColor
                        image[tr[1]:tr[1]+1,tr[0]:tr[0]+1]=idColor
                        image[bl[1]:bl[1]+1,bl[0]:bl[0]+1]=idColor
                        image[br[1]:br[1]+1,br[0]:br[0]+1]=idColor
                    maxX=max(maxX,tr[0],tl[0],br[0],bl[0])
                    minX=min(minX,tr[0],tl[0],br[0],bl[0])
                    maxY=max(maxY,tr[1],tl[1],br[1],bl[1])
                    minY=min(minY,tr[1],tl[1],br[1],bl[1])
            if useTextLines:
                maxX=int(maxX)
                minX=int(minX)
                maxY=int(maxY)
                minY=int(minY)
            minX-=2
            minY-=2
            maxX+=2
            maxY+=2
            lineWidth=2
            #color=(0.5,0,1)
            if len(group)>1:
                img_f.line(image,(minX,minY),(maxX,minY),color,lineWidth)
                img_f.line(image,(maxX,minY),(maxX,maxY),color,lineWidth)
                img_f.line(image,(maxX,maxY),(minX,maxY),color,lineWidth)
                img_f.line(image,(minX,maxY),(minX,minY),color,lineWidth)
                if verbosity>1:
                    image[minY:minY+3,minX:minX+3]=idColor
            if verbosity>1:
                image[maxY:maxY+1,minX:minX+1]=idColor
                image[maxY:maxY+1,maxX:maxX+1]=idColor
                image[minY:minY+1,maxX:maxX+1]=idColor
            groupCenters.append(((minX+maxX)//2,(minY+maxY)//2))



        #Draw pred pairings
        #draw_rel_thresh = relPred.max() * draw_rel_thresh_over
        numrelpred=0
        #hits = [False]*len(adjacency)
        edgesToDraw=[]
        if edgeIndexes is not None:
            for i,(g1,g2) in enumerate(edgeIndexes):
                
                #if score>draw_rel_thresh:
                x1,y1 = groupCenters[g1]
                x2,y2 = groupCenters[g2]
                if predTypes is not None and all([predType[i]=='TN' or predType[i]=='UN' for predType in predTypes[:4]]):
                    lineColor = (0,0,edgePred[i,-1,0].item()) #BLUE
                    img_f.line(image,(x1,y1),(x2,y2),lineColor,1)
                else:
                    edgesToDraw.append((i,x1,y1,x2,y2))

            if predTypes is not None and predTypes[0] is not None and len(predTypes[0])==len(edgeIndexes):
                if edgePred is None:
                    edgePred = torch.FloatTensor(len(predTypes[0]),1,1).fill_(1)
                if edgePred.size(2)>=len(predTypes):
                    edgeClassification = [(predTypes[i],edgePred[:,-1,i]) for i in range(len(predTypes))]
                else:
                    edgeClassification = [
                            (predTypes[0],torch.ones_like(edgePred[:,-1,0])),
                            (predTypes[1],edgePred[:,-1,0]),
                            (predTypes[2],edgePred[:,-1,1]),
                            (predTypes[3],edgePred[:,-1,2])
                            ]

                for i,x1,y1,x2,y2 in edgesToDraw:
                        if edgeClassification[0][0][i]=='TP':
                            lineColor = (0,edgePred[i,-1,0].item(),0)
                        elif edgeClassification[0][0][i]=='UP':
                            lineColor = (edgePred[i,-1,0].item(),0,edgePred[i,-1,0].item())
                        elif edgeClassification[0][0][i]=='FN':
                            lineColor = (edgePred[i,-1,0].item(),0,0)
                        else: #is false positive
                            #assert(edgeClassification[0][0][i]=='FP')
                            if edgeClassification[0][0][i]!='FP':
                                print('ERROR, edge classsification is {}, but expected to be FP'.format(edgeClassification[0][0][i]))
                                #import pdb;pdb.set_trace()
                            lineColor = (edgePred[i,-1,0].item(),edgePred[i,-1,0].item(),0)
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
                        img_f.line(image,(x1,y1),(x2,y2),lineColor,2)
                        cX = (x1+x2)//2
                        cY = (y1+y2)//2
                        
                        if verbosity>1:
                            #print('{} {},  {} {},  > {} {}'.format(x1,y1,x2,y2,cX,cY))
                            for i,(offsetX,offsetY,s) in enumerate([(-2,-2,3),(1,-2,3),(1,1,3),(-2,1,3)]):
                                if i>=len(boxColors):
                                    break
                                tX=cX+offsetX
                                tY=cY+offsetY
                                image[tY:tY+s,tX:tX+s]=boxColors[i]
                            #error
                            if len(boxColors)==5 and cY-3>=0 and cY+4<image.shape[0] and cX-3>=0 and cX+4<image.shape[1]:
                                image[cY-3,cX-2:cX+4]=boxColors[4]
                                image[cY-2:cY+4,cX-3]=boxColors[4]
            else:
                lineColor = (0,0.8,0)
                for i,x1,y1,x2,y2 in edgesToDraw:
                    img_f.line(image,(x1,y1),(x2,y2),lineColor,2)


        #Draw alginment between gt and pred bbs
        if verbosity>3:
            for bbI,bb in enumerate(outputBoxes):
                if useTextLines:
                    x1,y1 = bb.getCenterPoint()
                else:
                    x1=bb[1]
                    y1=bb[2]
                x1=int(x1)
                y1=int(y1)
                targI=bbAlignment[bbI].item()
                if targI>0:

                    x2 = round(targetBoxes[0,targI,0].item())
                    y2 = round(targetBoxes[0,targI,1].item())
                    img_f.line(image,(x1,y1),(x2,y2),(1,0,1),1)
                else:
                    #draw 'x', indicating not match
                    img_f.line(image,(x1-5,y1-5),(x1+5,y1+5),(.1,0,.1),1)
                    img_f.line(image,(x1+5,y1-5),(x1-5,y1+5),(.1,0,.1),1)
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
        #        img_f.line(image,(x1,y1),(x2,y2),gtcolor,wth)

        if verbosity>0 and missedGroups is not None:
            for mgi in missedGroups:
                maxX=maxY=0
                minX=minY=9999999999
                for bbi in targetGroups[mgi]:
                    if verbosity>1:
                        plotRect(image,(1,0.5,0),targetBoxes[0,bbi,0:5])
                    tr,tl,br,bl=getCorners(targetBoxes[0,bbi,0:5])
                    cls = targetBoxes[0,bbi,13:13+numClasses].argmax().item()
                    #image[tl[1]:tl[1]+2,tl[0]:tl[0]+2]=idColor
                    #image[tr[1]:tr[1]+1,tr[0]:tr[0]+1]=idColor
                    #image[bl[1]:bl[1]+1,bl[0]:bl[0]+1]=idColor
                    #image[br[1]:br[1]+1,br[0]:br[0]+1]=idColor
                    maxX=max(maxX,tr[0],tl[0],br[0],bl[0])
                    minX=min(minX,tr[0],tl[0],br[0],bl[0])
                    maxY=max(maxY,tr[1],tl[1],br[1],bl[1])
                    minY=min(minY,tr[1],tl[1],br[1],bl[1])
                minX-=5
                minY-=5
                maxX+=5
                maxY+=5
                lineWidth=2
                color=(0.82,0,0)
                img_f.line(image,(minX,minY),(maxX,minY),color,lineWidth)
                img_f.line(image,(maxX,minY),(maxX,maxY),color,lineWidth)
                img_f.line(image,(maxX,maxY),(minX,maxY),color,lineWidth)
                img_f.line(image,(minX,maxY),(minX,minY),color,lineWidth)
                #image[minY:minY+3,minX:minX+3]=idColor
                shade=1
                if cls==0:
                    color=(0,0,shade) #header
                elif cls==1:
                    color=(0,shade,shade) #question
                elif cls==2:
                    color=(shade,shade,0) #answer
                elif cls==3:
                    color=(shade,0,shade) #other
                else:
                    raise NotImplementedError('Only 4 colors/classes implemented for drawing')

                img_f.line(image,(maxX,maxY+1),(minX,maxY+1),color,2)
                img_f.line(image,(maxX,maxY+3),(minX,maxY+3),(0.82,0,0),1)


    

        #io.imsave(path,image)
        image*=255
        img_f.imwrite(path,image)


