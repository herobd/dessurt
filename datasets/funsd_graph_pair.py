import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math
from collections import defaultdict, OrderedDict
from utils.forms_annotations import fixAnnotations, convertBBs, getBBWithPoints, getStartEndGT, getResponseBBIdList_
import timeit
from .graph_pair import GraphPairDataset

import cv2

SKIP=['174']#['193','194','197','200']
ONE_DONE=[]


def collate(batch):
    assert(len(batch)==1)
    return batch[0]


class FUNSDGraphPair(GraphPairDataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FUNSDGraphPair, self).__init__(dirPath,split,config,images)

        self.only_types=None

        if images is not None:
            self.images=images
        else:
            splitFile = 'train_valid_test_split.json'
            splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                readFile = json.loads(f.read())
                if type(split) is str:
                    toUse = readFile[split]
                    imagesAndAnn = []
                    imageDir = os.path.join(dirPath,toUse['root'],'images')
                    annDir = os.path.join(dirPath,toUse['root'],'annotations')
                    for name in toUse['images']:
                        imagesAndAnn.append( (name+'.png',os.path.join(imageDir,name+'.png'),os.path.join(annDir,name+'.json')) )
                elif type(split) is list:
                    imagesAndAnn = []
                    for spstr in split:
                        toUse = readFile[spstr]
                        imageDir = os.path.join(dirPath,toUse['root'],'images')
                        annDir = os.path.join(dirPath,toUse['root'],'annotations')
                        for name in toUse['images']:
                            imagesAndAnn.append( (name+'.png',os.path.join(imageDir,name+'.png'),os.path.join(annDir,name+'.json')) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
            self.images=[]
            for imageName,imagePath,jsonPath in imagesAndAnn:
                org_path = imagePath
                if self.cache_resized:
                    path = os.path.join(self.cache_path,imageName)
                else:
                    path = org_path
                if os.path.exists(jsonPath):
                    rescale=1.0
                    if self.cache_resized:
                        rescale = self.rescale_range[1]
                        if not os.path.exists(path):
                            org_img = cv2.imread(org_path)
                            if org_img is None:
                                print('WARNING, could not read {}'.format(org_img))
                                continue
                            resized = cv2.resize(org_img,(0,0),
                                    fx=self.rescale_range[1], 
                                    fy=self.rescale_range[1], 
                                    interpolation = cv2.INTER_CUBIC)
                            cv2.imwrite(path,resized)
                    self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')]})
        self.only_types=None
        self.errors=[]






    def parseAnn(self,annotations,s):
        boxes = annotations['form']
        numClasses=4
        #if useBlankClass:
        #    numClasses+=1
        #if usePairedClass:
        #    numClasses+=1

        if self.split_to_lines:
            origIdToIndexes={}
            annotations['linking']=defaultdict(list)
            index=0
            groups=[]
            bbs=[]
            line=[]
            lineTrans=[]
            def combineLine():
                bb = np.empty(8+8+numClasses, dtype=np.float32)
                lXL = max([w[0] for w in line])
                rXL = max([w[2] for w in line])
                tYL = max([w[1] for w in line])
                bYL = max([w[3] for w in line])
                bb[0]=lXL*s
                bb[1]=tYL*s
                bb[2]=rXL*s
                bb[3]=tYL*s
                bb[4]=rXL*s
                bb[5]=bYL*s
                bb[6]=lXL*s
                bb[7]=bYL*s
                #we add these for conveince to crop BBs within window
                bb[8]=s*lXL
                bb[9]=s*(tYL+bYL)/2.0
                bb[10]=s*rXL
                bb[11]=s*(tYL+bYL)/2.0
                bb[12]=s*(lXL+rXL)/2.0
                bb[13]=s*tYL
                bb[14]=s*(rXL+lXL)/2.0
                bb[15]=s*bYL
                
                bb[16:]=0
                if boxinfo['label']=='header':
                    bb[16]=1
                elif boxinfo['label']=='question':
                    bb[17]=1
                elif boxinfo['label']=='answer':
                    bb[18]=1
                elif boxinfo['label']=='other':
                    bb[19]=1
                bbs.append(bb)
                trans.append(' '.join(lineTrans))
                nex = j<len(boxes)-1
                numNeighbors.append(len(boxinfo['linking'])+(1 if prev else 0)+(1 if nex else 0))
                prev=True
                index+=1

            #new line
            line=[]
            lineTrans=[]
            for j,boxinfo in enumerate(boxes):
                prev=False
                line=[]
                lineTrans=[]
                startIdx=len(bbs)
                for word in boxinfo['words']:
                    lX,tY,rX,bY = word['box']
                    if len(line)==0:
                        line.append(word['box']+[(lX+rX)/2,(tY+bY)/2])
                        lineTrans.append(word['text'])
                    else:
                        difX = lX-line[-1][2]
                        difY = (tY+bY)/2 - line[-1][5]
                        pW = lX-line[-1][2]-lX-line[-1][0]
                        pH = lX-line[-1][3]-lX-line[-1][1]
                        if difX<-pW*0.25 or difY>pH*0.75:
                            combineLine()
                        line.append(word['box']+[(lX+rX)/2,(tY+bY)/2])
                        lineTrans.append(word['text'])
                combineLine()
                endIdx=len(bbs)
                groups.append(list(range(startIdx,endIdx)))
                for idx in range(startIdx,endIdx-1):
                    annotations['linking'][idx].append(idx+1) #we link them in read order. The group supervises dense connections. Read order is how the NAF dataset is labeled.
                origIdToIndexes[j]=(startIdx,endIdx)

            for j,boxinfo in enumerate(boxes):
                for linkId in boxinfo['linking']:
                    j_first_x = np.mean(bbs[origIdToIndexes[j][0]][0:8:2])
                    j_first_y = np.mean(bbs[origIdToIndexes[j][0]][1:8:2])
                    link_first_x = np.mean(bbs[origIdToIndexes[linkId][0]][0:8:2])
                    link_first_y = np.mean(bbs[origIdToIndexeslinkIdj][0]][1:8:2])
                    j_last_x = np.mean(bbs[origIdToIndexes[j][1]][0:8:2])
                    j_last_y = np.mean(bbs[origIdToIndexes[j][1]][1:8:2])
                    link_last_x = np.mean(bbs[origIdToIndexes[linkId][1]][0:8:2])
                    link_last_y = np.mean(bbs[origIdToIndexeslinkIdj][1]][1:8:2])

                    above = link_last_y<j_first_y
                    below = link_first_y>j_last_y
                    left = left_last_x<j_first_x
                    right = link_first_x>j_last_x
                    if above or left:
                        annotations['linking'][origIdToIndexes[j][0]].append(origIdToIndexes[linkId][1])
                    elif below or right:
                        annotations['linking'][origIdToIndexes[j][1]].append(origIdToIndexes[linkId][0])
                    else:
                        print("!!!!!!!!")
                        print("Print odd para align, unhandeled case.")
                        import pdb;pdb.set_trace()
            bbs = np.stack(bbs,axis=0)
            bbs = bbs[None,...] #add batch dim



        else:
            bbs = np.empty((1,len(boxes), 8+8+numClasses), dtype=np.float32) #2x4 corners, 2x4 cross-points, n classes
            #pairs=set()
            numNeighbors=[]
            trans=[]
            for j,boxinfo in enumerate(boxes):
                lX,tY,rX,bY = boxinfo['box']
                bbs[:,j,0]=lX*s
                bbs[:,j,1]=tY*s
                bbs[:,j,2]=rX*s
                bbs[:,j,3]=tY*s
                bbs[:,j,4]=rX*s
                bbs[:,j,5]=bY*s
                bbs[:,j,6]=lX*s
                bbs[:,j,7]=bY*s
                #we add these for conveince to crop BBs within window
                bbs[:,j,8]=s*lX
                bbs[:,j,9]=s*(tY+bY)/2.0
                bbs[:,j,10]=s*rX
                bbs[:,j,11]=s*(tY+bY)/2.0
                bbs[:,j,12]=s*(lX+rX)/2.0
                bbs[:,j,13]=s*tY
                bbs[:,j,14]=s*(rX+lX)/2.0
                bbs[:,j,15]=s*bY
                
                bbs[:,j,16:]=0
                if boxinfo['label']=='header':
                    bbs[:,j,16]=1
                elif boxinfo['label']=='question':
                    bbs[:,j,17]=1
                elif boxinfo['label']=='answer':
                    bbs[:,j,18]=1
                elif boxinfo['label']=='other':
                    bbs[:,j,19]=1
                #for id1,id2 in boxinfo['linking']:
                #    pairs.add((min(id1,id2),max(id1,id2)))
                trans.append(boxinfo['text'])
                numNeighbors.append(len(boxinfo['linking']))
            groups = [[n] for n in range(len(boxes))]


        #self.pairs=list(pairs)
        return bbs, list(range(len(boxes))), numClasses, trans, groups


    def getResponseBBIdList(self,queryId,annotations):
        if self.split_to_lines:
            return annotations['linking'][queryId]
        else:
            boxes=annotations['form']
            cto=[]
            boxinfo = boxes[queryId]
            for id1,id2 in boxinfo['linking']:
                if id1==queryId:
                    cto.append(id2)
                else:
                    cto.append(id1)
            return cto




def getWidthFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[1]) + np.linalg.norm(bb[3]-bb[2]))/2
def getHeightFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[3]) + np.linalg.norm(bb[1]-bb[2]))/2



def polyIntersect(poly1, poly2):
    prevPoint = poly1[-1]
    for point in poly1:
        perpVec = np.array([ -(point[1]-prevPoint[1]), point[0]-prevPoint[0] ])
        perpVec = perpVec/np.linalg.norm(perpVec)
        
        maxPoly1=np.dot(perpVec,poly1[0])
        minPoly1=maxPoly1
        for p in poly1:
            p_onLine = np.dot(perpVec,p)
            maxPoly1 = max(maxPoly1,p_onLine)
            minPoly1 = min(minPoly1,p_onLine)
        maxPoly2=np.dot(perpVec,poly2[0])
        minPoly2=maxPoly2
        for p in poly2:
            p_onLine = np.dot(perpVec,p)
            maxPoly2 = max(maxPoly2,p_onLine)
            minPoly2 = min(minPoly2,p_onLine)

        if (maxPoly1<minPoly2 or minPoly1>maxPoly2):
            return False
        prevPoint = point
    return True

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def lineIntersection(lineA, lineB, threshA_low=10, threshA_high=10, threshB_low=10, threshB_high=10, both=False):
    a1=lineA[0]
    a2=lineA[1]
    b1=lineB[0]
    b2=lineB[1]
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    point = (num / denom.astype(float))*db + b1
    #check if it is on atleast one line segment
    vecA = da/np.linalg.norm(da)
    p_A = np.dot(point,vecA)
    a1_A = np.dot(a1,vecA)
    a2_A = np.dot(a2,vecA)

    vecB = db/np.linalg.norm(db)
    p_B = np.dot(point,vecB)
    b1_B = np.dot(b1,vecB)
    b2_B = np.dot(b2,vecB)
    
    ###rint('A:{},  B:{}, int p:{}'.format(lineA,lineB,point))
    ###rint('{:.0f}>{:.0f} and {:.0f}<{:.0f}  and/or  {:.0f}>{:.0f} and {:.0f}<{:.0f} = {} {} {}'.format((p_A+threshA_low),(min(a1_A,a2_A)),(p_A-threshA_high),(max(a1_A,a2_A)),(p_B+threshB_low),(min(b1_B,b2_B)),(p_B-threshB_high),(max(b1_B,b2_B)),(p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)),'and' if both else 'or',(p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B))))
    if both:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) and
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    else:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) or
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    return None

