import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from .graph_pair import GraphPairDataset

import utils.img_f as img_f

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

        self.split_to_lines = config['split_to_lines']

        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                splitFile = 'overfit_split.json'
            else:
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
                            org_img = img_f.imread(org_path)
                            if org_img is None:
                                print('WARNING, could not read {}'.format(org_img))
                                continue
                            resized = img_f.resize(org_img,(0,0),
                                    fx=self.rescale_range[1], 
                                    fy=self.rescale_range[1], 
                                    )
                            img_f.imwrite(path,resized)
                    self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')]})
        self.only_types=None
        self.errors=[]

        self.classMap={
                'header':16,
                'question':17,
                'answer': 18,
                'other': 19
                }
        self.index_class_map=[
                'header',
                'question',
                'answer',
                'other'
                ]





    def parseAnn(self,annotations,s):
        #if useBlankClass:
        #    numClasses+=1
        #if usePairedClass:
        #    numClasses+=1

        numClasses=len(self.classMap)
        if self.split_to_lines:
            bbs, numNeighbors, trans, groups = createLines(annotations,self.classMap,s)
        else:
            boxes = annotations['form']
            bbs = np.empty((1,len(boxes), 8+8+numClasses), dtype=np.float32) #2x4 corners, 2x4 cross-points, n classes
            #pairs=set()
            numNeighbors=[]
            trans=[]
            for j,boxinfo in enumerate(boxes):
                lX,tY,rX,bY = boxinfo['box']
                h=bY-tY
                w=rX-lX
                if h/w>5 and self.rotate: #flip labeling, since FUNSD doesn't label verticle text correctly
                    #I don't know if it needs rotated clockwise or countercw, so I just say countercw
                    bbs[:,j,0]=lX*s
                    bbs[:,j,1]=bY*s
                    bbs[:,j,2]=lX*s
                    bbs[:,j,3]=tY*s
                    bbs[:,j,4]=rX*s
                    bbs[:,j,5]=tY*s
                    bbs[:,j,6]=rX*s
                    bbs[:,j,7]=bY*s
                    #we add these for conveince to crop BBs within window
                    bbs[:,j,8]=s*(lX+rX)/2.0
                    bbs[:,j,9]=s*bY
                    bbs[:,j,10]=s*(lX+rX)/2.0
                    bbs[:,j,11]=s*tY
                    bbs[:,j,12]=s*lX
                    bbs[:,j,13]=s*(tY+bY)/2.0
                    bbs[:,j,14]=s*rX
                    bbs[:,j,15]=s*(tY+bY)/2.0
                else:
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

        word_boxes=[]
        word_trans=[]
        for entity in annotations['form']:
            for word in entity['words']:
                lX,tY,rX,bY = word['box']
                #cX = (lX+rX)/2
                #cY = (tY+bY)/2
                h = bY-tY +1
                w = rX-lX +1
                #word_boxes.append([cX,cY,0,h/2,w/2])
                bb=[None]*16
                if h/w>5 and self.rotate: #flip labeling, since FUNSD doesn't label verticle text correctly
                    #I don't know if it needs rotated clockwise or countercw, so I just say countercw
                    bb[0]=lX*s
                    bb[1]=bY*s
                    bb[2]=lX*s
                    bb[3]=tY*s
                    bb[4]=rX*s
                    bb[5]=tY*s
                    bb[6]=rX*s
                    bb[7]=bY*s
                    #w these for conveince to crop BBs within window
                    bb[8]=s*(lX+rX)/2.0
                    bb[9]=s*bY
                    bb[10]=s*(lX+rX)/2.0
                    bb[11]=s*tY
                    bb[12]=s*lX
                    bb[13]=s*(tY+bY)/2.0
                    bb[14]=s*rX
                    bb[15]=s*(tY+bY)/2.0
                else:
                    bb[0]=lX*s
                    bb[1]=tY*s
                    bb[2]=rX*s
                    bb[3]=tY*s
                    bb[4]=rX*s
                    bb[5]=bY*s
                    bb[6]=lX*s
                    bb[7]=bY*s
                    #w these for conveince to crop BBs within window
                    bb[8]=s*lX
                    bb[9]=s*(tY+bY)/2.0
                    bb[10]=s*rX
                    bb[11]=s*(tY+bY)/2.0
                    bb[12]=s*(lX+rX)/2.0
                    bb[13]=s*tY
                    bb[14]=s*(rX+lX)/2.0
                    bb[15]=s*bY
                word_boxes.append(bb)
                word_trans.append(word['text'])
        #word_boxes = torch.FloatTensor(word_boxes)
        word_boxes = np.array(word_boxes)
        #self.pairs=list(pairs)
        return bbs, list(range(bbs.shape[1])), numClasses, trans, groups, {}, {'word_boxes':word_boxes, 'word_trans':word_trans}


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

    def makeQuestions(self,bbs,transcription,groups,groups_adj):
        bbs=bbs[0]
        #get all question-answer relationships
        questions_gs=set()
        answers_gs=set()
        headers_gs=set()
        all_trans={}
        group_count = len(groups)
        for gi,group in enumerate(groups):
            cls = bbs[group[0],16:].argmax()
            trans_bb = []
            for bbi in group:
                trans_bb.append((bbs[bbi,1],transcription[bbi]))
                #if fullyKnown(transcription[bbi]): #need this for NAF
                #    trans_bb.append((bbs[bbi,1],transcription[bbi]))
                #else:
                #    continue

            trans_bb.sort(key=lambda a:a[0] )
            trans=trans_bb[0][1]
            for y,t in trans_bb[1:]:
                trans+=' '+t
            all_trans[gi]=trans
            #print('c:{} {},{} full group trans: {}'.format(cls,bbs[group[0],0],bbs[group[0],1],trans))

            if self.index_class_map[cls] == 'question':
                questions_gs.add(gi)
            elif self.index_class_map[cls] == 'answer':
                answers_gs.add(gi)
            elif self.index_class_map[cls] == 'header':
                headers_gs.add(gi)


        relationships_h_q=defaultdict(list)
        relationships_q_h={}
        for h_gi in headers_gs:
            for gi1,gi2 in groups_adj:
                if gi1 == h_gi and gi2 in questions_gs:
                    relationships_h_q[h_gi].append(gi2)
                    relationships_q_h[gi2]=h_gi
                elif gi2 == h_gi and gi1 in questions_gs:
                    relationships_h_q[h_gi].append(gi1)
                    relationships_q_h[gi1]=h_gi

        relationships_q_a=defaultdict(list)
        relationships_a_q=defaultdict(list)
        for q_gi in questions_gs:
            found=False
            for gi1,gi2 in groups_adj:
                if gi1 == q_gi and gi2 in answers_gs:
                    #q_a_pairs.append((gi1,gi2))
                    relationships_q_a[q_gi].append(gi2)
                    relationships_a_q[gi2].append(q_gi)
                    found=True
                elif gi2 == q_gi and gi1 in answers_gs:
                    #q_a_pairs.append((gi2,gi1))
                    relationships_q_a[q_gi].append(gi1)
                    relationships_a_q[gi1].append(q_gi)
                    found=True
            if not found:
                #q_a_pairs.append((q_gi,None))
                relationships_q_a[q_gi].append(None)


        #find duplicate labels and differentiate using header
        ambiguous=set()
        trans_to_gi=defaultdict(list)
        for gi,trans in all_trans.items():
            trans_to_gi[trans].append(gi)

        for trans,i_list in trans_to_gi.items():
            if len(i_list)>1:
                #print('possible ambig: {}'.format(trans))
                got=0
                for gi in i_list:
                    if gi in relationships_q_h:
                        got+=1
                        hi = relationships_q_h[gi]
                        all_trans[gi]= all_trans[hi]+' '+all_trans[gi]
                        #print('  ambig rename: {}'.format(all_trans[gi]))
                if got<len(i_list)-1:
                    ambiguous.add(trans)
                #else:
                    #print('  saved!')


        q_a_pairs=[]
        #table_map = {}
        #tables=[]
        table_values={}
        row_headers=set()
        col_headers=set()
        skip=set()

        for qi,ais in relationships_q_a.items():
            #if qi in table_map:
            #    continue
            if qi in skip:
                continue
            if len(ais)==1:
                ai=ais[0]
                if ai is None or len(relationships_a_q[ai])==1:
                    q_a_pairs.append((qi,ai))
                else:
                    #this is probably part of a table with single row/col
                    if len(relationships_a_q[ai])>2:
                        continue #too many question links, label error
                    q1,q2 = relationships_a_q[ai]
                    if q1==qi:
                        other_qi = q2
                    else:
                        other_qi = q1
                    success = addTableElement(table_values,row_headers,col_headers,ai,q1,q2,groups,bbs)
                    #if len(relationships_q_a[other_qi])>1: #be sure this is the non-single index
                    #    #assert ai not in table_map
                    #    #addTable(tables,table_map,goups,bbs,qi,ais,relationships_q_a,relationships_a_q)
                    #    addTableElement(table_values,row_headers,col_headers,ai,q1,q2,groups,bbs)
                    if not success and len(relationships_q_a[other_qi])==1:
                        #broken label?
                        gi = group_count
                        trans_bb = []
                        for qqi in [qi,other_qi]:
                            assert len(groups[qqi])==1
                            bbi = groups[qqi][0]
                            trans_bb.append((bbs[bbi,1],transcription[bbi]))
                        trans_bb.sort(key=lambda a:a[0] )
                        trans=trans_bb[0][1]
                        for y,t in trans_bb[1:]:
                            trans+=' '+t
                        all_trans[gi]=trans
                        group_count+=1
                        bb_ids = groups[qi]+groups[other_qi]
                        groups = groups+[bb_ids]

                        q_a_pairs.append((gi,ai))
                        skip.add(other_qi)
            else:
                #is this a misslabled multiline answer or a table?
                if all(len(relationships_a_q[ai])==1 for ai in ais):
                    #if must be a multiline answer
                    gi = group_count
                    trans_bb = []
                    bb_ids = []
                    for ai in ais:
                        #assert len(groups[ai])==1 this can be a list, in which  case we lose new lines
                        bbi = groups[ai][0]
                        trans_bb.append((bbs[bbi,1],transcription[bbi]))
                        bb_ids += groups[ai]
                    trans_bb.sort(key=lambda a:a[0] )
                    trans=trans_bb[0][1]
                    for y,t in trans_bb[1:]:
                        trans+=' '+t
                    all_trans[gi]=trans
                    group_count+=1
                    groups = groups+[bb_ids]

                    q_a_pairs.append((qi,gi))
                else:
                    #assert qi not in table_map
                    #addTable(tables,table_map,groups,bbs,qi,ais,relationships_q_a,relationships_a_q)
                    for ai in ais:
                        if len(relationships_a_q[ai])==2:
                            q1,q2 = relationships_a_q[ai]
                        elif len(relationships_a_q[ai])==1:
                            q1 = relationships_a_q[ai][0]
                            q2 = None
                        addTableElement(table_values,row_headers,col_headers,ai,q1,q2,groups,bbs)

        all_q_a=[]

        for qi,ai in q_a_pairs:
            trans_qi = all_trans[qi]
            #if 'Group' in trans_qi:
            #    import pdb;pdb.set_trace()
            if ai is not None:
                trans_ai = all_trans[ai]
                if trans_qi not in ambiguous:
                    all_q_a.append(('value for "{}"?'.format(trans_qi),trans_ai,[qi,ai]))
                if trans_ai not in ambiguous:
                    all_q_a.append(('label of "{}"?'.format(trans_ai),trans_qi,[qi,ai]))
            elif trans_qi not in ambiguous:
                all_q_a.append(('value for "{}"?'.format(trans_qi),'blank',[qi]))

        #addTable can cause two tables to be made in odd cases (uneven rows, etc), so we'll simply combine all the table information and generate questions from it.
        #print(tables)
        #for table in tables:
        #    for row_h in table['row_headers']:
        #        if row_h in ambiguous:
        #            continue
        #        q='row for "{}"?'.format(all_trans[row_h])
        #        a=''
        #        for col_h in table['col_headers']:
        #            v = table['values'][(col_h,row_h)]
        #            a+='{}: {}, '.format(all_trans[col_h],all_trans[v])
        #            if v not in ambiguous:
        #                all_q_a.append(('row that "{}" is in?'.format(all_trans[v]),all_trans[row_h]))
        #        a=a[:-2]#remove last ", "
        #        all_q_a.append((q,a))
        #    for col_h in table['col_headers']:
        #        if col_h in ambiguous:
        #            continue
        #        q='column for "{}"?'.format(all_trans[col_h])
        #        a=''
        #        for row_h in table['row_headers']:
        #            v = table['values'][(col_h,row_h)]
        #            a+='{}: {}, '.format(all_trans[row_h],all_trans[v])
        #            if v not in ambiguous:
        #                all_q_a.append(('column that "{}" is in?'.format(all_trans[v]),all_trans[col_h]))
        #        a=a[:-2]#remove last ", "
        #        all_q_a.append((q,a))

        #    for (col_h,row_h),v in table['values'].items():
        #        all_q_a.append(('value of "{}" and "{}"'.format(all_trans[row_h],all_trans[col_h]),all_trans[v]))
        #        all_q_a.append(('value of "{}" and "{}"'.format(all_trans[col_h],all_trans[row_h]),all_trans[v]))

        #we we'll aggregate the information and just make the questions
        col_vs=defaultdict(list)
        row_vs=defaultdict(list)
        for (col_h,row_h),v in table_values.items():
            if col_h is not None and row_h is not None:
                all_q_a.append(('value of "{}" and "{}"?'.format(all_trans[row_h],all_trans[col_h]),all_trans[v],[col_h,row_h,v]))
                all_q_a.append(('value of "{}" and "{}"?'.format(all_trans[col_h],all_trans[row_h]),all_trans[v],[col_h,row_h,v]))
            if all_trans[v] not in ambiguous:
                if row_h is not None:
                    all_q_a.append(('row that "{}" is in?'.format(all_trans[v]),all_trans[row_h],[v,row_h]))
                if col_h is not None:
                    all_q_a.append(('column that "{}" is in?'.format(all_trans[v]),all_trans[col_h],[v,col_h]))

            x,y = bbs[groups[v][0],0:2]
            if col_h is not None:
                col_vs[col_h].append((v,y))
            if row_h is not None:
                row_vs[row_h].append((v,x))

        for row_h, vs in row_vs.items():
            trans_row_h = all_trans[row_h]
            if trans_row_h not in ambiguous:
                vs.sort(key=lambda a:a[1])
                a=all_trans[vs[0][0]]
                for v,x in vs[1:]:
                    a+=', '+all_trans[v]
                all_q_a.append(('all values in row {}?'.format(trans_row_h),a,[row_h,vs[0][0]]))
        for col_h, vs in col_vs.items():
            trans_col_h = all_trans[col_h]
            if trans_col_h not in ambiguous:
                vs.sort(key=lambda a:a[1])
                a=all_trans[vs[0][0]]
                for v,y in vs[1:]:
                    a+=', '+all_trans[v]
                all_q_a.append(('all values in column {}?'.format(trans_col_h),a,[col_h,vs[0][0]]))

        new_all_q_a =[]
        for q,a,group_ids in all_q_a:
            bb_ids=[]
            for gid in group_ids:
                bb_ids+=groups[gid]
            new_all_q_a.append((q,a,bb_ids))
        return new_all_q_a


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

def addTable(tables,table_map,groups,bbs,qi,ais,relationships_q_a,relationships_a_q):
    other_qis=[]
    for ai in ais:
        if len(relationships_a_q[ai])==2:
            q1,q2 = relationships_a_q[ai]
            if q1==qi:
                cls = bbs[groups[q2],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q2][0],0:2]
                other_qis.append((q2,x,y))
            else:
                cls = bbs[groups[q1],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q1][0],0:2]
                other_qis.append((q1,x,y))
        else:
            assert len(relationships_a_q[ai])==1 #blank row/column header. Skipping for now

    other_set = set(q[0] for q in other_qis)
    if len(other_set)<len(other_qis):
        import pdb;pdb.set_trace()
        return #label error
    
    my_qis=[]
    debug_hit=False
    for ai in relationships_q_a[other_qis[0][0]]:
        if len(relationships_a_q[ai])==2:
            q1,q2 = relationships_a_q[ai]
            if q1==other_qis[0][0]:
                if q2 in other_set:
                    return
                cls = bbs[groups[q2],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q2][0],0:2]
                my_qis.append((q2,x,y))
                if q2==qi:
                    debug_hit=True
            else:
                if q1 in other_set:
                    return
                cls = bbs[groups[q1],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q1][0],0:2]
                my_qis.append((q1,x,y))
                if q1==qi:
                    debug_hit=True
        else:
            assert len(relationships_a_q[ai])==1
    assert debug_hit


    #which are rows, which are cols?
    other_mean_x = np.mean([q[1] for q in other_qis])
    other_mean_y = np.mean([q[2] for q in other_qis])
    my_mean_x = np.mean([q[1] for q in my_qis])
    my_mean_y = np.mean([q[2] for q in my_qis])

    if my_mean_x<other_mean_x and my_mean_y>other_mean_y:
        #my is row headers
        my_qis.sort(key=lambda a:a[2]) #sort by y
        other_qis.sort(key=lambda a:a[1]) #sort by x
        row_hs = [q[0] for q in my_qis]
        col_hs = [q[0] for q in other_qis]
        
    elif my_mean_x>other_mean_x and my_mean_y<other_mean_y:
        #my is col headers
        my_qis.sort(key=lambda a:a[1]) #sort by x
        other_qis.sort(key=lambda a:a[2]) #sort by y
        col_hs = [q[0] for q in my_qis]
        row_hs = [q[0] for q in other_qis]
    else:
        assert False, 'unknown case'


    values={}
    for row_h in row_hs:
        vs = relationships_q_a[row_h]
        for v in vs:
            try:
                q1,q2 = relationships_a_q[v]
                if q1==row_h:
                    col_h=q2
                else:
                    col_h=q1
                values[(col_h,row_h)] = v
            except ValueError:
                pass

    table = {
            "row_headers": row_hs,
            "col_headers": col_hs,
            "values": values
            }
    for row_h in row_hs:
        #assert row_h not in table_map
        table_map[row_h]=len(tables)
    for col_h in col_hs:
        #assert col_h not in table_map
        table_map[col_h]=len(tables)
    for v in values.values():
        #assert v not in table_map
        table_map[v]=len(tables)
    tables.append(table)

def addTableElement(table_values,row_headers,col_headers,ai,qi1,qi2,groups,bbs,threshold=5):
    ele_x,ele_y = bbs[groups[ai][0],0:2]
    q1_x,q1_y = bbs[groups[qi1][0],0:2]
    x_diff_1 = abs(ele_x-q1_x)
    y_diff_1 = abs(ele_y-q1_y)
    if qi2 is not None:
        #which question is the row, which is the header?
        q2_x,q2_y = bbs[groups[qi2][0],0:2]
        x_diff_2 = abs(ele_x-q2_x)
        y_diff_2 = abs(ele_y-q2_y)

        if abs(q1_x-q2_x)<threshold or abs(q1_y-q2_y)<threshold:
            return False

        if (x_diff_1<y_diff_1 or y_diff_2<x_diff_2) and y_diff_1>threshold and x_diff_2>threshold:
            row_h = qi2
            col_h = qi1
        elif (x_diff_2<y_diff_2 or y_diff_1<x_diff_1) and y_diff_2>threshold and x_diff_1>threshold:
            row_h = qi1
            col_h = qi2
        else:
            #IDK
            #import pdb;pdb.set_trace()
            return False
        
        table_values[(col_h,row_h)]=ai
        row_headers.add(row_h)
        col_headers.add(col_h)
    else:
        if x_diff_1>y_diff_1:
            row_headers.add(qi1)
            table_values[(None,qi1)]=ai
        elif x_diff_1<y_diff_1:
            col_headers.add(qi1)
            table_values[(qi1,None)]=ai
    return True
