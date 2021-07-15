import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from .qa import QADataset

import utils.img_f as img_f

def collate(batch):
    return {
            'img': torch.cat([b['img'] for b in batch],dim=0),
            'bb_gt': [b['bb_gt'] for b in batch], #torch.cat([b['bb_gt'] for b in batch],dim=0),
            'imgName': [b['imgName'] for b in batch],
            'scale': [b['scale'] for b in batch],
            'cropPoint': [b['cropPoint'] for b in batch],
            'transcription': [b['transcription'] for b in batch],
            'metadata': [b['metadata'] for b in batch],
            'form_metadata': [b['form_metadata'] for b in batch],
            'questions': [b['questions'] for b in batch],
            'answers': [b['answers'] for b in batch]
            }


class FUNSDQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FUNSDQA, self).__init__(dirPath,split,config,images)

        self.only_types=None

        self.split_to_lines = config['split_to_lines']
        self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0.15
        self.do_words = config['do_words']
        self.char_qs = config['char_qs'] if 'char_qs' in config else False

        self.min_start_read = 7

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
            bbs = bbs[0]
        else:
            boxes = annotations['form']
            bbs = np.empty((len(boxes), 8+8+numClasses), dtype=np.float32) #2x4 corners, 2x4 cross-points, n classes
            #pairs=set()
            numNeighbors=[]
            trans=[]
            for j,boxinfo in enumerate(boxes):
                lX,tY,rX,bY = boxinfo['box']
                h=bY-tY
                w=rX-lX
                if h/w>5 and self.rotate: #flip labeling, since FUNSD doesn't label verticle text correctly
                    #I don't know if it needs rotated clockwise or countercw, so I just say countercw
                    bbs[j,0]=lX*s
                    bbs[j,1]=bY*s
                    bbs[j,2]=lX*s
                    bbs[j,3]=tY*s
                    bbs[j,4]=rX*s
                    bbs[j,5]=tY*s
                    bbs[j,6]=rX*s
                    bbs[j,7]=bY*s
                    #we add these for conveince to crop BBs within window
                    bbs[j,8]=s*(lX+rX)/2.0
                    bbs[j,9]=s*bY
                    bbs[j,10]=s*(lX+rX)/2.0
                    bbs[j,11]=s*tY
                    bbs[j,12]=s*lX
                    bbs[j,13]=s*(tY+bY)/2.0
                    bbs[j,14]=s*rX
                    bbs[j,15]=s*(tY+bY)/2.0
                else:
                    bbs[j,0]=lX*s
                    bbs[j,1]=tY*s
                    bbs[j,2]=rX*s
                    bbs[j,3]=tY*s
                    bbs[j,4]=rX*s
                    bbs[j,5]=bY*s
                    bbs[j,6]=lX*s
                    bbs[j,7]=bY*s
                    #we add these for conveince to crop BBs within window
                    bbs[j,8]=s*lX
                    bbs[j,9]=s*(tY+bY)/2.0
                    bbs[j,10]=s*rX
                    bbs[j,11]=s*(tY+bY)/2.0
                    bbs[j,12]=s*(lX+rX)/2.0
                    bbs[j,13]=s*tY
                    bbs[j,14]=s*(rX+lX)/2.0
                    bbs[j,15]=s*bY
                
                bbs[j,16:]=0
                if boxinfo['label']=='header':
                    bbs[j,16]=1
                elif boxinfo['label']=='question':
                    bbs[j,17]=1
                elif boxinfo['label']=='answer':
                    bbs[j,18]=1
                elif boxinfo['label']=='other':
                    bbs[j,19]=1
                #for id1,id2 in boxinfo['linking']:
                #    pairs.add((min(id1,id2),max(id1,id2)))
                trans.append(boxinfo['text'])
                numNeighbors.append(len(boxinfo['linking']))
            groups = [[n] for n in range(len(boxes))]

        word_boxes=[]
        word_trans=[]
        word_groups=[]
        groups_adj=set()
        for group_id,entity in enumerate(annotations['form']):
            for linkIds in entity['linking']:
                groups_adj.add((min(linkIds),max(linkIds)))
                
            group=[]
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
                group.append(len(word_boxes))
                word_boxes.append(bb)
                word_trans.append(word['text'])
            word_groups.append(word_groups)
        #word_boxes = torch.FloatTensor(word_boxes)
        word_boxes = np.array(word_boxes)
        assert len(groups)==len(word_groups) #this should be identicle in alginment
        #self.pairs=list(pairs)

        if self.do_words:
            ocr = word_trans
            ocr_bbs = word_boxes
            ocr_groups = word_groups
        else:
            ocr = trans
            ocr_bbs = bbs[:,:16]
            ocr_groups = groups

        qa = self.makeQuestions(bbs,trans,groups,groups_adj,ocr_groups)

        ocr = [self.corrupt(text) for text in ocr]
        return ocr_bbs, list(range(ocr_bbs.shape[0])), ocr, {}, {}, qa


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

    def makeQuestions(self,bbs,transcription,groups,groups_adj,groups_id):
        all_q_a=[] #question-answers go here

        questions_gs=set()
        answers_gs=set()
        headers_gs=set()
        others_gs=set()
        all_trans={}
        group_count = len(groups)
        for gi,group in enumerate(groups):
            cls = bbs[group[0],16:].argmax()
            trans_bb = []
            class_qs=[]
            class_answer = '[ '+self.index_class_map[cls]+' ]'
            for bbi in group:
                trans_bb.append((bbs[bbi,1],transcription[bbi]))
                #if fullyKnown(transcription[bbi]): #need this for NAF
                #    trans_bb.append((bbs[bbi,1],transcription[bbi]))
                #else:
                #    continue
                if self.char_qs=='full':
                    #classify individual lines
                    class_qs.append(('cs~{}'.format(transcription[bbi]),class_answer,[gi])) #This can be ambigous, although generally the same text has the same class

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
            else:
                others_gs.add(gi)

            if self.char_qs=='full':
                #classify all together
                class_qs.append(('cs~{}'.format(trans),class_answer,[gi])) #This can be ambigous, although generally the same text has the same class

                all_q_a.append(random.choice(class_qs))

                #complete (read)
                if len(trans)>2:
                    if len(trans)>self.min_start_read+1:
                        start_point = random.randrange(self.min_start_read,len(trans)-1)
                    else:
                        start_point = random.randrange(len(trans)//2,len(trans)-1)
                    start_text = trans[:start_point]
                    finish_text = trans[start_point:]
                    all_q_a.append(('re~{}'.format(start_text),finish_text,[gi]))


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
                typ = bbs[groups[i_list[0]][0],16:].argmax()
                amb=True
                for i in i_list[1:]:
                    bbi = groups[i][0]
                    if bbs[bbi,16:].argmax() != typ:
                        amb=False
                        break
                if not amb:
                    break
                #import pdb;pdb.set_trace()
                #print('possible ambig: {}'.format(trans))
                got=0
                #for gi in i_list:
                #    if gi in relationships_q_h:
                #        got+=1
                #        hi = relationships_q_h[gi]
                #        all_trans[gi]= all_trans[hi]+' '+all_trans[gi]
                #        #print('  ambig rename: {}'.format(all_trans[gi]))
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

                        groups_id.append(groups_id[qi]+groups_id[other_qi])

                        q_a_pairs.append((gi,ai))
                        skip.add(other_qi)
            else:
                #is this a misslabled multiline answer or a table?
                if all(len(relationships_a_q[ai])==1 for ai in ais):
                    #if must be a multiline answer
                    gi = group_count
                    trans_bb = []
                    bb_ids = []
                    real_ids = []
                    for ai in ais:
                        #assert len(groups[ai])==1 this can be a list, in which  case we lose new lines
                        bbi = groups[ai][0]
                        trans_bb.append((bbs[bbi,1],transcription[bbi]))
                        bb_ids += groups[ai]
                        real_ids += groups_id[ai]
                    trans_bb.sort(key=lambda a:a[0] )
                    trans=trans_bb[0][1]
                    for y,t in trans_bb[1:]:
                        trans+=' '+t
                    all_trans[gi]=trans
                    group_count+=1
                    groups = groups+[bb_ids]

                    groups_id.append(real_ids)

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


        for qi,ai in q_a_pairs:
            trans_qi = all_trans[qi]
            #if 'Group' in trans_qi:
            #    import pdb;pdb.set_trace()
            if ai is not None:
                trans_ai = all_trans[ai]
                if trans_qi not in ambiguous:
                    if self.char_qs:
                        all_q_a.append(('l~{}'.format(trans_qi),trans_ai,[qi,ai]))
                    else:
                        all_q_a.append(('value for "{}"?'.format(trans_qi),trans_ai,[qi,ai]))
                if trans_ai not in ambiguous:
                    if self.char_qs:
                        all_q_a.append(('v~{}'.format(trans_ai),trans_qi,[qi,ai]))
                    else:
                        all_q_a.append(('label of "{}"?'.format(trans_ai),trans_qi,[qi,ai]))
            elif trans_qi not in ambiguous:
                if self.char_qs:
                    all_q_a.append(('l~{}'.format(trans_qi),'[ blank ]',[qi]))
                else:
                    all_q_a.append(('value for "{}"?'.format(trans_qi),'blank',[qi]))


        if self.char_qs=="full":

            #Do header and qestions
            for hi,qis in relationships_h_q.items():
                trans_h = all_trans[hi]
                trans_qs=[]
                for qi in qis:
                    trans_q = all_trans[qi]
                    if len(qis)>1:
                        x=y=h=0
                        for bbi in groups[qi]:
                            x += bbs[bbi,0]
                            x += bbs[bbi,4]
                            y += bbs[bbi,1]
                            y += bbs[bbi,5]
                            h += bbs[bbi,5]-bbs[bbi,1]+1
                        xc = x/(2*len(groups[qi]))
                        yc = y/(2*len(groups[qi]))
                        h = h/len(groups[qi])
                    else:
                        xc=yc=h=-1
                    trans_qs.append((trans_q,xc,yc,h))
                    all_q_a.append(('qu~{}'.format(trans_q),trans_h,[hi,qi]))

                #Now we need to put all the questions into read order
                if len(trans_qs)>1:
                    rows=[]
                    rows_mean_y=[]
                    for trans_q,x,y,h in trans_qs:
                        row_i=None
                        for r,mean_y in enumerate(rows_mean_y):
                            if abs(mean_y-y)<h*0.6:
                                row_i=r
                                break
                        if row_i is None:
                            rows.append([(trans_q,x,y)])
                            rows_mean_y.append(y)
                        else:
                            rows[row_i].append((trans_q,x,y))
                            mean_y=0
                            for (trans_q2,x2,y2) in rows[r]:
                                mean_y+=y2
                            rows_mean_y[row_i] = mean_y/len(rows[row_i])
                    trans_qs = []
                    rows = list(zip(rows,rows_mean_y))
                    rows.sort(key=lambda a:a[1])
                    for row,mean_y in rows:
                        row.sort(key=lambda a:a[1])
                        for trans_q,x,y in row:
                            trans_qs.append(trans_q)
                    trans_qs = ', '.join(trans_qs)
                else:
                    trans_qs = trans_qs[0][0]
                all_q_a.append(('hd~{}'.format(trans_h),trans_qs,[hi]+qis))

            for gi in others_gs:
                trans_gi = all_trans[qi]
                if trans_gi not in ambiguous:
                    all_q_a.append(('l~{}'.format(trans_gi),'[ np ]',None))
                    all_q_a.append(('v~{}'.format(trans_gi),'[ np ]',None))
                    all_q_a.append(('hd~{}'.format(trans_gi),'[ np ]',None))
                    all_q_a.append(('qu~{}'.format(trans_gi),'[ np ]',None))

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
        tables={}
        header_to_table_id={}
        cur_table_id=0
        for (col_h,row_h),v in table_values.items():
            if col_h is not None and row_h is not None:
                if col_h in header_to_table_id:
                    col_tab = header_to_table_id[col_h]
                else:
                    col_tab = None
                if row_h in header_to_table_id:
                    row_tab = header_to_table_id[row_h]
                else:
                    row_tab = None

                if col_tab is None and row_tab is None:
                    tables[cur_table_id] = [[col_h],[row_h]]
                    header_to_table_id[col_h]=cur_table_id
                    header_to_table_id[row_h]=cur_table_id
                    cur_table_id+=1
                elif col_tab is None:
                    tables[row_tab][0].append(col_h)
                    header_to_table_id[col_h]=row_tab
                elif row_tab is None:
                    tables[col_tab][1].append(row_h)
                    header_to_table_id[row_h]=col_tab
                else:
                    if col_tab!=row_tab:
                        #merge tables
                        tables[col_tab][0]+=tables[row_tab][0]
                        tables[col_tab][1]+=tables[row_tab][1]
                        for h in tables[row_tab][0]+tables[row_tab][1]:
                            header_to_table_id[h]=col_tab
                        del tables[row_tab]

            if col_h is not None and row_h is not None:
                if self.char_qs:
                    all_q_a.append(('t~{}~~{}'.format(all_trans[row_h],all_trans[col_h]),all_trans[v],[col_h,row_h,v]))
                    all_q_a.append(('t~{}~~{}'.format(all_trans[col_h],all_trans[row_h]),all_trans[v],[col_h,row_h,v]))
                else:
                    all_q_a.append(('value of "{}" and "{}"?'.format(all_trans[row_h],all_trans[col_h]),all_trans[v],[col_h,row_h,v]))
                    all_q_a.append(('value of "{}" and "{}"?'.format(all_trans[col_h],all_trans[row_h]),all_trans[v],[col_h,row_h,v]))
            if all_trans[v] not in ambiguous:
                if row_h is not None:
                    if self.char_qs:
                        all_q_a.append(('ri~{}'.format(all_trans[v]),all_trans[row_h],[v,row_h]))
                    else:
                        all_q_a.append(('row that "{}" is in?'.format(all_trans[v]),all_trans[row_h],[v,row_h]))
                if col_h is not None:
                    if self.char_qs:
                        all_q_a.append(('ci~{}'.format(all_trans[v]),all_trans[col_h],[v,col_h]))
                    else:
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
                if self.char_qs:
                    all_q_a.append(('ar~{}'.format(trans_row_h),a,[row_h,vs[0][0]]))
                else:
                    all_q_a.append(('all values in row "{}"?'.format(trans_row_h),a,[row_h,vs[0][0]]))
        for col_h, vs in col_vs.items():
            trans_col_h = all_trans[col_h]
            if trans_col_h not in ambiguous:
                vs.sort(key=lambda a:a[1])
                a=all_trans[vs[0][0]]
                for v,y in vs[1:]:
                    a+=', '+all_trans[v]
                if self.char_qs: 
                    all_q_a.append(('ac~{}'.format(trans_col_h),a,[col_h,vs[0][0]]))
                else:
                    all_q_a.append(('all values in column "{}"?'.format(trans_col_h),a,[col_h,vs[0][0]]))

        if self.char_qs:
            all_q_a.append(('t#>',str(len(tables)),list(col_vs.keys())+list(row_vs.keys())))
            for i,(col_hs, row_hs) in enumerate(tables.values()):
                col_hs = [(h,bbs[groups[h][0]][0]) for h in col_hs]
                col_hs.sort(key=lambda a:a[1])
                col_hs = [h[0] for h in col_hs]
                col_h_strs = [all_trans[h] for h in col_hs]
                row_hs = [(h,bbs[groups[h][0]][1]) for h in row_hs]
                row_hs.sort(key=lambda a:a[1])
                row_hs = [h[0] for h in row_hs]
                row_h_strs = [all_trans[h] for h in row_hs]
                
                all_q_a.append(('ch~{}'.format(i),', '.join(col_h_strs),col_hs))
                all_q_a.append(('rh~{}'.format(i),', '.join(row_h_strs),row_hs))




        #Convert the group IDs on each QA pair to be BB IDs.
        #   This uses groups_id, which can be the word BB ids
        new_all_q_a =[]
        for q,a,group_ids in all_q_a:
            bb_ids=[]
            for gid in group_ids:
                bb_ids+=groups_id[gid]
            new_all_q_a.append((q,a,bb_ids))
        return new_all_q_a


    def corrupt(self,s):
        new_s=''
        for c in s:
            r = random.random()
            if r<self.corruption_p/3:
                pass
            elif r<self.corruption_p*2/3:
                new_s+=random.choice(string.ascii_letters)
            elif r<self.corruption_p:
                if random.random()<0.5:
                    new_s+=c+random.choice(string.ascii_letters)
                else:
                    new_s+=random.choice(string.ascii_letters)+c
            else:
                new_s+=c
        return new_s





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

