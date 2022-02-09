import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines, fixNAF
import timeit
from data_sets.form_qa import FormQA,collate, Line, Entity, Table
from utils.forms_annotations import fixAnnotations

from utils import img_f



class NAFQA(FormQA):
    """
    Class for reading forms dataset and preping for FormQA format
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(NAFQA, self).__init__(dirPath,split,config,images)



        self.min_start_read = 7

        self.extra_np = 0.05

        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                split_file = 'overfit_split.json'
            else:
                split_file = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,split_file)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                read_file = json.loads(f.read())
                if type(split) is str:
                    to_use = read_file[split]
                    imagesAndAnn = []
                    imageDir = os.path.join(dirPath,to_use['root'],'images')
                    annDir = os.path.join(dirPath,to_use['root'],'annotations')
                    for name in to_use['images']:
                        imagesAndAnn.append( (name+'.png',os.path.join(imageDir,name+'.png'),os.path.join(annDir,name+'.json')) )
                elif type(split) is list:
                    imagesAndAnn = []
                    for spstr in split:
                        to_use = read_file[spstr]
                        imageDir = os.path.join(dirPath,to_use['root'],'images')
                        annDir = os.path.join(dirPath,to_use['root'],'annotations')
                        for name in to_use['images']:
                            imagesAndAnn.append( 
                                (name+'.png',os.path.join(imageDir,name+'.png'),
                                os.path.join(annDir,name+'.json')) 
                                )
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
            self.images=[]
            for image_name,imagePath,jsonPath in imagesAndAnn:
                ##DEBUG
                #if '0011973451' not in image_name:
                #    continue
                #if len(self.images)==0 and '01073843' not in image_name:
                #    continue
                ##DEBUG

                org_path = imagePath
                if self.cache_resized:
                    path = os.path.join(self.cache_path,image_name)
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
                    if self.train:
                        self.images.append({'id':image_name, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':image_name[:image_name.rfind('.')]})
                    else:
                        assert self.rescale_range[1]==self.rescale_range[0]
                        assert self.questions==1
                        #create questions for each image
                        with open(jsonPath) as f:
                            annotations = json.load(f)
                        s=1
                        bbs, numNeighbors, trans, groups = createLines(annotations,self.classMap,s)
                        bbs = bbs[0]

                        groups_adj=set()
                        for group_id,entity in enumerate(annotations['form']):
                            for linkIds in entity['linking']:
                                groups_adj.add((min(linkIds),max(linkIds)))
                                
                        entities,entity_link,tables = self.prepareForm(bbs,trans,groups,groups_adj)
                        full_entities,full_entity_link_dict = self.prepareFormRaw(bbs,trans,groups,groups_adj)
                        qa = self.makeQuestions(self.rescale_range[1],entities,entity_link,tables,full_entities,full_entity_link_dict)
                        #import pdb;pdb.set_trace()
                        for _qa in qa:
                            _qa['bb_ids']=None
                            self.images.append({'id':image_name, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':image_name[:image_name.rfind('.')], 'qa':[_qa]})
        self.errors=[]



    def parseAnn(self,annotations,s):

        all_entities,entity_link,tables,proses = self.getEntitiesAndSuch(annotations)

        #run through all entites to build bbs, assign bbid, and find ambiguity
        boxes = []
        text_line_counts = defaultdict(list)
        for ei,entity in enumerate(all_entities):
            for li,line in enumerate(entity.lines):
                text = self.punc_regex.sub('',line.text.lower())
                text_line_counts[text].append((ei,li))
                bbid = len(boxes)
                boxes.append(self.convertBB(s,line.box))
                line.bbid = bbid

        bbs = np.array(boxes)

        #assign ambiguity
        for line_ids in text_line_counts.values():
            if len(line_ids)>1:
                for ei,li in line_ids:
                    all_entities[ei].lines[li].ambiguous = True

        link_dict=entity_link
        entity_link=[(e1,list(e2s) if e2s is not None else None) for e1,e2s in link_dict.items()]
        #now set up a full linking dictionary
        for e1,e2s in entity_link:
            if e2s is not None:
                for e2 in e2s:
                    if e2 is not None:
                        if link_dict[e2] is None:
                            link_dict[e2]=[]
                        link_dict[e2].append(e1)
            elif link_dict[e1] is None or len(link_dict[e1])==0:
                del link_dict[e1]
        #Add all the link for tables
        for table in tables:
            for r,r_header in enumerate(table.row_headers):
                r_index = all_entities.index(r_header)
                for c,c_header in enumerate(table.col_headers):
                    c_index = all_entities.index(c_header)
                    v=table.cells[r][c]
                    if v is not None:
                        v_index = all_entities.index(v)
                        link_dict[r_index].append(v_index)
                        link_dict[c_index].append(v_index)
                        link_dict[v_index].append(r_index)
                        link_dict[v_index].append(c_index)
        #actually prose will already have links in entity_link
        #for prose in proses:
        #    prev_idx = all_entities.index(prose.entities[0])
        #    for entity in prose.entities[1:]:
        #        this_idx = all_entities.index(entity)
        #        link_dict[prev_idx] = this_idx
        #        link_dict[this_idx] = prev_idx
                

        link_dict = self.sortLinkDict(all_entities,link_dict)

        if self.train:
            qa = self.makeQuestions(1.0,all_entities,entity_link,tables,all_entities,link_dict)
        else:
            qa = None #This is pre-computed

        ocr=None
        #ocr = [self.corrupt(text) for text in ocr]
        return bbs, list(range(bbs.shape[0])), ocr, {}, {}, qa


    def convertBB(self,s,box):
        assert s==1
        if isinstance(box,list):
            return box
        return box.tolist()



    def getEntitiesAndSuch(self,annotations):
        fixAnnotations(None,annotations) #fixes somethings
        #we need to both produce the json and the entities + links + tables

        all_bbs = annotations['byId']#annotations['fieldBBs']+annotations['textBBs']
        #all_bbs = {bb['id']:bb for bb in all_bbs}
        for bb in all_bbs.values():
            bb['poly_points'] = np.array(bb['poly_points'])
        all_pairs = annotations['pairs']#+annotations['samePairs']
        transcriptions = annotations['transcriptions']
        
        #group paragraphs / multiline
        #I think this is whenever they have the same type
        #also para
        groups = defaultdict(list)
        group_num=0
        id_to_group = defaultdict(lambda: None)
        for id1,id2 in all_pairs:
            if all_bbs[id1]['type']==all_bbs[id2]['type'] or ('P' in all_bbs[id1]['type'] and 'P' in all_bbs[id2]['type']):
                assert 'Row' not in all_bbs[id1]['type'] and 'Row' not in all_bbs[id2]['type']
                assert 'Col' not in all_bbs[id1]['type'] and 'Col' not in all_bbs[id2]['type']
                group1 = id_to_group[id1]
                group2 = id_to_group[id2]
                if group1 is None and group2 is None:
                    groups[group_num] = putInReadOrder(id1,all_bbs[id1]['poly_points'],id2,all_bbs[id2]['poly_points'])
                    id_to_group[id1]=group_num
                    id_to_group[id2]=group_num
                    group_num+=10
                elif group1 is None:
                    groups[group2].append(id1)
                    id_to_group[id1]=group2
                elif group2 is None:
                    groups[group1].append(id2)
                    id_to_group[id2]=group1
                elif group1 != group2:
                    #merge
                    for idx in groups[group2]:
                        id_to_group[idx] = group1
                    #append in read order
                    order_a = putInReadOrder(1,groups[group1][0],2,groups[group2][-1])
                    order_b = putInReadOrder(1,groups[group1][-1],2,groups[group2][0])
                    assert order_a[0] == order_b[0]
                    if order_a[0]==1:
                        groups[group1]+=groups[group2]
                    else:
                        groups[group1]=groups[group2]+groups[group1]
                    del groups[group2]
                #else nothing needed

        #put all unsed bbs in their own groups
        used_ids = set()
        e_groups = []
        entities = []
        proses = []
        prose_groups = []

        for group in groups:
            if all(all_bbs[group[0]]['type'] == all_bbs[bb_id]['type'] for bb_id in group[1:]):
                #same class
                if 'Circle' in all_bbs[group[0]]['type']:
                    cls = 'circle'
                elif 'text' in all_bbs[group[0]]['type']:
                    cls = 'question'
                elif 'field' in all_bbs[group[0]]['type']:
                    cls = 'answer'
                else:
                    assert False
                lines=[]
                for bb_id in group:
                    lines.append(Line(transcriptions[bb_id],all_bbs[bb_id]['poly_points']))
                entities.append(Entity(cls,lines))
                e_groups.append(group)
            else:
                #should be para
                assert all('P' in all_bbs[bb_id]['type'] for bb_id in group)
                p_entities = []
                lines = []
                prev_type = all_bbs[group[0]]['type']
                cls = 'question' if 'text' in prev_type else 'answer'
                for bb_id in group:
                    l = Line(transcriptions[bb_id],all_bbs[bb_id]['poly_points'  ])
                    if all_bbs[bb_id]['type']==cls:
                        lines.append(l)
                    else:
                        p_entities.append(Entity(cls,lines))
                        lines=[l]
                        prev_type = all_bbs[bb_id]['type']
                        cls = 'question' if 'text' in prev_type else 'answer'
                p_entities.append(Entity(cls,lines))
                proses.append(FillInProse(p_entities))
                prose_groups.append(group)




##################################

#helper functions
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


def intersection(bb1,bb2):
    points = bb1['poly_points']
    if bb1['type']=='fieldRow':
        line1 = np.array([(points[0]+points[3])/2,(points[1]+points[2])/2])
    else:
        line1 = np.array([(points[0]+points[1])/2,(points[3]+points[2])/2])
    points = bb2['poly_points']
    if bb2['type']=='fieldRow':
        line2 = np.array([(points[0]+points[3])/2,(points[1]+points[2])/2])
    else:
        line2 = np.array([(points[0]+points[1])/2,(points[3]+points[2])/2])
    return lineIntersection(line1,line2) is not None

def getAngle(poly):
    p1 = (poly[0]+poly[3])/2
    p2 = (poly[1]+poly[2])/2
    return math.atan2(p2[1]-p1[1],p2[0]-p1[0])
def getHorzReadPosition(poly,angle):
    new_poly = np.array([poly[3],poly[0],poly[1],poly[2]])
    return getVertReadPosition(new_poly)
def getVertReadPosition(poly,angle):
    if angle is None:
        angle = getAngle(poly)
    angle *= -1
    slope = -math.tan(angle)
    #xc,yc = self.center_point
    xc = poly[::2].mean()
    yc = poly[1::2].mean()

    if math.isinf(slope):
        if angle==math.pi/2:
            return xc
        elif angle==-math.pi/2:
            return -xc
        else:
            assert(False)
    elif slope == 0:
        if angle==0:
            return yc
        elif angle==-math.pi:
            return -yc
        else:
            assert(False)
    else:
        b=yc-slope*xc

        #we define a parameteric line which defines the read direction (perpediculat to this text lines slope) and find the location parametrically. Smaller values of t are earlier in read order
        #x = +/- sqrt(1/(1+1/slope**2)) * t
        #y = +/- sqrt(1/(1+slope**2)) * t
        #The +/- must be determined using the actual slope

        if angle<math.pi/2 and angle>=0:
            sign_x=1
            sign_y=1
        elif angle>math.pi/2: # and angle<=math.pi:
            sign_x=1
            sign_y=-1
        elif angle>-math.pi/2 and angle<0:
            sign_x=-1
            sign_y=1
        elif angle<-math.pi/2:
            sign_x=-1
            sign_y=-1
        else:
            assert(False)



        t = b/(sign_y*math.sqrt(1/(1+slope**2))-slope*sign_x*math.sqrt(1/(1+(1/slope**2))))
        return t.item()
def putInReadOrder(a,a_poly,b,b_poly,final=False):
    angle = (getAngle(a_poly) + getAngle(b_poly))/2

    height_a = getHeight(a_poly)
    pos_a = getVertReadPosition(a_poly, angle)
    height_b = getHeight(b_poly)
    pos_b = getVertReadPosition(b_poly, angle)

    diff = pos_b-pos_a
    if final or abs(diff)<0.5*(height_a+height_b)/2:
        if diff>0:
            return [a,b]
        else:
            return [b,a]
    else:
        new_poly_a = np.array([poly_a[3],poly_a[0],poly_a[1],poly_a[2]])
        new_poly_b = np.array([poly_b[3],poly_b[0],poly_b[1],poly_b[2]])
        return putInPlace(a,new_poly_a,b,new_poly_b,final=True)

