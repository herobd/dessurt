import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string
from collections import defaultdict, OrderedDict
import timeit
#from data_sets.form_qa import FormQA,collate, Line, Entity, Table, FillInProse, MinoredField
from data_sets.qa import QADataset, collate
from utils.forms_annotations import fixAnnotations
from utils.read_order import getVertReadPosition,getHorzReadPosition,putInReadOrder,sortReadOrder, intersection,getHeight,sameLine

from utils import img_f


SKIP=['174']#['193','194','197','200']

class NAFRead(QADataset):
    """
    Class for reading forms dataset and preping for FormQA format
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(NAFRead, self).__init__(dirPath,split,config,images)
        only = config.get('only',False)
        self.only_handwriting = only=='handwriting'
        self.only_print = only=='print'

        balance = self.train and config.get('balance',False)

        self.do_masks=True
        self.cased=True

        self.extra_np = 0.05


        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                split_file = 'overfit_split.json'
            else:
                split_file = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,split_file)) as f:
                readFile = json.loads(f.read())
                if type(split) is str:
                    groups_to_use = readFile[split]
                elif type(split) is list:
                    groups_to_use = {}
                    for spstr in split:
                        newGroups = readFile[spstr]
                        groups_to_use.update(newGroups)
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
            self.images=[]
            group_names = list(groups_to_use.keys())
            group_names.sort()
            
            #to prevent bad BBs
            assert self.rescale_range[1]==self.rescale_range[0]
            assert self.rescale_range[1]==1

            for groupName in group_names:
                imageNames=groups_to_use[groupName]
                
                if groupName in SKIP:
                    print('Skipped group {}'.format(groupName))
                    continue
                for imageName in imageNames:
                    ###DEBUG(')
                    #imageName = '004713434_00277.jpg'

                    org_path = os.path.join(dirPath,'groups',groupName,imageName)
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName)
                    else:
                        path = org_path
                    jsonPath = org_path[:org_path.rfind('.')]+'.json'
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
                        #if self.train:
                        #    name = imageName[:imageName.rfind('.')]
                        #    self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':name})
                        #
                        #else:
                        assert self.questions==1
                        #create questions for each image
                        with open(jsonPath) as f:
                            annotations = json.load(f)
                        #all_entities,entity_link,tables,proses,minored_fields,bbs,link_dict = self.getEntitiesAndSuch(annotations,rescale)
                        #qa = self.makeQuestions(self.rescale_range[1],all_entities,entity_link,tables,all_entities,link_dict,proses=proses,minored_fields=minored_fields)
                        qa = self.makeQuestions(annotations,self.rescale_range[1])
                        #import pdb;pdb.set_trace()
                        for _qa in qa:
                            _qa['bb_ids']=None
                            self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')], 'qa':[_qa]})
                            if balance:
                                if len(_qa['answer'])> 20:
                                    #add again
                                    self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')], 'qa':[dict(_qa)]})
                                if len(dict(_qa)['answer'])> 75:
                                    #add third time
                                    self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')], 'qa':[dict(_qa)]})
                                if len(dict(_qa)['answer'])> 110:
                                    #add fourth time
                                    self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')], 'qa':[dict(_qa)]})

                                #len_bin = len(_qa['answer'])//10

                                #bins[len_bin].append((_qa,imageName,path,jsonPath,rescale))
        #if balance:
            #max_len = max(len(b) for b in bins.values())
            #for l in bins:
            #    if len(bins[l])<max_len:
            #        num = max_len//len(bins[l])
            #        orig = bins[l]
            #        for n in range(nums-1):
            #            bins[l]+=orig
            #    for _qa,imageName,path,jsonPath,rescale in bins[l]:
            #        self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')], 'qa':[_qa]})



    def parseAnn(self,annotations,s):
        qa = None #This is pre-computed
        return None,None,None,None, qa

    def makeQuestions(self,annotations,scale):
        fixAnnotations(None,annotations)
        all_bbs = annotations['byId']
        qa=[]
        for bb_id,bb in all_bbs.items():
            (tlX,tlY),(trX,trY),(brX,brY),(blX,blY) = bb['poly_points']
            #typ = bb['type']
            draw_type = bb['isBlank']

            if draw_type=='blank' or (self.only_handwriting and draw_type!='handwriting' and draw_type!='signature') or (self.only_print and draw_type!='print'):
                continue

            question = 'w0>'
            if bb_id not in annotations['transcriptions']:
                continue
            response = annotations['transcriptions'][bb_id]
            if response=='' or '§' in response:
                continue

            if '¿' in response:
                count = sum(1 if c=='¿' else 0 for c in response)
                if count/len(response)>=0.1:
                    continue

            #get mid points
            lX = (tlX+blX)/2.0
            lY = (tlY+blY)/2.0
            rX = (trX+brX)/2.0
            rY = (trY+brY)/2.0
            d=np.sqrt((lX-rX)**2 + (lY-rY)**2)

            if (d==0).any():
                print('ERROR: zero length bb {}'.format(bbs[0][d[0]==0]))
                d[d==0]=1

            hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpo  se horz run
            hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transp  ose horz run
            h = (hl+hr)/2.0

            cX = (lX+rX)/2.0
            cY = (lY+rY)/2.0
            rot = np.arctan2(-(rY-lY),rX-lX)
            height = np.abs(h)    #this is half height
            width = d/2.0 #and half width


            #height[ np.logical_or(np.isnan(height),height==0) ] =1
            #width[ np.logical_or(np.isnan(width),width==0) ] =1
            topX = cX-np.sin(rot)*height
            botX = cX+np.sin(rot)*height
            topY = cY-np.cos(rot)*height
            botY = cY+np.cos(rot)*height
            leftX = lX
            leftY = lY
            rightX = rX
            rightY = rY
            inmask = [
                    [tlX*scale, tlY*scale, trX*scale, trY*scale, brX*scale, brY*scale, blX*scale, blY*scale,
                    leftX*scale,leftY*scale,rightX*scale,rightY*scale,topX*scale,topY*scale,botX*scale,botY*scale],
                    ]
            #qa.append([question,response,inmask+outmask,inmask,outmask,None])
            self.qaAdd(qa,question,response,None,inmask)

        for _qa in qa:
            assert len(_qa['in_bbs'])==1

        return qa

