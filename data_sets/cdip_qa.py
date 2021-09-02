import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate

import utils.img_f as img_f


class CDIPQA(ParaQADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(CDIPQA, self).__init__(dirPath,split,config,images)

        self.cache_resized = False
        #NEW the document must have a block_score above thresh for anything useing blocks (this is newline following too)
        self.block_score_thresh = 0.73 #eye-balled this one


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
                if split in readFile:
                    subdirs = readFile[split]
                    toUse=[]
                    for subdir in subdirs:
                        with open(os.path.join(dirPath,subdir+'.list')) as lst:
                            toUse += [path.strip() for path in lst.readlines()]
                    imagesAndAnn = []
                    for path in toUse:#['images']:
                        try:
                            name = path[path.rindex('/')+1:]
                        except ValueError:
                            name = path
                        imagesAndAnn.append( (name,os.path.join(dirPath,path+'.png'),os.path.join(dirPath,path+'.json')) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit(1)
            self.images=[]
            for imageName,imagePath,jsonPath in imagesAndAnn:
                if os.path.exists(jsonPath):
                    org_path = imagePath
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName+'.png')
                    else:
                        path = org_path

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
                    self.images.append({'id':imageName, 'imageName':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale })
                else:
                    print('{} does not exist'.format(jsonPath))
                    print('No json found for {}'.format(imagePath))
                    #exit(1)
        self.errors=[]




    def parseAnn(self,ocr,s):
        image_h=ocr['height']
        image_w=ocr['width']
        ocr=ocr['blocks']

        block_score_sum=0
        line_count=0
        for block in ocr:
            t,l,b,r = block['box']
            h=b-t
            w=r-l
            if w==0 or h==0:
                continue
            squareness = min(0.4,h/w)
            area_whole = h*w
            area_covered = 0 #we'll assume lines don't overlap
            num_lines=0
            for para in block['paragraphs']:
                for line in para['lines']:
                    num_lines+=1
                    for word in line['words']:
                        top,left,bottom,right = word['box']
                        height = bottom-top
                        width = right-left
                        area_covered+=height*width
            if num_lines>1:
                area_score = area_covered/area_whole
            else:
                area_score = 0
            total_score = area_score+squareness
            block_score_sum += total_score*num_lines
            line_count += num_lines
        block_score = block_score_sum/line_count
        use_blocks = block_score>self.block_score_thresh
        #print('block_score: {} {}'.format(block_score,'good!' if use_blocks else 'bad'))
        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s,use_blocks)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, {}, {}, qa

