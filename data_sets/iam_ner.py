import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.parseIAM import getWordAndLineBoundaries
import timeit
from data_sets.qa import QADataset, collate

import utils.img_f as img_f


class IAMNER(QADataset):
    """
    Named entity recognition task on IAM
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(IAMNER, self).__init__(dirPath,split,config,images)

        self.do_masks=True
        self.crop_to_data=True
        split_by = 'rwth'
        self.cache_resized = False

        self.current_crop=None
        self.word_id_to_cls={}

        if images is not None:
            self.images=images
        else:
            split_file = os.path.join(dirPath,'ne_annotations','iam',split_by,'iam_{}_{}_6_all.txt'.format(split,split_by))
            doc_set = set()
            with open(split_file) as f:
                lines = f.readlines()
            for line in lines:
                parts = line.split('-')
                if len(parts)>1:
                    name = '-'.join(parts[:2])
                    doc_set.add(name)

                    word_id, cls = line.strip().split(' ')
                    self.word_id_to_cls[word_id]=cls
            rescale=1.0
            self.images=[]
            for name in doc_set:
                xml_path = os.path.join(dirPath,'xmls',name+'.xml')
                image_path = os.path.join(dirPath,'forms',name+'.png')
                if self.train:
                    self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':xml_path, 'rescaled':rescale })
                else:
                    qas,bbs = self.makeQuestions(xml_path,rescale)
                    for qa in qas:
                        qa['bb_ids']=None
                        self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':xml_path, 'rescaled':rescale,'qa':[qa]})






    def getCrop(self,xmlfile):
        W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xmlfile)
        #W_lines is list of lists
        # inner list has ([minY,maxY,minX,maxX],text,id) id=gt for NER

        #We need to crop out the prompt text
        #We'll do that by cropping to only the handwriting area
        maxX=0
        maxY=0
        minX=image_w
        minY=image_h
        for words in W_lines:
            ocr_words=[]
            for word in words:
                minX = min(minX,word[0][2])
                minY = min(minY,word[0][0])
                maxX = max(maxX,word[0][3])
                maxY = max(maxY,word[0][1])
        crop = [max(0,round(minX-40)),
                max(0,round(minY-40)),
                min(image_h,round(maxX+40)),
                min(image_w,round(maxY+40))]
        self.current_crop=crop[:2]
        return crop

    def makeQuestions(self,xmlfile,s):
        W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xmlfile)
        #W_lines is list of lists
        # inner list has ([minY,maxY,minX,maxX],text,id) id=gt for NER
        if self.current_crop is None:
            self.getCrop(xmlfile)
        crop_x,crop_y = self.current_crop
        self.current_crop = None
        qas=[]
        bbs = []
        for words in W_lines:
            for word in words:
                cls = self.word_id_to_cls[word[2]]
                tY,bY,lX,rX = word[0]
                tY-=crop_y
                bY-=crop_y
                lX-=crop_x
                rX-=crop_x
                bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                            s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/ 2.0, s*bY]
                inmask = [bb]
                if self.train and random.random()<0.5:
                    q='ne>'
                    a='['+cls+']'+word[1]
                else:
                    q='ne~'+word[1]
                    a='['+cls+']'
                self.qaAdd(qas,q,a,[len(bbs)],inmask)
                bbs.append(bb)
        return qas,bbs

    def parseAnn(self,xmlfile,s):
        qas,bbs = self.makeQuestions(xmlfile,s)
        bbs = np.array(bbs)
        return bbs, list(range(bbs.shape[0])), None, {}, {}, qas

