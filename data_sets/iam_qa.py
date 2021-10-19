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
from data_sets.para_qa_dataset import ParaQADataset, collate

import utils.img_f as img_f


class IAMQA(ParaQADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(IAMQA, self).__init__(dirPath,split,config,images)

        self.crop_to_data=True
        split_by = 'rwth'
        self.cache_resized = False
        #NEW the document must have a block_score above thresh for anything useing blocks (this is newline following too)
        self.block_score_thresh = 0.73 #eye-balled this one


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
            rescale=1.0
            self.images=[]
            for name in doc_set:
                xml_path = os.path.join(dirPath,'xmls',name+'.xml')
                image_path = os.path.join(dirPath,'forms',name+'.png')
                self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':xml_path, 'rescaled':rescale })





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
        for words,line in zip(W_lines,lines):
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

    def parseAnn(self,xmlfile,s):
        W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xmlfile)
        #W_lines is list of lists
        # inner list has ([minY,maxY,minX,maxX],text,id) id=gt for NER

        crop_x,crop_y = self.current_crop
        self.current_crop=None
        maxX=0
        maxY=0
        minX=image_w
        minY=image_h
        ocr_lines=[]
        for words,line in zip(W_lines,lines):
            ocr_words=[]
            for word in words:
                ocr_word={'box':[word[0][2]-crop_x,word[0][0]-crop_y,word[0][3]-crop_x,word[0][1]-crop_y],
                      'text':word[1]}
                ocr_words.append(ocr_word)
                minX = min(minX,word[0][2])
                minY = min(minY,word[0][0])
                maxX = max(maxX,word[0][3])
                maxY = max(maxY,word[0][1])
                        
            ocr_line = {'box':[line[0][2]-crop_x,line[0][0]-crop_y,line[0][3]-crop_x,line[0][1]-crop_y],
                    'text':line[1],
                    'words':ocr_words}
            ocr_lines.append(ocr_line)
        ocr=[{'paragraphs':[{'lines':ocr_lines}],
              'box': [minX-crop_x,minY-crop_y,maxX-crop_x,maxY-crop_y]
              }]



        use_blocks = False
        #print('block_score: {} {}'.format(block_score,'good!' if use_blocks else 'bad'))
        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s,use_blocks)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, {}, {}, qa

