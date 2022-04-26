import torch.utils.data
import numpy as np
import json
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils import grid_distortion
from utils.parseIAM import getWordAndLineBoundaries
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate

import utils.img_f as img_f


class IAMQA(ParaQADataset):
    """
    Class for training on IAM dataset.
    This doesn't define a specific set of tasks, rather the mode in ParaQADataset will.
    For training IAM full page/paragraph recognition use "mode": "IAM_para"
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(IAMQA, self).__init__(dirPath,split,config,images)

        self.augment_shade = config['augment_shade'] if 'augment_shade' in config else True
        
        self.crop_to_data=True
        self.warp_lines = config.get('warp_lines',0.999 if self.train else None) 
        split_by = config.get('data_split','rwth') #custom / standard
        self.cache_resized = False


        if images is not None:
            self.images=images
        if split_by in ['standard','Coquenet']:
            split_file = './data_sets/iam_{}_splits.json'.format(split_by)
            with open(split_file) as f:
                splits = json.load(f)
            doc_set = splits[split]
        else:
            #split of IAM NER
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




    #This gets the crop for the handwriting regiong (else the model can cheat and read the text prompt at the top of the page)
    def getCropAndLines(self,xmlfile,shape):
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
                round(maxX+40),
                round(maxY+40)]
        self.current_crop=crop[:2]

        crop_x,crop_y = self.current_crop
        line_bbs=[]
        for line in lines:
            line_bbs.append([line[0][2]-crop_x,line[0][0]-crop_y,line[0][3]-crop_x,line[0]  [1]-crop_y])
        return crop, line_bbs


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
        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s,use_blocks)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, None, qa

    def doLineWarp(self,img,bbs):
        pad=5
        std = (random.random()*1.5) + 1.5 #warp document more or less randomly (a std has a disticntive "look")
        for x1,y1,x2,y2 in bbs:
            sub_img = img[y1-pad:y2+pad,x1-pad:x2+pad]
            sub_img = grid_distortion.warp_image(sub_img, w_mesh_std=std, h_mesh_std=std) 
            img[y1-pad:y2+pad,x1-pad:x2+pad] = sub_img
