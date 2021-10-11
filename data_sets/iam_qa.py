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





    def parseAnn(self,xmlfile,s):
        W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xmlfile)
        #W_lines is list of lists
        # inner list has ([minY,maxY,minX,maxX],text,id) id=gt for NER
        ocr_lines=[]
        for words,line in zip(W_lines,lines):
            ocr_words=[]
            for word in words:
                ocr_word={'box':[word[0][2],word[0][0],word[0][3],word[0][1]],
                      'text':word[1]}
                ocr_words.append(ocr_word)
                        
            ocr_line = {'box':[line[0][2],line[0][0],line[0][3],line[0][1]],
                    'text':line[1],
                    'words':ocr_words}
            ocr_lines.append(ocr_line)
        ocr=[{'paragraphs':[{'lines':ocr_lines}]}]

        use_blocks = False
        #print('block_score: {} {}'.format(block_score,'good!' if use_blocks else 'bad'))
        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s,use_blocks)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, {}, {}, qa

