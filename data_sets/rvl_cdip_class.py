import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
import timeit
from data_sets.qa import QADataset, collate

import utils.img_f as img_f


class RVLCDIPClass(QADataset):
    """
    Document classification on RVL-CDIP dataset
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(RVLCDIPClass, self).__init__(dirPath,split,config,images)

        self.do_masks=True
        self.cased = True

        if split=='valid':
            split = 'val'

        self.str_lookup=[
                'letter',
                'form',
                'email',
                'handwritten',
                'advertisement',
                'scientific_report',
                'scientific_publication',
                'specification',
                'file_folder',
                'news_article',
                'budget',
                'invoice',
                'presentation',
                'questionnaire',
                'resume',
                'memo',]
        self.str_lookup = ['C:'+cls for cls in self.str_lookup]

        split_file = os.path.join(dirPath,'labels',f'{split}.txt')
        self.images=[]
        with open(split_file) as f:
            for line in f.readlines():
                path, cls = line.strip().split(' ')
                path = os.path.join(dirPath,'images',path)
                cls = int(cls)
                qa=[]
                self.qaAdd(qa,'classify>',self.str_lookup[cls])
                self.images.append({'imageName':path,'imagePath':path,'annotationPath':cls,'qa':qa})


        if split=='val':
            #takes too long to go through whole val set during training, shorten. 1/5?
            self.images = self.images[::5]




    def parseAnn(self,class_index,s):
        class_str = self.str_lookup[class_index]
        qas=[]
        self.qaAdd(qas,'classify>',class_str)
        return None,None,None,None, qas

