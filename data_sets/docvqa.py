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


class DocVQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(DocVQA, self).__init__(dirPath,split,config,images)

        self.do_masks=True

        if split=='valid':
            split='val'

        qa_file = os.path.join(dirPath,split,split+'_v1.0.json')
        with open(qa_file) as f:
            data = json.load(f)['data']

        self.images=[]
        for instance in data:
            image_path = os.path.join(dirPath,split,instance['image'])
            #answer = random.choice(instance['answers'])
            qa = (instance['question'],instance.get('answers'))
            self.images.append({'id':instance['questionId'], 'imageName':instance['image'], 'imagePath':image_path, 'annotationPath':qa, 'rescaled':1 })

        if config.get('half',False):
            self.images = self.images[::2]





    def parseAnn(self,qa,s):
        question,answers = qa
        qa=[]
        self.qaAdd(qa,
            'natural_q~'+question,
            random.choice(answers) if answers is not None else None
            )
        
        form_metadata={'all_answers':[answers]}
        return np.zeros(0), [], None, {}, form_metadata, qa

