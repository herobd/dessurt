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


class DocVQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(IAMQA, self).__init__(dirPath,split,config,images)

        if split=='valid':
            split='val'

        qa_file = os.path.join(dirPath,split,split+'_v1.0.json')
        with open(qa_file) as f:
            data = json.load(f)['data']

        for instance in data:
            image_path = os.path.join(dirPath,split,instance['image'])
            #answer = random.choice(instance['answers'])
            qa = (instance['question'],instance['answers'])
            self.images.append({'id':instance['questionId'], 'imageName':instance['image'], 'imagePath':image_path, 'annotationPath':qa, 'rescaled':1 })





    def parseAnn(self,qa,s):
        question,answers = qa
        qa=[]
        self.qaAdd(qa,
            'natural_q~'+question,
            random.choice(answers)
            )

        return np.zeros(0), [], None, {}, {}, qa

