import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils import grid_distortion
from utils.parseIAM import getWordAndLineBoundaries
import timeit
from data_sets.qa import QADataset, collate

import utils.img_f as img_f


class SROIE(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(SROIE, self).__init__(dirPath,split,config,images)
        self.do_masks=True
        self.augment_shade = config['augment_shade'] if 'augment_shade' in config else True
        self.cased = config.get('cased',True)
        self.rescale_to_crop_size_first = config.get('rescale_to_crop_size_first',True)
        
        assert images is None

        split_file = os.path.join(dirPath,'splits.json')
        with open(split_file) as f:
            splits = json.load(f)
        doc_set = splits[split]

        rescale=1.0
        self.images=[]
        for name in doc_set:
            json_path = os.path.join(dirPath,name+'.json')
            image_path = os.path.join(dirPath,name+'.jpg')
            #if self.train:
            self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':json_path, 'rescaled':rescale })
            #else:
            #    _,_,_,_,_,qa = self.parseAnn(xml_path,rescale)
            #    #qa = self.makeQuestions(rescale,entries))
            #    import pdb;pdb.set_trace()
            #    for _qa in qa:
            #        _qa['bb_ids']=None
            #        self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':xml_path, 'rescaled':rescale, 'qa':[_qa]})





    def parseAnn(self,gt,s):
        del gt['XX_imageName']
        
        question='sroie>'
        answer = json.dumps(gt)+'‡'
        qa=[]
        self.qaAdd(qa,question,answer)

        return None,None,None,None, qa

