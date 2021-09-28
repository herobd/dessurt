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


class CensusQA(RecordQA):
    """
    For 1930 census from FamilySearch
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(CensusQA, self).__init__(dirPath,split,config,images)

        self.cache_resized = False
        self.do_masks=True


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
                    instances = readFile[split]
                    imagesAndAnn = []
                    for image_path,json_path in instances:
                        try:
                            name = image_path[path.rindex('/')+1:]
                        except ValueError:
                            name = image_path
                        imagesAndAnn.append( (name,os.path.join(dirPath,image_path),os.path.join(dirPath,json_path)) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit(1)
            self.images=[]
            for imageName,imagePath,jsonPath in imagesAndAnn:
                #if os.path.exists(jsonPath):
                #    org_path = imagePath
                #    if self.cache_resized:
                #        path = os.path.join(self.cache_path,imageName+'.png')
                #    else:
                #        path = org_path

                #    rescale=1.0
                #    if self.cache_resized:
                #        rescale = self.rescale_range[1]
                #        if not os.path.exists(path):
                #            org_img = img_f.imread(org_path)
                #            if org_img is None:
                #                print('WARNING, could not read {}'.format(org_img))
                #                continue
                #            resized = img_f.resize(org_img,(0,0),
                #                    fx=self.rescale_range[1], 
                #                    fy=self.rescale_range[1], 
                #                    )
                #            img_f.imwrite(path,resized)
                path=imagePath
                rescale=1
                self.images.append({'id':imageName, 'imageName':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale })
                #else:
                #    print('{} does not exist'.format(jsonPath))
                #    print('No json found for {}'.format(imagePath))
                #    #exit(1)

        self.all_fields=set(['line no.','household','name','previous','race','relationship','sex'])
        self.id_fields=set(['line no.','name'])


    def parseAnn(self,data,s):
        if isinstance(data,dict):
            recognition = data['recognition'] if self.use_recognition else None
            data = data['indexed']
        else:
            recognition = None

        entries =[
                {
                    'line no.': entry['LINE_NBR'],
                    'household': entry['HOUSEHOLD_ID'],
                    'name': entry['PR_NAME'],
                    'previous': entry['PR_PREV_RESIDENCE_PLACE'],
                    'race': entry['PR_RACE_OR_COLOR'],
                    'relationship': entry['PR_RELATIONSHIP_TO_HEAD'],
                    'sex': entry['PR_SEX_CODE'],

                    } for entry in data]
        qa, qa_bbs = self.makeQuestions(s,entries)


        return np.zeros(0), [], None, {}, {}, qa

