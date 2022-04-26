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
from data_sets.record_qa import RecordQA, collate

import utils.img_f as img_f


class CensusQA(RecordQA):
    """
    For 1930 census data provided by FamilySearch
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(CensusQA, self).__init__(dirPath,split,config,images)

        self.use_recognition = config['use_recognition'] if 'use_recognition' in config else False

        self.cache_resized = False
        self.do_masks=True
        self.valid = 'valid'==split
        
        self.max_records=40
        self.all_fields=set(['line no.','household','name','previous','race','relationship','sex', 'given name', 'age', 'birthplace'])
        self.id_fields=set(['line no.','name'])
        self.next_name_ids = ['given name','name']
        self.ordered_ids = ['line no.','household','name','relationship','sex','race','age','birthplace']
        self.main_id_field = 'line no.'
        self.name_to_id = {
                    'line no.': 'LINE_NBR',
                    'household': 'HOUSEHOLD_ID',
                    'name': 'PR_NAME',
                    'previous': 'PR_PREV_RESIDENCE_PLACE',
                    'race': 'PR_RACE_OR_COLOR',
                    'relationship': 'PR_RELATIONSHIP_TO_HEAD',
                    'sex': 'PR_SEX_CODE',
                    'given name': 'PR_NAME_GN',
                    'age': 'PR_AGE',
                    'birthplace': 'PR_BIRTH_PLACE'
                    } 

        self.pretrain = config.get('pretrain',False)
        if self.pretrain=='all':
            self.cased=True
            self.all_fields.remove('previous')
            if self.train:
                self.q_types =         ['all-name','all-age','whole-doc','get-field']
                self.q_type_weights =  [1,          1,          2, 0.5]
                self.q_types_single =         ['all-name','all-age','whole-doc','get-field']
                self.q_type_single_weights =  [1,1,2, 0.5]
            else:
                #these are what we'll use to actually score
                #(not actually looked at as it isn't sampling)
                self.q_types =         ['whole-doc']
        elif self.pretrain:
            self.cased=True
            self.all_fields.remove('previous')
            if self.train:
                self.q_types =         ['all-name','all-age','whole-doc']
                self.q_type_weights =  [1,          1,          2]
                self.q_types_single =         ['all-name','all-age','whole-doc']
                self.q_type_single_weights =  [1,1,2]
            else:
                #these are what we'll use to actually score
                #(not actually looked at as it isn't sampling)
                self.q_types =         ['whole-doc']



        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                splitFile = 'overfit_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                readFile = json.loads(f.read())
                if split in readFile:
                    instances = readFile[split]
                    imagesAndAnn = []
                    for image_path,json_path in instances:
                        try:
                            name = image_path[image_path.rindex('/')+1:]
                        except ValueError:
                            name = image_path
                        imagesAndAnn.append( (name,os.path.join(dirPath,image_path),os.path.join(dirPath,json_path)) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit(1)
            self.images=[]
            if not self.train:
                imagesAndAnn = imagesAndAnn[::2]
            for imageName,imagePath,jsonPath in imagesAndAnn:
                path=imagePath
                rescale=1
                if self.train:
                    self.images.append({'id':imageName, 'imageName':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale })
                else:
                    with open(jsonPath) as f:
                        data = json.load(f)
                    _,_,_,_,_,qa = self.parseAnn(data,rescale)
                    for _qa in qa:
                        _qa['bb_ids']=None
                        self.images.append({'id':imageName, 'imageName':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'qa':[_qa]})




    def parseAnn(self,data,s):
        if isinstance(data,dict):
            data = data['indexed']

        data = data[:self.max_records]
        
        if self.valid and not self.pretrain:
            #the validation set is too big to run through frequently. So instead we'll only take every other entry
            data = data[::10]
            #This allows us to cover more variety in handwriting than just making the validation set smaller

        entries = [ {key:entry[self.name_to_id[key]] if self.name_to_id[key] in entry else None for key in self.all_fields} for entry in data]
        qa = self.makeQuestions(s,entries)
        
        return None,None,None,None, qa

