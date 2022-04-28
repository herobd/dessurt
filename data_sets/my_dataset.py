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


class MyDataset(QADataset):
    """
    Generic Query-Response dataset
    
    This is intended to make it easy to fine-tune Dessurt on your data. You define the queries and answers

    Expects dirPath to point to a directory contiaing a "train", "valid", and optionally "test" subdirectory
    Each of these as image files (png,jpg,jpeg,tiff) and then either a single "qa.json" or a .json for each image file
    "qa.json" has a map of {"imagefile/path.png": [ {"question": "TOK~context text",
                                                     "answer": "response text"},
                                                    {"question": "TOK2>",
                                                     "answer": "other response text"},
                                                     ...
                                                  ],
                            ...}
    Or the individual json per image instead just has [ {"question": "TOK~the context text",
                                                        "answer": "the response text"},
                                                        ...
                                                      ]
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(MyDataset, self).__init__(dirPath,split,config,images)

        self.do_masks=True

        directory = os.path.join(dirPath,split)

        all_annotations=None
        min_path=999999999
        images=[]
        for root, dirs, files in os.walk(directory):
            path = root.split(os.sep)
            min_path = min(min_path,len(path))
            for file in files:
                
                if file=='qa.json' and len(path)==min_path:
                    with open(os.path.join(root,file)) as f:
                        all_annotations=json.load(f)
                if any(file.lower().endswith(ext) for ext in ['.png','.jpg','.jpeg','.tiff']):
                    no_ext = file[:file.rfind('.')]
                    json_path = os.path.join(root,no_ext+'.json')
                    if os.path.exists(json_path):
                        images.append((os.path.join(root,file),json_path))
                    else:
                        images.append((os.path.join(root,file),None))



        

        self.images=[]
        for image_path,json_path in images:
            if json_path is not None:
                assert all_annotations is None
                #json_path = os.path.join(directory,json_path)
            else:
                if all_annotations is None:
                    print('WARNING! There was no json found for: '+json_path) 
                    continue
                image = image_path[image_path.find('/'+split+'/')+len(split)+2:]
                json_path = all_annotations[image]
            #image_path = os.path.join(directory,image_path)
            self.images.append({'imagePath':image_path, 'annotationPath':json_path, 'rescaled':1 })

        if len(self.images)==0:
            print('ERROR, no images in dataset at {}'.format(directory))
            exit()




    def parseAnn(self,qas,s):
        ret=[]
        for qa in qas:
            self.qaAdd(ret,
                qa['question'],
                qa['answer'] if isinstance(qa['answer'],str) else random.choice(qa['answer'])
                )
        if isinstance(qa['answer'],str):
            metadata={'all_answers':[qa['answer']]}
        else:
            metadata={'all_answers':qa['answer']}
        return None,None,None,metadata,ret

