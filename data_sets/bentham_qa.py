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


class BenthamQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(BenthamQA, self).__init__(dirPath,split,config,images)

        self.do_masks=True

        split=''

        qa_file = os.path.join(dirPath,'BenthamQA_'+split+'_1.0.json')
        with open(qa_file) as f:
            data = json.load(f)['data']

        self.images=[]
        for instance in data:
            if 'qas' in instance:
                #image_path = os.path.join(dirPath,'BenthamQA_document_images',instance['document_id']+'.jpg')
                image_path = os.path.join(dirPath,instance['document_image']['document_image'])
                #answer = random.choice(instance['answers'])
                for qa in instance['qas']:
                    question = qa['question']
                    for ans in qa['answers']:
                        answer=[]
                        start_word = ans['answer_start_word_no']
                        end_word = ans['answer_end_word_no']
                        last_word_no=-1
                        for word in instance['document_image']['gold_standard_transcription']:
                            if word['wordno']>=start_word:
                                answer.append(word['text'])
                            assert word['wordno']==last_word_no+1
                            last_word_no=word['wordno']
                        answer = ' '.join(answer)

                        self.images.append({'id':qa['question_id'], 'imageName':image_path, 'imagePath':image_path, 'annotationPath':(question,answer), 'rescaled':1 })






    def parseAnn(self,qa,s):
        question,answer = qa
        qa=[]
        self.qaAdd(qa,
            'natural_q~'+question,
            answer
            )
        
        form_metadata={'all_answers':[[answer]]}
        return None,None,None form_metadata, qa

