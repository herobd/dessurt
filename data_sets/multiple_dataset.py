import json

import torch
from torch.utils.data import Dataset

from collections import defaultdict
import os
import utils.img_f as cv2
import numpy as np
import math


import random

from .qa import collate
from .synth_qadoc_dataset import SynthQADocDataset
from .synth_form_dataset import SynthFormDataset
from .synth_para_qa import SynthParaQA
from .funsd_qa import FUNSDQA
from .cdip_qa import CDIPQA
from .cdip_cloud_qa import CDIPCloudQA
from .distil_bart import DistilBartDataset
from .squad import SQuAD
from .iam_qa import IAMQA
from .synth_hw_qa import SynthHWQA
from .test_qa import TestQA



class MultipleDataset(Dataset):
    def __init__(self, dirPath, split, config):

        self.train = split=='train'

        frequencies=[]
        self.data_sets=[]
        total_freq=0
        for dataset in config['datasets']:
            frequencies.append(dataset['freq'])
            total_freq += dataset['freq']
            d_config = dataset['config']
            d_config['super_computer'] = config['super_computer'] #this is currently only needed by CDIPCloudQA
            self.data_sets.append(
                    eval(d_config['data_set_name'])(dirPath=d_config['data_dir'], split=split, config=d_config)
                    )
        if total_freq != 1:
            frequencies=[f/total_freq for f in frequencies]
        self.d_ranges=[frequencies[0]]
        for f in frequencies[1:]:
            self.d_ranges.append(f+self.d_ranges[-1])

        #if self.train:
        self.lens = [len(d) for d in self.data_sets]
        #else:
        #    self.lens = [min(len(d) for d in self.data_sets)]*len(self.datasets)

        if self.train:
            self.orders=[]
            for dataset in self.data_sets:
                dataset_len = len(dataset)
                self.orders.append(random.sample(range(dataset_len),k=dataset_len))

    def max_len(self):
        return None

    def __len__(self):
        if self.train:
            return sum(self.lens)*10
        else:
            return sum(self.lens)

    def __getitem__(self, idx):
        #import pdb;pdb.set_trace()
        if self.train:
            choice = random.random()
            dataset_i=0
            while choice>self.d_ranges[dataset_i]:
                dataset_i+=1

            if len(self.orders[dataset_i])==0:
                dataset_len = len(self.data_sets[dataset_i])
                self.orders[dataset_i]=random.sample(range(dataset_len),k=dataset_len)
            index = self.orders[dataset_i].pop()
            #index = random.randrange(0,self.lens[dataset_i])
            ret = self.data_sets[dataset_i][index]
        else:
            dataset_i=0
            while idx>=self.lens[dataset_i]:
                idx-=self.lens[dataset_i]
                dataset_i+=1
            ret = self.data_sets[dataset_i][idx]

        ret['imgName'] = 'd{}>{}'.format(dataset_i,ret['imgName'])
        return ret
