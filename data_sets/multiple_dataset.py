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
from .synth_form_dataset import SynthFormDataset
from .synth_para_qa import SynthParaQA
from .funsd_qa import FUNSDQA
from .cdip_cloud_qa import CDIPCloudQA
from .distil_bart import DistilBartDataset
from .squad import SQuAD
from .census_qa import CensusQA
from .iam_qa import IAMQA
from .iam_mixed import IAMMixed
from .synth_hw_qa import SynthHWQA
from .test_qa import TestQA

#This handles balancing multiple datasets in training.
#All the dataset definitions are given to this and it creates all the datasets

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

        self.lens = [len(d) for d in self.data_sets]

        if self.train:
            #We track which instances haven;t been used to have closer to sampling-without-replacment
            self.orders=[]
            for dataset in self.data_sets:
                dataset_len = len(dataset)
                self.orders.append(random.sample(range(dataset_len),k=dataset_len))

        self.debug = config.get('DEBUG',False)

    def max_len(self):
        return None

    def __len__(self):
        if self.train:
            return sum(self.lens)*10
        else:
            return sum(self.lens)

    def __getitem__(self, idx):
        if self.train:
            choice = random.random()
            dataset_i=0
            while choice>self.d_ranges[dataset_i]:
                dataset_i+=1


            if len(self.orders[dataset_i])==0:
                #Reset this dataset
                dataset_len = len(self.data_sets[dataset_i])
                self.orders[dataset_i]=random.sample(range(dataset_len),k=dataset_len)

            index = self.orders[dataset_i].pop()
            ret = self.data_sets[dataset_i][index]
            if ret is None:
                return self.__getitem__(idx)
        else:
            dataset_i=0
            while idx>=self.lens[dataset_i]:
                idx-=self.lens[dataset_i]
                dataset_i+=1
            ret = self.data_sets[dataset_i][idx]

        ret['imgName'] = 'd{}>{}'.format(dataset_i,ret['imgName'])
        return ret
