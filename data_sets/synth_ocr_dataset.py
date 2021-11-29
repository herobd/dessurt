import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
from utils import img_f
import numpy as np
import math

from utils import grid_distortion
from utils.util import ensure_dir
from utils import string_utils, augmentation
from data_sets.gen_daemon import GenDaemon

import random
PADDING_CONSTANT = -1

def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['image'].shape[0] for b in batch])) != 1:
        print('Error in images sizes:')
        for b in batch:
            print(b['image'].shape)
        return None
    assert len(set([b['image'].shape[0] for b in batch])) == 1
    assert len(set([b['image'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['image'].shape[0]
    dim1 = max([b['image'].shape[1] for b in batch])
    dim2 = batch[0]['image'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['image']
        input_batch[i,:,0:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    #all_labels = np.concatenate(all_labels)
    label_lengths = torch.IntTensor(label_lengths)
    max_len = label_lengths.max()
    all_labels = [np.pad(l,((0,max_len-l.shape[0]),),'constant') for l in all_labels]
    all_labels = np.stack(all_labels,axis=1)


    images = input_batch.transpose([0,3,1,2])
    images = torch.from_numpy(images)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    #label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "image": images,
        "label": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
        "name": [b['name'] for b in batch],
    }

class SynthOCRDataset(Dataset):
    def __init__(self, dirPath, split, config):

        self.img_height = config['img_height']
        self.min_text_height = config['min_text_height']
        self.max_width = config['max_width']

        self.generator = GenDaemon(dirPath)

        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']

        self.augmentation = config['augmentation'] if 'augmentation' in config else 'warp low'
        self.generated_words=[]

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        
        word_height = random.randrange(self.min_text_height,self.img_height+1)
        em_approx = word_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
        min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/chara  cter-design-standards/whitespace
        max_space = 0.5*em_approx
        space_width = round(random.random()*(max_space-min_space) + min_space)
        space = np.zeros((self.img_height,space_width),dtype=np.uint8)

        image = np.zeros((self.img_height,self.max_width),dtype=np.uint8)
        x=random.randrange(0,space_width)
        y=random.randrange(0,1+self.img_height-word_height) if word_height!=self.img_height else 0
        gt=[]
        while True:
            #import pdb;pdb.set_trace()
            if len(self.generated_words)==0:
                self.generated_words = self.generator.generate()
            text,word_img = self.generated_words[0]
            scale = word_height/word_img.shape[0]
            new_width = round(scale*word_img.shape[1])
            if word_img.shape[0]<6 or word_img.shape[1]<6 or new_width<4:
                self.generated_words = self.generated_words[1:]
                continue
            if x+new_width > self.max_width:
                break #end here
            word_img = img_f.resize(word_img,(word_height,new_width))
            image[y:y+word_height,x:x+new_width] = word_img
            gt.append(text)
            x+=space_width+new_width

            self.generated_words = self.generated_words[1:]
        gt = ' '.join(gt)
        gt = gt.replace('¶','')

        img = image

        if self.augmentation is not None and (type(self.augmentation) is not str or 'warp' in self.augmentation):
            #img = augmentation.apply_random_color_rotation(img)
            if type(self.augmentation) is str and "low" in self.augmentation:
                if random.random()>0.1:
                    img = img[:,:,None]
                    img = augmentation.apply_tensmeyer_brightness(img)
                if random.random()>0.01:
                    if len(img.shape)==3:
                        img = img[:,:,0]
                    img = grid_distortion.warp_image(img,w_mesh_std=0.7,h_mesh_std=0.7)
            else:
                img = augmentation.apply_tensmeyer_brightness(img)
                img = grid_distortion.warp_image(img)
        if len(img.shape)==2:
            img = img[...,None]

        img = img.astype(np.float32)
        img = (img / 128)-1 #already has 0 as background

        if len(gt) == 0:
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            "image": img,
            "gt": gt,
            "gt_label": gt_label,
            "name": 'generated',
        }
