import json

import torch
from torch.utils.data import Dataset

from collections import defaultdict
import os
import utils.img_f as img_f
import numpy as np
import math, re

from utils import grid_distortion
from utils.util import ensure_dir

from utils import string_utils, augmentation
import itertools, pickle

import random
PADDING_CONSTANT = -1
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def collate(batch):
    if len(batch)==1:
        batch[0]['a_batch_size']=batch[0]['image'].size(0)
        return batch[0]
    batch = [b for b in batch if b is not None]
    a_batch_size = len(batch[0]['gt'])

    dim1 = batch[0]['image'].shape[1]
    dim3 = max([b['image'].shape[3] for b in batch])
    dim2 = batch[0]['image'].shape[2]


    max_label_len = max([b['label'].size(0) for b in batch])

    input_batch = torch.FloatTensor(len(batch)*a_batch_size, dim1, dim2, dim3).fill_(PADDING_CONSTANT)
    labels_batch = torch.IntTensor(max_label_len,len(batch)*a_batch_size).fill_(0)

    for i in range(len(batch)):
        b_img = batch[i]['image']
        l = batch[i]['label']
        #toPad = (dim3-b_img.shape[3])
        input_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = b_img
        labels_batch[0:l.size(0),i*a_batch_size:(i+1)*a_batch_size] = l



    toRet= {
        "image": input_batch,
        "label": labels_batch,
        "label_lengths": torch.cat([b['label_lengths'] for b in batch],dim=0),
        "gt": [l for b in batch for l in b['gt']],
        "author": [l for b in batch for l in b['author']],
        "name": [l for b in batch for l in b['name']],
        "a_batch_size": a_batch_size
    }
    return toRet
class FormlinesATRDataset(Dataset):
    def __init__(self, dirPath, split, config):
        if 'split' in config:
            split = config['split']
        if split!='train':
            dirPath=os.path.join(dirPath,split)
        subdir = split

        self.img_height = config['img_height']
        self.max_width = config['max_width'] if  'max_width' in config else 1000
        #assert(config['batch_size']==1)
        wtype = config['type']

        self.pad_ends = config['pad_ends'] if 'pad_ends' in config else False


        #with open(os.path.join(dirPath,'sets.json')) as f:
        gt_file = os.path.join(dirPath,'{}.txt'.format(wtype if len(wtype)>0 else 'text'))

        with open(gt_file) as f:
            gt_list = f.readlines()
        self.image_list = [(os.path.join(dirPath,wtype,a[:a.index('|')]),a[a.index('|')+1:].strip()) for a in gt_list]
        self.max_char_len=max([len(p[1]) for p in self.image_list])

        self.warning=False



        char_set_path = config['char_file']
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']
        self.augmentation = config['augmentation'] if 'augmentation' in config else None

        #DEBUG
        if 'overfit' in config and config['overfit']:
            self.lineIndex = self.lineIndex[:10]




    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_path, gt = self.image_list[idx]
        img = 255-img_f.imread(img_path,0)#read as grayscale
        if img is None:
            return None
        if img.shape[1]<25 and img.shape[0]>56:
            img=img_f.rotate(img,img_f.ROTATE_90_COUNTERCLOCKWISE)

        if img.shape[0]< self.img_height:
            pad = self.img_height-img.shape[0]
            img = np.pad(img,((pad//2+pad%2,pad//2),(0,0)),'constant',constant_values=255)

        if self.pad_ends:
            to_pad = self.img_height//2
            img = np.pad(img,((0,0),(to_pad,to_pad)),'constant',constant_values=255)


        if img.shape[0] != self.img_height:
            assert( img.shape[0] > self.img_height)
            percent = float(self.img_height) / img.shape[0]
            #if img.shape[1]*percent > self.max_width:
            #    percent = self.max_width/img.shape[1]
            img = img_f.resize(img, (0,0), fx=percent, fy=percent)
            #if img.shape[0]<self.img_height:
            #    diff = self.img_height-img.shape[0]
            #    img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=255)
        if 'UNKNOWN' in gt and img.shape[1]>self.max_width:
            diff = img.shape[1]-self.max_width
            start = random.randint(0,diff-1)
            img = img[:,start:start+self.max_width]


        if len(img.shape)==2:
            img = img[...,None]

        if self.augmentation is not None:
            #img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)
            img = grid_distortion.warp_image(img)
            if len(img.shape)==2:
                img = img[...,None]

        img = img.astype(np.float32)
        img = 1.0 - img / 128.0
        #img = (img / 128.0)-1


        #if len(gt) == 0:
        #    gt_label=None
        #else:
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)

        name = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
        toAppend = {
            "image": img,
            "gt": gt,
            "gt_label": gt_label,
            "name": name,
            "author": 'unknown'
        }
        batch=[toAppend]

        #vvv This is all reduntend, but left in for developement speed vvv
        #batch = [b for b in batch if b is not None]
        #These all should be the same size or error
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
            toPad = (dim1-b_img.shape[1])
            if 'center' in batch[0] and batch[0]['center']:
                toPad //=2
            else:
                toPad = 0
            input_batch[i,:,toPad:toPad+b_img.shape[1],:] = b_img
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
        
        toRet = {
            "image": images,
            "label": labels,
            "label_lengths": label_lengths,
            "gt": [b['gt'] for b in batch],
            "name": [b['name'] for b in batch],
            "author": [b['author'] for b in batch],
        }
        return toRet

    def max_len(self):
        return self.max_char_len
