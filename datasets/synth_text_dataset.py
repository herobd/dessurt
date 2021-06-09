import json, pickle
import timeit
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
from glob import iglob
import os
import utils.img_f as cv2
import numpy as np
import math, time

from utils import grid_distortion

from utils import string_utils, augmentation
from utils.util import ensure_dir
#import pyexiv2
#import piexif

from multiprocessing import Pool, TimeoutError

import random, pickle
PADDING_CONSTANT = -1


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
    if batch[0]['spaced_label'] is not None:
        max_spaced_label_len = max([b['spaced_label'].size(0) for b in batch])
    else:
        max_spaced_label_len = None

    input_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
    mask_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
    if 'fg_mask' in batch[0]:
        fg_masks = torch.full((len(batch)*a_batch_size, 1, dim2, dim3), 0)
    if 'changed_image' in batch[0]:
        changed_batch = torch.full((len(batch)*a_batch_size, dim1, dim2, dim3), PADDING_CONSTANT)
    top_and_bottom_batch = torch.full((len(batch)*a_batch_size,2,dim3), 0)
    center_line_batch = torch.full((len(batch)*a_batch_size,dim3), dim2/2)
    labels_batch = torch.IntTensor(max_label_len,len(batch)*a_batch_size).fill_(0)
    if max_spaced_label_len is not None:
        spaced_labels_batch = torch.IntTensor(max_spaced_label_len,len(batch)*a_batch_size).fill_(0)
    else:
        spaced_labels_batch = None

    for i in range(len(batch)):
        b_img = batch[i]['image']
        b_mask = batch[i]['mask']
        b_top_and_bottom = batch[i]['top_and_bottom']
        b_center_line = batch[i]['center_line']
        l = batch[i]['label']
        #toPad = (dim3-b_img.shape[3])
        input_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = b_img
        mask_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = b_mask
        if 'fg_mask' in batch[i]:
            fg_masks[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = batch[i]['fg_mask']
        if 'changed_image' in batch[i]:
            changed_batch[i*a_batch_size:(i+1)*a_batch_size,:,:,0:b_img.shape[3]] = batch[i]['changed_image']
        if b_top_and_bottom is not None:
            top_and_bottom_batch[i*a_batch_size:(i+1)*a_batch_size,:,0:b_img.shape[3]] = b_top_and_bottom
        else:
            top_and_bottom_batch=None
        if b_center_line is not None:
            center_line_batch[i*a_batch_size:(i+1)*a_batch_size,0:b_img.shape[3]] = b_center_line
        else:
            center_line_batch=None
        labels_batch[0:l.size(0),i*a_batch_size:(i+1)*a_batch_size] = l
        if max_spaced_label_len is not None:
            sl = batch[i]['spaced_label']
            spaced_labels_batch[0:sl.size(0),i*a_batch_size:(i+1)*a_batch_size] = sl


    if batch[0]['style'] is None:
        style=None
    else:
        style=torch.cat([b['style'] for b in batch],dim=0)

    toRet = {
        "image": input_batch,
        "mask": mask_batch,
        "top_and_bottom": top_and_bottom_batch,
        "center_line": center_line_batch,
        "label": labels_batch,
        "style": style,
	#"style": torch.cat([b['style'] for b in batch],dim=0),
        #"label_lengths": [l for b in batch for l in b['label_lengths']],
        "label_lengths": torch.cat([b['label_lengths'] for b in batch],dim=0),
        "gt": [l for b in batch for l in b['gt']],
        "spaced_label": spaced_labels_batch,
        "author": [l for b in batch for l in b['author']],
        "name": [l for b in batch for l in b['name']],
        "a_batch_size": a_batch_size
    }
    if 'fg_mask' in batch[0]:
        toRet['fg_mask']=fg_masks
    if 'changed_image' in batch[0]:
        toRet['changed_image']=changed_batch
    return toRet

def create_image(x):
    synth_gen,img_height,blank_size,seed = x
    random.seed(seed) #ugh, didn't realize the processes inherit the same random state
    np.random.seed(random.randint(0,99999999))
    while True:
        img, gt, font_idx = synth_gen.getSample()
        if img_height is not None and img.shape[0] != img_height:
            percent = float(img_height) / img.shape[0]
            if percent<=0:
                continue
            #if img.shape[1]*percent > max_width:
            #    percent = max_width/img.shape[1]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent)
            if img.shape[0]<img_height:
                diff = img_height-img.shape[0]
                img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=0)
        img = (255*(1-img)).astype(np.uint8)
        if blank_size is not None:
            page = np.ones((blank_size,blank_size),dtype=np.uint8)
            page*=255
            x = random.randrange(0,blank_size-img.shape[1])
            y = random.randrange(0,blank_size-img.shape[0])
            page[y:y+img.shape[0],x:x+img.shape[1]]=img
            img=page
        break
    return gt,img

class SynthTextDataset(QADataset):
    def __init__(self, dirPath, split, config):
        from synthetic_text_gen import SyntheticText
        self.batch_size=1
        self.single = config['single'] if 'single' in config else False
        self.spaced_by_name=None
        self.img_height = config['img_height']
        self.max_width = config['max_width'] if 'max_width' in config else 10000
        self.clip_width = config['clip_width'] if 'clip_width' in config else 99999999999999999
        self.blank_size = config['blank_size'] if 'blank_size' in config else None
        text_len = config['max_chars'] if 'max_chars' in config else 35
        self.text_max_len=text_len
        char_set_path = config['char_file']
        self.directory = dirPath
        self.gt_filename = os.path.join(self.directory,'gt.txt')
        self.mask_post = config['mask_post'] if 'mask_post' in config else []
        self.mask_random = config['mask_random'] if 'mask_random' in config else False
        with open(char_set_path) as f:
            char_set = json.load(f)
        self.char_to_idx = char_set['char_to_idx']
        self.warp_freq = 1.0
        if split=='train':
            self.train=True
            self.augmentation = config['augmentation'] if 'augmentation' in config else None
	    
            fontdir = config['fontdir'] if 'fontdir' in config else '../data/fonts/textfonts'
            textdir = config['textdir'] if 'textdir' in config else '../data/OANC_text'
            text_len = config['max_chars'] if 'max_chars' in config else 35
            self.text_max_len=text_len
            text_min_len = config['min_chars'] if 'min_chars' in config else 33

            cropped_aug = config['cropped_aug'] if 'cropped_aug' in config else False
            gen_type = config['gen_type'] if 'gen_type' in config else 'normal'

            if cropped_aug or gen_type=='cropped_aug':
                self.synth_gen= SyntheticText(fontdir,textdir,line_prob=0.8,line_thickness=70,line_var=30,pad=20,gaus_noise=0.15,hole_prob=0.6, hole_size=400,neighbor_gap_var=25,rot=2.5,text_len=text_len, text_min_len=text_min_len, use_warp=0.4,warp_std=[1,1.4])
            elif gen_type=='clean':
                self.synth_gen= SyntheticText(fontdir,textdir,text_len=text_len,text_min_len=text_min_len,line_prob=0.0,line_thickness=70,line_var=0,mean_pad=10,pad=0,gaus_noise=0.001,hole_prob=0.0, hole_size=400,neighbor_gap_var=0,rot=0, use_warp=0.0,warp_std=[1,1.4], linesAboveAndBelow=False)
            elif gen_type=='veryclean':
                self.synth_gen= SyntheticText(fontdir,textdir,text_len=text_len,text_min_len=text_min_len,line_prob=0.0,line_thickness=70,line_var=0,mean_pad=10,pad=0,gaus_noise=0,gaus_std=0.0000001,blur_std=0.000001,hole_prob=0.0, hole_size=400,neighbor_gap_var=0,rot=0, use_warp=0.0,warp_std=[1,1.4], linesAboveAndBelow=False,useBrightness=False)
            elif gen_type=='normal':
                self.synth_gen= SyntheticText(fontdir,textdir,text_len=text_len,text_min_len=text_min_len,line_prob=0.0,line_thickness=70,line_var=30,mean_pad=10,pad=15,gaus_noise=0.05,hole_prob=0.0, hole_size=400,neighbor_gap_var=25,rot=0, use_warp=0.4,warp_std=[1,1.4], linesAboveAndBelow=False)
            elif gen_type=='small noise':
                self.synth_gen= SyntheticText(fontdir,textdir,text_len=text_len,text_min_len=text_min_len,line_prob=0.0,line_thickness=70,line_var=30,mean_pad=20,pad=5,gaus_noise=0.06,hole_prob=0.0, hole_size=400,neighbor_gap_var=25,rot=0, use_warp=0.3,warp_std=[1,1.4], linesAboveAndBelow=False)
            elif gen_type=='NAF':
                self.synth_gen= SyntheticText(fontdir,textdir,text_len=text_len,text_min_len=text_min_len,line_prob=0.8,line_thickness=70,line_var=30,mean_pad=0,pad=17,gaus_noise=0.1,hole_prob=0.6, hole_size=400,neighbor_gap_var=25,rot=5, use_warp=0.0, linesAboveAndBelow=True, clean=True)
                self.warp_freq=0.1
            else:
                raise NotImplementedError('SynthTextGen unknown gen_type: {}'.format(gen_type))


            self.set_size = config['set_size']
            self.use_before_refresh = config['use_before_refresh']
            self.used=-1
            self.used_instances=0
            self.num_processes = config['num_processes']
            self.per_process = config['per_process'] if 'per_process' in config else 100

            self.use_fg_mask = config['use_fg_mask'] if 'use_fg_mask' in config else False
            self.max_strech=0.4
            self.max_rot_rad= 45/180 * math.pi

            self.include_stroke_aug = config['include_stroke_aug'] if 'include_stroke_aug' in config else False

            
            ensure_dir(self.directory)

            self.labels = [None]*self.set_size

            self.init_size=0
            cur_files = list(os.listdir(self.directory))
            try:
                with open(self.gt_filename) as f:
                    labels =f.readlines()

                for i in range(min(self.set_size,len(labels))):
                    if '{}.png'.format(i) in cur_files:
                        self.init_size+=1
                        self.labels[i]=labels[i].strip()
                    else:
                        break
                if self.init_size>0:
                    print('Found synth images 0-{} with labels'.format(self.init_size-1))
            except:
                self.init_size=0
            if self.init_size<self.set_size:
                self.used=self.use_before_refresh
                self.used_instances=self.set_size

            self.refresh_self=config['refresh_self'] if 'refresh_self' in config else False
            if self.refresh_self:
                assert 'num_workers' not in config or config['num_workers']<2
        else:
            self.set_size=min(config['set_size'],2000)
            self.train=False
            self.augmentation=None
            self.include_stroke_aug=False
            self.use_fg_mask=False



    def __len__(self):
        return self.set_size

    def max_len(self):
        return self.text_max_len

    def __getitem__(self, idx):


        if self.train and self.refresh_self:
            if self.used_instances>=self.set_size:
                self.refresh_data(None,None,logged=False)
                self.used_instances=0
            self.used_instances+=self.batch_size

        if self.augmentation is not None and 'affine' in self.augmentation:
            strech = (self.max_strech*2)*np.random.random() - self.max_strech +1
            skew = (self.max_rot_rad*2)*np.random.random() - self.max_rot_rad
        if self.include_stroke_aug:
            thickness_change= np.random.randint(-4,5)
            fg_shade = np.random.random()*0.25 + 0.75
            bg_shade = np.random.random()*0.2
            blur_size = np.random.randint(2,4)
            noise_sigma = np.random.random()*0.02




        img_path = os.path.join(self.directory,'{}.png'.format(idx+b))
        img = cv2.imread(img_path,0)
        if img is None:
            print('Error, could not read {}'.format(img_path))
            return self[(idx+1)%len(self)]
        
        if self.blank_size is None:
            if img.shape[0] != self.img_height:
                scale = float(self.img_height) / img.shape[0]
                if scale*img.shape[1] > self.max_width:
                    scale = self.max_width/img.shape[1]
                img = cv2.resize(img, (0,0), fx=scale, fy=scale)
                if img.shape[0]<self.img_height: #it was too long. We need to pad it vertically
                    diff = self.img_height-img.shape[0]
                    img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=255)

            if self.augmentation=='affine':
                if img.shape[1]*strech > self.max_width:
                    strech = self.max_width/img.shape[1]
            if img.shape[1] > self.max_width:
                percent = float(self.max_width) / img.shape[1]
                img = cv2.resize(img, (0,0), fx=percent, fy=1)

            if img.shape[1] > self.clip_width:
                img = img[:,:self.clip_width]

        if self.use_fg_mask:
            th,fg_mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            fg_mask = 255-fg_mask
            ele = cv2.getStructuringElement(  cv2.MORPH_ELLIPSE, (9,9) )
            fg_mask = cv2.dilate(fg_mask,ele)
            fg_mask = fg_mask/255
        else:
            fg_mask = None

        if len(img.shape)==2:
            img = img[...,None]
        if self.augmentation is not None:
            #img = augmentation.apply_random_color_rotation(img)
            if 'affine' in self.augmentation:
                img,fg_mask = augmentation.affine_trans(img,fg_mask,skew,strech)
            if 'brightness' in self.augmentation:
                img = augmentation.apply_tensmeyer_brightness(img)
                assert(fg_mask is None)
            if 'warp' in self.augmentation and random.random()<self.warp_freq:
                try:
                    img = grid_distortion.warp_image(img)
                except cv2.error as e:
                    print(e)
                    print(img.shape)
                assert(fg_mask is None)
            if 'invert' in self.augmentation and random.random()<0.25:
                img = 1-img

        if self.include_stroke_aug:
            new_img = augmentation.change_thickness(img,thickness_change,fg_shade,bg_shade,blur_size,noise_sigma)
            new_img = new_img*2 -1.0

        if len(img.shape)==2:
            img = img[...,None]


        img = img.astype(np.float32)
        img = 1.0 - img / 128.0
        
        if self.train:
            gt = self.labels[idx]
        else:
            with open(self.gt_filename) as f:
                for i in range(0,idx+1):
                    gt=f.readline()
            gt=gt.strip()
        if gt is None:
            print('Error unknown label for image: {}'.format(img_path))
            return self.__getitem__((idx+7)%self.set_size)

        gt_label = string_utils.str2label_single(gt, self.char_to_idx)

        font_idx='?'
        toRet= {
            "image": img,
            "gt": gt,
            "gt_label": gt_label,
            "author": font_idx,
            "name": '{}_{}'.format(idx+b,font_idx),
            "style": None,
            "spaced_label": None
        }
        if self.use_fg_mask:
            toRet['fg_mask'] = fg_mask
        if self.include_stroke_aug:
            toRet['changed_image'] = new_img
        return toRet

    def sample(self):
        #ri = np.random.choice(self.num_styles,[self.gen_batch_size,2],replace=False)
        #mix = np.random.random(self.gen_batch_size)
        #if self.extrapolate>0:
        #    mix = (2*self.extrapolate+1)*mix - self.extrapolate
        #style = self.styles[ri[:,0]]*mix + self.styles[ri[:,1]]*(1-mix)

        authors = np.random.choice(self.num_authors,[self.gen_batch_size,2],replace=True)
        mix = np.random.random(self.gen_batch_size)
        style = []
        for b in range(self.gen_batch_size):
            style0_i = np.random.choice(len(self.styles[authors[b,0]]))
            style1_i = np.random.choice(len(self.styles[authors[b,1]]))
            style0 = self.styles[authors[b,0]][style0_i]
            style1 = self.styles[authors[b,1]][style1_i]
            style.append(style0*mix[b] + style1*(1-mix[b]))
        style = np.stack(style,axis=0)
        style = torch.from_numpy(style).float()



        all_labels = []
        label_lengths = []
        gt=[]

        for i in range(self.gen_batch_size):
            idx = np.random.randint(0,len(self.text))
            text =  self.text[idx]
            gt.append(text)
            l = string_utils.str2label_single(text, self.char_to_idx)
            all_labels.append(l)
            label_lengths.append(len(l))

        #all_labels = np.concatenate(all_labels)
        label_lengths = torch.IntTensor(label_lengths)
        max_len = label_lengths.max()
        all_labels = [np.pad(l,((0,max_len-l.shape[0]),),'constant') for l in all_labels]
        all_labels = np.stack(all_labels,axis=1)

        label = torch.from_numpy(all_labels.astype(np.int32))

        return style,label,label_lengths,gt


    #def save(self,i,image,gt):
    #    image = (255*(1-image)).astype(np.uint8)
    #    #f = open(self.gt_filename,'a')
    #    with open(self.gt_filename,'a') as f:
    #        idx = i
    #        filename = os.path.join(self.directory,'{}.png'.format(idx))
    #        cv2.imwrite(filename,image)
    #        self.labels[idx] = gt
    #        #metadata = pyexiv2.ImageMetadata(filename)
    #        #metadata.read()
    #        #metadata.write()
    #        f.write(gt+'\n')
    #    #f.close()

    #def refresh_data(self,generator,gpu,logged=True):
    #    self.used+=1
    #    if self.used >= self.use_before_refresh:
    #        if self.init_size==0:
    #            with open(self.gt_filename,'w') as f:
    #                f.write('') #erase or start the gt file
    #        if logged:
    #            print('refreshing sythetic')
    #        for i in range(self.init_size,self.set_size):
    #            img, gt, font_idx = self.synth_gen.getSample()
    #            if img.shape[0] != self.img_height:
    #                percent = float(self.img_height) / img.shape[0]
    #                #if img.shape[1]*percent > self.max_width:
    #                #    percent = self.max_width/img.shape[1]
    #                img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
    #                if img.shape[0]<self.img_height:
    #                    diff = self.img_height-img.shape[0]
    #                    img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=0)

    #            if not logged:
    #                print('refreshing sythetic: {}/{}'.format(i,self.set_size), end='\r')
    #            self.save(i,img,gt)
    #        self.init_size=0
    #        self.used=0

        

    def refresh_data(self,generator,gpu,logged=True):
        self.used+=1
        if self.used >= self.use_before_refresh:
            if self.init_size==0:
                with open(self.gt_filename,'w') as f:
                    f.write('') #erase or start the gt file
            #if logged:
            print('refreshing sythetic')
            images_to_do = self.set_size-self.init_size
            images_per_process = math.ceil(images_to_do/self.num_processes)
            #rounds = math.ceil(images_per_process/self.per_process)
            idx = self.init_size
            if idx<self.set_size:
                tic=timeit.default_timer()
                pool = Pool(processes=self.num_processes)
                #for r in range(rounds):
                chunk = min(20,math.ceil(self.set_size/(4*self.num_processes)))
                jobs = [(self.synth_gen,self.img_height,self.blank_size,time.time()+random.randint(0,999999)) for i in range(self.set_size-self.init_size)]
                created = pool.imap_unordered(create_image, jobs, chunksize=chunk)#images_per_process//(4*self.num_processes))
                with open(self.gt_filename,'a') as f:
                    for gt,img in created:
                        gt=gt.strip() #spaces on ends shouldn't be GT
                        if idx>=self.set_size:
                            break
                        filename = os.path.join(self.directory,'{}.png'.format(idx))
                        if len(img.shape)==3 and img.shape[2]==1:
                            img = img[:,:,0]
                        cv2.imwrite(filename,img)
                        #if not success:
                        #    print('ERROR, failed to write file {} {}'.format(filename,img.shape))
                        #    import pdb;pdb.set_trace()
                        self.labels[idx] = gt
                        f.write(gt+'\n')
                        if not logged:
                            print('refreshing sythetic: {}/{}'.format(idx,self.set_size), end='\r')
                        idx+=1
                        if idx>=self.set_size:
                            break
                #print('all: '+str(timeit.default_timer()-tic))
                pool.terminate()
                print('done refreshing: '+str(timeit.default_timer()-tic))
                    
            self.init_size=0
            self.used=0
