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
from .qa import QADataset
#import pyexiv2
#import piexif

from multiprocessing import Pool, TimeoutError

import random, pickle
PADDING_CONSTANT = -1


def collate(batch):
    return {
            'img': torch.cat([b['img'] for b in batch],dim=0),
            'bb_gt': torch.cat([b['bb_gt'] for b in batch],dim=0),
            'imgName': [b['imgName'] for b in batch],
            'scale': [b['scale'] for b in batch],
            'cropPoint': [b['cropPoint'] for b in batch],
            'transcription': [b['transcription'] for b in batch],
            'metadata': [b['metadata'] for b in batch],
            'form_metadata': [b['form_metadata'] for b in batch],
            'questions': [b['questions'] for b in batch],
            'answers': [b['answers'] for b in batch]
            }

def create_image(x):
    synth_gen,text_height,image_size,seed = x
    random.seed(seed) #ugh, didn't realize the processes inherit the same random state
    np.random.seed(random.randint(0,99999999))
    while True:
        img, gt, font_idx = synth_gen.getSample()
        if text_height is not None and img.shape[0] != text_height:
            percent = float(text_height) / img.shape[0]
            if percent<=0:
                continue
            #if img.shape[1]*percent > max_width:
            #    percent = max_width/img.shape[1]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent)
            if img.shape[0]<text_height:
                diff = text_height-img.shape[0]
                img = np.pad(img,((diff//2,diff//2+diff%2),(0,0)),'constant',constant_values=0)
        img = (255*(1-img)).astype(np.uint8)
        if image_size is not None:
            page = np.ones((image_size,image_size),dtype=np.uint8)
            page*=255
            x = random.randrange(0,image_size-img.shape[1])
            y = random.randrange(0,image_size-img.shape[0])
            page[y:y+img.shape[0],x:x+img.shape[1]]=img
            img=page
        break
    return gt,img

class SynthQADataset(QADataset):
    def __init__(self, dirPath, split, config):
        super(SynthQADataset, self).__init__(dirPath,split,config)
        from synthetic_text_gen import SyntheticText
        self.color=False
        self.text_height = config['text_height']
        self.image_size = config['image_size'] if 'image_size' in config else None
        text_len = config['max_chars'] if 'max_chars' in config else 35
        self.text_max_len=text_len
        char_set_path = config['char_file']
        self.directory = dirPath
        self.gt_filename = os.path.join(self.directory,'gt.txt')
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

            gen_type = config['gen_type'] if 'gen_type' in config else 'normal'

            if gen_type=='cropped_aug':
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
            self.used=-1
            self.used_instances=0
            self.num_processes = config['num_processes']
            self.per_process = config['per_process'] if 'per_process' in config else 100
            
            ensure_dir(self.directory)

            self.labels = [None]*self.set_size
            self.images= [None]*self.set_size

            self.init_size=0
            cur_files = set(os.listdir(self.directory))
            try:
                with open(self.gt_filename) as f:
                    labels =f.readlines()

                for i in range(min(self.set_size,len(labels))):
                    if '{}.png'.format(i) in cur_files:
                        self.init_size+=1
                        self.labels[i]=labels[i].strip()
                        self.images[i]={
                            'id':'{}'.format(i),
                            'imagePath': os.path.join(self.directory,'{}.png'.format(i)),
                            'annotationPath':i,
                            'rescaled':1.0,
                            'imageName':'{}'.format(i)
                            }

                    else:
                        break
                if self.init_size>0:
                    print('Found synth images 0-{} with labels'.format(self.init_size-1))
                    if self.init_size!=self.set_size:
                        print('Need to finish making dataset!')
                        exit(1)
                else:
                    print('Need to create dataset!')
                    exit(1)
            except:
                self.init_size=0


        else:
            self.images=[]
            self.set_size=min(config['set_size'],2000)
            self.train=False
            self.augmentation=None
            self.include_stroke_aug=False
            self.use_fg_mask=False



    def __len__(self):
        return len(self.images)

    def max_len(self):
        return self.text_max_len

    def parseAnn(self,annotations,s):
        bbs = np.zeros([0,0])
        
        ocr = []
        qa = [('>',self.labels[annotations],None)]
        return bbs, list(range(bbs.shape[1])), ocr, {}, {}, qa



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

    def refresh_data(self,logged=True):
        if self.init_size==0:
            with open(self.gt_filename,'w') as f:
                f.write('') #erase or start the gt file
        #if logged:
        images_to_do = self.set_size-self.init_size
        images_per_process = math.ceil(images_to_do/self.num_processes)
        #rounds = math.ceil(images_per_process/self.per_process)
        idx = self.init_size
        if idx<self.set_size:
            print('refreshing sythetic')
            tic=timeit.default_timer()
            pool = Pool(processes=self.num_processes)
            #for r in range(rounds):
            chunk = min(20,math.ceil(self.set_size/(4*self.num_processes)))
            jobs = [(self.synth_gen,self.text_height,self.image_size,time.time()+random.randint(0,999999)) for i in range(self.set_size-self.init_size)]
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
                    self.images[idx]={
                        'id':'{}'.format(idx),
                        'imagePath': filename,
                        'annotationPath':idx,
                        'rescaled':1.0,
                        'imageName':'{}'.format(idx)
                        }
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
