import json
import timeit
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
from glob import iglob
import os
import utils.img_f as img_f
import numpy as np
import math, time
import random, string

from utils import grid_distortion

from utils import string_utils, augmentation
from utils.util import ensure_dir
from utils.yolo_tools import allIOU
from .qa import QADataset
#import pyexiv2
#import piexif

from multiprocessing import Pool, TimeoutError

import random, pickle
PADDING_CONSTANT = -1


def collate(batch):
    return {
            'img': torch.cat([b['img'] for b in batch],dim=0),
            'bb_gt': [b['bb_gt'] for b in batch], #torch.cat([b['bb_gt'] for b in batch],dim=0),
            'imgName': [b['imgName'] for b in batch],
            'scale': [b['scale'] for b in batch],
            'cropPoint': [b['cropPoint'] for b in batch],
            'transcription': [b['transcription'] for b in batch],
            'metadata': [b['metadata'] for b in batch],
            'form_metadata': [b['form_metadata'] for b in batch],
            'questions': [b['questions'] for b in batch],
            'answers': [b['answers'] for b in batch]
            }
def get_height_width_from_list(imgs,pad):
    height = 0
    width = 0
    for img in imgs[1:]:
        height += pad + img.shape[0]
        width = max(img.shape[1],width)
    #height-=pad
    return height,width


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
            img = img_f.resize(img, (0,0), fx=percent, fy=percent)
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

class SynthQADocDataset(QADataset):
    def __init__(self, dirPath, split, config):
        super(SynthQADocDataset, self).__init__(dirPath,split,config)
        from synthetic_text_gen import SyntheticText
        self.color=False
        self.ocr = config['include_ocr'] if 'include_ocr' in config else False
        self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0.15
        self.text_height = config['text_height']
        self.image_size = config['image_size'] if 'image_size' in config else None
        if type(self.image_size) is int:
            self.image_size = (self.image_size,self.image_size)
        self.min_entries = config['min_entries'] if 'min_entries' in config else self.questions
        self.max_entries = config['max_entries'] if 'max_entries' in config else self.questions
        self.change_size = config['change_size'] if 'change_size' in config else False
        self.min_text_height = config['min_text_height'] if 'min_text_height' in config else 8
        self.max_text_height = config['max_text_height'] if 'max_text_height' in config else 32
        self.np_qs = 0.02
        self.use_hw = config['use_hw'] if 'use_hw' in config else False
        self.word_questions = config['word_questions'] if 'word_questions' in config else False
        self.multiline = config['multiline'] if 'multiline' in config else False
        self.max_num_lines = config['max_num_lines'] if 'max_num_lines' in config else 6
        self.tables = config['tables'] if 'tables' in config else False
        assert not self.tables or self.word_questions
        assert not (self.use_hw and self.tables)
        assert not self.tables or self.min_entries is None
        if self.word_questions:
            self.ask_for_value = [
                    'value for "{}"?',
                    'value of "{}"?',
                    'answer for "{}"?',
                    'answer of "{}"?',
                    'response for "{}"?',
                    'response of "{}"?'
                    ]
            self.ask_for_label = [
                    'label for "{}"?',
                    'label of "{}"?',
                    'question for "{}"?',
                    'question of "{}"?',
                    'prompt for "{}"?',
                    'prompt of "{}"?'
                    ]
        if self.use_hw or self.tables:
            self.header_dir = config['header_dir']
            header_gt_filename = os.path.join(self.header_dir,'gt.txt')
        if self.use_hw:
            self.hw_dir = config['hw_dir']
            hw_gt_filename = os.path.join(self.hw_dir,'gt.txt')
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


            self.used=-1
            self.used_instances=0
            self.num_processes = config['num_processes']
            self.per_process = config['per_process'] if 'per_process' in config else 100
            
            if 'create' in config:
                ensure_dir(self.directory)


        self.set_size = config['set_size']
        self.labels, self.init_size = self.readGT(self.gt_filename,self.directory,self.set_size)
        
        if self.init_size>0:
            print('Found synth images 0-{} with labels'.format(self.init_size-1))
            if self.init_size!=self.set_size and 'create' not in config:
                print('Need to finish making dataset!')
                exit(1)
        elif 'create' not in config:
            print('Need to create dataset! {}'.format(self.directory))
            exit(1)

        if self.use_hw or self.tables:
            self.header_labels,header_size = self.readGT(header_gt_filename,self.header_dir)
        if self.use_hw:
            self.hw_labels,hw_size = self.readGT(hw_gt_filename,self.hw_dir)
            
        
        self.images=[]
        for i in range(config['batch_size']*100): #we just randomly generate instances on the fly
            self.images.append({'id':'{}'.format(i), 'imagePath':None, 'annotationPath':0, 'rescaled':1.0, 'imageName':'0'})


    def readGT(self,gt_filename,directory,set_size=None):
        init_size=0
        cur_files = set(os.listdir(directory))
        if os.path.exists(gt_filename):
            with open(gt_filename) as f:
                labels =f.readlines()
            if set_size is None:
                set_size=len(labels)
            store = [None]*set_size

            for i in range(min(set_size,len(labels))):
                if '{}.png'.format(i) in cur_files:
                    init_size+=1
                    store[i]=labels[i].strip().lower()

                elif set_size is None:
                    print('Dataset at {} is missing image {}'.format(directory,i))
                    exit(1)
                else:
                    break
        elif set_size is not None:
            store = [None]*set_size
        else:
            store = []
        return store,init_size


    def __len__(self):
        return len(self.images)

    def max_len(self):
        return self.text_max_len

    def parseAnn(self,annotations,s):
        #make the image
        if self.min_entries is not None:
            num_entries = random.randrange(self.min_entries,self.max_entries+1)
        else:
            num_entries = self.max_entries

        entries=[]
        heights=[]

        if self.use_hw:
            num_hw_entries = int(self.use_hw*num_entries)
            num_entries = num_entries-num_hw_entries

            hw_labels = random.sample(list(enumerate(self.header_labels)),k=num_hw_entries)
            hw_labels = [a+(self.header_dir,True) for a in hw_labels]
            hw_values = random.sample(list(enumerate(self.hw_labels)),k=num_hw_entries)
            hw_values = [a+(self.hw_dir,False) for a in hw_values]


        if self.multiline:
            assert not self.ocr
            labels_linec = [random.randrange(1,self.max_num_lines) if random.random()<self.multiline else 1 for i in range(num_entries)]
            values_linec = [random.randrange(1,self.max_num_lines) if random.random()<self.multiline else 1 for i in range(num_entries)]
        else:
            labels_linec = [1]*num_entries
            values_linec = [1]*num_entries
        total_entries = sum(labels_linec) + sum(values_linec)
        selected = random.sample(list(enumerate(self.labels)),k=total_entries)
        selected = [a+(self.directory,True) for a in selected]

        #labels = selected[:num_entries]
        #values = selected[num_entries:]
        i=0
        labels = []
        for num in labels_linec:
            labels.append(selected[i:i+num])
            i+=num
        values = []
        for num in values_linec:
            values.append(selected[i:i+num])
            i+=num

        if self.use_hw:
            labels+=hw_labels
            values+=hw_values

        #for (label_img_idx,label_text,label_dir,resize_l),(value_img_idx,value_text,value_dir,resize_v) in zip(labels,values):
        for l,v in zip(labels,values):
            pad = random.randrange(5)
            label_imgs=[pad]
            label_texts=[]
            l_h=0
            if self.change_size:
                label_height = random.randrange(self.min_text_height,self.max_text_height)
            for label_img_idx,label_text,label_dir,resize_l in l:
                label_img_path = os.path.join(label_dir,'{}.png'.format(label_img_idx))
                label_img = img_f.imread(label_img_path,False)


                if self.change_size:
                    assert resize_l
                    label_width = round(label_img.shape[1]*label_height/label_img.shape[0])
                    try:
                        label_img = img_f.resize(label_img,(label_height,label_width))
                    except OverflowError as e:
                        print(e)
                        print('image {} to {}'.format(label_img.shape,(label_height,label_width)))
                l_h += label_img.shape[0]+pad
                label_imgs.append(label_img)
                label_texts.append(label_text)
            l_h-=pad
    
            pad = random.randrange(5)
            value_imgs=[pad]
            value_texts=[]
            v_h=0
            if self.change_size:
                if v[0][3]:
                    value_height = random.randrange(self.min_text_height,self.max_text_height)
                else:
                    value_height = random.randrange(value_img.shape[0]-5,value_img.shape[0]+5)
            for value_img_idx,value_text,value_dir,resize_v in v:
                value_img_path = os.path.join(value_dir,'{}.png'.format(value_img_idx))
                value_img = img_f.imread(value_img_path,False)
                if self.change_size:
                    value_width = round(value_img.shape[1]*value_height/value_img.shape[0])
                    value_img = img_f.resize(value_img,(value_height,value_width))
                v_h += value_img.shape[0]+pad
                value_imgs.append(value_img)
                value_texts.append(value_text)
            v_h-=pad

            heights.append(max(l_h,v_h))
            rel_x = random.randrange(10)
            rel_y = random.randrange(-10,10)
            if not self.multiline or len(l)==1:
                entries.append((label_img,label_text,value_img,value_text,rel_x,rel_y))
            else:
                entries.append((label_imgs,label_texts,value_imgs,value_texts,rel_x,rel_y))

        image = np.full((self.image_size[0],self.image_size[1]),255,np.uint8)

        boxes=[]
        trans=[]
        qa=[]
        if self.min_entries is not None:

            y_pos=0
            for ei,(label_img,label_text,value_img,value_text,rel_x,rel_y) in enumerate(entries):
                width = label_img.shape[1] + value_img.shape[1] + rel_x
                if width>=self.image_size[1]:
                    x=0
                else:
                    x = random.randrange(self.image_size[1] - width)
                room_y = self.image_size[0] - sum(heights[ei:])
                assert room_y >= 0
                if ei == len(entries)-1:
                    y = random.randrange(y_pos,room_y)
                else:
                    y = int(random.triangular(y_pos,room_y,y_pos+1))
                
                value_x = x + label_img.shape[1] + rel_x
                value_y = y + rel_y
                value_x = max(0,value_x)
                value_y = max(0,value_y)
                

                vert = value_img.shape[0]-max(value_y+value_img.shape[0]-self.image_size[0],0)
                horz = value_img.shape[1]-max(value_x+value_img.shape[1]-self.image_size[1],0)
                if horz<=0 or vert<=0:
                    continue
                l_vert = label_img.shape[0]-max(y+label_img.shape[0]-self.image_size[0],0)
                l_horz = label_img.shape[1]-max(x+label_img.shape[1]-self.image_size[1],0)
                if l_horz<=0 or l_vert<=0:
                    continue


                image[value_y:value_y+vert,value_x:value_x+horz] = value_img[:vert,:horz]

                image[y:y+l_vert,x:x+l_horz] = label_img[:l_vert,:l_horz]

                y_pos=y+heights[ei]
                
                if self.word_questions=='simple':
                    qa.append(('l~{}'.format(label_text),value_text,None))
                    qa.append(('v~{}'.format(value_text),label_text,None))
                elif self.word_questions:
                    question_to_value = random.choice(self.ask_for_value)
                    question_to_label = random.choice(self.ask_for_label)
                    qa.append((question_to_value.format(label_text),value_text,None))
                    qa.append((question_to_label.format(value_text),label_text,None))
                else:
                    qa.append((label_text,value_text,None))

                if self.ocr:
                    self.addText(label_text,x,y,l_horz,l_vert,value_text,value_x,value_y,horz,vert,boxes,trans,s)
        else:
            #TABLE
            if self.tables and random.random()<self.tables:
                table_x,table_y,table_width,table_height,row_hs,col_hs = self.addTable(image,qa,boxes if self.ocr else None,trans if self.ocr else None)
                if table_x is None:
                    did_table=False
                    row_hs=col_hs=[]
                else:
                    did_table=True
            else:
                did_table=False
                row_hs=col_hs=[]

            #Assign random positions and collect bounding boxes
            full_bbs=torch.FloatTensor(len(entries)+(1 if did_table else 0),4)
            pad_v=1
            pad_h=30
            for ei,(label_img,label_text,value_img,value_text,rel_x,rel_y) in enumerate(entries):
                if type(label_img) is list:
                    label_height, label_width = get_height_width_from_list(label_img[1:],label_img[0])
                    extra_height = label_height*(len(label_img[1:])-1)/len(label_img[1:])
                else:
                    label_height, label_width = label_img.shape
                    extra_height=0
                if type(value_img) is list:
                    value_height, value_width = get_height_width_from_list(value_img[1:],value_img[0])
                    extra_height += value_height*(len(value_img[1:])-1)/len(value_img[1:])
                else:
                    value_height, value_width = value_img.shape

                height = max(label_height,value_height)+abs(rel_y)+pad_v*2 + extra_height
                width = label_width+rel_x+value_width+pad_h*2
                
                if width>=self.image_size[1]:
                    x=0
                else:
                    x = random.randrange(int(self.image_size[1]-width))
                if height>=self.image_size[0]:
                    y=0
                else:
                    y = random.randrange(int(self.image_size[0]-height))
                full_bbs[ei,0]=x
                full_bbs[ei,1]=y
                full_bbs[ei,2]=x+width+1
                full_bbs[ei,3]=y+height+1


            if did_table:
                full_bbs[-1,0]=table_x
                full_bbs[-1,1]=table_y
                full_bbs[-1,2]=table_x+table_width+1
                full_bbs[-1,3]=table_y+table_height+1


            #find intersections and remove an entry until no more intersections
            intersections = allIOU(full_bbs,full_bbs,x1y1x2y2=True)>0
            intersections.diagonal()[:]=0

            def print_iter(intersections):
                s=''
                for r in range(intersections.size(0)):
                    for c in range(intersections.size(1)):
                        s += 'o' if intersections[r,c] else '.'
                    s+='\n'
                print(s)
            
            intersections_per = intersections.sum(dim=0)
            if did_table:
                intersections_per[-1]=0 #clear Table so it is not selected
            removed = set()
            while intersections_per.sum()>0:
                #print_iter(intersections)
                worst_offender = intersections_per.argmax()
                #print('removing {} i:{} (of {})'.format(worst_offender,intersections_per[worst_offender],len(entries)))
                removed.add(worst_offender.item())
                intersections[:,worst_offender]=0
                intersections[worst_offender,:]=0
                intersections_per = intersections.sum(dim=0)
                if did_table:
                    intersections_per[-1]=0 #clear Table so it is not selected

            #removed.sort(reverse=True)
            #for r in removed:
            #    del entries[r]
            not_present_qs=[]
            selected_labels_stripped = []#[l[1].strip() for i,l in enumerate(labels) if i not in removed]
            for ei,(label_img,label_text,value_img,value_text,rel_x,rel_y) in enumerate(entries):
                if ei not in removed:
                    if type(label_img) is list:
                        label_height, label_width = get_height_width_from_list(label_img[1:],label_img[0])
                    else:
                        label_height, label_width = label_img.shape

                    label_x = int(full_bbs[ei,0].item())+pad_h//2
                    label_y = int(full_bbs[ei,1].item())+pad_v//2
                    value_x = label_x + label_width + rel_x
                    value_y = label_y + rel_y

                    if value_y<0:
                        if type(value_img) is list:
                            value_img[1] = value_img[1][-value_y:]
                        else:
                            value_img = value_img[-value_y:]
                        value_y=0


                    if type(label_img) is list:
                        l_imgs = label_img[1:]
                        pad = label_img[0]
                    else:
                        l_imgs = [label_img]
                        pad = 0
                    cur_y=label_y
                    for img in l_imgs:
                        l_vert = img.shape[0]-max(cur_y+img.shape[0]-self.image_size[0],0)
                        l_horz = img.shape[1]-max(label_x+img.shape[1]-self.image_size[1],0)
                        if l_horz<=0 or l_vert<=0:
                            continue
                        image[cur_y:cur_y+l_vert,label_x:label_x+l_horz] = img[:l_vert,:l_horz]
                        cur_y += l_vert+pad

                    cur_y -= l_vert+pad - rel_y
                    if cur_y<0:
                        cur_y=0
                    if type(value_img) is list:
                        v_imgs = value_img[1:]
                        pad = value_img[0]
                    else:
                        v_imgs = [value_img]
                        pad = 0
                    for img in v_imgs:
                        v_vert = img.shape[0]-max(cur_y+img.shape[0]-self.image_size[0],0)
                        v_horz = img.shape[1]-max(value_x+img.shape[1]-self.image_size[1],0)
                        if v_horz<=0 or v_vert<=0:
                            continue
                        image[cur_y:cur_y+v_vert,value_x:value_x+v_horz] = img[:v_vert,:v_horz]
                        cur_y += v_vert+pad
                    
                    if type(label_text) is list:
                        label_text = ' '.join(label_text)
                    selected_labels_stripped.append(label_text.strip())
                    if type(value_text) is list:
                        value_text = ' '.join(value_text)

                    if self.word_questions=='simple':
                        qa.append(('l~{}'.format(label_text),value_text,None))
                        qa.append(('v~{}'.format(value_text),label_text,None))
                    elif self.word_questions:
                        question_to_value = random.choice(self.ask_for_value)
                        question_to_label = random.choice(self.ask_for_label)
                        qa.append((question_to_value.format(label_text),value_text,None))
                        qa.append((question_to_label.format(value_text),label_text,None))
                    else:
                        qa.append((label_text,value_text,None))

                    if self.ocr:
                        self.addText(label_text,label_x,label_y,l_horz,l_vert,value_text,value_x,value_y,v_horz,v_vert,boxes,trans,s)
                else:
                    if type(label_text) is list:
                        label_text = ' '.join(label_text)
                    if label_text.strip() not in selected_labels_stripped:
                        not_present_qs.append(label_text)




        bbs = np.array(boxes)
        
        ocr = trans

        #Add possible not present questions
        to_add_np = len(qa)*self.np_qs #?0.02
        num_np = math.floor(to_add_np)
        if random.random()<to_add_np-num_np: #handle non-whole number correctly
            num_np+=1
        for i in range(num_np):
            if self.tables and random.random()<(self.tables*0.5) and self.word_questions=='simple':
                if random.random()<0.3:
                    while True:
                        q_text = random.choice(self.labels)
                        if q_text.strip() not in col_hs:
                            break
                    qa.append(('ac~{}'.format(q_text),'[ np ]',None))
                elif random.random()<0.6:
                    while True:
                        q_text = random.choice(self.labels)
                        if q_text.strip() not in row_hs:
                            break
                    qa.append(('ar~{}'.format(q_text),'[ np ]',None))
                else:
                    while True:
                        c_text = random.choice(self.labels)
                        if c_text.strip() not in col_hs+row_hs:
                            break
                    while True:
                        r_text = random.choice(self.labels)
                        if r_text.strip() not in row_hs+col_hs:
                            break
                    qa.append(('t~{}~~{}'.format(r_text,c_text),'[ np ]',None))
            else:
                if len(not_present_qs)>0:
                    q_text = not_present_qs.pop()
                else:
                    while True:
                        q_text = random.choice(self.labels)
                        if q_text.strip() not in selected_labels_stripped:
                            break
                if self.word_questions=='simple':
                    qa.append(('l~{}'.format(q_text),'[ np ]',None))
                elif self.word_questions:
                    question = random.choice(self.ask_for_value)
                    qa.append((question.format(q_text),'[ np ]',None))
                else:
                    qa.append((q_text,'[ np ]',None))
        
        if self.tables and self.word_questions=='simple':
            if did_table:
                qa.append(('t>','yes',None))
                rows = ', '.join(row_hs)
                qa.append(('rh>',rows,None))
                cols = ', '.join(col_hs)
                qa.append(('ch>',cols,None))
            else:
                qa.append(('t>','no',None))



        if self.questions<len(qa):
            qa = random.sample(qa,k=self.questions)

        return bbs, list(range(bbs.shape[0])), ocr, {'image':image}, {}, qa

    
    def addText(self,label_text,label_x,label_y,l_horz,l_vert,value_text=None,value_x=None,value_y=None,v_horz=None,v_vert=None,boxes=None,trans=None,s=1):
        #corrupt text
        label_text = self.corrupt(label_text)
        if value_text is not None:
            value_text = self.corrupt(value_text)
        
        if random.random()<0.5 and value_text is not None:
            #single line
            if random.random()>0.1:
                full_text = label_text+' '+value_text
                lX = label_x
                rX = value_x+v_horz
                tY = label_y
                bY = max(label_y+l_vert,value_y+v_vert)
                lX+=random.gauss(0,5)
                rX+=random.gauss(0,5)
                tY+=random.gauss(0,5)
                bY+=random.gauss(0,5)
                if lX>rX:
                    tmp=lX
                    lX=rX
                    rX=tmp
                if tY>bY:
                    tmp=tY
                    tY=bY
                    bY=tmp

                bb=[None]*16
                bb[0]=lX*s
                bb[1]=bY*s
                bb[2]=lX*s
                bb[3]=tY*s
                bb[4]=rX*s
                bb[5]=tY*s
                bb[6]=rX*s
                bb[7]=bY*s
                bb[8]=s*(lX+rX)/2.0
                bb[9]=s*bY
                bb[10]=s*(lX+rX)/2.0
                bb[11]=s*tY
                bb[12]=s*lX
                bb[13]=s*(tY+bY)/2.0
                bb[14]=s*rX
                bb[15]=s*(tY+bY)/2.0
                boxes.append(bb)
                trans.append(full_text)
        else:
            #seperate
            if random.random()>0.1:
                #label
                lX = label_x
                rX = label_x+l_horz
                tY = label_y
                bY = label_y+l_vert
                lX+=random.gauss(0,5)
                rX+=random.gauss(0,5)
                tY+=random.gauss(0,5)
                bY+=random.gauss(0,5)
                if lX>rX:
                    tmp=lX
                    lX=rX
                    rX=tmp
                if tY>bY:
                    tmp=tY
                    tY=bY
                    bY=tmp

                bb=[None]*16
                bb[0]=lX*s
                bb[1]=bY*s
                bb[2]=lX*s
                bb[3]=tY*s
                bb[4]=rX*s
                bb[5]=tY*s
                bb[6]=rX*s
                bb[7]=bY*s
                bb[8]=s*(lX+rX)/2.0
                bb[9]=s*bY
                bb[10]=s*(lX+rX)/2.0
                bb[11]=s*tY
                bb[12]=s*lX
                bb[13]=s*(tY+bY)/2.0
                bb[14]=s*rX
                bb[15]=s*(tY+bY)/2.0
                boxes.append(bb)
                trans.append(label_text)

            if random.random()>0.1 and value_text is not None:
                #value
                lX = value_x
                rX = value_x+v_horz
                tY = value_y
                bY = value_y+v_vert
                lX+=random.gauss(0,5)
                rX+=random.gauss(0,5)
                tY+=random.gauss(0,5)
                bY+=random.gauss(0,5)
                if lX>rX:
                    tmp=lX
                    lX=rX
                    rX=tmp
                if tY>bY:
                    tmp=tY
                    tY=bY
                    bY=tmp

                bb=[None]*16
                bb[0]=lX*s
                bb[1]=bY*s
                bb[2]=lX*s
                bb[3]=tY*s
                bb[4]=rX*s
                bb[5]=tY*s
                bb[6]=rX*s
                bb[7]=bY*s
                bb[8]=s*(lX+rX)/2.0
                bb[9]=s*bY
                bb[10]=s*(lX+rX)/2.0
                bb[11]=s*tY
                bb[12]=s*lX
                bb[13]=s*(tY+bY)/2.0
                bb[14]=s*rX
                bb[15]=s*(tY+bY)/2.0
                boxes.append(bb)
                trans.append(value_text)


    def refresh_data(self,logged=False):
        if self.init_size==0:
            assert not os.path.exists(self.gt_filename,)
            #with open(self.gt_filename,'w') as f:
            #    f.write('') #erase or start the gt file
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
            jobs = [(self.synth_gen,self.text_height,None,time.time()+random.randint(0,999999)) for i in range(self.set_size-self.init_size)]
            created = pool.imap_unordered(create_image, jobs, chunksize=chunk)#images_per_process//(4*self.num_processes))
            with open(self.gt_filename,'a') as f:
                for gt,img in created:
                    gt=gt.strip() #spaces on ends shouldn't be GT
                    if idx>=self.set_size:
                        break
                    filename = os.path.join(self.directory,'{}.png'.format(idx))
                    if len(img.shape)==3 and img.shape[2]==1:
                        img = img[:,:,0]
                    img_f.imwrite(filename,img)
                    self.labels[idx] = gt
                    f.write(gt+'\n')
                    if not logged:
                        print('refreshing sythetic: {}/{}'.format(idx,self.set_size), end='\r')
                    if idx%100==0:
                        f.flush()
                    idx+=1
                    if idx>=self.set_size:
                        break
            #print('all: '+str(timeit.default_timer()-tic))
            pool.terminate()
            print('done refreshing: '+str(timeit.default_timer()-tic))
                    
            self.init_size=0
    def corrupt(self,s):
        new_s=''
        for c in s:
            r = random.random()
            if r<self.corruption_p/3:
                pass
            elif r<self.corruption_p*2/3:
                new_s+=random.choice(string.ascii_letters)
            elif r<self.corruption_p:
                if random.random()<0.5:
                    new_s+=c+random.choice(string.ascii_letters)
                else:
                    new_s+=random.choice(string.ascii_letters)+c
            else:
                new_s+=c
        return new_s

    def addTable(self,image,all_q_a,boxes,trans):
        #Taken from FUNSD_QA
        num_rows=random.randrange(1,15)
        num_cols=random.randrange(1,10)

        mean_height = random.randrange(self.min_text_height+1,self.max_text_height)

        table_entries = random.choices(list(enumerate(self.header_labels)),k=num_rows*num_cols+num_rows+num_cols)
        #table_entries = [(img_f.imread(os.path.join(self.header_dir,'{}.png'.format(e[0]))),e[1]) for e in table_entries]
        table_entries_1d = []
        for num,label in table_entries:
            img = img_f.imread(os.path.join(self.header_dir,'{}.png'.format(num)))
            if self.change_size:
                height = int(random.gauss(mean_height,4))#random.randrange(self.min_text_height,img.shape[0])
                width = round(img.shape[1]*height/img.shape[0])
                if height>1 and width>1:
                    img = img_f.resize(img,(height,width))
            table_entries_1d.append((img,label))
        row_headers = table_entries_1d[-num_rows:]
        col_headers = table_entries_1d[-(num_rows+num_cols):-num_rows]
        table_entries = table_entries_1d[:-(num_rows+num_cols)]
        table_entries_2d = []
        for r in range(num_rows):
            table_entries_2d.append(table_entries_1d[r*num_cols:(r+1)*num_cols])
        table_entries = table_entries_2d


        table_x = random.randrange(self.image_size[1]*0.75)
        table_y = random.randrange(self.image_size[0]*0.75)

        padding = random.randrange(0,30)

        #find the height of each row and cut rows that spill off page
        max_height=0
        for c in range(num_cols):
            max_height = max(max_height,col_headers[c][0].shape[0])
        total_height = max_height+padding
        height_col_heading = max_height+padding
        
        if total_height+table_y >= self.image_size[0]:
            #NO TABLE
            return None,None,None,None, None, None
        height_row=[0]*num_rows
        for r in range(num_rows):
            max_height = row_headers[r][0].shape[0]
            for c in range(num_cols):
                max_height = max(max_height,table_entries[r][c][0].shape[0])
            height_row[r] = max_height+padding
            total_height+= max_height+padding

            if total_height+table_y >= self.image_size[0]:
                num_rows = r
                if num_rows==0:
                    return None,None,None,None,None,None #NO TABLE
                total_height -= height_row[r]
                row_headers=row_headers[:num_rows]
                height_row=height_row[:num_rows]
                break

        #find the width of each rowumn and cut rowumns that spill off page
        max_width=0
        for r in range(num_rows):
            max_width = max(max_width,row_headers[r][0].shape[1])
        total_width = max_width+padding
        width_row_heading = max_width+padding
        
        if total_width+table_x >= self.image_size[1]:
            #NO TABLE
            return None,None,None,None, None, None
        width_col=[0]*num_cols
        for c in range(num_cols):
            max_width = col_headers[c][0].shape[1]
            for r in range(num_rows):
                max_width = max(max_width,table_entries[r][c][0].shape[1])
            width_col[c] = max_width+padding
            total_width+= max_width+padding

            if total_width+table_x >= self.image_size[1]:
                num_cols = c
                if num_cols==0:
                    return None,None,None,None, None, None#NO TABLE
                total_width -= width_col[c]
                col_headers=col_headers[:num_cols]
                width_col=width_col[:num_cols]
                break
    
        #build table
        table_values=[]

        #put row headers in image
        cur_y = height_col_heading+table_y
        for r in range(num_rows):
            if width_row_heading-padding==row_headers[r][0].shape[1]:
                x=table_x
            else:
                x=table_x + random.randrange(0,width_row_heading-padding-row_headers[r][0].shape[1])
            if height_row[r]-padding==row_headers[r][0].shape[0]:
                y=cur_y
            else:
                y=cur_y + random.randrange(0,height_row[r]-padding-row_headers[r][0].shape[0])
            cur_y += height_row[r]

            image[y:y+row_headers[r][0].shape[0],x:x+row_headers[r][0].shape[1]] = row_headers[r][0]
            if boxes is not None:
                self.addText(row_headers[r][1],x,y,row_headers[r][0].shape[1],row_headers[r][0].shape[0],boxes=boxes,trans=trans)

        #put col headers in image
        cur_x = width_row_heading+table_x
        for c in range(num_cols):
            if height_col_heading-padding==col_headers[c][0].shape[0]:
                y=table_y
            else:
                y=table_y + random.randrange(0,height_col_heading-padding-col_headers[c][0].shape[0])
            if width_col[c]-padding==col_headers[c][0].shape[1]:
                x=cur_x
            else:
                x=cur_x + random.randrange(0,width_col[c]-padding-col_headers[c][0].shape[1])
            cur_x += width_col[c]

            image[y:y+col_headers[c][0].shape[0],x:x+col_headers[c][0].shape[1]] = col_headers[c][0]
            if boxes is not None:
                self.addText(col_headers[c][1],x,y,col_headers[c][0].shape[1],col_headers[c][0].shape[0],boxes=boxes,trans=trans)

        #put in entries
        cur_x = width_row_heading+table_x
        for c in range(num_cols):
            cur_y = height_col_heading+table_y
            for r in range(num_rows):
                if random.random()>0.2: #sometimes skip an entry
                    if width_col[c]-padding==table_entries[r][c][0].shape[1]:
                        x=cur_x
                    else:
                        x=cur_x + random.randrange(0,width_col[c]-padding-table_entries[r][c][0].shape[1])
                    if height_row[r]-padding==table_entries[r][c][0].shape[0]:
                        y=cur_y
                    else:
                        y=cur_y + random.randrange(0,height_row[r]-padding-table_entries[r][c][0].shape[0])

                    image[y:y+table_entries[r][c][0].shape[0],x:x+table_entries[r][c][0].shape[1]] = table_entries[r][c][0]
                    table_values.append((col_headers[c][1],row_headers[r][1],table_entries[r][c][1],x,y))
                    if boxes is not None:
                        self.addText(table_entries[r][c][1],x,y,table_entries[r][c][0].shape[1],table_entries[r][c][0].shape[0],boxes=boxes,trans=trans)
                else:
                    #table_values.append((col_headers[c][1],row_headers[r][1],'[np]',cur_x,cur_y))
                    if self.word_questions=='simple':
                        all_q_a.append(('t~{}~~{}'.format(row_headers[r][1],col_headers[c][1]),'[ blank ]',None))
                        all_q_a.append(('t~{}~~{}'.format(col_headers[c][1],row_headers[r][1]),'[ blank ]',None))
                    else:
                        all_q_a.append(('value of "{}" and "{}"?'.format(row_headers[r][1],col_headers[c][1]),'[np]',None))
                        all_q_a.append(('value of "{}" and "{}"?'.format(col_headers[c][1],row_headers[r][1]),'[np]',None))
                
                cur_y += height_row[r]
            cur_x += width_col[c]

        #add lines for headers
        line_thickness_h = random.randrange(1,max(2,min(10,padding)))
        #top
        img_f.line(image,
                (max(0,table_x+random.randrange(-5,5)),table_y+height_col_heading-random.randrange(0,1+padding)),
                (min(self.image_size[1]-1,table_x+total_width+random.randrange(-5,5)),table_y+height_col_heading-random.randrange(0,1+padding)),
                random.randrange(0,100),
                line_thickness_h
                )
        #side
        img_f.line(image,
                (table_x+width_row_heading-random.randrange(0,padding+1),max(0,table_y+random.randrange(-5,5))),
                (table_x+width_row_heading-random.randrange(0,padding+1),min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                random.randrange(0,100),
                line_thickness_h
                )

        #outside of headers?
        if random.random()<0.5:
            line_thickness = random.randrange(1,max(2,min(10,padding)))
            #top
            img_f.line(image,
                    (max(0,table_x+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)),
                    (min(self.image_size[1]-1,table_x+total_width+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)),
                    random.randrange(0,100),
                    line_thickness
                    )
            #side
            img_f.line(image,
                    (table_x-random.randrange(0,padding+1),max(0,table_y+random.randrange(-5,5))),
                    (table_x-random.randrange(0,padding+1),min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                    random.randrange(0,100),
                    line_thickness
                    )

        #value outline?
        if random.random()<0.5:
            line_thickness = random.randrange(1,max(2,min(10,padding)))
            #bot
            img_f.line(image,
                    (max(0,table_x+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+total_height),
                    (min(self.image_size[1]-1,table_x+total_width+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+total_height),
                    random.randrange(0,100),
                    line_thickness
                    )
            #right
            img_f.line(image,
                    (table_x-random.randrange(0,padding+1)+total_width,max(0,table_y+random.randrange(-5,5))),
                    (table_x-random.randrange(0,padding+1)+total_width,min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                    random.randrange(0,100),
                    line_thickness
                    )

        #all inbetween lines?
        if random.random()<0.5:
            line_thickness = random.randrange(1,max(2,line_thickness_h))
            #horz
            cur_height = height_col_heading
            for r in range(num_rows-1):
                cur_height += height_row[r]
                img_f.line(image,
                        (max(0,table_x+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+cur_height),
                        (min(self.image_size[1]-1,table_x+total_width+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+cur_height),
                        random.randrange(0,100),
                        line_thickness
                        )
            #right
            cur_width = width_row_heading
            for c in range(num_cols-1):
                cur_width += width_col[c]
                img_f.line(image,
                        (table_x-random.randrange(0,padding+1)+cur_width,max(0,table_y+random.randrange(-5,5))),
                        (table_x-random.randrange(0,padding+1)+cur_width,min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                        random.randrange(0,100),
                        line_thickness
                        )

        
        #now, optionally add the other lines

        
        #add all possible questions
        col_vs=defaultdict(list)
        row_vs=defaultdict(list)
        ambiguous=set()
        for (col_h,row_h,v,x,y) in table_values:
            if col_h is not None and row_h is not None:
                if self.word_questions=='simple':
                    all_q_a.append(('t~{}~~{}'.format(row_h,col_h),v,None))
                    all_q_a.append(('t~{}~~{}'.format(col_h,row_h),v,None))
                else:
                    if random.random()<0.5:
                        all_q_a.append(('value of "{}" and "{}"?'.format(row_h,col_h),v,None))
                        all_q_a.append(('value of "{}" and "{}"?'.format(col_h,row_h),v,None))
                    else:
                        all_q_a.append(('value in "{}" and "{}"?'.format(row_h,col_h),v,None))
                        all_q_a.append(('value in "{}" and "{}"?'.format(col_h,row_h),v,None))
            if v not in ambiguous:
                if self.word_questions=='simple':
                    if row_h is not None:
                        all_q_a.append(('ri~{}'.format(v),row_h,None))
                    if col_h is not None:
                        all_q_a.append(('ci~{}'.format(v),col_h,None))
                else:
                    if row_h is not None:
                        all_q_a.append(('row that "{}" is in?'.format(v),row_h,None))
                    if col_h is not None:
                        all_q_a.append(('column that "{}" is in?'.format(v),col_h,None))

            if col_h is not None:
                col_vs[col_h].append((v,y))
            if row_h is not None:
                row_vs[row_h].append((v,x))

        for row_h, vs in row_vs.items():
            if row_h not in ambiguous:
                vs.sort(key=lambda a:a[1])
                a=vs[0][0]
                for v,x in vs[1:]:
                    a+=', '+v
                if self.word_questions=='simple':
                    all_q_a.append(('ar~{}'.format(row_h),a,None))
                else:
                    if random.random()<0.5:
                        all_q_a.append(('all values in row "{}"?'.format(row_h),a,None))
                    else:
                        all_q_a.append(('all values of row "{}"?'.format(row_h),a,None))
        for col_h, vs in col_vs.items():
            if col_h not in ambiguous:
                vs.sort(key=lambda a:a[1])
                a=vs[0][0]
                for v,y in vs[1:]:
                    a+=', '+v
                if self.word_questions=='simple':
                    all_q_a.append(('ac~{}'.format(col_h),a,None))
                else:
                    if random.random()<0.5:
                        all_q_a.append(('all values in column "{}"?'.format(col_h),a,None))
                    else:
                        all_q_a.append(('all values of column "{}"?'.format(col_h),a,None))
        return table_x-10, table_y-10, total_width+20, total_height+20, [r[1].strip() for r in row_headers], [c[1].strip() for c in col_headers]
