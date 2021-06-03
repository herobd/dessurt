import json, pickle
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
        self.text_height = config['text_height']
        self.image_size = config['image_size'] if 'image_size' in config else None
        self.min_entries = config['min_entries'] if 'min_entries' in config else self.questions
        self.max_entries = config['max_entries'] if 'max_entries' in config else self.questions
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

            self.init_size=0
            cur_files = set(os.listdir(self.directory))
            if os.path.exists(self.gt_filename):
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
                    if self.init_size!=self.set_size and 'create' not in config:
                        print('Need to finish making dataset!')
                        exit(1)
                elif 'create' not in config:
                    print('Need to create dataset!')
                    exit(1)
            
            self.images=[]
            for i in range(config['batch_size']*100): #we just randomly generate instances on the fly
                self.images.append({'id':'{}'.format(i), 'imagePath':None, 'annotationPath':0, 'rescaled':1.0, 'imageName':'0'})


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
        #make the image
        if self.min_entries is not None:
            num_entries = random.randrange(self.min_entries,self.max_entries+1)
        else:
            num_entries = self.max_entries
        selected = random.choices(list(enumerate(self.labels)),k=num_entries*2)
        labels = selected[:num_entries]
        values = selected[num_entries:]
        entries=[]
        heights=[]
        for ei in range(num_entries):
            label_img_idx,label_text = labels[ei]
            label_img_path = os.path.join(self.directory,'{}.png'.format(label_img_idx))
            label_img = img_f.imread(label_img_path,False)
            value_img_idx,value_text = values[ei]
            value_img_path = os.path.join(self.directory,'{}.png'.format(value_img_idx))
            value_img = img_f.imread(value_img_path,False)

            heights.append(max(label_img.shape[0],value_img.shape[0]))
            rel_x = random.randrange(10)
            rel_y = random.randrange(-10,10)
            entries.append((label_img,label_text,value_img,value_text,rel_x,rel_y))

        image = np.full((self.image_size,self.image_size),255,np.uint8)

        boxes=[]
        trans=[]
        qa=[]
        if self.min_entries is not None:

            y_pos=0
            for ei,(label_img,label_text,value_img,value_text,rel_x,rel_y) in enumerate(entries):
                width = label_img.shape[1] + value_img.shape[1] + rel_x
                if width>=self.image_size:
                    x=0
                else:
                    x = random.randrange(self.image_size - width)
                room_y = self.image_size - sum(heights[ei:])
                assert room_y >= 0
                if ei == len(entries)-1:
                    y = random.randrange(y_pos,room_y)
                else:
                    y = int(random.triangular(y_pos,room_y,y_pos+1))
                
                value_x = x + label_img.shape[1] + rel_x
                value_y = y + rel_y
                value_x = max(0,value_x)
                value_y = max(0,value_y)
                

                vert = value_img.shape[0]-max(value_y+value_img.shape[0]-self.image_size,0)
                horz = value_img.shape[1]-max(value_x+value_img.shape[1]-self.image_size,0)
                if horz<=0 or vert<=0:
                    continue
                l_vert = label_img.shape[0]-max(y+label_img.shape[0]-self.image_size,0)
                l_horz = label_img.shape[1]-max(x+label_img.shape[1]-self.image_size,0)
                if l_horz<=0 or l_vert<=0:
                    continue


                image[value_y:value_y+vert,value_x:value_x+horz] = value_img[:vert,:horz]

                image[y:y+l_vert,x:x+l_horz] = label_img[:l_vert,:l_horz]

                y_pos=y+heights[ei]

                qa.append((label_text,value_text,None))

                if self.ocr:
                    self.addText(label_text,x,y,l_horz,l_vert,value_text,value_x,value_y,horz,vert,boxes,trans,s)
        else:
            #Assign random positions and collect bounding boxes
            full_bbs=torch.FloatTensor(len(entries),4)
            pad_v=1
            pad_h=30
            for ei,(label_img,label_text,value_img,value_text,rel_x,rel_y) in enumerate(entries):
                height = max(label_img.shape[0],value_img.shape[0])+abs(rel_y)+pad_v*2
                width = label_img.shape[1]+rel_x+value_img.shape[1]+pad_h*2
                
                if width>=self.image_size:
                    x=0
                else:
                    x = random.randrange(self.image_size-width)
                if height>=self.image_size:
                    y=0
                else:
                    y = random.randrange(self.image_size-height)
                full_bbs[ei,0]=x
                full_bbs[ei,1]=y
                full_bbs[ei,2]=x+width+1
                full_bbs[ei,3]=y+height+1

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
            removed = set()
            while intersections_per.sum()>0:
                #print_iter(intersections)
                worst_offender = intersections_per.argmax()
                #print('removing {} i:{} (of {})'.format(worst_offender,intersections_per[worst_offender],len(entries)))
                removed.add(worst_offender.item())
                intersections[:,worst_offender]=0
                intersections[worst_offender,:]=0
                intersections_per = intersections.sum(dim=0)

            #removed.sort(reverse=True)
            #for r in removed:
            #    del entries[r]
            
            for ei,(label_img,label_text,value_img,value_text,rel_x,rel_y) in enumerate(entries):
                if ei in removed:
                    continue
                #    label_img = 256-label_img
                #    value_img = 256-value_img

                label_x = int(full_bbs[ei,0].item())+pad_h//2
                label_y = int(full_bbs[ei,1].item())+pad_v//2
                value_x = label_x + label_img.shape[1] + rel_x
                value_y = label_y + rel_y

                if value_y<0:
                    value_img = value_img[-value_y:]
                    value_y=0

                v_vert = value_img.shape[0]-max(value_y+value_img.shape[0]-self.image_size,0)
                v_horz = value_img.shape[1]-max(value_x+value_img.shape[1]-self.image_size,0)
                if v_horz<=0 or v_vert<=0:
                    continue
                l_vert = label_img.shape[0]-max(label_y+label_img.shape[0]-self.image_size,0)
                l_horz = label_img.shape[1]-max(label_x+label_img.shape[1]-self.image_size,0)
                if l_horz<=0 or l_vert<=0:
                    continue

                image[value_y:value_y+v_vert,value_x:value_x+v_horz] = value_img[:v_vert,:v_horz]

                image[label_y:label_y+l_vert,label_x:label_x+l_horz] = label_img[:l_vert,:l_horz]

                qa.append((label_text,value_text,None))

                if self.ocr:
                    self.addText(label_text,label_x,label_y,l_horz,l_vert,value_text,value_x,value_y,v_horz,v_vert,boxes,trans,s)




        bbs = np.array(boxes)
        
        ocr = trans
        if self.questions<len(qa):
            qa = random.sample(qa,k=self.questions)
        return bbs, list(range(bbs.shape[0])), ocr, {'image':image}, {}, qa

    
    def addText(self,label_text,label_x,label_y,l_horz,l_vert,value_text,value_x,value_y,v_horz,v_vert,boxes,trans,s):
        #corrupt text
        label_text = self.corrupt(label_text)
        value_text = self.corrupt(value_text)
        
        if random.random()<0.5:
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

            if random.random()>0.1:
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
            random
            if r<0.1:
                pass
            elif r<0.2:
                new_s+=random.choice(string.ascii_letters)
            elif r<0.3:
                if random.random()<0.5:
                    new_s+=c+random.choice(string.ascii_letters)
                else:
                    new_s+=random.choice(string.ascii_letters)+c
            else:
                new_s+=c
        return new_s
