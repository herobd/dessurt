import torch
import numpy as np
import json
import logging
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
import timeit
from data_sets.gen_daemon import GenDaemon
from data_sets.para_qa_dataset import makeMLMInstance
from utils import augmentation

from transformers import BartTokenizer, BartForConditionalGeneration

import utils.img_f as img_f


def collate(batch):
    return {
            'img': torch.cat([b['img'] for b in batch],dim=0),
            'imgName': [b['imgName'] for b in batch],
            'questions': [b['questions'] for b in batch],
            'answers': [b['answers'] for b in batch],
            'mask_label': None,
            'mask_labels_batch_mask': None,
            "bart_logits": torch.cat([b['bart_logits'] for b in batch],dim=0) if 'bart_logits' in batch[0] and batch[0]['bart_logits'] is not None else None,
            "bart_last_hidden": torch.cat([b['bart_last_hidden'] for b in batch],dim=0) if 'bart_last_hidden' in batch[0] and batch[0]['bart_last_hidden'] is not None else None,
            "distill_loss_mask": torch.cat([b['distill_loss_mask'] for b in batch],dim=0) if 'distill_loss_mask' in batch[0] and batch[0]['distill_loss_mask'] is not None else None,
            }
#This used to perform distillation. It generates the images and pre-comutes the BART logits on the text
class DistilBartDataset(torch.utils.data.Dataset):


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(DistilBartDataset, self).__init__()

        if split=='train':
            no_distill = config.get('no_distill',False) #was used to test just Text Infilling Task, ended up moving it as a task in ParaQADataset
            self.loss_mask = config.get('loss_mask',True) #only compute the loss on the masked tokens, rather than all
            
            if os.path.exists('./cache_huggingface/bart-large'):
                model_id = './cache_huggingface/bart-large'
            else:
                model_id = 'facebook/bart-large'

            if not no_distill or self.loss_mask:
                self.tokenizer = BartTokenizer.from_pretrained(model_id)
                self.blank_token = 50264
            if not no_distill:
                self.model = BartForConditionalGeneration.from_pretrained(model_id)
                self.model.eval()
            else:
                self.model = None

        self.max_auto_tokens = config['max_auto_tokens'] #max BART is allowed to do

        #maybe should have turned these off...
        self.augment_shade = config.get('augment_shade',True)
        self.rotate_std_dev = (math.pi/180) * config.get('rotate_std_dev',1)
        self.scale_std_dev = config.get('scale_std_dev',0.05)
        self.aug_params = config.get('aug_params',{})


        font_dir = dirPath
        self.gen_daemon = GenDaemon(font_dir,clear_fonts=config.get('clear_fonts',False))
        self.prev_words = None

        self.image_size = config['image_size']
        self.min_text_height = config['min_text_height'] if 'min_text_height' in config else 8
        self.max_text_height = config['max_text_height'] if 'max_text_height' in config else 32
        self.min_para_width = config.get('min_para_width', 0.5)


        self.held_instance=None
        self.used_held = 0
        self.max_used_held = config['prefetch_factor']//2 if 'prefetch_factor' in config else 2
        self.logger = logging.getLogger(self.__class__.__name__)

    def __len__(self):
        return 1000

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,recur=0):
        if recur>15:
            return None #checking to be sure an infinite loop wasn't happening

        image_h,image_w = self.image_size

        if self.held_instance is not None:
            image,ocr = self.held_instance
            self.used_held+=1
            if self.used_held>=self.max_used_held:
                self.held_instance=None
        else:
            
            image = np.zeros([image_h,image_w],dtype=np.uint8)

            ocr=[] #blocks
            success=True
            while success:
                #we'll add as many as we can fit (without trying too hard)
                success = self.addBlock(ocr,image)

            self.held_instance= (image,ocr)
            self.used_held=1
        

        words,to_remove,target_string,block = makeMLMInstance(ocr)
        if words is None:
            image=None
            return self.getitem(index,recur+1)



        if self.model is not None or self.loss_mask:
            bart_input_string = []
            for word in words:
                if word is not None:
                    bart_input_string.append(word[0]['text'])
                else:
                    bart_input_string.append('<mask>')
            bart_input_string = ' '.join(bart_input_string)

            input_ids = self.tokenizer([bart_input_string], return_tensors='pt')['input_ids']
            gt_input_ids = self.tokenizer([target_string], return_tensors='pt')['input_ids']
            gt_input_ids = gt_input_ids[:,:self.max_auto_tokens]

        if self.model is not None:
            with torch.no_grad():
                try:
                    #seems fine to do this on CPU
                    bart_out = self.model(input_ids, labels=gt_input_ids,output_hidden_states=True)
                except IndexError as e:
                    print(e)
                    print('bad index, probably: {} or {}'.format(input_ids.max(),gt_input_ids.max()))
                    print(bart_input_string)
                    print(target_string)
                    image=None
                    return self.getitem(index,recur+1)
            logits = bart_out.logits
            last_hidden = bart_out.decoder_hidden_states[-1]
        else:
            logits = None
            last_hidden = None

        if self.loss_mask:
            #Ideally, we only compute the loss on masked tokens
            #however, we have to break the masks on words (for visual masking) which doesn't correspond to the tokenization
            #So we'll compute the insertions and deletions in the Levenstein distance betweem the target and input
            #From there we'll imply which tokens in the target are important and shouldn't be masked.
            #This isn't perfect, but should be closer than no mask at all
            #We also will ignore the insertion masks (at beginning and end only in our data) as it's unclear that the distribution of words would be the same between the models
            dynamic_prog = [None]*(input_ids.shape[1])
            for ii,input_id in enumerate(input_ids[0]):
                dynamic_prog[ii] = [None]*(gt_input_ids.shape[1])


                for ti,targ_id in enumerate(gt_input_ids[0]):
                    same = input_id.item()==targ_id.item()
                    blank = input_id.item()==self.blank_token and ii>1 and ii<input_ids.shape[1]-1
                    possible_paths = []
                    if ii>0 and ti>0:
                        past_score,mask = dynamic_prog[ii-1][ti-1]
                        possible_paths.append(
                                (past_score+(0 if same else (1 if blank else 1.1)),
                                    mask+[blank],
                                    #past_path+[(ii,ti)]
                                    ))
                    elif ii==0 and ti==0:
                        possible_paths.append(
                                ((0 if same else 1),
                                    [blank],
                                    #[(0,0)]
                                    ))
    
                    if ii>0:
                        past_score,mask = dynamic_prog[ii-1][ti]
                        possible_paths.append(
                                (past_score+1.2,
                                    mask,#+['skip'],
                                    #past_path+[(ii,ti)]
                                    ))
                    if ti>0:
                        past_score,mask = dynamic_prog[ii][ti-1]
                        possible_paths.append(
                                (past_score+(0.9 if blank else 1.1),
                                    mask+[blank],
                                    #past_path+[(ii,ti)]
                                    ))

                    possible_paths.sort(key=lambda a:a[0])
                    
                    dynamic_prog[ii][ti] = possible_paths[0]
            score,loss_mask = dynamic_prog[-1][-1]

            assert len(loss_mask) == gt_input_ids.shape[1]
            loss_mask = torch.BoolTensor(loss_mask)[None,:] #tensor and add batch dim

            if not loss_mask.any():
                loss_mask = None




        image = 255-image
        if len(image.shape)==2:
            image = image[:,:,None]
        if self.augment_shade and self.augment_shade>random.random():
            if image.shape[2]==3:
                image = augmentation.apply_random_color_rotation(image)
                image = augmentation.apply_tensmeyer_brightness(image,**self.aug_params)
            else:
                
                image = augmentation.apply_tensmeyer_brightness(image,**self.aug_params)

        image=np.concatenate((image,np.zeros([image_h,image_w,1],dtype=np.float32)),axis=2)
        #highlight lines to make it easier to locate
        for paragraph in block['paragraphs']:
            for line in paragraph['lines']:
                x1,y1,x2,y2 = line['box']
                image[round(y1):round(y2),round(x1):round(x2),1] = 255 #highlight block we're working on
        for x1,y1,x2,y2 in to_remove:
            x1=round(x1)
            y1=round(y1)
            x2=round(x2)
            y2=round(y2)
            image[y1:y2,x1:x2,0] = 128 #we mask to 0 [middle of range] in qa dataset
            image[y1:y2,x1:x2,1] = -255 #flip mask

        #rotate image
        rot_amount = np.random.normal(0,self.rotate_std_dev)
        scale_amount = 1+np.random.normal(0,self.scale_std_dev)
        rot = np.array([  [math.cos(rot_amount), -math.sin(rot_amount), 0],
                        [math.sin(rot_amount), math.cos(rot_amount), 0],
                        [0,0,1] ])
        scale = np.array([ [scale_amount,0,0],
                           [0,scale_amount,0],
                           [0,0,1] ])
        center = np.array([ [1,0,-image.shape[1]/2],
                            [0,1,-image.shape[0]/2],
                            [0,0,1] ])
        uncenter = np.array([   [1,0,image.shape[1]/2],
                                [0,1,image.shape[0]/2],
                                [0,0,1] ])
        M=center
        M = scale.dot(M)
        M = rot.dot(M)
        M = uncenter.dot(M)
        M=M[:2] #opencv didn't want 3x3
        image = img_f.warpAffine(image,M,(image.shape[0],image.shape[1]))



        img = image.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,  col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img[:,0] = 1.0 - img[:,0] / 128.0 #ideally the median value would be 0
        mask_on = img[:,1]>0
        mask_masked = img[:,1]<-1
        mask_off = (~mask_on)*(~mask_masked)
        img[:,1][mask_on] = 1
        img[:,1][mask_off] = 0
        img[:,1][mask_masked] = -1

        ret = {
                "img": img,
                "imgName": "generated",
                "scale": 1,
                "questions": ["mlm>"],
                "answers": [target_string],
                "mask_label": None,
                }
        if logits is not None and (not self.loss_mask or loss_mask is not None):
            ret["bart_logits"] = logits
            ret["bart_last_hidden"] = last_hidden
        if self.loss_mask and loss_mask is not None:
            ret['distill_loss_mask'] = loss_mask

        return ret



    #This is largely copied from SynthParaQA
    def addBlock(self,ocr,image):
        image_h,image_w = image.shape

        if self.prev_words is not None:
            words = self.prev_words
            self.prev_words = None
            use_prev=True
        else:
            words = []
            while len(words)==0:
                words = self.gen_daemon.generate()
            use_prev=False
        word_height = random.randrange(self.min_text_height,self.max_text_height)
        scale = word_height / words[0][1].shape[0]

        #layout the Paragraph to find it's height
        para_width = random.randrange(round(image_w*self.min_para_width),image_w-10) #min width wider than synth para
        em_approx = word_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
        min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/character-design-standards/whitespace
        max_space = 0.5*em_approx
        while True:
            space_width = round(random.random()*(max_space-min_space) + min_space)
            newline_height = random.randrange(1,word_height) + word_height
            tab_min = round(0.6*em_approx)
            tab_max = round(3*em_approx)

            paras=[]
            cur_lines=[]
            indent=0
            if random.random()<0.2:
                start_x=0
                if random.random()<0.1:
                    indent=random.randrange(tab_min,tab_max)
            else:
                start_x=random.randrange(tab_min,tab_max)
            cur_line = [words[0]+(start_x,0)]
            x=start_x+words[0][1].shape[1]*scale
            y=0
            new_para=False
            for text,img in words[1:]:
                if new_para:
                    #newparagraph
                    if indent==0 and start_x==0:
                        #if no indent, extra verticle space
                        y+=round(random.random()*newline_height)
                    y+=newline_height
                    x=start_x + random.randrange(space_width)
                    cur_lines.append(cur_line)
                    cur_line = []
                    paras.append(cur_lines)
                    cur_lines=[]
                else:
                    #add space
                    x+=space_width
                    if x+img.shape[1]*scale>=para_width:
                        #newline!
                        x=indent + random.randrange(space_width)
                        y+=newline_height
                        cur_lines.append(cur_line)
                        cur_line = []
                cur_line.append((text,img,int(x),int(y)))
                x+=img.shape[1]*scale

                new_para=text[-1]=='¶'

            if len(cur_line)>0:
                cur_lines.append(cur_line)
            if len(cur_lines)>0:
                paras.append(cur_lines)

            para_height = y+word_height

            if para_height<image_h:
                break
            #else loop again if smaller para_width
            para_width  = round(para_width*1.2)

            #unless we're too wide, then start removing words
            if para_width >= image_w:
                para_width = image_w-10
                words = words[:round(len(words)*0.8)] #remove some words to make it shorter

        #find somewhere for the paragraph
        search_step_size=15
        start_x = random.randrange(0,image_w-para_width)
        start_y = random.randrange(0,image_h-para_height)

        #we'll do this by stepping it in different directions as long as the ink in the area is decreasing
        initial_overlap = overlap = image[start_y:start_y+para_height,start_x:start_x+para_width].sum()
        step=0
        directions=[(0,search_step_size),(search_step_size,0),(0,-search_step_size),(-search_step_size,0)] #just 4-direction
        while overlap>0 and step<150 and len(directions)>0:
            step+=1
            to_remove=[]
            for di,(xd,yd) in enumerate(directions):
                step_y = start_y + step*yd
                step_x = start_x + step*xd
                if step_x<0 or step_y<0 or step_x+para_width>=image_w or step_y+para_height>=image_h:
                    to_remove.append(di)
                else:
                    overlap = image[step_y:step_y+para_height,step_x:step_x+para_width].sum()
                    if overlap==0:
                        break
                    elif overlap>= initial_overlap:
                        to_remove.append(di)
            to_remove.reverse()
            for r in to_remove:
                del directions[r]

        if overlap>0: #couldn't fit
            self.prev_words=words #save to use in next image
            return False
        else:
            if step>0:
                start_x = step_x
                start_y = step_y

            #Actually draw in the paragraph and build ocr
            for i,lines in enumerate(paras):
                if i>0 or (use_prev and len(ocr)>0):
                    to_append = self.addPara(lines,image,scale,start_x,start_y)
                    last_para = ocr[-1]
                    new_box = [
                            min(last_para['box'][0],to_append['box'][0]),
                            min(last_para['box'][1],to_append['box'][1]),
                            max(last_para['box'][2],to_append['box'][2]),
                            max(last_para['box'][3],to_append['box'][3])
                            ]
                    last_para['box'] = new_box
                    last_para['paragraphs']+=to_append['paragraphs']
                else:
                    ocr.append(self.addPara(lines,image,scale,start_x,start_y))
            return True

    def addPara(self,lines,image,scale,start_x,start_y):
        para_min_x = 999999
        para_max_x = 0
        para_min_y = 999999
        para_max_y = 0
        ocr_lines=[]
        for line in lines:
            line_min_x = 999999
            line_max_x = 0
            line_min_y = 999999
            line_max_y = 0
            ocr_words=[]
            line_text=[]
            for text,img,x_off,y_off in line:
                if text[-1]=='¶':
                    text=text[:-1]
                img = img_f.resize(img,fx=scale,fy=scale)
                x1 = min(image.shape[1]-1,start_x+x_off)
                y1 = min(image.shape[0]-1,start_y+y_off)
                x2 = min(image.shape[1],x1+img.shape[1])
                y2 = min(image.shape[0],y1+img.shape[0])
                image[y1:y2,x1:x2]=img[:y2-y1,:x2-x1]
                ocr_words.append({'text':text, 'box':[x1,y1,x2,y2]})
                line_text.append(text)
                para_min_x = min(para_min_x,x1)
                para_max_x = max(para_max_x,x2)
                para_min_y = min(para_min_y,y1)
                para_max_y = max(para_max_y,y2)
                line_min_x = min(line_min_x,x1)
                line_max_x = max(line_max_x,x2)
                line_min_y = min(line_min_y,y1)
                line_max_y = max(line_max_y,y2)
            ocr_lines.append({'text':' '.join(line_text),'box':[line_min_x,line_min_y,line_max_x,line_max_y],'words':ocr_words})

        #Following how I modified the Tesseract output with the CDIP dataset, blocks and paragrphs are identical
        return {
                 'box':[para_min_x,para_min_y,para_max_x,para_max_y],
                 'paragraphs': [{
                     'box':[para_min_x,para_min_y,para_max_x,para_max_y],
                     'lines':ocr_lines
                     }]
                }
