import torch.utils.data
import numpy as np
import json
import os
import math, random
from utils.crop_transform import CropBoxTransform
from utils import augmentation
from collections import defaultdict, OrderedDict
from utils.forms_annotations import convertBBs

import utils.img_f as img_f

#This collate function is for any child class of QADataset
def collate(batch):
    if any(b['mask_label'] is not None for b in batch):
        mask_labels = []
        mask_labels_batch_mask = torch.FloatTensor(len(batch))
        for bi,b in enumerate(batch):
            if b['mask_label'] is None:
                mask_labels_batch_mask[bi]=00
                mask_labels.append( torch.FloatTensor(1,1,b['img'].shape[2],b['img'].shape[3]).fill_(0))
            else:
                mask_labels_batch_mask[bi]=1
                mask_labels.append( b['mask_label'] )
        mask_labels = torch.cat(mask_labels,dim=0)
    else:
        mask_labels = None
        mask_labels_batch_mask = None

    return {
            'img': torch.cat([b['img'] for b in batch],dim=0),
            'imgName': [b.get('imgName') for b in batch],
            'id': [b.get('id') for b in batch],
            'scale': [b.get('scale') for b in batch],
            'cropPoint': [b.get('cropPoint') for b in batch],
            'questions': [b.get('questions') for b in batch],
            'answers': [b.get('answers') for b in batch],
            'metadata': [b.get('metadata') for b in batch],
            'mask_label': mask_labels,
            'mask_labels_batch_mask': mask_labels_batch_mask,
            "bart_logits": torch.cat([b['bart_logits'] for b in batch],dim=0) if 'bart_logits' in batch[0] else None,
            "bart_last_hidden": torch.cat([b['bart_last_hidden'] for b in batch],dim=0) if 'bart_last_hidden' in batch[0] else None,
            "distill_loss_mask": torch.cat([b['distill_loss_mask'] for b in batch],dim=0) if 'distill_loss_mask' in batch[0] and batch[0]['distill_loss_mask'] is not None else None,
            "noise_token_mask": torch.cat([b['noise_token_mask'] for b in batch],dim=0) if 'noise_token_mask' in batch[0] and batch[0]['noise_token_mask'] is not None else None,
            }

#Make a mask channel
def getMask(shape,boxes):
    mask = torch.FloatTensor(1,1,shape[2],shape[3]).fill_(0)
    for box in boxes:
        if isinstance(box,list):
            box = np.array(box)
        points = box[0:8].reshape(4,2)
        img_f.fillConvexPoly(mask[0,0],points,1)
    return mask

#Parent class of almost all datasets used by Dessurt
#It defines the augmentation and prepares the data (masks and such)
#Dessurt works with a query input and respose, or Question and Answer
class QADataset(torch.utils.data.Dataset):


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        self.train = split=='train'
        self.questions = config.get('questions',1)
        self.max_qa_len_in = config['max_qa_len_in'] if 'max_qa_len_in' in config else None
        self.max_qa_len_out = config['max_qa_len_out'] if 'max_qa_len_out' in config else None
        if self.max_qa_len_in is None and self.max_qa_len_out is None and 'max_qa_len' in config:
            self.max_qa_len_in = config['max_qa_len']
            self.max_qa_len_out = config['max_qa_len']

        self.cased = config.get('cased',True)

        self.color = config['color'] if 'color' in config else False #everything with Dessurt is done with grayscale images
        self.rotate = config['rotation'] if 'rotation' in config else False #wether BBs are not axis aligned, not really used

        if 'crop_params' in config and config['crop_params'] is not None:
            self.transform = CropBoxTransform(config['crop_params'],self.rotate)
        else:
            self.transform = None

        self.rescale_range = config['rescale_range']
        self.rescale_to_crop_size_first = config['rescale_to_crop_size_first'] if 'rescale_to_crop_size_first' in config else False
        self.rescale_to_crop_width_first = config['rescale_to_crop_width_first'] if 'rescale_to_crop_width_first' in config else False
        self.rescale_to_crop_height_first = config['rescale_to_crop_height_first'] if 'rescale_to_crop_height_first' in config else False
        if self.rescale_to_crop_size_first or self.rescale_to_crop_width_first or self.rescale_to_crop_height_first:
            self.crop_size = config['crop_params']['crop_size']
        if type(self.rescale_range) is float:
            self.rescale_range = [self.rescale_range,self.rescale_range]

        self.rearrange_tall_images = False #used by HW-SQuAD

        if 'cache_resized_images' in config:
            #This wasn't used for any of Dessurt's stuff
            self.cache_resized = config['cache_resized_images']
            if self.cache_resized:
                if self.rescale_to_crop_size_first:
                    self.cache_path = os.path.join(dirPath,'cache_match{}x{}'.format(*config['crop_params']['crop_size']))
                elif self.rescale_to_crop_width_first:
                    self.cache_path = os.path.join(dirPath,'cache_matchHx{}'.format(config['crop_params']['crop_size'][1]))
                else:
                    assert not self.rescale_to_crop_width_first
                    self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
                if not os.path.exists(self.cache_path):
                    os.mkdir(self.cache_path)
        else:
            self.cache_resized = False

        self.augment_shade = config['augment_shade'] if 'augment_shade' in config else False #Do brightness/contrast augmentation
        self.aug_params = config['additional_aug_params'] if 'additional_aug_params' in config else {}


        self.do_masks=True

        #These are based on EasyOCR, which I did some experiments with
        self.ocr_out_dim = 97
        self.char_to_ocr = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_ocr = {char:i+1 for i,char in enumerate(self.char_to_ocr)} #+1 as 0 is the blank token
        self.one_hot_conf = 0.9


        self.crop_to_data = False
        self.crop_to_q = config.get('crop_to_q',False) #Used for training recognition on NAF
        if self.crop_to_q: 
            self.min_text_height = config['min_text_height'] #Min text height in rescaling




    def __len__(self):
        return len(self.images)


    #helper function for adding a question-answer pair
    def qaAdd(self,qa,question,answer,bb_ids=None,in_bbs=[],out_bbs=None,mask_bbs=[],noise_token_mask=None):
        #aif all([(pair['question']!=question or  for pair in qa]): #prevent duplicate q
        qa.append({
            'question':question,
            'answer':answer,
            'bb_ids':bb_ids,
            'in_bbs':in_bbs,
            'out_bbs':out_bbs,
            'mask_bbs':mask_bbs,
            'noise_token_mask':noise_token_mask
            })

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,scaleP=None,cropPoint=None):
        imagePath = self.images[index]['imagePath']
        imageName = self.images[index].get('imageName',imagePath)

        annotationPath = self.images[index]['annotationPath']
        #This was originally just the json, but as different datasets have different data, it can be something else
        
        rescaled = self.images[index].get('rescaled',1)
        if isinstance(annotationPath,str) and  annotationPath.endswith('.json'):
            try: 
                with open(annotationPath) as annFile:
                    annotations = json.loads(annFile.read())
                    if isinstance(annotations,dict):
                        annotations['XX_imageName']=imageName #so I have it
            except FileNotFoundError:
                print("ERROR, could not open "+annotationPath)
                return self.__getitem__((index+1)%self.__len__())
            except json.decoder.JSONDecodeError as e:
                print(e)
                print('Error reading '+annotationPath)
                return self.__getitem__((index+1)%self.__len__())
        else:
            annotations=annotationPath

        #Load image
        if imagePath is not None:
            try:
                np_img = img_f.imread(imagePath, 1 if self.color else 0)#*255.0
            except FileNotFoundError as e:
                print(e)
                print('ERROR, could not find: '+imagePath)
                return self.__getitem__((index+1)%self.__len__())
            if np_img is None or np_img.shape[0]==0:
                print("ERROR, could not open "+imagePath)
                return self.__getitem__((index+1)%self.__len__())
            if np_img.max()<=1:
                np_img*=255
        else:
            np_img = None #will get generated from parseAnn


        if self.crop_to_data:
            #This is used for the IAM dataset so we don't include the form prompt text (which would be easy to cheat from)
            #This is used by the NAF dataset to sometimes cut landscape documents in half to have better resolution
            crop, line_bbs = self.getCropAndLines(annotations,np_img.shape)
            x1,y1,x2,y2 = crop
            np_img = np_img[y1:y2,x1:x2]
            
            if self.warp_lines is not None and random.random()<self.warp_lines:
                #apply warp grid augmentation to each handwriting line
                self.doLineWarp(np_img,line_bbs)


        #
        if scaleP is None:
            s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        else:
            s = scaleP

        if self.rescale_to_crop_size_first:
            if rescaled!=1:
                raise NotImplementedError('havent implemented caching with match resizing')
        
            scale_height = self.crop_size[0]/np_img.shape[0]
            scale_width = self.crop_size[1]/np_img.shape[1]

            scale = min(scale_height, scale_width)

            if self.rearrange_tall_images:
                #compute the scale if we split the image vertically and put the two halfs side by side
                alt_w = np_img.shape[1]*2
                alt_h = np_img.shape[0]//2
                scale_height = self.crop_size[0]/alt_h
                scale_width = self.crop_size[1]/alt_w
                alt_scale = min(scale_height, scale_width)
                if alt_scale>scale:
                    #The scale is better, so actually to the transformation
                    left = np_img[:np_img.shape[0]//2]
                    right = np_img[np_img.shape[0]//2:]
                    if right.shape[0]>left.shape[0]:
                        left = np.pad(left,(0,1))
                    np_img = np.concatenate((left,right),axis=1)
                    #recompute scale
                    scale_height = self.crop_size[0]/np_img.shape[0]
                    scale_width = self.crop_size[1]/np_img.shape[1]
                    scale = min(scale_height, scale_width)
                    
                    #Note, this did not change any bounding boxes. This means it can only be used on a dataset that doesn't use bounding boxes (like a question-answering dataset)

            partial_rescale = s*scale
            s=partial_rescale

        elif self.rescale_to_crop_width_first:
            if rescaled!=1:
                raise NotImplementedError('havent implemented caching with match resizing')
            scale = self.crop_size[1]/np_img.shape[1]
            partial_rescale = s*scale
            s=partial_rescale
        elif self.rescale_to_crop_height_first:
            if rescaled!=1:
                raise NotImplementedError('havent implemented caching with match resizing')
            scale = self.crop_size[0]/np_img.shape[0]
            partial_rescale = s*scale
            s=partial_rescale
        else:
            partial_rescale = s/rescaled
        

        #Parse annotation file
        bbs,ids, gen_img, metadata, questions_and_answers = self.parseAnn(annotations,s)
        if bbs is None:
            assert ids is None
            bbs = np.zeros(0)
            ids = []

        if self.crop_to_q:
            #crop the image to focus on the text line related to the question
            questions_and_answers = self.images[index]['qa']
            assert len(questions_and_answers)==1
            qa = questions_and_answers[0]
            assert len(qa['in_bbs'])==1
            bb = qa['in_bbs'][0]
            assert len(bb)==16
            bb_height = math.sqrt( ((bb[-4]-bb[-2])**2) + ((bb[-3]-bb[-1])**2) )
            bb_width = math.sqrt( ((bb[-8]-bb[-6])**2) + ((bb[-7]-bb[-5])**2) )
            if bb_height*s < self.min_text_height:
                s=partial_rescale = self.min_text_height/bb_height

                if s*bb_width>self.crop_size[1]:
                    s=partial_rescale = self.crop_size[1]/bb_width
            
            bb_x = bb[-4]*s
            bb_y = bb[1]*s
            cropPoint_x = max(0,round(bb_x-self.crop_size[1]/2))
            cropPoint_y = max(0,round(bb_y-self.crop_size[0]/2))

            if cropPoint_x + self.crop_size[1]>int(s*np_img.shape[1]):
                cropPoint_x -= cropPoint_x + self.crop_size[1] - s*np_img.shape[1]
                if cropPoint_x<0:
                    cropPoint_x = 0
            if cropPoint_y + self.crop_size[0]>int(s*np_img.shape[0]):
                cropPoint_y -= cropPoint_y + self.crop_size[0] - s*np_img.shape[0]
                if cropPoint_y<0:
                    cropPoint_y = 0

            cropPoint = (int(cropPoint_x),int(cropPoint_y))
                
        if (not self.train or self.crop_to_q) and 'qa' in self.images[index]:
            #override questions_and_answers returned by parseAnn
            questions_and_answers = self.images[index]['qa']
            #But the scale doesn't match! So fix it
            for qa in questions_and_answers:
                for bb_name in ['in_bbs','out_bbs','mask_bbs']:
                    if qa[bb_name] is not None:
                        qa[bb_name] = [ [s*v for v in bb] for bb in qa[bb_name] ]
            



        if np_img is None:
            np_img=gen_img #generated image

        if partial_rescale!=1:
            np_img = img_f.resize(np_img,(0,0),
                    fx=partial_rescale,
                    fy=partial_rescale,
            )


        if len(np_img.shape)==2:
            np_img=np_img[...,None] #add 'color' channel
        if self.color and np_img.shape[2]==1:
            np_img = np.repeat(np_img,3,axis=2) #make color image
        
        #set up for cropping
        # The cropping needs to be aware of bounding boxes
        outmasks=False
        if self.do_masks:
            assert self.questions==1 #only allow 1 qa pair if using masking
            mask_bbs=[]
            mask_ids=[]
            for i,qa in enumerate(questions_and_answers):
                inmask_bbs = qa['in_bbs']
                outmask_bbs = qa['out_bbs']
                blank_bbs = qa['mask_bbs']
                if outmask_bbs is not None:
                    outmasks=True
                    mask_bbs+=inmask_bbs+outmask_bbs+blank_bbs
                    mask_ids+=  ['in{}_{}'.format(i,ii) for ii in range(len(inmask_bbs))] + \
                                ['out{}_{}'.format(i,ii) for ii in range(len(outmask_bbs))] + \
                                ['blank{}_{}'.format(i,ii) for ii in range(len(blank_bbs))]
                else:
                    mask_bbs+=inmask_bbs+blank_bbs
                    mask_ids+=  ['in{}_{}'.format(i,ii) for ii in range(len(inmask_bbs))] + \
                                ['blank{}_{}'.format(i,ii) for ii in range(len(blank_bbs))]

            mask_bbs = np.array(mask_bbs)

        #Do crop
        if self.transform is not None:
            if self.do_masks and len(mask_bbs.shape)==2:
                if (bbs is not None and bbs.shape[0]>0) and mask_bbs.shape[0]>0:
                    crop_bbs = np.concatenate([bbs,mask_bbs])
                elif mask_bbs.shape[0]>0:
                    crop_bbs = mask_bbs
                else:
                    crop_bbs = bbs
                crop_ids = ids+mask_ids
            else:
                crop_bbs = bbs
                crop_ids = ids

            out, cropPoint = self.transform({
                "img": np_img,
                "bb_gt": crop_bbs[None,...],
                'bb_auxs':crop_ids,
                
            }, cropPoint)
            np_img = out['img'] #cropped image


            #Get the adjusted bounding boxes
            new_q_inboxes=defaultdict(list)
            if outmasks:
                new_q_outboxes=defaultdict(list)
            else:
                new_q_outboxes=None
            new_q_blankboxes=defaultdict(list)
            new_recog_boxes={}
            if self.do_masks:
                orig_idx=0
                for ii,(bb_id,bb) in enumerate(zip(out['bb_auxs'],out['bb_gt'][0])):
                    if type(bb_id) is int:
                        assert orig_idx==ii
                        orig_idx+=1
                    elif bb_id.startswith('in'):
                        nums = bb_id[2:].split('_')
                        i=int(nums[0])
                        new_q_inboxes[i].append(bb)
                    elif bb_id.startswith('out'):
                        nums = bb_id[3:].split('_')
                        i=int(nums[0])
                        new_q_outboxes[i].append(bb)
                    elif bb_id.startswith('blank'):
                        nums = bb_id[5:].split('_')
                        i=int(nums[0])
                        new_q_blankboxes[i].append(bb)
                    elif bb_id.startswith('recog'):
                        i=int(bb_id[5:])
                        new_recog_boxes[i]=bb
                bbs = out['bb_gt'][0,:orig_idx]
                ids= out['bb_auxs'][:orig_idx]

                #Put boxes back in questions_and_answers
                for i in range(len(questions_and_answers)):
                    questions_and_answers[i]['in_bbs'] = new_q_inboxes[i]
                    if outmasks:
                        questions_and_answers[i]['out_bbs'] = new_q_outboxes[i]
                    questions_and_answers[i]['mask_bbs'] = new_q_blankboxes[i]
            else:
                bbs = out['bb_gt'][0]
                ids= out['bb_auxs']


            if questions_and_answers is not None:
                questions=[]
                answers=[]
                questions_and_answers = [qa for qa in questions_and_answers if qa['bb_ids'] is None or all((i in ids) for i in qa['bb_ids'])] #filter out q-a pairs that were cropped out

        if questions_and_answers is not None:
            if len(questions_and_answers) > self.questions:
                #select the q-a pairs used for this image
                #Dessurt only uses 1 q-a pairs, as each could have a different input mask
                questions_and_answers = random.sample(questions_and_answers,k=self.questions)
            if len(questions_and_answers)==0:
                #Had no questions...
                #weird crops might cause this
                return self.getitem((index+1)%len(self))
            
            new_q_inboxes= [qa['in_bbs'] for qa in questions_and_answers]
            new_q_outboxes= [qa['out_bbs'] for qa in questions_and_answers]
            new_q_blankboxes= [qa['mask_bbs'] for qa in questions_and_answers]
            if self.cased:
                questions = [qa['question'] for qa in questions_and_answers]
                answers = [qa['answer'] for qa in questions_and_answers]
            else:
                questions = [qa['question'].lower() for qa in questions_and_answers]
                answers = [qa['answer'].lower() for qa in questions_and_answers]

            if questions_and_answers[0]['noise_token_mask'] is not None:
                assert len(questions_and_answers)==1
                noise_token_mask = questions_and_answers[0]['noise_token_mask']
            else:
                noise_token_mask = None
        else:
            questions=answers=noise_token_mask=None




        if self.augment_shade and self.augment_shade>random.random():
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)
            else:
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)

        img = np_img.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0

        if self.do_masks:
            assert len(new_q_inboxes)<=1
            assert new_q_outboxes is None or len(new_q_outboxes)<=1

            mask = getMask(img.shape,new_q_inboxes[0])
            img = torch.cat((img,mask),dim=1)
            for blank_box in new_q_blankboxes[0]:
                assert(img.shape[1]==2)
                x1,y1,x2,y2,x3,y3,x4,y4 = blank_box[:8]
                img_f.polylines(img[0,0],np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)]),True,0) #blank on image
                img_f.polylines(img[0,-1],np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)]),True,-1) #flip mask to indicate it was blanked

            if outmasks and new_q_outboxes[0] is not None:
                mask_label = getMask(img.shape,new_q_outboxes[0])
            else:
                mask_label = None
        else:
            mask_label = None


        if bbs is not None:
            bbs = convertBBs(bbs[None,...],self.rotate,0)
            if bbs is not None:
                bbs=bbs[0]
            else:
                bbs = torch.FloatTensor(1,0,5+8+1)
        else:
            bbs = torch.FloatTensor(1,0,5+8+1)



        

        return {
                "img": img,
                "imgName": imageName,
                "id": self.images[index].get('id'),
                "scale": s,
                "cropPoint": cropPoint,
                "questions": questions,
                "answers": answers,
                "noise_token_mask": noise_token_mask,
                "mask_label": mask_label,
                "metadata": metadata
                }


