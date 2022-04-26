import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.parseIAM import getWordAndLineBoundaries
import timeit
from data_sets.qa import QADataset, collate
from transformers import BartTokenizer
import editdistance

import utils.img_f as img_f


class IAMNER(QADataset):
    """
    Named entity recognition task on IAM
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(IAMNER, self).__init__(dirPath,split,config,images)

        self.do_masks=True
        self.crop_to_data=True
        split_by = config.get('data_split','rwth')
        self.cache_resized = False
        self.warp_lines = None
        self.full = config.get('full',True)
        self.class_first = config.get('class_first',False) #do class first half of the time
        self.short_class = config.get('short_class',False) #if not using special tokens, this is a good idea
        if self.full:
            assert self.cased
        self.eval_full = config.get('eval_full',True)
        self.eval_class_before = config.get('eval_class_before',False)

        task = config['task'] if 'task' in config else 6


        self.use_noise = config.get('use_noise',False)
        if self.use_noise:
            self.tokenizer = BartTokenizer.from_pretrained('./cache_huggingface/BART')
            with open('data_sets/wordsEn.txt') as f:
                self.random_words = f.read().split('\n')
                random.shuffle(self.random_words)

        all_classes = set()

        self.current_crop=None
        self.word_id_to_cls={}
        
        if images is not None:
            self.images=images
        else:
            split_file = os.path.join(dirPath,'ne_annotations','iam',split_by,'iam_{}_{}_{}_all.txt'.format(split,split_by,task))
            doc_set = set()
            with open(split_file) as f:
                lines = f.readlines()
            for line in lines:
                parts = line.split('-')
                if len(parts)>1:
                    name = '-'.join(parts[:2])
                    doc_set.add(name)

                    word_id, cls = line.strip().split(' ')
                    self.word_id_to_cls[word_id]=cls
                    all_classes.add(cls)
            rescale=1.0
            self.images=[]
            for name in doc_set:
                xml_path = os.path.join(dirPath,'xmls',name+'.xml')
                image_path = os.path.join(dirPath,'forms',name+'.png')
                if self.train:
                    self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':xml_path, 'rescaled':rescale })
                else:
                    qas,bbs = self.makeQuestions(xml_path,rescale)
                    for qa in qas:
                        qa['bb_ids']=None
                        self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':xml_path, 'rescaled':rescale,'qa':[qa]})




    #Find crop to handwriting area and return text line boxes
    def getCropAndLines(self,xmlfile,shape):
        W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xmlfile)
        #W_lines is list of lists
        # inner list has ([minY,maxY,minX,maxX],text,id) id=gt for NER

        #We need to crop out the prompt text
        #We'll do that by cropping to only the handwriting area
        maxX=0
        maxY=0
        minX=image_w
        minY=image_h
        for words in W_lines:
            ocr_words=[]
            for word in words:
                minX = min(minX,word[0][2])
                minY = min(minY,word[0][0])
                maxX = max(maxX,word[0][3])
                maxY = max(maxY,word[0][1])
                #print(word)
        crop = [max(0,round(minX-40)),
                max(0,round(minY-40)),
                round(maxX+40),
                round(maxY+40)]
                #min(image_h,round(maxX+40)),
                #min(image_w,round(maxY+40))]
        self.current_crop=crop[:2]

        crop_x,crop_y = self.current_crop
        line_bbs=[]
        for line in lines:
            line_bbs.append([line[0][2]-crop_x,line[0][0]-crop_y,line[0][3]-crop_x,line[0]  [1]-crop_y])
        #print(crop)
        return crop, line_bbs

    def makeQuestions(self,xmlfile,s):
        W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xmlfile)
        #W_lines is list of lists
        # inner list has ([minY,maxY,minX,maxX],text,id) id=gt for NER
        if self.current_crop is None:
            self.getCropAndLines(xmlfile,(image_h,image_w))
        crop_x,crop_y = self.current_crop
        self.current_crop = None
        qa_by_class = defaultdict(list)
        bbs = []
        qas=[]


        if self.full:
            class_after = (not self.train) or random.random()<0.5 or (not self.class_first)
            if self.eval_class_before and not self.train:
                class_after=False
            if (self.eval_full and not self.train) or (self.train and random.random()<0.5):
                #Full document 
                q='ner_full>' if class_after else 'ner_full_c1>'
                corrupt_a=[]
                a=[]
                for words in W_lines:
                    minX=minY = 9999999999
                    maxX=maxY = -1
                    for word in words:
                        cls = self.word_id_to_cls[word[2]]
                        tY,bY,lX,rX = word[0]
                        tY-=crop_y
                        bY-=crop_y
                        lX-=crop_x
                        rX-=crop_x

                        minX = min(minX,lX)
                        maxX = max(maxX,rX)
                        minY = min(minY,tY)
                        maxY = max(maxY,bY)
                        #«»       
                        if class_after:
                            if self.short_class:
                                cls = '['+cls+']' #when doing class after and before, the bracketing is different
                            else:
                                cls = '[NE:'+cls+']'
                            a.append(word[1]+cls)
                            if self.use_noise and self.train:
                                corrupt_a.append(self.corrupt(word[1])+cls)
                        else:
                            if self.short_class:
                                cls = '{'+cls+'}'
                            else:
                                cls = '{NE:'+cls+'}'
                            a.append(cls+word[1])
                            if self.use_noise and self.train:
                                corrupt_a.append(cls+self.corrupt(word[1]))


                    lX=minX
                    rX=maxX
                    tY=minY
                    bY=maxY
                    bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                            s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/ 2.0, s*bY]
                    bbs.append(bb)
                a = ' '.join(a)
                if self.use_noise and self.train:
                    corrupt_a=' '.join(corrupt_a)
                    mask = self.getChangeMask(a,corrupt_a)
                    self.qaAdd(qas,q,corrupt_a,None,bbs,noise_token_mask=mask)
                else:
                    self.qaAdd(qas,q,a,None,bbs)
            else:
                #Single line

                for words in W_lines:
                    q='ner_line>' if class_after else 'ner_line_c1>'
                    a=[]
                    corrupt_a=[]
                    minX=minY = 9999999999
                    maxX=maxY = -1
                    all_O = True
                    for word in words:
                        cls = self.word_id_to_cls[word[2]]
                        if cls != 'O':
                            all_O=False
                        tY,bY,lX,rX = word[0]
                        tY-=crop_y
                        bY-=crop_y
                        lX-=crop_x
                        rX-=crop_x

                        minX = min(minX,lX)
                        maxX = max(maxX,rX)
                        minY = min(minY,tY)
                        maxY = max(maxY,bY)
                        
                        if class_after:
                            if self.short_class:
                                class_str = '['+cls+']'
                            else:
                                class_str = '[NE:'+cls+']'
                            a.append(word[1]+class_str)
                            if self.use_noise and self.train:
                                corrupt_a.append(self.corrupt(word[1])+class_str)
                        else:
                            if self.short_class:
                                class_str = '{'+cls+'}'
                            else:
                                class_str = '{NE:'+cls+'}'
                            a.append(class_str+word[1])
                            if self.use_noise and self.train:
                                corrupt_a.append(class_str+self.corrupt(word[1]))

                    if self.train and all_O and random.random()<0.5:
                        #This has only "Other" (no) class words
                        continue #skip this so we see more instances of Named Entities

                    lX=minX
                    rX=maxX
                    tY=minY
                    bY=maxY
                    bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                            s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/ 2.0, s*bY]
                    a=' '.join(a)
                    if self.use_noise and self.train:
                        corrupt_a=' '.join(corrupt_a)
                        mask = self.getChangeMask(a,corrupt_a)
                        self.qaAdd(qas,q,corrupt_a,[len(bbs)],[bb],noise_token_mask=mask)
                    else:
                        self.qaAdd(qas,q,a,[len(bbs)],[bb])
                    bbs.append(bb)
        else:
            #This does it by word, which is quite inefficient. We don't use this
            for words in W_lines:
                for word in words:
                    cls = self.word_id_to_cls[word[2]]
                    tY,bY,lX,rX = word[0]
                    tY-=crop_y
                    bY-=crop_y
                    lX-=crop_x
                    rX-=crop_x
                    bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                                s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/ 2.0, s*bY]
                    inmask = [bb]
                    if not self.train or random.random()<0.5:
                        q='ne>'
                        a='['+cls+']'+word[1]
                    else:
                        q='ne~'+word[1]
                        a='['+cls+']'
                    qa_by_class[cls].append((q,a,[len(bbs)],inmask))
                    #self.qaAdd(qas,q,a,[len(bbs)],inmask)
                    bbs.append(bb)

            if self.train:
                #balance by class 
                classes = list(qa_by_class.keys())
                random.shuffle(classes)
                for qa_cls in qa_by_class.values():
                    random.shuffle(qa_cls)
                i=0
                some_added=True
                while len(qas)<3*self.questions and some_added:
                    some_added = False
                    for cls in classes:
                        if len(qa_by_class[cls])>i:
                            self.qaAdd(qas,*qa_by_class[cls][i])
                            some_added=True
                    i+=1
            else:
                for qa_cls in qa_by_class.values():
                    for qa in qa_cls:
                        self.qaAdd(qas,*qa)
        return qas,bbs

    def parseAnn(self,xmlfile,s):
        qas,bbs = self.makeQuestions(xmlfile,s)
        bbs = np.array(bbs)
        return bbs, list(range(bbs.shape[0])), None, None, qas


    #Sometimes return a word with a relatively close editdistance
    #We do this by taking a set of 25000 random words and returning the one with lowest editdistance
    #This also tries to match the captialization stype of the input word.
    #Don't corrupt punctuation
    def corrupt(self,word):
        if random.random() < self.use_noise and word!=',' and word!='.' and word!=';' and word!='?' and word!='!' and word!='"' and word!=':' and word!='(' and word!=')':
            first_cap = not word.islower()
            all_cap = word.isupper() and len(word)>1
            word = word.lower()

            best_ed=99999999
            best_sub=None
            start_i=random.randrange(len(self.random_words))
            for i in range(25000):
                sub = self.random_words[(i+start_i)%len(self.random_words)]
                ed = editdistance.eval(word,sub)
                if ed>0:
                    if ed==1:
                        best_sub = sub
                        break #cant get better than 1
                    elif ed<best_ed:
                        best_ed = ed
                        best_sub = sub
            if all_cap:
                best_sub = best_sub.upper()
            elif first_cap and len(best_sub)>0:
                best_sub = best_sub[0].upper() + best_sub[1:]
            return best_sub

                
        else:
            return word

    #We substitute words when corrupting and don't want to backprop on their error, but the tokenization of the corrupted text doesn't correspond directly to the token representation of the GT text.
    #This function returns the mask for all changed tokens (from the the tokenization of having all the correct words)
    def getChangeMask(self,gt,changed):
		#we'll compute the insertions and deletions in the Levenstein distance betweem the target and input
		#From there we'll imply which tokens in the target are the same

        noise_ids = self.tokenizer([changed], return_tensors='pt')['input_ids']
        gt_ids = self.tokenizer([gt], return_tensors='pt')['input_ids']


        dynamic_prog = [None]*(gt_ids.shape[1])
        for ii,input_id in enumerate(gt_ids[0]):
            dynamic_prog[ii] = [None]*(noise_ids.shape[1])


            for ti,targ_id in enumerate(noise_ids[0]):
                same = input_id.item()==targ_id.item()

                possible_paths = []
                if ii>0 and ti>0:
                    past_score,mask = dynamic_prog[ii-1][ti-1]
                    possible_paths.append(
                            (past_score+(0 if same else 1),
                                mask+[same],
                                #past_path+[(ii,ti)]
                                ))
                elif ii==0 and ti==0:
                    possible_paths.append(
                            ((0 if same else 1),
                                [same],
                                #[(0,0)]
                                ))

                if ii>0:
                    past_score,mask = dynamic_prog[ii-1][ti]
                    possible_paths.append(
                            (past_score+1.1,
                                mask,#+['skip'],
                                #past_path+[(ii,ti)]
                                ))
                if ti>0:
                    past_score,mask = dynamic_prog[ii][ti-1]
                    possible_paths.append(
                            (past_score+0.9,
                                mask+[same],
                                #past_path+[(ii,ti)]
                                ))

                possible_paths.sort(key=lambda a:a[0])
                
                dynamic_prog[ii][ti] = possible_paths[0]
        score,loss_mask = dynamic_prog[-1][-1]
        assert len(loss_mask) == noise_ids.shape[1]
        loss_mask = torch.BoolTensor(loss_mask)[None,:] #tensor and add batch dim
        return loss_mask
