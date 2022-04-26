import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate
from data_sets.gen_daemon import GenDaemon

import utils.img_f as img_f

#This class defines an on-the-fly synthetic dataset of documents with Wikipedia articles
class SynthParaQA(ParaQADataset):
    """
    Class for creating synthetic paragraph type documents
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(SynthParaQA, self).__init__(dirPath,split,config,images)

        font_dir = dirPath
        self.simple_vocab = config.get('simple_vocab') #for debugging
        if self.simple_vocab:
            self.min_read_start_no_mask=4
            self.min_read_start_with_mask=4
        self.gen_daemon = GenDaemon(font_dir,simple=self.simple_vocab) #actually generates word images
        self.prev_words = None


        self.image_size = config['image_size']
        self.min_text_height = config['min_text_height'] if 'min_text_height' in config else 8
        self.max_text_height = config['max_text_height'] if 'max_text_height' in config else 32

        self.images=[]
        for i in range(config['batch_size']*config.get('num_batches',100)): #we just randomly generate instances on the fly
            #these are just to match the QADataset format
            self.images.append({'id':'{}'.format(i), 'imagePath':None, 'annotationPath':i, 'rescaled':1.0, 'imageName':'0'})

        self.held_instance=None
        self.used_held = 0
        self.max_used_held = config['prefetch_factor']//2 if 'prefetch_factor' in config else 2 #how many times to reuse the same image with different questions

    def parseAnn(self,seed,s):
        if not self.train:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        image_h,image_w = self.image_size

        if self.held_instance is not None:
            image,ocr = self.held_instance #reuse image
            self.used_held+=1
            if self.used_held>=self.max_used_held:
                self.held_instance=None #make a new image next time
        else:
            
            image = np.zeros([image_h,image_w],dtype=np.uint8)

            ocr=[] #blocks
            success=True
            while success:
                #we'll add as many as we can fit (without trying too hard)
                success = self.addBlock(ocr,image)
                if success and self.simple_vocab: 
                    break

            self.held_instance= (image,ocr)
            self.used_held=1
            

        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s)

        return qa_bbs, list(range(qa_bbs.shape[0])), 255-image, None, qa

    
    #Add (part of) a Wikipedia article to the image
    def addBlock(self,ocr,image):
        image_h,image_w = image.shape

        if self.prev_words is not None:
            words = self.prev_words
            self.prev_words = None
        else:
            words = []
            while len(words)==0:
                words = self.gen_daemon.generate() #get text and word images

        word_height = random.randrange(self.min_text_height,self.max_text_height) #text height
        scale = word_height / words[0][1].shape[0]

        para_width = random.randrange(image_w//5,image_w-10) #how wide will we allow it to be?
        #layout the Paragraph to find it's height
        em_approx = word_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
        min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/character-design-standards/whitespace
        max_space = 0.5*em_approx
        while True: #we'll be increasing the para_width if the paragraph ends up longer than the page
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
                    #inverse indent, where firstline is not indented, but later ones are
                    indent=random.randrange(tab_min,tab_max)
                #else no indent, gets extra verticle space
            else:
                #normal paragraph indentation
                start_x=random.randrange(tab_min,tab_max)

            #init text line with first word
            cur_line = [words[0]+(start_x,0)]
            
            #position other words
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

                new_para=text[-1]=='¶' #special character returned by GenDaemon

            if len(cur_line)>0:
                cur_lines.append(cur_line)
            if len(cur_lines)>0:
                paras.append(cur_lines)

            para_height = y+word_height

            if para_height<image_h:
                break #success!
            #else loop again with bigger para_width
            para_width  = round(para_width*1.2)

            #unless we're too wide, then start removing words
            if para_width >= image_w:
                para_width = image_w-10
                words = words[:round(len(words)*0.8)] #remove some words to make it shorter

        #find somewhere for the paragraph
        search_step_size=15

        #start somewhere random
        start_x = random.randrange(0,image_w-para_width)
        start_y = random.randrange(0,image_h-para_height)

        #we'll do this by stepping it in different directions as long as the ink in the area is decreasing
        # the ink being the sum of the pixel values
        initial_overlap = overlap = image[start_y:start_y+para_height,start_x:start_x+para_width].sum()
        step=0
        directions=[(0,search_step_size),(search_step_size,0),(0,-search_step_size),(-search_step_size,0)] #just 4-direction search

        while overlap>0 and step<150 and len(directions)>0:
            #at each iteration, step all remaining directions once
            step+=1
            to_remove=[]
            for di,(xd,yd) in enumerate(directions):
                step_y = start_y + step*yd
                step_x = start_x + step*xd
                if step_x<0 or step_y<0 or step_x+para_width>=image_w or step_y+para_height>=image_h:
                    to_remove.append(di) #no room, this direction won't work
                else:
                    overlap = image[step_y:step_y+para_height,step_x:step_x+para_width].sum()
                    if overlap==0:
                        break #this works, we're done!
                    elif overlap>= initial_overlap:
                        to_remove.append(di) #getting worse, don't do this direction more
            to_remove.reverse()
            for r in to_remove:
                del directions[r]

        if overlap>0: #couldn't fit
            self.prev_words=words #save to use in next image
            return False
        else:
            if step>0:
                #because we used bread, the step variables have the right step
                start_x = step_x
                start_y = step_y

            #Actually draw in the paragraph and build ocr
            for lines in paras:
                ocr.append(self.addPara(lines,image,scale,start_x,start_y))
            return True


    #resize the word images and put them on the page. Also build data structure the same as Tesseract output
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
                
