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
from data_sets.qa import QADataset, collate
from data_sets.gen_daemon import GenDaemon

import utils.img_f as img_f


class SQuAD(QADataset):
    """
    Class for rendering SQuAD to images
    This is not HW-SQuAD, but my own varient. 
    It just renders the context text like is done for Synthetic Wikipedia (SynthParaQA)
    I was going to pretrain on this for DocVQA, but found it wasn't needed.
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(SQuAD, self).__init__(dirPath,split,config,images)

        self.do_masks=True

        font_dir = config.get('font_dir','../data/fonts/')
        self.gen_daemon = GenDaemon(font_dir,no_wiki=True)
        self.prev_words = None

        self.gt_ocr = config['gt_ocr'] if 'gt_ocr' in config else False

        self.image_size = config['image_size']
        self.min_text_height = config['min_text_height'] if 'min_text_height' in config else 8
        self.max_text_height = config['max_text_height'] if 'max_text_height' in config else 32
        if self.train:
            json_path = os.path.join(dirPath,'train-v2.0.json')
        else:
            #I don't have the test set implemented as I was just pre-training
            json_path = os.path.join(dirPath,'dev-v2.0.json')
        with open(json_path) as f:
            data = json.load(f)['data']

        assert split!='test'
        
        

        self.images=[]
        for article in data:
            name = article['title']
            if not self.train and len(article['paragraphs'])>6:
                    article['paragraphs']=article['paragraphs'][:3]+article['paragraphs'][-3:] #cut down size to make evaluation faster. First 3 and last 3 paragraphs
            for p,para in enumerate(article['paragraphs']):
                context = para['context']
                for qa in para['qas']:
                    data = {
                            'seed': len(self.images),
                            'context': context,
                            'question': qa['question'],
                            'answers': [a['text'] for a in qa['answers']] if not qa['is_impossible'] else ['№']
                            }
                    self.images.append({'id':'{}{}'.format(name,qa['id']), 'imagePath':None, 'annotationPath':data, 'rescaled':1.0, 'imageName':'{}{}'.format(name,qa['id'])})


    def parseAnn(self,instance,s):
        if not self.train:
            seed = instance['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        image_h,image_w = self.image_size
        image = np.zeros([image_h,image_w],dtype=np.uint8)
        self.addBlock(instance['context'],image)

            

        #qa, qa_bbs = self.makeQuestions(para['qas'],image_h,image_w,s)
        qa=[]
        self.qaAdd(qa,
                'natural_q~'+instance['question'],
                random.choice(instance['answers'])
                )
        metadata={'all_answers':[instance['answers']]}

        return None, None, 255-image, metadata, qa


    def addBlock(self,text,image):
        image_h,image_w = image.shape

        words = self.gen_daemon.generate(text)
        word_height = random.randrange(self.min_text_height,self.max_text_height)
        scale = word_height / words[0][1].shape[0]

        #layout the Paragraph to find it's height
        para_width = random.randrange(image_w//2,image_w-10)
        em_approx = word_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
        min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/character-design-standards/whitespace
        max_space = 0.5*em_approx
        while True:#para_width<image_w: #we'll be increasing the para_width if the paragraph ends up longer than the page
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

            #unless we're too wide, then start shrinking text
            if para_width >= image_w:
                para_width = image_w-10
                word_height *= 0.9
                word_height = round(word_height)
                scale = word_height / words[0][1].shape[0]
                em_approx = word_height*1.6 
                min_space = 0.2*em_approx 
                max_space = 0.5*em_approx


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
            assert False
            self.prev_words=words #save to use in next image
            return False
        else:
            if step>0:
                start_x = step_x
                start_y = step_y

            #Actually draw in the paragraph and build ocr
            for lines in paras:
                #ocr.append(self.addPara(lines,image,scale,start_x,start_y))
                self.addPara(lines,image,scale,start_x,start_y)
            return True

    def addPara(self,lines,image,scale,start_x,start_y):
        para_min_x = 999999
        para_max_x = 0
        para_min_y = 999999
        para_max_y = 0
        #ocr_lines=[]
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
            #ocr_lines.append({'text':' '.join(line_text),'box':[line_min_x,line_min_y,line_max_x,line_max_y],'words':ocr_words})

        #Following how I modified the Tesseract output with the CDIP dataset, blocks and paragrphs are identical
        #return {
        #         'box':[para_min_x,para_min_y,para_max_x,para_max_y],
        #         'paragraphs': [{
        #             'box':[para_min_x,para_min_y,para_max_x,para_max_y],
        #             'lines':ocr_lines
        #             }]
        #        }
                
