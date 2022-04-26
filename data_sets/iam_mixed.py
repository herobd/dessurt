import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils import grid_distortion
from utils.parseIAM import getWordAndLineBoundaries
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate

import utils.img_f as img_f
from utils import grid_distortion


class IAMMixed(ParaQADataset):
    """
    Presents IAM data as two lists of words in random order, sampling from three IAM documents per synthetic document
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(IAMMixed, self).__init__(dirPath,split,config,images)

        self.augment_shade = config['augment_shade'] if 'augment_shade' in config else True
        self.min_text_height = config['min_text_height'] if 'min_text_height' in config else   18
        self.max_text_height = config['max_text_height'] if 'max_text_height' in config else   48
        self.image_size = config['image_size']
        
        split_by = 'rwth'
        self.cased = config.get('cased',True)


        if images is not None:
            self.images=images
        else:
            split_file = os.path.join(dirPath,'ne_annotations','iam',split_by,'iam_{}_{}_6_all.txt'.format(split,split_by))
            doc_set = set()
            with open(split_file) as f:
                lines = f.readlines()
            for line in lines:
                parts = line.split('-')
                if len(parts)>1:
                    name = '-'.join(parts[:2])
                    doc_set.add(name)
            self.images=[]
            if self.train:
                doc_set = list(doc_set)
                used=set()
                #for i1,name1 in enumerate(doc_set[:-2]):
                #    for i2,name2 in enumerate(doc_set[i1+1:-1]):
                #        for i3,name3 in enumerate(doc_set[i2+1:]):
                for i in range(4*len(doc_set)):
                    while True:
                        name1,name2,name3 = random.sample(doc_set,3)
                        tup = (name1,name2,name3)
                        if tup not in used:
                            break
                    used.add(tup)
                    xml_path1 = os.path.join(dirPath,'xmls',name1+'.xml')
                    image_path1 = os.path.join(dirPath,'forms',name1+'.png')
                    xml_path2 = os.path.join(dirPath,'xmls',name2+'.xml')
                    image_path2 = os.path.join(dirPath,'forms',name2+'.png')
                    xml_path3 = os.path.join(dirPath,'xmls',name3+'.xml')
                    image_path3 = os.path.join(dirPath,'forms',name3+'.png')
                    self.images.append({ 'imageName':name1+name2+name3, 'imagePath':None, 'annotationPath':[(image_path1,xml_path1),(image_path2,xml_path2),(image_path3,xml_path3)], })
            else:
                for name1 in doc_set:
                    xml_path1 = os.path.join(dirPath,'xmls',name1+'.xml')
                    image_path1 = os.path.join(dirPath,'forms',name1+'.png')
                    self.images.append({ 'imageName':name1, 'imagePath':None, 'annotationPath':[(image_path1,xml_path1)]})

        if self.train:
            self.q_types = {
                    'read_block':0.5,
                    'read_block0':0.5
                    }
        else:
            self.q_types = {
                    'read_block0':1
                    }


        self.punctuation = ['.',',','!','?','"',"'",')','(',';']

    def parseAnn(self,xmlfiles,s):
        all_words=[]
        images=[]
        for image_path,xmlfile in xmlfiles:
            W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xmlfile)
            for line in W_lines:
                for coords,text,identifier in line:
                    if text not in self.punctuation:
                        all_words.append((coords,text,len(images)))
            images.append(img_f.imread(image_path))


        if self.train:
            random.shuffle(all_words)
        else:
            #psuedo shuffle so LM can't help
            all_words = all_words[0::3]+all_words[1::3]+all_words[2::3]

        image_h,image_w = self.image_size
        image = np.full([image_h,image_w],253,dtype=np.uint8)

        if self.train:
            start_x = random.randrange(2,150)
            cur_y = random.randrange(2,150)
            y_spacing = random.randrange(4,self.min_text_height)
        else:
            start_x=50
            cur_y=50
            y_spacing=round(self.min_text_height*.66)

        para_min_x = start_x
        para_max_x = start_x
        para_min_y = cur_y
        para_max_y = cur_y

        
        max_w = 0
        lines=[]
        for (minY,maxY,minX,maxX),text,image_id in all_words:

            #pad out since IAM cropping is really tight
            minY-=4
            maxY+=4
            minX-=4
            maxX+=4

            #select text height
            if self.train:
                text_height = random.randrange(self.min_text_height,self.max_text_height+1)
            else:
                text_height = (self.min_text_height+self.max_text_height)//2

            w_img = images[image_id][minY:maxY,minX:maxX]
            if w_img.shape[0]==0:
                continue #skip. something is wrong

            if text_height != w_img.shape[0]:
                #reize word image
                n_width = round(w_img.shape[1]*text_height/w_img.shape[0])
                w_img = img_f.resize(w_img,(text_height,n_width))

            max_w = max(max_w,w_img.shape[1])

            if cur_y+text_height>image_h:
                if self.train:
                    start_x += max(max_w + round((0.1+random.random())*max_w),random.randrange(image_w//2-20,image_w//2+150))
                    cur_y = random.randrange(2,150)
                    y_spacing = random.randrange(4,self.min_text_height)
                else:
                    start_x += image_w//2
                    cur_y=50
                    y_spacing = round(self.min_text_height*.66)
                if start_x+30>=image_w:
                    break #can't do this
            
            if self.train:
                x=random.randrange(max(2,start_x-30),min(image_w,start_x+30))
            else:
                x=start_x

            if x+w_img.shape[1]>image_w:
                continue #can't draw this word
            
            if self.train:
                #Warp augmentation
                std = (random.random()*1.5) + 1.5 #warp differen words to a different degree
                w_img = grid_distortion.warp_image(w_img, w_mesh_std=std, h_mesh_std=std)

            #put word image into synthetic document
            image[cur_y:cur_y+w_img.shape[0],x:x+w_img.shape[1]] = w_img
            
            lines.append({'text':text,
                'box':[x,cur_y,x+w_img.shape[1],cur_y+w_img.shape[0]],
                'words': [{'text':text,'box':[x,cur_y,x+w_img.shape[1],cur_y+w_img.shape[0]]}]
                })
            para_min_x = min(x,para_min_x)
            para_min_y = min(cur_y,para_min_y)
            para_max_x = max(x+w_img.shape[1],para_max_x)
            para_max_y = max(cur_y+text_height,para_max_y)

            cur_y += text_height + y_spacing

        #this ocr is wrong, but works for how read_block parses the ocr to have it read the two lists
        ocr=[{
                'box':[para_min_x,para_min_y,para_max_x,para_max_y],
                'paragraphs': [{
                    'box':[para_min_x,para_min_y,para_max_x,para_max_y],
                    'lines': lines
                    }]
                }]

        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s)
        for pair in qa:
            pair['bb_ids']=None

        return None, None, image, None, qa

