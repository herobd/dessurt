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
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate

import utils.img_f as img_f


class SynthHWQA(ParaQADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(SynthHWQA, self).__init__(dirPath,split,config,images)
        self.image_dir = dirPath

        self.image_size = config['image_size']
        self.min_text_height = config['min_text_height'] if 'min_text_height' in config else 18
        self.max_text_height = config['max_text_height'] if 'max_text_height' in config else 48

        self.augment_shade = config['augment_shade'] if 'augment_shade' in config else True
        
        #self.warp_lines = config.get('warp_lines',0.999)

        self.images=[]
        with open(os.path.join(dirPath,'OUT.txt')) as f:
                lines = f.readlines()
        for l in lines:
            loc = l.find(':')
            index = int(l[:loc])
            text = l[loc+1:].strip()
            #image_path = os.path.join(dirPath,'sample_{}.png'.format(index))
            #if self.train:
            self.images.append({'imageName':index, 'imagePath':None, 'annotationPath':(len(self.images),index,text)})
                #else:
                #    _,_,_,_,_,qa = self.parseAnn(xml_path,rescale)
                #    #qa = self.makeQuestions(rescale,entries))
                #    import pdb;pdb.set_trace()
                #    for _qa in qa:
                #        _qa['bb_ids']=None
                #        self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':xml_path, 'rescaled':rescale, 'qa':[_qa]})



        self.q_types = {
                'read_block':0.5,
                'read_block0':0.5
                }



    def parseAnn(self,data,s):
        pos = data[0]
        image_h,image_w = self.image_size
        
        text_height = random.randrange(self.min_text_height,self.max_text_height)
        newline_height = random.randrange(4,text_height)

        num_lines = random.randrange(1,image_h//(text_height+newline_height))


        ocr_lines=[]
        image = np.full([image_h,image_w],253,dtype=np.uint8)
        
        read_lines=[]
        max_width=0
        while len(read_lines)==0:
            lines = self.images[pos:pos+num_lines]
            for line in lines:
                pos,index,text = line[  'annotationPath']
                try:
                    line_img = img_f.imread(os.path.join(self.image_dir,'sample_{}.png'.format(index)))
                except FileNotFoundError:
                    continue
                new_width = round(line_img.shape[1]*(text_height/line_img.shape[0]))
                read_lines.append([text,line_img,new_width])
                max_width = max(max_width,new_width)

            if len(read_lines)==0:
                pos=random.randrange(len(self.images)-1)


        if max_width>(image_w-1):
            #change resize to fit imagesize
            change = (image_w-1)/max_width
            text_height = round(text_height*change)
            newline_height = round(newline_height*change)
            for i in range(len(read_lines)):
                read_lines[i][2] = round(read_lines[i][2]*change)
            max_width = image_w-1


        start_x = random.randrange(0,image_w-max_width)
        start_y = random.randrange(0,image_h-(len(read_lines)*(text_height+newline_height)))
        max_x=0
        min_x=image_w
        cur_y = start_y
        for text,line_img,width in read_lines:
            line_img = img_f.resize(line_img,(text_height,width))
            #warp line image with Curtis's augmentation
            std = (random.random()*1.5) + 1.5
            line_img = grid_distortion.warp_image(line_img, w_mesh_std=std, h_mesh_std=std)
            
            room_after = image_w-(start_x+width)
            cur_x = start_x
            if start_x>0 or room_after>0:
                #little bit of horizontal wiggle
                cur_x += random.randrange(max(-start_x,-10),min(room_after,10))
            image[cur_y:cur_y+text_height,cur_x:cur_x+width]=line_img

            ocr_line = {'box':[cur_x,cur_y,cur_x+width,cur_y+text_height],
                        'text': text,
                        'words': None}
            ocr_lines.append(ocr_line)

            if cur_x<min_x:
                min_x=cur_x
            if cur_x+width>max_x:
                max_x=cur_x+width

            cur_y += text_height+newline_height

        ocr=[{'paragraphs':[{'lines':ocr_lines}],
              'box': [min_x,start_y,max_x,cur_y-newline_height]
              }]



        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s)
        for pair in qa:
            pair['bb_ids']=None
        return None,None, image, None, qa
