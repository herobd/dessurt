import torch.utils.data
import numpy as np
import json, re
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from .qa import QADataset

import utils.img_f as img_f

SKIP=['174']#['193','194','197','200']
ONE_DONE=[]

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


class NobrainQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(NobrainQA, self).__init__(dirPath,split,config,images)

        self.images=[]
        for i in range(config['batch_size']*100):
            self.images.append({'id':'{}'.format(i), 'imagePath':'../data/FUNSD/training_data/images/12825369.png', 'annotationPath':'../data/english_char_set.json', 'rescaled':1.0, 'imageName':'0'})

        if 'textfile' in config:
            with open(config['textfile']) as f:
                text = f.read()
        else:
            #text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque faucibus ligula accumsan dui hendrerit, sit amet egestas risus suscipit. Mauris vel euismod lacus. Quisque fermentum sed tortor eleifend congue. Donec at odio non diam rutrum posuere at bibendum erat. Sed id tempus ligula. Proin nec elit eget ligula dignissim varius a in libero. Integer scelerisque sem iaculis magna lobortis, non porta dui auctor. Integer efficitur quam vel ex fermentum sagittis. Donec quis pharetra mi. Vestibulum fringilla vitae tellus vel volutpat. Nam augue felis, lacinia quis mi a, semper dictum justo. Maecenas varius sollicitudin augue, nec pellentesque orci faucibus et. Praesent eget quam nibh. Fusce euismod neque sit amet mollis finibus. Cras posuere purus non diam cursus vestibulum. Cras ut posuere diam, et facilisis eros.'
            text='this is some sample text and should be very easy to solve just do it already please I am rather frustrated with you music dog cat train girl boy food pants shirt table chair bed go come take leave off on apart together close far big small round smooth flat touch press hot cold warm cool floor wall red blue green purple pink yellow orange hair long short spear sword grab pull push hard soft her his high low ask'
        text=re.sub('\s+',' ',text)
        self.words = text.strip().lower().split(' ')
        #else:
        #    self.words = None

        self.additional_doc_len = config['additional_doc_len'] if 'additional_doc_len' in config else 0

        self.shuffle_doc = config['shuffle_doc'] if 'shuffle_doc' in config else False

        self.repeat_after_me=config['repeat_after_me'] if 'repeat_after_me' in config else False
        self.not_present_freq=0.5

        self.only_present = config['only_present'] if 'only_present' in config else False

        self.lm = False



    def parseAnn(self,annotations,s):



        word_boxes=[]
        word_trans=[]

        if self.words is None:
            bb=[None]*16
            lX=0
            rX=10
            tY=0
            bY=10
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
            word_boxes.append(bb)
            word_trans.append('name:')

            bb=[None]*16
            lX=10
            rX=20
            tY=0
            bY=10
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
            word_boxes.append(bb)
            word_trans.append('Skynet')

            bb=[None]*16
            lX=0
            rX=10
            tY=10
            bY=20
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
            word_boxes.append(bb)
            word_trans.append('Month:')

            bb=[None]*16
            lX=10
            rX=20
            tY=10
            bY=20
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
            word_boxes.append(bb)
            word_trans.append('May')
        else:
            self.qa=[]
            cY=0
            questions=[]
            skipped=[]
            if self.lm:
                q_s=[]
                a_s=[]
                for i in range(self.questions):
                    start_i = random.randrange(0,len(self.words)-self.lm)
                    end_i = start_i+self.lm
                    q_s.append(self.words[start_i])
                    a_s.append(' '.join(self.words[start_i+1,end_i]))
            else:
                q_s = random.sample(self.words,k=2*self.questions + self.additional_doc_len)
                a_s = random.sample(self.words,k=2*self.questions + self.additional_doc_len)
            for i in range(self.questions):
                q=q_s[i]
                a=a_s[i]
                if random.random()<self.not_present_freq:
                    skipped.append(q)
                else:


                    bb=[None]*16
                    lX=0
                    rX=10
                    tY=cY
                    bY=cY+10
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
                    word_boxes.append(bb)
                    if self.only_present:
                        word_trans.append(q)
                    else:
                        word_trans.append('['+q+']')

                        bb=[None]*16
                        lX=10
                        rX=20
                        tY=cY
                        bY=cY+10
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
                        word_boxes.append(bb)
                        word_trans.append(a)

                    if self.repeat_after_me:
                        a=q
                    elif self.lm:
                        a=a
                    elif self.only_present:
                        a='>'
                        questions.append(q)
                    else:
                        a='> {}'.format(a)
                        questions.append(q)

                    self.qa.append((q,a,None))

                    cY+=11

            #keep document size the same by adding redherring words
            for i in range(len(skipped)+self.additional_doc_len):
                q=q_s[-(i+1)]
                a=a_s[-(i+1)]
                bb=[None]*16
                lX=0
                rX=10
                tY=cY
                bY=cY+10
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
                word_boxes.append(bb)
                if self.only_present:
                    word_trans.append(q)
                else:
                    word_trans.append('['+q+']')

                    bb=[None]*16
                    lX=10
                    rX=20
                    tY=cY
                    bY=cY+10
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
                    word_boxes.append(bb)
                    word_trans.append(a)

                cY+=11
            not_present = [w for w in skipped if w not in questions]
            diff = self.questions - len(self.qa)
            for w in not_present[:diff]:
                self.qa.append((w,'~',None))

        if self.shuffle_doc=='pairs':
            pairs = [(word_boxes[a],word_boxes[a+1],word_trans[a],word_trans[a+1]) for a in range(0,len(word_boxes),2)]
            random.shuffle(pairs)
            word_boxes1, word_boxes2, word_trans1, word_trans2 = zip(*pairs)
            word_boxes=[]
            word_trans=[]
            for b1,b2,t1,t2 in pairs:
                word_boxes+=(b1,b2)
                word_trans+=(t1,t2)
        elif  self.shuffle_doc:
            b_and_t = zip(word_boxes,word_trans)
            random.shuffle(b_and_t)
            word_boxes,word_trans = zip(*b_and_t)


        word_boxes = np.array(word_boxes)
        #trans = []
        #groups = []

        bbs=word_boxes[None,...]
        trans=word_trans

        return bbs, list(range(bbs.shape[1])), trans, {}, {}, self.qa
