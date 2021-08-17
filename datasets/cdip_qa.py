import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from .qa import QADataset
from .synth_qadoc_dataset import addRead, breakLong

import utils.img_f as img_f

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


class CDIPQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(CDIPQA, self).__init__(dirPath,split,config,images)
        assert self.questions==1 #current set up (with masks being appended to image) requires only 1 qa pair per image
        self.do_masks=True
        self.cache_resized = False

        self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0

        self.extra_np = 0.05

        self.min_read_start_no_mask=9
        self.min_read_start_with_mask=4

        self.num_question_types=1 #15
        #question types
        #0. Read from prompt (no highlight) including new lines (stops at block end) and draw where you read
        # . Read from prompt (with highlight) including new lines (stops at block end) and draw where you read
        #1. Read backwards from prompt (no highlight) including new lines (stops at block end) and draw where you read
        # . Read backwards from prompt (with highlight) including new lines (stops at block end) and draw where you read
        #2. Return blanked words (no highlight) and draw where it is "the [blank] chased the cat" > "cat"
        # . Return blanked words (with highlight) and draw where it is "the [blank] chased the cat" > "cat"
        #3. Return replaced word (no highlight) and draw where it is "the industrial chased the cat" > "cat"
        # . Return replaced word (with highlight) and draw where it is "the industrial chased the cat" > "cat"
        #4. Read line above (no highlight)and draw where it is. based on position, not just para/block
        # . Read line above (with highlight) and draw where it is. based on position, not just para/block
        #5. Read line below (no highlight)and draw where it is. based on position, not just para/block
        # . Read line below (with highlight) and draw where it is. based on position, not just para/block
        #6. bool is start of block (no highlight) (line) if no, hightlight next line
        # . bool is start of block (with highlight) (line) if no, hightlight next line
        #7. bool is end of block (no highlight) if no, hightlight next line
        # . bool is end of block (with highlight) if no, hightlight next line
        #8. draw the line this is in 
        #9. draw the para this is in
        #10. draw the block this is in
        #=========
        #11. draw where this text is 
        #12. Read highlighted section
        #13. Read highlighted section filling in masked word
        #14. guess masked word (HARD!)

        #input mask. 0 everywhere, 1 is highlight, -1 where removed
        #  Multi-channel for multiple questions?

        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                splitFile = 'overfit_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                readFile = json.loads(f.read())
                if type(split) is str:
                    toUse = readFile[split]
                    imagesAndAnn = []
                    for path in toUse:#['images']:
                        try:
                            name = path[path.rindex('/')+1:]
                        except ValueError:
                            name = path
                        imagesAndAnn.append( (name,os.path.join(dirPath,path+'.png'),os.path.join(dirPath,name+'.json')) )
                elif type(split) is list:
                    imagesAndAnn = []
                    for spstr in split:
                        toUse = readFile[spstr]
                        for path in toUse:#['images']:
                            name = path[path.rindex('/')+1:]
                            imagesAndAnn.append( (name,os.path.join(dirPath,path+'.png'),os.path.join(dirPath,name+'.json')) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
            self.images=[]
            for imageName,imagePath,jsonPath in imagesAndAnn:
                if os.path.exists(jsonPath):
                    org_path = imagePath
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName+'.png')
                    else:
                        path = org_path

                    rescale=1.0
                    if self.cache_resized:
                        rescale = self.rescale_range[1]
                        if not os.path.exists(path):
                            org_img = img_f.imread(org_path)
                            if org_img is None:
                                print('WARNING, could not read {}'.format(org_img))
                                continue
                            resized = img_f.resize(org_img,(0,0),
                                    fx=self.rescale_range[1], 
                                    fy=self.rescale_range[1], 
                                    )
                            img_f.imwrite(path,resized)
                    self.images.append({'id':imageName, 'imageName':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale })
                else:
                    print('{} does not exist'.format(jsonPath))
                    print('No json found for {}'.format(imagePath))
                    exit(1)
        self.errors=[]



    def parseAnn(self,ocr,s):
        image_h=ocr['height']
        image_w=ocr['width']
        ocr=ocr['blocks']
        qa = self.makeQuestions(ocr,image_h,image_w)

        qbbs=[]
        t_id_to_bb={}
        new_qa = []
        for q,a,t_ids,in_mask,out_mask,blank_mask in qa:
            new_ids=[]
            for t_id in t_ids:
                if t_id in t_id_to_bb:
                    bb_id = t_id_to_bb[t_id]
                else:
                    b,p,l,w = t_id
                    inst = ocr[b]
                    if p is not None:
                        inst = inst['paragraphs'][p]
                        if l is not None:
                            inst = inst['lines'][l]
                            if w is not None:
                                inst = inst['words'][w]
                    
                    box = inst['box']
                    #text = inst['text']
                    

                    lX,tY,rX,bY = box
                    h=bY-tY
                    w=rX-lX
                    if h/w>5 and self.rotate: #flip labeling, since FUNSD doesn't label verticle text correctly
                        assert False #Do i need to do this?
                        #I don't know if it needs rotated clockwise or countercw, so I just say countercw
                        bbs[j,0]=lX*s
                        bbs[j,1]=bY*s
                        bbs[j,2]=lX*s
                        bbs[j,3]=tY*s
                        bbs[j,4]=rX*s
                        bbs[j,5]=tY*s
                        bbs[j,6]=rX*s
                        bbs[j,7]=bY*s
                        #we add these for conveince to crop BBs within window
                        bbs[j,8]=s*(lX+rX)/2.0
                        bbs[j,9]=s*bY
                        bbs[j,10]=s*(lX+rX)/2.0
                        bbs[j,11]=s*tY
                        bbs[j,12]=s*lX
                        bbs[j,13]=s*(tY+bY)/2.0
                        bbs[j,14]=s*rX
                        bbs[j,15]=s*(tY+bY)/2.0
                    else:
                        bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                               s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/2.0, s*bY]  #we add these for conveince to crop BBs within window
                        bb_id = len(qbbs)
                        new_ids.append(bb_id)
                        qbbs.append(bb)

            new_inmask = getAllBBs(ocr,in_mask,s)
            new_outmask = getAllBBs(ocr,out_mask,s)
            print(out_mask)
            print(new_outmask)
            new_blankmask = getAllBBs(ocr,blank_mask,s)


            new_qa.append((q,a,new_ids,new_inmask,new_outmask,new_blankmask))
        qa_bbs = np.array(qbbs)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, {}, {}, new_qa


    def getResponseBBIdList(self,queryId,annotations):
        if self.split_to_lines:
            return annotations['linking'][queryId]
        else:
            boxes=annotations['form']
            cto=[]
            boxinfo = boxes[queryId]
            for id1,id2 in boxinfo['linking']:
                if id1==queryId:
                    cto.append(id2)
                else:
                    cto.append(id1)
            return cto

    def makeQuestions(self,ocr,image_h,image_w):
        wordmap=None
        qa=[]
        for i in range(self.questions):
            question_type = random.randrange(self.num_question_types)
            print('make question {} of type {}'.format(i,question_type))
            if question_type ==0:
                #0. Read from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read from prompt (with highlight) including new lines (stops at block end) and draw where you read
                use_highlight = random.random()<0.5
                if use_highlight:
                    min_read_start = self.min_read_start_with_mask
                else:
                    min_read_start = self.min_read_start_no_mask
                if wordmap is None:
                    wordmap = makeWordmap(ocr)
                start_word_idx = random.randrange(len(wordmap))
                goal_prompt_len = random.randrange(min_read_start,self.max_qa_len+1)
                
                words_in_prompt=[start_word_idx]
                start_word = wordmap[start_word_idx]
                prompt = ocr[start_word[0]]['paragraphs'][start_word[1]]['lines'][start_word[2]]['words'][start_word[3]]['text']
                start_block = start_word[0]
                next_word_idx = start_word_idx+1
                if next_word_idx>=len(wordmap):
                    next_word=(None,None,None,None)
                else:
                    next_word = wordmap[next_word_idx]
                    next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                while len(prompt)+1+len(next_text)<self.max_qa_len and next_word[0]==start_block:
                    prompt+=' '+next_text
                    words_in_prompt.append(next_word_idx)
                    next_word_idx = next_word_idx+1
                    if next_word_idx>=len(wordmap):
                        next_word=(None,None,None,None)
                    else:
                        next_word = wordmap[next_word_idx]
                        next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                    if len(prompt)>=goal_prompt_len:
                        break
                if len(prompt)>self.max_qa_len:
                    prompt = prompt[-self.max_qa_len:]

                if next_word[0]!=start_block:
                    response='‡'
                    words_in_response=[]
                else:
                    goal_response_len = self.max_qa_len
                    response=next_text
                    words_in_response=[next_word_idx]
                    print(words_in_response)
                    next_word_idx = next_word_idx+1
                    if next_word_idx>=len(wordmap):
                        next_word=(None,None,None,None)
                    else:
                        next_word = wordmap[next_word_idx]
                        next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                    while len(response)+1+len(next_text)<self.max_qa_len and next_word[0]==start_block:
                        response+=' '+next_text
                        words_in_response.append(next_word_idx)
                        print(words_in_response)
                        next_word_idx = next_word_idx+1
                        if next_word_idx>=len(wordmap):
                            next_word=(None,None,None,None)
                        else:
                            next_word = wordmap[next_word_idx]
                            next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']

                        if len(response)>=goal_response_len:
                            break
                    if len(response)>self.max_qa_len:
                        response = response[-self.max_qa_len:]
                    if next_word[0]!=start_block and len(response)<self.max_qa_len:
                        response+='‡'#if we can, add an end-of-block sign
                if use_highlight:
                    question = 'r0~'
                    inmask =  [wordmap[i] for i in words_in_prompt]
                    #inmask = highlightAll(image_h,image_w,ocr,indexes)
                    #inmask = allBoxes(ocr,indexes)
                else:
                    question = 're~'
                    inmask = []

                #outmask = highlightAll(image_h,image_w,ocr,[wordmap[i] for i in words_in_response])
                outmask = [wordmap[i] for i in words_in_response]
                #outmask = allBoxes(ocr,indexes)

                qa.append([question+prompt,response,[wordmap[i] for i in words_in_prompt+words_in_response],inmask,outmask,None])

                #1. Read backwards from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read backwards from prompt (with highlight) including new lines (stops at block end) and draw where you read
                #2. Return blanked words (no highlight) and draw where it is "the [blank] chased the cat" > "cat"
                # . Return blanked words (with highlight) and draw where it is "the [blank] chased the cat" > "cat"
                #3. Return replaced word (no highlight) and draw where it is "the industrial chased the cat" > "cat"
                # . Return replaced word (with highlight) and draw where it is "the industrial chased the cat" > "cat"
                #4. Read line above (no highlight)and draw where it is. based on position, not just para/block
                # . Read line above (with highlight) and draw where it is. based on position, not just para/block
                #5. Read line below (no highlight)and draw where it is. based on position, not just para/block
                # . Read line below (with highlight) and draw where it is. based on position, not just para/block
                #6. bool is start of block (no highlight) (line) if no, hightlight next line
                # . bool is start of block (with highlight) (line) if no, hightlight next line
                #7. bool is end of block (no highlight) if no, hightlight next line
                # . bool is end of block (with highlight) if no, hightlight next line
                #8. draw the line this is in 
                #9. draw the para this is in
                #10. draw the block this is in
                #=========
                #11. draw where this text is 
                #12. Read highlighted section
                #13. Read highlighted section filling in masked word
                #14. guess masked word (HARD!)
        return qa

#def highlightAll(image_h,image_w,ocr,indexes,value=1):
#    mask = torch.FloatTensor(image_h,image_w).fill_(0)
#    for idx in indexes:
#        if idx[3] is not None:
#            l,t,r,b = ocr[idx[0]]['paragraphs'][idx[1]]['lines'][idx[2]]['words'][idx[3]]['box']
#        elif idx[2] is not None:
#            l,t,r,b = ocr[idx[0]]['paragraphs'][idx[1]]['lines'][idx[2]]['box']
#        elif idx[1] is not None:
#            l,t,r,b = ocr[idx[0]]['paragraphs'][idx[1]]['box']
#        else:
#            l,t,r,b = ocr[idx[0]]['box']
#
#        mask[t:b+1,l:r+1]=value
#    return mask

def allBoxes(ocr,indexes):
    ret=[]
    for idx in indexes:
        if idx[3] is not None:
            bb = ocr[idx[0]]['paragraphs'][idx[1]]['lines'][idx[2]]['words'][idx[3]]['box']
        elif idx[2] is not None:
            bb = ocr[idx[0]]['paragraphs'][idx[1]]['lines'][idx[2]]['box']
        elif idx[1] is not None:
            bb = ocr[idx[0]]['paragraphs'][idx[1]]['box']
        else:
            bb = ocr[idx[0]]['box']
        ret.append(bb)
    return ret

def makeWordmap(ocr):
    wordmap=[]
    for b,block in enumerate(ocr):
        for p,para in enumerate(block['paragraphs']):
            for l,line in enumerate(para['lines']):
                for w in range(len(line['words'])):
                    wordmap.append((b,p,l,w))
    return wordmap

def getAllBBs(ocr,t_ids,s):
    bbs=[]
    if t_ids is not None:
        for t_id in t_ids:
            b,p,l,w = t_id
            inst = ocr[b]
            if p is not None:
                inst = inst['paragraphs'][p]
                if l is not None:
                    inst = inst['lines'][l]
                    if w is not None:
                        inst = inst['words'][w]
            
            box = inst['box']
            lX,tY,rX,bY = box
            bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                   s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/2.0, s*bY]  #we add these for conveince to crop BBs within window
            bbs.append(bb)
    return bbs
