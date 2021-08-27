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
from .qa import QADataset, collate
from .synth_qadoc_dataset import addRead, breakLong

import utils.img_f as img_f


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


        sub_vocab_file = config['sub_vocab_file'] if 'sub_vocab_file' in config else '../data/wordsEn.txt'
        with open(sub_vocab_file) as f:
            #self.vocab = [w.strip() for w in f.readlines()]
            self.vocab = defaultdict(list)
            for w in f.readlines():
                w=w.strip()
                self.vocab[len(w)].append(w)


        #NEW the document must have a block_score above thresh for anything useing blocks (this is newline following too)
        self.block_score_thresh = 0.73 #eye-balled this one
        self.num_question_types_all=12 #15
        self.num_question_types_noblock=9
        #question types
        #0. Return blanked words (no highlight) and draw where it is "the [blank] chased the cat" > "dog"
        # . Return blanked words (with highlight) and draw where it is "the [blank] chased the cat" > "dog"
        #1. Return replaced word (no highlight) and draw where it is "the industrial chased the cat" > "dog"
        # . Return replaced word (with highlight) and draw where it is "the industrial chased the cat" > "dog"
        #2. Read line above (no highlight)and draw where it is. based on position, not just para/block
        # . Read line above (with highlight) and draw where it is. based on position, not just para/block
        # . Read line below (no highlight)and draw where it is. based on position, not just para/block
        # . Read line below (with highlight) and draw where it is. based on position, not just para/block
        # . (if using blocks) Read line above (no highlight)and draw where it is. based on para/block
        # . (if using blocks) Read line above (with highlight) and draw where it is. based on para/block
        # . (if using blocks) Read line below (no highlight)and draw where it is. based on  para/block
        # . (if using blocks) Read line below (with highlight) and draw where it is. based on  para/block
        #3. draw the line this is in  TODO, same task as below (4)
        #4. draw where this text is 
        #5. Read highlighted section
        #6. Read highlighted section filling in masked word
        #7. guess masked word (HARD!)
        #8. given a word a several masked spots, hightlight which masked spot this belongs in
        #=========
        #9. Read from prompt (no highlight) including new lines (stops at block end) and draw where you read
        # . Read from prompt (with highlight) including new lines (stops at block end) and draw where you read
        #10. Read backwards from prompt (no highlight) including new lines (stops at block end) and draw where you read
        # . Read backwards from prompt (with highlight) including new lines (stops at block end) and draw where you read
        #11. draw the block this is in

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
                if split in readFile:
                    subdirs = readFile[split]
                    toUse=[]
                    for subdir in subdirs:
                        with open(os.path.join(dirPath,subdir+'.list')) as lst:
                            toUse += [path.strip() for path in lst.readlines()]
                    imagesAndAnn = []
                    for path in toUse:#['images']:
                        try:
                            name = path[path.rindex('/')+1:]
                        except ValueError:
                            name = path
                        imagesAndAnn.append( (name,os.path.join(dirPath,path+'.png'),os.path.join(dirPath,path+'.json')) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit(1)
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
                    #exit(1)
        self.errors=[]

        self.punc_regex = re.compile('[%s]' % re.escape(string.punctuation))



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
                    if h==0 or w ==0:
                        continue
                    if h/w>5 and self.rotate: #flip labeling, since FUNSD doesn't label verticle text correctly
                        #assert False #Do i need to do this?
                        #I don't know if it needs rotated clockwise or countercw, so I just say countercw
                        bb = [lX*s, bY*s, lX*s, tY*s, rX*s, tY*s,rX*s, bY*s,
                                s*(lX+rX)/2.0, s*bY, s*(lX+rX)/2.0, s*tY, s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0]
                    else:
                        bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                               s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/2.0, s*bY]  #we add these for conveince to crop BBs within window
                        bb_id = len(qbbs)
                        new_ids.append(bb_id)
                        qbbs.append(bb)

            new_inmask = getAllBBs(ocr,in_mask,s)
            new_outmask = getAllBBs(ocr,out_mask,s)
           
            new_blankmask = getAllBBs(ocr,blank_mask,s,expand=True) #make a bit bigger to get all ink


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
        block_score_sum=0
        line_count=0
        for block in ocr:
            t,l,b,r = block['box']
            h=b-t
            w=r-l
            if w==0 or h==0:
                continue
            squareness = min(0.4,h/w)
            area_whole = h*w
            area_covered = 0 #we'll assume lines don't overlap
            num_lines=0
            for para in block['paragraphs']:
                for line in para['lines']:
                    num_lines+=1
                    for word in line['words']:
                        top,left,bottom,right = word['box']
                        height = bottom-top
                        width = right-left
                        area_covered+=height*width
            if num_lines>1:
                area_score = area_covered/area_whole
            else:
                area_score = 0
            total_score = area_score+squareness
            block_score_sum += total_score*num_lines
            line_count += num_lines
        block_score = block_score_sum/line_count
        use_blocks = block_score>self.block_score_thresh
        print('block_score: {} {}'.format(block_score,'good!' if use_blocks else 'bad'))
        wordmap = makeWordmap(ocr)
        linemap = makeLinemap(ocr)
        qa=[]
        for i in range(self.questions):
            #question_type = random.randrange(self.num_question_types)
            if use_blocks:
                question_type = random.randrange(self.num_question_types_all)
            else:
                question_type = random.randrange(self.num_question_types_noblock)
            if question_type == 0 or question_type == 1 or question_type == 6:

                #0. Return blanked words (no highlight) and draw where it is "the [blank] chased the cat" > "dog"
                # . Return blanked words (with highlight) and draw where it is "the [blank] chased the cat" > "dog"
                #1. Return replaced word (no highlight) and draw where it is "the industrial chased the cat" > "dog"
                # . Return replaced word (with highlight) and draw where it is "the industrial chased the cat" > "dog"
                #6. Read highlighted section filling in masked word

                read_with_masked = question_type == 6
                use_highlight = random.random()<0.5 or read_with_masked
                blank = question_type == 0
                blank_word_idx = random.randrange(len(wordmap))
                outmask = [wordmap[blank_word_idx]]
                blank_word_map = wordmap[blank_word_idx]
                start_word_idx = blank_word_idx
                start_word_map = wordmap[start_word_idx]
                last_start_block = start_word_map[0]
                last_start_line = start_word_map[0:3]
                end_word_idx = blank_word_idx
                end_word_map = wordmap[end_word_idx]
                last_end_block = end_word_map[0]
                last_end_line = end_word_map[0:3]
                at_front_end = start_word_idx==0
                at_back_end = end_word_idx==len(wordmap)-1

                response=ocr[blank_word_map[0]]['paragraphs'][blank_word_map[1]]['lines'][blank_word_map[2]]['words'][blank_word_map[3]]['text']
                if read_with_masked:
                    prompt = response
                elif blank:
                    prompt='ø'
                else:
                    vocab = self.vocab[len(response)] + self.vocab[len(response)-1] + self  .vocab[len(response)+1]
                    if len(vocab)<=1:
                        vocal += self.vocab[len(response)-2] + self.vocab[len(response)+2]
                        if len(vocab)<=1:
                            vocal += self.vocab[len(response)-3] + self.vocab[len(response)+3]
                    prompt=random.choice(vocab)
                    no_punc_response = self.punc_regex.sub('',response)
                    while prompt==no_punc_response:
                        prompt=random.choice(vocab)
                words_in_prompt=[blank_word_idx]
                while len(prompt)<self.max_qa_len-1: #-1 to account for adding a space
                    if not at_front_end and not at_back_end:
                        step_front = random.random()<0.5
                    elif at_front_end and not at_back_end:
                        step_front=False
                    elif not at_front_end and at_back_end:
                        step_front=True
                    else:
                        break

                    if step_front:
                        start_word_idx -= 1
                        start_word_map = wordmap[start_word_idx]
                        changed_block = last_start_block != start_word_map[0]
                        changed_line = last_start_line != start_word_map[0:3]
                        if start_word_idx==0:
                            at_front_end=True
                        if changed_block or (changed_line and not use_blocks):
                            at_front_end=True
                        else:
                            text = ocr[start_word_map[0]]['paragraphs'][start_word_map[1]]['lines'][start_word_map[2]]['words'][start_word_map[3]]['text']
                            if len(text)+len(prompt)+1<=self.max_qa_len:
                                last_start_block = start_word_map[0]
                                last_start_line = start_word_map[0:3]

                                if changed_line:
                                    prompt = text+'\\'+prompt
                                else:
                                    prompt = text+' '+prompt
                                words_in_prompt.append(start_word_idx)
                            else:
                                at_front_end=True #force check back if a word will fit
                    else:
                        end_word_idx += 1
                        end_word_map = wordmap[end_word_idx]
                        changed_block = last_end_block != end_word_map[0]
                        changed_line = last_end_line != end_word_map[0:3]
                        if end_word_idx==len(wordmap)-1:
                            at_back_end=True
                        if changed_block or (changed_line and not use_blocks):
                            at_back_end=True
                        else:
                            text = ocr[end_word_map[0]]['paragraphs'][end_word_map[1]]['lines'][end_word_map[2]]['words'][end_word_map[3]]['text']
                            if len(text)+len(prompt)+1<=self.max_qa_len:
                                last_end_block = end_word_map[0]
                                last_end_line = end_word_map[0:3]

                                if changed_line:
                                    prompt += '\\'+text
                                else:
                                    prompt += ' '+text
                                words_in_prompt.append(end_word_idx)
                            else:
                                at_back_end=True #force check front if a word will fit
                    
                if read_with_masked:
                    question = 'rm>'
                    inmask =  [wordmap[i] for i in words_in_prompt]
                    response=prompt
                    prompt=''
                    maskmask = [blank_word_map]
                    outmask = []
                elif use_highlight:
                    question = 'k0~' if blank else 's0~'
                    inmask =  [wordmap[i] for i in words_in_prompt]
                    maskmask = None
                else:
                    question = 'kb~' if blank else 'su~'
                    inmask = []
                    maskmask = None
                qa.append([question+prompt,response,[wordmap[i] for i in words_in_prompt+[blank_word_idx]],inmask,outmask,maskmask])

            elif question_type == 2:
                #2. Read line above (no highlight)and draw where it is. based on position, not just para/block
                # . Read line above (with highlight) and draw where it is. based on position, not just para/block
                # . Read line below (no highlight)and draw where it is. based on position, not just para/block
                # . Read line below (with highlight) and draw where it is. based on position, not just para/block
                # . Read line above (no highlight)and draw where it is. based on para/block
                # . Read line above (with highlight) and draw where it is. based on para/block
                # . Read line below (no highlight)and draw where it is. based on  para/block
                # . Read line below (with highlight) and draw where it is. based on  para/block
                use_highlight = random.random()<0.5
                above = random.random()<0.5
                beyond_block = random.random()<0.5 if use_blocks else True

                if len(linemap)==0:
                    continue
                line_idx = random.randrange(len(linemap))
                line_map = linemap[line_idx]
                if above:
                    if line_idx==0:
                        next_line_map = getLineAboveBlock(ocr,linemap,line_idx) if beyond_block else None
                    else:
                        next_line_idx = line_idx-1
                        next_line_map = linemap[next_line_idx]
                        if line_map[0] != next_line_map[0]:
                            #get line above line_idx
                            next_line_map = getLineAboveBlock(ocr,linemap,line_idx) if beyond_block else None

                else:
                    if line_idx==len(linemap)-1:
                        next_line_map = getLineAboveBlock(ocr,linemap,line_idx,below=True) if beyond_block else None
                    else:
                        next_line_idx = line_idx+1
                        next_line_map = linemap[next_line_idx]
                        if line_map[0:2] != next_line_map[0:2]:
                            #get line above line_idx
                            next_line_map = getLineAboveBlock(ocr,linemap,line_idx,below=True) if beyond_block else None
                
                prompt = ocr[line_map[0]]['paragraphs'][line_map[1]]['lines'][line_map[2]]['text']
                if next_line_map is not None:
                    response = ocr[next_line_map[0]]['paragraphs'][next_line_map[1]]['lines'][next_line_map[2]]['text']
                    outmask = [next_line_map+(i,) for i in range(len(ocr[next_line_map[0]]['paragraphs'][next_line_map[1]]['lines'][next_line_map[2]]['words']))]
                else:
                    response= 'ø'
                    outmask = []

                if len(prompt)>self.max_qa_len:
                    prompt = prompt[:self.max_qa_len-2] +'>>'
                if len(response)>self.max_qa_len:
                    response = response[:self.max_qa_len-2] +'>>'


                if use_highlight:
                    inmask = [line_map+(i,) for i in range(len(ocr[line_map[0]]['paragraphs'][line_map[1]]['lines'][line_map[2]]['words']))]
                    if beyond_block:
                        question = 'u0~' if above else 'd0~'
                    else:
                        question = '^0~' if above else 'v0~'
                else:
                    inmask = []
                    if beyond_block:
                        question = 'up~' if above else 'dn~'
                    else:
                        question = '^^~' if above else 'vv~'

                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])


            elif question_type == 3:
                #3. draw the line this is in 
                line_idx = random.randrange(len(linemap))
                line_map = linemap[line_idx]
                line = ocr[line_map[0]]['paragraphs'][line_map[1]]['lines'][line_map[2]]
                num_words = len(line['words'])
                
                if num_words>self.min_read_start_no_mask:
                    use_words = random.randrange(self.min_read_start_no_mask,num_words+1)
                    start_word_idx = random.randrange(num_words-use_words+1)
                else:
                    use_words = num_words
                    start_word_idx = 0
                end_word_idx = start_word_idx+use_words-1

                if random.random() < 0.5:
                    #build prompt forwards
                    prompt = line['words'][start_word_idx]['text']
                    word_idx =start_word_idx+1
                    while word_idx<=end_word_idx and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len:
                        prompt+= ' '+line['words'][word_idx]['text']
                        word_idx+=1
                else:
                    #build prompt backwards
                    prompt = line['words'][end_word_idx]['text']
                    word_idx =end_word_idx-1
                    while word_idx>=0 and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len:
                        prompt= line['words'][word_idx]['text']+' '+prompt
                        word_idx-=1
                response = ''
                
                #outmask = [line_map+(i,) for i in range(num_words)]
                outmask = [line_map+(None,)]
                inmask=[]
                question = '0l~'
                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])

            elif question_type == 4 or question_type == 5:
                #4. draw where this text is 
                #5. Read highlighted section
                do_draw = question_type == 4
                line_idx = random.randrange(len(linemap))
                line_map = linemap[line_idx]
                line = ocr[line_map[0]]['paragraphs'][line_map[1]]['lines'][line_map[2]]
                num_words = len(line['words'])
                
                if num_words>self.min_read_start_no_mask:
                    use_words = random.randrange(self.min_read_start_no_mask,num_words+1)
                    start_word_idx = random.randrange(num_words-use_words+1)
                else:
                    use_words = num_words
                    start_word_idx = 0
                end_word_idx = start_word_idx+use_words-1
                if random.random() < 0.5:
                    #build prompt forwards
                    prompt = line['words'][start_word_idx]['text']
                    word_idxs=[start_word_idx]

                    word_idx =start_word_idx+1
                    while word_idx<=end_word_idx and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len:
                        prompt+= ' '+line['words'][word_idx]['text']
                        word_idxs.append(word_idx)
                        word_idx+=1
                else:
                    #build prompt backwards
                    prompt = line['words'][end_word_idx]['text']
                    word_idxs = [end_word_idx]

                    word_idx =end_word_idx-1
                    while word_idx>=0 and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len:
                        prompt= line['words'][word_idx]['text']+' '+prompt
                        word_idxs.append(word_idx)
                        word_idx-=1

                if do_draw: 
                    response = ''
                    
                    outmask = [line_map+(i,) for i in word_idxs]
                    inmask=[]
                    question = '0w~'
                else: #read
                    response = prompt
                    prompt = ''
                    outmask = []
                    inmask = [line_map+(i,) for i in word_idxs]
                    question = 'w0>'
                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])
            #question_type 6 is under the first
            elif question_type == 7:
                #7. guess masked word (HARD!)
                blank_word_idx = random.randrange(len(wordmap))
                maskmask = [wordmap[blank_word_idx]]
                blank_word_map = wordmap[blank_word_idx]
                response=ocr[blank_word_map[0]]['paragraphs'][blank_word_map[1]]['lines'][blank_word_map[2]]['words'][blank_word_map[3]]['text']
                question = 'mk>'
                qa.append([question,response,maskmask,maskmask,None,maskmask])

            elif question_type == 8:
                #8. given a word a several masked spots, hightlight which masked spot this belongs in
                num_blanks = random.randrange(2,9)
                it_word_idx = random.randrange(len(wordmap))
                it_word_map = wordmap[it_word_idx]
                prompt=ocr[it_word_map[0]]['paragraphs'][it_word_map[1]]['lines'][it_word_map[2]]['words'][it_word_map[3]]['text']
                close_len_words=[]
                for i,this_word_map in enumerate(wordmap):
                    if i != it_word_idx:
                        text = ocr[this_word_map[0]]['paragraphs'][this_word_map[1]]['lines'][this_word_map[2]]['words'][this_word_map[3]]['text']
                        if abs(len(text)-len(prompt))<2:
                            close_len_words.append(this_word_map)
                if len(close_len_words)>0:
                    num_blanks = min(num_blanks-1,len(close_len_words))
                    if num_blanks == len(close_len_words):
                        allmaps = close_len_words
                    else:
                        allmaps = random.sample(close_len_words,num_blanks)
                else:
                    word_idxs = random.sample(range(len(wordmap)),num_blanks-1)
                    allmaps = [wordmap[i] for i in word_idxs]

                allmaps.append(it_word_map)
                response=''
                
                question = 'mm~'
                qa.append([question+prompt,response,allmaps,None,[it_word_map],allmaps])

            elif question_type ==9 or question_type==10:
                #9. Read from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read from prompt (with highlight) including new lines (stops at block end) and draw where you read
                #10. Read backwards from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read backwards from prompt (with highlight) including new lines (stops at block end) and draw where you read
                forward = question_type ==9
                use_highlight = random.random()<0.5
                if use_highlight:
                    min_read_start = self.min_read_start_with_mask
                else:
                    min_read_start = self.min_read_start_no_mask
                start_word_idx = random.randrange(len(wordmap))
                goal_prompt_len = random.randrange(min_read_start,self.max_qa_len+1)
                
                words_in_prompt=[start_word_idx]
                start_word = wordmap[start_word_idx]
                prompt = ocr[start_word[0]]['paragraphs'][start_word[1]]['lines'][start_word[2]]['words'][start_word[3]]['text']
                if not forward:
                    prompt=prompt[::-1]
                start_block = start_word[0]
                last_line_id = start_word[0:3]
                next_word_idx = start_word_idx+(1 if forward else -1)
                if next_word_idx>=len(wordmap) or next_word_idx<0:
                    next_word=(None,None,None,None)
                else:
                    next_word = wordmap[next_word_idx]
                    next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                    if not forward:
                        next_text = next_text[::-1]
                    next_line_id = next_word[0:3]
                while next_word[0]==start_block and len(prompt)+1+len(next_text)<self.max_qa_len:
                    if last_line_id==next_line_id:
                        prompt+=' '+next_text
                    else:
                        prompt+='\\'+next_text
                    words_in_prompt.append(next_word_idx)
                    next_word_idx = next_word_idx+(1 if forward else -1)
                    last_line_id=next_line_id
                    if next_word_idx>=len(wordmap):
                        next_word=(None,None,None,None)
                    else:
                        next_word = wordmap[next_word_idx]
                        next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                        if not forward:
                            next_text = next_text[::-1]
                        next_line_id = next_word[0:3]
                    if len(prompt)>=goal_prompt_len:
                        break
                if len(prompt)>self.max_qa_len:
                    prompt = prompt[-self.max_qa_len:]

                if next_word[0]!=start_block:
                    response='‡'
                    words_in_response=[]
                else:
                    goal_response_len = self.max_qa_len
                    if last_line_id==next_line_id:
                        response=next_text
                    else:
                        response='\\'+next_text
                    words_in_response=[next_word_idx]
                    next_word_idx = next_word_idx+(1 if forward else -1)
                    last_line_id=next_line_id
                    if next_word_idx>=len(wordmap):
                        next_word=(None,None,None,None)
                    else:
                        next_word = wordmap[next_word_idx]
                        next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                        if not forward:
                            next_text = next_text[::-1]
                        next_line_id = next_word[0:3]
                    while len(response)+1+len(next_text)<self.max_qa_len and next_word[0]==start_block:
                        if last_line_id==next_line_id:
                            response+=' '+next_text
                        else:
                            response+='\\'+next_text
                        words_in_response.append(next_word_idx)
                        next_word_idx = next_word_idx+(1 if forward else -1)
                        last_line_id=next_line_id
                        if next_word_idx>=len(wordmap):
                            next_word=(None,None,None,None)
                        else:
                            next_word = wordmap[next_word_idx]
                            next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                            if not forward:
                                next_text = next_text[::-1]
                            next_line_id = next_word[0:3]

                        if len(response)>=goal_response_len:
                            break
                    if len(response)>self.max_qa_len:
                        response = response[-self.max_qa_len:]
                    if next_word[0]!=start_block and len(response)<self.max_qa_len:
                        response+='‡'#if we can, add an end-of-block sign
                if use_highlight:
                    question = 'r0~' if forward else 'b0~'
                    inmask =  [wordmap[i] for i in words_in_prompt]
                    #inmask = highlightAll(image_h,image_w,ocr,indexes)
                    #inmask = allBoxes(ocr,indexes)
                else:
                    question = 're~' if forward else 'bk~'
                    inmask = []

                #outmask = highlightAll(image_h,image_w,ocr,[wordmap[i] for i in words_in_response])
                outmask = [wordmap[i] for i in words_in_response]
                #outmask = allBoxes(ocr,indexes)

                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])


            elif question_type == 11:
                #11. draw the block this is in
                use_highlight = random.random()<0.5
                block_idx = random.randrange(len(ocr))
                lines = ocr[block_idx]['paragraphs'][0]['lines']
                line_idx = random.randrange(len(lines))

                line = ocr[block_idx]['paragraphs'][0]['lines'][line_idx]
                num_words = len(line['words'])
                
                if num_words>self.min_read_start_no_mask:
                    use_words = random.randrange(self.min_read_start_no_mask,num_words+1)
                    start_word_idx = random.randrange(num_words-use_words+1)
                else:
                    use_words = num_words
                    start_word_idx = 0
                end_word_idx = start_word_idx+use_words-1

                if random.random() < 0.5:
                    #build prompt forwards
                    prompt = line['words'][start_word_idx]['text']
                    word_idxs = [start_word_idx]
                    word_idx =start_word_idx+1
                    while word_idx<=end_word_idx and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len:
                        prompt+= ' '+line['words'][word_idx]['text']
                        word_idxs.append(word_idx)
                        word_idx+=1
                else:
                    #build prompt backwards
                    prompt = line['words'][end_word_idx]['text']
                    word_idxs = [end_word_idx]
                    word_idx =end_word_idx-1
                    while word_idx>=0 and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len:
                        prompt= line['words'][word_idx]['text']+' '+prompt
                        word_idxs.append(word_idx)
                        word_idx-=1
                response = ''
                
                outmask = [(block_idx,None,None,None)]
                #for ln,line in enumerate(lines):
                #    outmask += [(block_idx,0,ln,i) for i in range(len(line['words']))]
                if use_highlight:
                    inmask=[(block_idx,0,line_idx,i) for i in word_idxs]
                    question = '00~'
                else:
                    inmask=[]
                    question = '0p~'
                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])

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
def makeLinemap(ocr):
    linemap=[]
    for b,block in enumerate(ocr):
        for p,para in enumerate(block['paragraphs']):
            for l in range(len(para['lines'])):
                linemap.append((b,p,l))
    return linemap

def getAllBBs(ocr,t_ids,s,expand=False):
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
            if expand:
                lX-=2
                tY-=2
                rX+=2
                bY+=2
            bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                   s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/2.0, s*bY]  #we add these for conveince to crop BBs within window
            bbs.append(bb)
    return bbs

def getLineAboveBlock(ocr,linemap,line_idx,below=False):
    line_map = linemap[line_idx]
    #get block width
    #x1=999999
    #x2=0
    #for line in ocr[line_map[0]]['lines']:
    #    l,t,r,b = line['box']
    #    x1 = min(x1,l)
    #    x2 = max(x2,r)
    l,t,r,b = ocr[line_map[0]]['box']

    closest_block=None
    cb_dist=99999999
    for bid,block in enumerate(ocr):
        if bid == line_map[0]:
            continue
        l2,t2,r2,b2 = block['box']
        covered_horz = max(min(r,r2) - max(l,l2),0)
        covered_horz/=min(r2-l2,r-l)
        if below:
            dist = t2-b
        else:
            dist = t-b2
        if covered_horz>0.7 and dist>=-2 and dist<cb_dist:
            cb_dist = dist
            closest_block=bid
    if closest_block is None:
        return None

    lowest_line=0
    assert len(ocr[closest_block]['paragraphs'])==1
    ll_y=ocr[closest_block]['paragraphs'][0]['lines'][0]['box'][1 if below else 3]
    for l in range(1,len(ocr[closest_block]['paragraphs'][0]['lines'])):
        y = ocr[closest_block]['paragraphs'][0]['lines'][l]['box'][1 if below else 3]
        if (below and y<ll_y) or (not below and y>ll_y):
            lowest_line=l
            ll_y=y

    return (closest_block,0,lowest_line)



