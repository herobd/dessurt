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



class ParaQADataset(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(ParaQADataset, self).__init__(dirPath,split,config,images)
        assert self.questions==1 #current set up (with masks being appended to image) requires only 1 qa pair per image
        self.do_masks=True

        self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0

        self.extra_np = 0.05

        self.min_read_start_no_mask=5
        self.min_read_start_with_mask=1


        self.punc_regex = re.compile('[%s]' % re.escape(string.punctuation))
        sub_vocab_file = config['sub_vocab_file'] if 'sub_vocab_file' in config else '../data/wordsEn.txt'
        with open(sub_vocab_file) as f:
            #self.vocab = [w.strip() for w in f.readlines()]
            self.vocab = defaultdict(list)
            for w in f.readlines():
                w=w.strip()
                self.vocab[len(w)].append(w)

        self.q_types =      ['read_blanked','read_replaced','read_with_masked','read_line','highlight_text','read_highlighted','masked_lm','put_in_place','read_on','read_backwards','highlight_block']
        self.q_type_weights=[0.5,           0.5,            1.0,               0.5,        1.0,             0.5,               4.0,        1.0,            0.5,      0.5,             1.0]
        self.q_types_noblock =      ['read_blanked','read_replaced','read_with_masked','read_line','highlight_text','read_highlighted','masked_lm','put_in_place']
        self.q_type_weights_noblock=[0.5,           0.5,            1.0,               0.5,        1.0,             0.5,               4.0,        1.0]

        #self.num_question_types_all=11 #15
        #self.num_question_types_noblock=8
        #question types
        #0. Return blanked words (no highlight) and draw where it is "the [blank] chased the cat" > "dog" [kb]
        # . Return blanked words (with highlight) and draw where it is "the [blank] chased the cat" > "dog" [k0]
        #1. Return replaced word (no highlight) and draw where it is "the industrial chased the cat" > "dog" [su]
        # . Return replaced word (with highlight) and draw where it is "the industrial chased the cat" > "dog" [s0]
        #2. Read line above (no highlight)and draw where it is. based on position, not just para/block [up]
        # . Read line above (with highlight) and draw where it is. based on position, not just para/block [u0]
        # . Read line below (no highlight)and draw where it is. based on position, not just para/block  [dn]
        # . Read line below (with highlight) and draw where it is. based on position, not just para/block [d0]
        # . (if using blocks) Read line above (no highlight)and draw where it is. based on para/block [^^]
        # . (if using blocks) Read line above (with highlight) and draw where it is. based on para/block [^0]
        # . (if using blocks) Read line below (no highlight)and draw where it is. based on  para/block [vv]
        # . (if using blocks) Read line below (with highlight) and draw where it is. based on  para/block [v0]
        #3a. draw the line this is in  same task as below (4) [0l]
        # b. draw where this text is [0w]
        #4. Read highlighted section [w0]
        # . Read highlighted line [l0]
        #5. Read highlighted section filling in masked word [rm>]
        #6. guess masked word (HARD!) [mk]
        #7. given a word a several masked spots, hightlight which masked spot this belongs in [mm]
        #=========
        #8. Read from prompt (no highlight) including new lines (stops at block end) and draw where you read [re]
        # . Read from prompt (with highlight) including new lines (stops at block end) and draw where you read [r0]
        #9. Read backwards from prompt (no highlight) including new lines (stops at block end) and draw where you read [bk]
        # . Read backwards from prompt (with highlight) including new lines (stops at block end) and draw where you read [b0]
        #10. draw the block this is in [0p or 00]

        #input mask. 0 everywhere, 1 is highlight, -1 where removed
        #  Multi-channel for multiple questions?





    def makeQuestions(self,ocr,image_h,image_w,s,use_blocks=True):
        wordmap = makeWordmap(ocr)
        if len(wordmap)==0:
            return [],np.array([])
        linemap = makeLinemap(ocr)
        if use_blocks:
            q_types = random.choices(self.q_types,self.q_type_weights,k=self.questions*50)
        else:
            q_types = random.choices(self.q_types_noblock,self.q_types_weights_noblock,k=self.questions*50)

        qa=[]
        #for i in range(self.questions*10): #return extra in case the cropping/rotations clips some words
        for question_type in q_types:
            #if use_blocks:
            #    question_type = random.randrange(self.num_question_types_all)
            #else:
            #    question_type = random.randrange(self.num_question_types_noblock)
            #if question_type == 0 or question_type == 1 or question_type == 5:
            if question_type == 'read_blanked' or question_type == 'read_replaced' or question_type == 'read_with_masked':

                #0. Return blanked words (no highlight) and draw where it is "the [blank] chased the cat" > "dog"
                # . Return blanked words (with highlight) and draw where it is "the [blank] chased the cat" > "dog"
                #1. Return replaced word (no highlight) and draw where it is "the industrial chased the cat" > "dog"
                # . Return replaced word (with highlight) and draw where it is "the industrial chased the cat" > "dog"
                #5. Read highlighted section filling in masked word

                read_with_masked = question_type == 'read_with_masked'
                use_highlight = random.random()<0.5 or read_with_masked
                blank = question_type == 'read_blanked'
                for i in range(10):
                    blank_word_idx = random.randrange(len(wordmap))
                    outmask = [wordmap[blank_word_idx]]
                    blank_word_map = wordmap[blank_word_idx]
                    response=ocr[blank_word_map[0]]['paragraphs'][blank_word_map[1]]['lines'][blank_word_map[2]]['words'][blank_word_map[3]]['text']
                    if len(response)<15:
                        break
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

                if read_with_masked:
                    prompt = response
                elif blank:
                    prompt='ø'
                else:
                    vocab = self.vocab[len(response)] + self.vocab[len(response)-1] + self.vocab[len(response)+1]
                    if len(vocab)<=1:
                        vocab += self.vocab[len(response)-2] + self.vocab[len(response)+2]
                        if len(vocab)<=1:
                            vocab += self.vocab[len(response)-3] + self.vocab[len(response)+3]
                            if len(vocab)<=1:
                                vocab += self.vocab[len(response)-4] + self.vocab[len(response)+4]
                                if len(vocab)<=1:
                                    vocab += self.vocab[len(response)-5] + self.vocab[len(response)+5]
                    if len(vocab)==0:
                        print('no vocab for string of length {} ({})'.format(len(response),response))
                        continue
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
                    outmask = [wordmap[i] for i in words_in_prompt]
                elif use_highlight:
                    question = 'k0~' if blank else 's0~'
                    inmask =  [wordmap[i] for i in words_in_prompt]
                    maskmask = None
                else:
                    question = 'kb~' if blank else 'su~'
                    inmask = []
                    maskmask = None
                qa.append([question+prompt,response,[wordmap[i] for i in words_in_prompt+[blank_word_idx]],inmask,outmask,maskmask])

            #elif question_type == 2:
            elif question_type == 'read_line':
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
                    response= '№'
                    outmask = []

                if len(prompt)>self.max_qa_len:
                    prompt = prompt[:self.max_qa_len-2] +'>>'
                if len(response)>self.max_qa_len:
                    response = response[:self.max_qa_len]
                elif len(response)+1 < self.max_qa_len and response!='№':
                    response = response+'‡'


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


            #elif question_type == 3 and random.random()<0.5:
            elif question_type == 'highlight_text' and random.random()<0.5:
                #3a. draw the line this is in 
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
                question = '0;~'
                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])

            #elif question_type == 3 or question_type == 4:
            elif question_type == 'highlight_text' or question_type == 'read_highlighted':
                #3b. draw where this text is 
                #4. Read highlighted section
                do_draw = question_type == 'highlight_text'
                if not do_draw:
                    whole_line = random.random()<0.5
                else:
                    whole_line = False
                line_idx = random.randrange(len(linemap))
                line_map = linemap[line_idx]
                line = ocr[line_map[0]]['paragraphs'][line_map[1]]['lines'][line_map[2]]
                num_words = len(line['words'])

                if whole_line:
                    assert not do_draw
                    response = line['text']
                    if len(response)>self.max_qa_len:
                        response = response[:self.max_qa_len]
                    elif len(response)+1 <= self.max_qa_len:
                        response += '‡'
                    prompt = ''
                    question = ';0>'
                    inmask = [line_map+(None,)]
                    outmask = [line_map+(i,) for i in range(num_words)]
                else:
                    if num_words>self.min_read_start_no_mask:
                        use_words = random.randrange(self.min_read_start_no_mask,num_words+1)
                        build_forwards = random.random() < 0.5
                        if build_forwards:
                            start_word_idx = random.randrange(num_words-use_words+1)
                        else:
                            start_word_idx = random.randrange(use_words-1,num_words)
                    else:
                        build_forwards = True
                        use_words = num_words
                        start_word_idx = 0
                    if build_forwards:
                        #build prompt forwards
                        end_word_idx = start_word_idx+use_words-1
                        prompt = line['words'][start_word_idx]['text']
                        word_idxs=[start_word_idx]

                        word_idx =start_word_idx+1
                        while word_idx<=end_word_idx and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len:
                            prompt+= ' '+line['words'][word_idx]['text']
                            word_idxs.append(word_idx)
                            word_idx+=1
                    else:
                        #build prompt backwards
                        end_word_idx = start_word_idx-use_words
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
                        outmask = [line_map+(i,) for i in word_idxs] #want it in habit of outputing where it reads
                        inmask = [line_map+(i,) for i in word_idxs]
                        question = 'w0>'
                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])
            #question_type 5 is under the first
            #elif question_type == 6:
            elif question_type == 'masked_lm':
                #6. guess masked word (HARD!)
                blank_word_idx = random.randrange(len(wordmap))
                maskmask = [wordmap[blank_word_idx]]
                blank_word_map = wordmap[blank_word_idx]
                response=ocr[blank_word_map[0]]['paragraphs'][blank_word_map[1]]['lines'][blank_word_map[2]]['words'][blank_word_map[3]]['text']
                question = 'mk>'
                qa.append([question,response,maskmask,maskmask,None,maskmask])

            #elif question_type == 7 and len(wordmap)>1:
            elif question_type == 'put_in_place' and len(wordmap)>1:
                #7. given a word a several masked spots, hightlight which masked spot this belongs in
                num_blanks = random.randrange(2,9)
                it_word_idx = random.randrange(len(wordmap))
                it_word_maps = [wordmap[it_word_idx]]
                prompt=ocr[it_word_maps[0][0]]['paragraphs'][it_word_maps[0][1]]['lines'][it_word_maps[0][2]]['words'][it_word_maps[0][3]]['text']
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
                    if num_blanks-1 > len(wordmap):
                        num_blanks = 2
                    word_idxs = random.sample(range(len(wordmap)),num_blanks-1)
                    allmaps = [wordmap[i] for i in word_idxs if i != it_word_idx]

                #We may have blanked other instances of the same word. We'll just say they are right answers too.
                for this_word_map in allmaps:
                    text = ocr[this_word_map[0]]['paragraphs'][this_word_map[1]]['lines'][this_word_map[2]]['words'][this_word_map[3]]['text']
                    if text.lower()==prompt.lower():
                        it_word_maps.append(this_word_map)

                allmaps.append(it_word_maps[0]) #add original 'it'
                response=''
                
                question = 'mm~'
                qa.append([question+prompt,response,allmaps,None,it_word_maps,allmaps])

            #elif question_type ==8 or question_type==9:
            elif question_type =='read_on' or question_type=='read_backwards':
                #8. Read from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read from prompt (with highlight) including new lines (stops at block end) and draw where you read
                #9. Read backwards from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read backwards from prompt (with highlight) including new lines (stops at block end) and draw where you read
                forward = question_type =='read_on'
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
                    if next_word_idx>=len(wordmap) or next_word_idx<0:
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
                        if next_word_idx>=len(wordmap) or next_word_idx<0: 
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
                        response = response[:self.max_qa_len]
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


            #elif question_type == 10:
            elif question_type == 'highlight_block':
                #10. draw the block this is in
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

            if len(qa)>=self.questions*10:
                break

        #return qa

        #Set up bounding boxes and masks
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


            self.qaAdd(new_qa,q,a,new_ids,new_inmask,new_outmask,new_blankmask)
        qa_bbs = np.array(qbbs)


        return new_qa, qa_bbs

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
        covered_horz/=min(max(r2-l2,1),max(r-l,1))
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



