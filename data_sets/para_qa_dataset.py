import torch.utils.data
import numpy as np
import json
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from data_sets.qa import QADataset, collate
from transformers import BartTokenizer

#The creates a MLM run instance (based on BART)
#used by distillation dataset too
def getMinMaxWidth(words):
    max_w=0
    min_w=99999999
    x_diffs=[]
    last_x=None
    for w in words:
        if w is not None:
            lx,ty,rx,by = w[0]['box']
            w=rx-lx
            max_w = max(max_w,w)
            min_w = min(min_w,w)
            if last_x is not None and last_x<lx:
                x_diffs.append(lx-last_x)
            last_x = rx
    space = np.mean(x_diffs) if len(x_diffs)>1 else 10
    return min_w,max_w,space

def makeMLMInstance(ocr):           
    ##Make mlm instances
    words = []

    #get a good block of text
    while len(words)<4 and len(ocr)>1:
        words = []
        block_i = random.randrange(len(ocr))
        block = ocr[block_i]
        ocr = ocr[:block_i]+ocr[block_i+1:] #remove
        target_string = []
        for p,para in enumerate(block['paragraphs']):
            for l,line in enumerate(para['lines']):
                for word in line['words']:
                    words.append((word,p,l))
                    target_string.append(word['text'])
        target_string = ' '.join(target_string)
    if len(words)<4:
        return None,None,None,None

    num_spans=random.randrange(1,max(len(words)//8,2)) #how many mask spans
    to_remove=[]
    for i in range(num_spans):
        num = np.random.poisson(3)  #number of words in this span
        num = min(num,len(words)-1)

        #Find a suitable area to mask
        good_spot = False
        for i in range(50 if num>0 else 20):
            loc = random.randrange(0,1+len(words)-num)
            if num==0:
                #If this is an empty span, it can only be at begging or end of block
                if (loc==0 and words[0] is not None) or (loc==len(words) and words[-1] is not None):
                    good_spot = True
                    break
                elif words[0] is None and words[-1] is None:
                    good_spot = False
                    break
            else:
                before=max(0,loc-1)
                after=min(len(words),loc+num+1)
                good_spot = all((w is not None) for w in words[before:after])
                if good_spot:
                    break
        

        if good_spot:
            #get the bounding box to mask out of image
            rm_words = words[loc:loc+num]
            pre_add_none = words
            words = words[:loc]+[None]+words[loc+num:]

            if len(rm_words)>0:
                #group the words by line
                lines=defaultdict(list)
                for word,p,l in rm_words:
                    lines[(p,l)].append(word)
                for line in lines.values():
                    #get bb. We can assume words are in read order
                    #I'll assume left-right read order
                    left_x = line[0]['box'][0] -1
                    right_x = line[-1]['box'][2] +1
                    top_y = min(w['box'][1] for w in line) -1
                    bot_y = max(w['box'][3] for w in line) +1
                    to_remove.append((round(left_x),round(top_y),round(right_x),round(bot_y)))
            else:
                #empty span, add "removed region" to beginning or end of block
                word = pre_add_none[loc] if loc==0 else pre_add_none[loc-1]
                if word is not None:
                    word_min_w,word_max_w,space = getMinMaxWidth(pre_add_none)
                    word,p,l = word
                    line=block['paragraphs'][p]['lines'][l]
                    line_x1,line_y1,line_x2,line_y2 = line['box']
                    if loc == 0:
                        right_x,top_y,_,bot_y = word['box']
                        right_x -= space
                        left_x = max(0,right_x - random.randrange(word_min_w,word_max_w+1))
                        line['box'] = (round(left_x),round(line_y1),round(line_x2),round(line_y2))
                    else:
                        assert loc == len(pre_add_none)
                        _,top_y,left_x,bot_y = word['box']
                        left_x += space
                        right_x = left_x + random.randrange(word_min_w,word_max_w+1)
                        line['box'] = (round(line_x1),round(line_y1),round(right_x),round(line_y2))
                    to_remove.append((round(left_x),round(top_y),round(right_x),round(bot_y)))


    
    return words,to_remove,target_string,block


#This class is the paraent for synthetic Wikipedia and IIT-CDIP data
#It defines the tasks with the makeQuestions function
#makeQuestions recieves the textual information of the document formatted in the TessearactOCR output style:
# List of blocks, each block has a "box": (l,t,r,b) and "paragraphs": list, though we always assume there is exactly one paragraph per block,
# A paragraph has a box and "lines": list
# A line has a box, "text": string, and "words": list
# A word as a box and text
class ParaQADataset(QADataset):


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(ParaQADataset, self).__init__(dirPath,split,config,images)
        assert self.questions==1 #current set up (with masks being appended to image) requires only 1 qa pair per image
        self.do_masks=True
        self.use_highlight = config.get('use_highlight',True)

        self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0 #not used

        self.extra_np = 0.05

        self.min_read_start_no_mask=5
        self.min_read_start_with_mask=1

        self.end_token = '‡'

        self.max_q_tokens = config.get('max_q_tokens',20)
        self.max_a_tokens = config.get('max_a_tokens',800)

        if os.path.exists('./cache_huggingface/BART'):
            model_id = './cache_huggingface/BART'
        else:
            model_id = 'facebook/bart-base'
        self.tokenizer = BartTokenizer.from_pretrained(model_id) #needed for correct read-on


        self.punc_regex = re.compile('[%s]' % re.escape(string.punctuation))

        #We want a length sorted vocab for subtituting words
        sub_vocab_file = config['sub_vocab_file'] if 'sub_vocab_file' in config else 'data_sets/wordsEn.txt'
        with open(sub_vocab_file) as f:
            self.vocab = defaultdict(list)
            for w in f.readlines():
                w=w.strip()
                self.vocab[len(w)].append(w)

        mode = 'easy' if ('easy' in config and config['easy']) else 'hard'
        mode = config['mode'] if 'mode' in config else mode
        #mode defines the tasks and their frequencies
        #"hard_word" is the mode used in Dessurt's pretraining
        if mode == 'blind':
            self.q_types = {
                    'read_blanked':1,
                    'read_line':0.75,
                    'highlight_text':0.1,
                    'read_on':0.5,
                    'read_backwards':0.5,
                    'highlight_block':0.1}
            self.q_types_noblock = {
                    'read_blanked':1,
                    'read_line':0.76,
                    'highlight_text':0.1,
                    }
        elif mode == 'echo':
            self.q_types = {
                    'echo':1,
                    }
            self.q_types_noblock = {
                    'echo':1,
                    }
        elif mode == 'echo2':
            self.q_types = {
                    'echo2':1,
                    }
            self.q_types_noblock = {
                    'echo2':1,
                    }
        elif mode == 'simple':
            self.q_types = {
                    'read_blanked':1,
                    'read_on':1,
                    }
            self.q_types_noblock = {
                    'read_blanked':1,
                    'read_on':1,
                    }
        elif mode == 'easy':
            self.q_types = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'read_with_masked':1,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    'put_in_place':1.0,
                    'read_on':0.5,
                    'read_backwards':0.5,
                    'highlight_block':1.0}
            self.q_types_noblock = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'read_with_masked':1.0,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    'put_in_place':1.0}
        elif mode == 'easy_word':
            self.q_types = {
                    'read_blanked':1,
                    'proper_read_replaced':1,
                    'long_mlm':1.0,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    'put_in_place':1.0,
                    'read_on':1.0,
                    'highlight_block':1.0}
            self.q_types_noblock = {
                    'read_blanked':1,
                    'proper_read_replaced':1,
                    'long_mlm':1.0,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    'put_in_place':1.0}
        elif mode == 'IAM':
            self.q_types = {
                    'read_blanked':1,
                    'proper_read_replaced':1,
                    'read_with_masked':1,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    'put_in_place':1.0,
                    'read_on':1.0,
                    'read_block': 1.0,
                    'read_block0': 1.0,
                    }
            self.q_types_noblock = {
                    'read_blanked':1,
                    'proper_read_replaced':1,
                    'read_with_masked':1.0,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    'put_in_place':1.0,
                    'read_block': 1.0,
                    'read_block0': 1.0,
                    }
        elif mode == 'IAM_valid':
            self.q_types = {
                    'read_block0': 1.0
                    }
            self.q_types_noblock = {
                    'read_block0': 1.0
                    }
        elif mode == 'IAM_para':
            self.q_types = {
                    'read_block': 1.0
                    }
            self.q_types_noblock = {
                    'read_block': 1.0
                    }
        elif mode == 'easy_bart':
            self.q_types = {
                    'text_infilling_read':1,
                    'proper_read_replaced':1,
                    'read_line':1,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    'read_on':1.0,
                    }
            self.q_types_noblock = {
                    'text_infilling_read':1,
                    'proper_read_replaced':1,
                    'read_line':1,
                    'read_highlighted':1,
                    'masked_lm':1.0,
                    }
        elif mode == 'pretrain':
            self.q_types = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'read_on':0.5,
                    'read_backwards':0.5,
                    }
            self.q_types_noblock = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'read_line':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    }
        elif mode == 'pretrain2':
            self.q_types = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'read_on':0.5,
                    'read_backwards':0.5,
                    }
            self.q_types_noblock = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    }
        elif mode == 'pretrain_word':
            self.q_types = {
                    'read_blanked':1,
                    'proper_read_replaced':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    'read_on':1.0,
                    }
            self.q_types_noblock = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    }
        elif mode == 'pretrain_bart':
            self.q_types = {
                    'text_infilling_read':1,
                    'proper_read_replaced':1,
                    'read_highlighted':1,
                    'read_on':1,
                    }
            self.q_types_noblock = {
                    'text_infilling_read':1,
                    'proper_read_replaced':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    }
        elif mode == 'pretrain_nomask':
            self.q_types = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'read_highlighted':1,
                    'read_on':0.5,
                    'read_backwards':0.5,
                    }
            self.q_types_noblock = {
                    'read_blanked':1,
                    'read_replaced':1,
                    'highlight_text':1.0,
                    'read_highlighted':1,
                    }
        elif mode == 'hard':
            self.q_types = {
                    'read_blanked':0.5,
                    'read_replaced':0.5,
                    'read_with_masked':1.0,
                    'read_line':0.5,
                    'highlight_text':1.0,
                    'read_highlighted':0.5,
                    'masked_lm':4.0,
                    'put_in_place':1.0,
                    'read_on':0.4,
                    'read_backwards':0.5,
                    'highlight_block':1.0}
            self.q_types_noblock = {
                    'read_blanked':0.5,
                    'read_replaced':0.5,
                    'read_with_masked':1.0,
                    'read_line':0.5,
                    'highlight_text':1.0,
                    'read_highlighted':0.5,
                    'masked_lm':4.0,
                    'put_in_place':1.0}
        elif mode == 'hard_word':
            self.q_types = {
                    'read_blanked':0.5,
                    'proper_read_replaced':0.5,
                    'read_line':0.1,
                    'highlight_text': 0.1,
                    'read_highlighted':0.1,
                    'masked_lm':4.0,
                    'long_mlm':16.0,
                    'put_in_place':1.0,
                    'read_on':0.9,
                    'highlight_block':1.0}
            self.q_types_noblock = {
                    'read_blanked':0.5,
                    'proper_read_replaced':0.5,
                    'read_line':0.1,
                    'highlight_text': 0.1,
                    'read_highlighted':0.1,
                    'masked_lm':4.0,
                    'long_mlm':12.0,
                    'put_in_place':1.0}
        elif mode == 'streamlined':
            self.q_types = {
                    'masked_lm':0.08,
                    'long_mlm':0.82,
                    'put_in_place':0.08,
                    'read_on no_highlight':0.02,
                    }
            self.q_types_noblock = {
                    'masked_lm':0.08,
                    'long_mlm':0.82,
                    'put_in_place':0.08,
                    'read_on no_highlight':0.02
                    }
        elif mode == 'mk_only':
            self.q_types = {'masked_lm':4.0}
            self.q_types_noblock = {'masked_lm':4.0}
        elif mode == 'test':
            self.q_types = {'long_mlm':4.0}
            self.q_types_noblock = {'long_mlm':4.0}
        else:
            raise ValueError('Unknown para qa mode: {}'.format(mode))




    #return the list of question-answer (task-respone) pairs
    def makeQuestions(self,ocr,image_h,image_w,s,use_blocks=True):
        wordmap = makeWordmap(ocr)
        linemap = makeLinemap(ocr)
        if len(wordmap)==0 and len(linemap)==0:
            return [],np.array([])
        if use_blocks:
            q_types = random.choices(list(self.q_types.keys()),self.q_types.values(),k=self.questions*50)
        else:
            q_types = random.choices(list(self.q_types_noblock.keys()),self.q_types_noblock.values(),k=self.questions*50)

        qa=[]
        for question_type in q_types:
            if question_type == 'read_blanked' or question_type == 'read_replaced' or question_type == 'read_with_masked' or question_type == 'proper_read_replaced':
                #Get Blanked = read_blanked
                #Re-read Replaced = proper_read_replaced

                #0. Return blanked words (no highlight) and draw where it is "the [blank] chased the cat" > "dog"
                # . Return blanked words (with highlight) and draw where it is "the [blank] chased the cat" > "dog"
                #1. Return replaced word (no highlight) and draw where it is "the industrial chased the cat" > "dog"
                # . Return replaced word (with highlight) and draw where it is "the industrial chased the cat" > "dog"
                #5. Read highlighted section filling in masked word

                #"proper" means the response includes all of the input text, not just the replaced word (similar to the Text Infilling task)

                read_with_masked = question_type == 'read_with_masked'
                use_highlight = (random.random()<0.5 and self.use_highlight) or read_with_masked
                blank = question_type == 'read_blanked'
                proper = question_type.startswith('proper')
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
                max_prompt_len = self.max_qa_len_in if not read_with_masked else self.max_qa_len_out

                proper_response=response
                while len(prompt)<max_prompt_len-1: #-1 to account for adding a space
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
                            if len(text)+len(prompt)+1<=max_prompt_len:
                                last_start_block = start_word_map[0]
                                last_start_line = start_word_map[0:3]

                                if changed_line:
                                    prompt = text+'\\'+prompt
                                    proper_response = text+'\\'+proper_response
                                else:
                                    prompt = text+' '+prompt
                                    proper_response = text+' '+proper_response
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
                            if len(text)+len(prompt)+1<=max_prompt_len:
                                last_end_block = end_word_map[0]
                                last_end_line = end_word_map[0:3]

                                if changed_line:
                                    prompt += '\\'+text
                                    proper_response += '\\'+text
                                else:
                                    prompt += ' '+text
                                    proper_response += ' '+text
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
                if proper:
                    question = 'proper_'+question
                    response = proper_response
                qa.append([question+prompt,response,[wordmap[i] for i in words_in_prompt+[blank_word_idx]],inmask,outmask,maskmask])

            elif question_type == 'read_line':
                #Read Line Above
                #Read Line Below
                #  This both have variations where it respects or doesn't respect blocks

                # . Read line above (no highlight)and draw where it is. based on position, not just para/block
                # . Read line above (with highlight) and draw where it is. based on position, not just para/block
                # . Read line below (no highlight)and draw where it is. based on position, not just para/block
                # . Read line below (with highlight) and draw where it is. based on position, not just para/block
                # . Read line above (no highlight)and draw where it is. based on para/block
                # . Read line above (with highlight) and draw where it is. based on para/block
                # . Read line below (no highlight)and draw where it is. based on  para/block
                # . Read line below (with highlight) and draw where it is. based on  para/block
                use_highlight = random.random()<0.5 and self.use_highlight
                above = random.random()<0.5 #or below
                beyond_block = random.random()<0.5 if use_blocks else True #whether it should look beyond the block of text or not

                if len(linemap)==0:
                    continue #no lines

                line_idx = random.randrange(len(linemap))
                line_map = linemap[line_idx]
                if above:
                    if line_idx==0:
                        next_line_map = getLineAboveBlock(ocr,linemap,line_idx) if beyond_block else None
                    else:
                        next_line_idx = line_idx-1
                        next_line_map = linemap[next_line_idx]
                        if line_map[0] != next_line_map[0]: #if no longer same block
                            next_line_map = getLineAboveBlock(ocr,linemap,line_idx) if beyond_block else None

                else:
                    if line_idx==len(linemap)-1:
                        next_line_map = getLineAboveBlock(ocr,linemap,line_idx,below=True) if beyond_block else None
                    else:
                        next_line_idx = line_idx+1
                        next_line_map = linemap[next_line_idx]
                        if line_map[0:2] != next_line_map[0:2]: #if no longer same block
                            next_line_map = getLineAboveBlock(ocr,linemap,line_idx,below=True) if beyond_block else None
                
                prompt = ocr[line_map[0]]['paragraphs'][line_map[1]]['lines'][line_map[2]]['text']
                if next_line_map is not None:
                    response = ocr[next_line_map[0]]['paragraphs'][next_line_map[1]]['lines'][next_line_map[2]]['text']
                    outmask = [next_line_map+(i,) for i in range(len(ocr[next_line_map[0]]['paragraphs'][next_line_map[1]]['lines'][next_line_map[2]]['words']))]
                else:
                    response= '№'
                    outmask = []

                #if len(prompt)>self.max_qa_len_in:
                #    prompt = prompt[:self.max_qa_len_in-2] +'>>'
                #if len(response)>self.max_qa_len_out:
                #    response = response[:self.max_qa_len_out]
                #elif len(response)+1 < self.max_qa_len_out and response!='№':
                response = response+self.end_token


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


            elif question_type == 'highlight_text' and random.random()<0.5:
                #Height Text (one variant)
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
                    while word_idx<=end_word_idx and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len_in:
                        prompt+= ' '+line['words'][word_idx]['text']
                        word_idx+=1
                else:
                    #build prompt backwards
                    prompt = line['words'][end_word_idx]['text']
                    word_idx =end_word_idx-1
                    while word_idx>=0 and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len_in:
                        prompt= line['words'][word_idx]['text']+' '+prompt
                        word_idx-=1
                response = ''
                
                outmask = [line_map+(None,)]
                inmask=[]
                question = '0;~'
                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])

            elif question_type == 'highlight_text' or question_type == 'read_highlighted':
                #Height Text (other variant)
                #Read Highlighted

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
                    if len(response)>self.max_qa_len_out:
                        response = response[:self.max_qa_len_out]
                    elif len(response)+1 <= self.max_qa_len_out:
                        response += self.end_token
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
                        while word_idx<=end_word_idx and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len_in:
                            prompt+= ' '+line['words'][word_idx]['text']
                            word_idxs.append(word_idx)
                            word_idx+=1
                    else:
                        #build prompt backwards
                        end_word_idx = start_word_idx-use_words
                        prompt = line['words'][end_word_idx]['text']
                        word_idxs = [end_word_idx]

                        word_idx =end_word_idx-1
                        while word_idx>=0 and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len_in:
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

            elif question_type == 'masked_lm':
                #Word Infilling
                # A single word from the image is masked and the model must predict it
                # (HARD!)
                blank_word_idx = random.randrange(len(wordmap))
                maskmask = [wordmap[blank_word_idx]]
                blank_word_map = wordmap[blank_word_idx]
                response=ocr[blank_word_map[0]]['paragraphs'][blank_word_map[1]]['lines'][blank_word_map[2]]['words'][blank_word_map[3]]['text']
                question = 'mk>'
                qa.append([question,response,maskmask,maskmask,None,maskmask])

            elif question_type == 'put_in_place' and len(wordmap)>1:
                #Place Word
                #7. given a word and several masked spots, hightlight which masked spot this belongs in
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
                if len(allmaps)==0:
                    continue

                #We may have blanked other instances of the same word. We'll just say they are right answers too.
                for this_word_map in allmaps:
                    text = ocr[this_word_map[0]]['paragraphs'][this_word_map[1]]['lines'][this_word_map[2]]['words'][this_word_map[3]]['text']
                    if text.lower()==prompt.lower():
                        it_word_maps.append(this_word_map)

                allmaps.append(it_word_maps[0]) #add original 'it'
                response=''
                
                question = 'mm~'
                qa.append([question+prompt,response,allmaps,None,it_word_maps,allmaps])

            elif 'read_on' in question_type or question_type=='read_backwards':
                #Read On

                #8. Read from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read from prompt (with highlight) including new lines (stops at block end) and draw where you read
                #9. Read backwards from prompt (no highlight) including new lines (stops at block end) and draw where you read
                # . Read backwards from prompt (with highlight) including new lines (stops at block end) and draw where you read
                forward = 'read_on' in question_type #only character based models should learn to read backwards (not Dessurt)
                use_highlight = random.random()<0.5 and self.use_highlight and 'no_highlight' not in question_type
                if use_highlight:
                    min_read_start = self.min_read_start_with_mask
                else:
                    min_read_start = self.min_read_start_no_mask
                start_word_idx = random.randrange(len(wordmap))
                start_word = wordmap[start_word_idx]
                start_block = start_word[0]

                #first, get length of paragraph
                last_line_id = start_word[0:3]
                prompt = ocr[start_word[0]]['paragraphs'][start_word[1]]['lines'][start_word[2]]['words'][start_word[3]]['text']
                next_word_idx = start_word_idx+(1 if forward else -1)
                if next_word_idx>=len(wordmap) or next_word_idx<0:
                    next_word=(None,None,None,None)
                else:
                    next_word = wordmap[next_word_idx]
                    next_text = ocr[next_word[0]]['paragraphs'][next_word[1]]['lines'][next_word[2]]['words'][next_word[3]]['text']
                    if not forward:
                        next_text = next_text[::-1]
                    next_line_id = next_word[0:3]
                while next_word[0]==start_block and len(prompt)+1+len(next_text)<self.max_qa_len_in:
                    if last_line_id==next_line_id:
                        prompt+=' '+next_text
                    else:
                        prompt+='\\'+next_text
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

                #length of paragraph
                para_len = min(self.max_qa_len_in,len(prompt))
                
                if para_len>min_read_start:
                    goal_prompt_len = random.randrange(min_read_start,para_len)
                else:
                    goal_prompt_len = min_read_start
                
                words_in_prompt=[start_word_idx]
                start_word = wordmap[start_word_idx]
                prompt = ocr[start_word[0]]['paragraphs'][start_word[1]]['lines'][start_word[2]]['words'][start_word[3]]['text']
                if not forward:
                    prompt=prompt[::-1]
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
                while next_word[0]==start_block and len(prompt)+1+len(next_text)<self.max_qa_len_in:
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
                if len(prompt)>self.max_qa_len_in:
                    prompt = prompt[-self.max_qa_len_in:]

                if next_word[0]!=start_block:
                    response=self.end_token
                    words_in_response=[]
                else:
                    goal_response_len = self.max_qa_len_out
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
                    while len(response)+1+len(next_text)<self.max_qa_len_out and next_word[0]==start_block:
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
                    if len(response)>self.max_qa_len_out:
                        response = response[:self.max_qa_len_out]
                    if next_word[0]!=start_block and len(response)<self.max_qa_len_out:
                        response+=self.end_token#if we can, add an end-of-block sign
                if use_highlight:
                    question = 'r0~' if forward else 'b0~'
                    inmask =  [wordmap[i] for i in words_in_prompt]
                else:
                    question = 're~' if forward else 'bk~'
                    inmask = []

                outmask = [wordmap[i] for i in words_in_response]

                ####getting the right tokens
                prompt_tokens = self.tokenizer(prompt,return_tensors="pt")['input_ids']
                tok_len = prompt_tokens.shape[1] +1#for task token
                if tok_len>self.max_q_tokens:
                    over = tok_len-self.max_q_tokens
                    prompt_tokens = prompt_tokens[0,:-(over+1)] #+1 becuase trimming the end (SEP) token doesn't count
                    prompt = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(prompt_tokens,skip_special_tokens=True))
                #######

                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])


            elif question_type == 'highlight_block':
                #10. draw the block this is in
                use_highlight = random.random()<0.5 and self.use_highlight
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
                    while word_idx<=end_word_idx and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len_in:
                        prompt+= ' '+line['words'][word_idx]['text']
                        word_idxs.append(word_idx)
                        word_idx+=1
                else:
                    #build prompt backwards
                    prompt = line['words'][end_word_idx]['text']
                    word_idxs = [end_word_idx]
                    word_idx =end_word_idx-1
                    while word_idx>=0 and len(prompt)+len(line['words'][word_idx]['text'])<self.max_qa_len_in:
                        prompt= line['words'][word_idx]['text']+' '+prompt
                        word_idxs.append(word_idx)
                        word_idx-=1
                response = ''
                
                outmask = [(block_idx,None,None,None)]
                if use_highlight:
                    inmask=[(block_idx,0,line_idx,i) for i in word_idxs]
                    question = '00~'
                else:
                    inmask=[]
                    question = '0p~'
                qa.append([question+prompt,response,inmask+outmask,inmask,outmask,None])


            elif question_type == 'echo':
                words = []
                for b,p,l,w in wordmap:
                    words.append(ocr[b]['paragraphs'][p]['lines'][l]['words'][w]['text'])
                response = ' '.join(words)
                qa.append(['>',response,[],None,None,None])
            elif question_type == 'echo2':
                words = []
                firstb=None
                firstl=None
                for b,p,l,w in wordmap:
                    if firstb is None or (b==firstb and l==firstl):
                        firstb = b
                        firstl = l
                        continue
                    words.append(ocr[b]['paragraphs'][p]['lines'][l]['words'][w]['text'])
                response = ' '.join(words) + self.end_token
                qa.append(['>',response,[],None,None,None])


            elif question_type == 'text_infilling_read':
                block_idx = random.randrange(len(ocr))
                words = []
                allwords = []
                
                response=''
                for p,para in enumerate(ocr[block_idx]['paragraphs']):
                    for l,line in enumerate(para['lines']):
                        for w,word in enumerate(line['words']):
                            words.append((word,p,l,w))
                            allwords.append((block_idx,p,l,w))
                            response+=word['text']+' '
                        response=response[:-1]+'\\' #change space to newline
                response=response[:-1]+self.end_token #remove newline
                
                to_mask=[]
                num_spans=random.randrange(1,max(len(words)//8,2))
                for i in range(num_spans):
                    num = np.random.poisson(3)
                    if num>= len(words):
                        if len(words)>1:
                            num=1
                        else:
                            continue
                    loc = random.randrange(0,len(words)-num)
                    to_mask.append((loc,num))


                to_mask.sort(key=lambda a:a[0],reverse=True)
                new_words = words
                for loc,num in to_mask:
                    new_words = new_words[:loc]+[None]+new_words[loc+num:]

                prompt = ''
                last_paraline = None
                last_blank=False
                for word_info in new_words:
                    if word_info is not None:
                        last_blank = False
                        word,p,l,w=word_info
                        text = word['text']
                        if last_paraline is None:
                            space=''
                        elif last_paraline==(p,l):
                            space=' '
                        else:
                            space='\\'
                        last_paraline=(p,l)
                    else:
                        if last_blank:
                            continue #merge blank
                        text='<mask> '
                        space = ' '
                        last_paraline = None

                    prompt+=space+text

                if random.random()<0.5: 
                    question = 'infillread~'
                    inmask =  []
                else:
                    question = 'infillread0~'
                    inmask = [(block_idx,None,None,None)]
                qa.append([question+prompt,response,allwords,inmask,None,None])


            elif question_type == 'read_block' or question_type == 'read_block0':
                #Task used by full-page recognition
                if len(ocr)==1:
                    response = []
                    inmask = [] #for text lines
                    for p,paragraph in enumerate(ocr[0]['paragraphs']):
                        for l,line in enumerate(paragraph['lines']):
                            response.append(line['text'])
                            inmask.append((0,p,l,None))

                    response = '\\'.join(response)
                    
                    all_lines = inmask
                    if question_type == 'read_block':
                        inmask = None
                    qa.append([question_type+'>',response,all_lines,inmask,None,None])

            elif question_type == 'long_mlm':
                #Text Infilling (BART based)
                # a random block is selected
                # multiple spans of block are masked
                # The model must output the entire block with blanks filled in
                words,to_remove,target_string,block = makeMLMInstance(ocr)
                if words is None:
                    continue

                inmask=[]
                for para in block['paragraphs']:
                    inmask += [line['id'] for line in para['lines']]
                all_words=[w[0]['id'] for w in words if w is not None]

                to_remove = [bb+(None,) for bb in to_remove] #ugh, special "marker" that this is a bb not id into ocr

                qa.append(['mlm>',target_string,all_words,inmask,None,to_remove])


            #else:
            #    raise NotImplementedError('Unknown question type: {}'.format(question_type))

            if len(qa)>=self.questions*10:
                break


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
                        #Do i need to do this?
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

#Get the bounding boxes for mixed word/line/para ids
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
                if line['words'] is not None:
                    for w in range(len(line['words'])):
                        wordmap.append((b,p,l,w))
                        line['words'][w]['id']=(b,p,l,w)
    return wordmap
def makeLinemap(ocr):
    linemap=[]
    for b,block in enumerate(ocr):
        for p,para in enumerate(block['paragraphs']):
            for l in range(len(para['lines'])):
                linemap.append((b,p,l))
                para['lines'][l]['id'] = (b,p,l,None)
    return linemap

def getAllBBs(ocr,t_ids,s,expand=False):
    bbs=[]
    if t_ids is not None:
        for t_id in t_ids:
            if len(t_id)==4:
                b,p,l,w = t_id #block, paragraph, line ,word
                inst = ocr[b]
                if p is not None:
                    inst = inst['paragraphs'][p]
                    if l is not None:
                        inst = inst['lines'][l]
                        if w is not None:
                            inst = inst['words'][w]
                
                box = inst['box']
                lX,tY,rX,bY = box
            else:
                lX,tY,rX,bY,_ = t_id
                assert _ is None
            if expand:
                lX-=2
                tY-=2
                rX+=2
                bY+=2
            bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
                   s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/2.0, s*bY]  #we add these for conveince to crop BBs within window
            bbs.append(bb)
    return bbs

#This finds the line above or below the block the line index belongs to
def getLineAboveBlock(ocr,linemap,line_idx,below=False):
    line_map = linemap[line_idx]
    l,t,r,b = ocr[line_map[0]]['box']

    closest_block=None
    cb_dist=99999999
    for bid,block in enumerate(ocr):
        if bid == line_map[0]:
            continue
        l2,t2,r2,b2 = block['box']

        #Do the blocks align horizontally?
        covered_horz = max(min(r,r2) - max(l,l2),0)
        covered_horz/=min(max(r2-l2,1),max(r-l,1))
        
        #verticle distance
        if below:
            dist = t2-b
        else:
            dist = t-b2

        if covered_horz>0.7 and dist>=-2 and dist<cb_dist:
            cb_dist = dist
            closest_block=bid
    if closest_block is None:
        return None

    #Now get the line in the found block that is closest
    lowest_line=0
    assert len(ocr[closest_block]['paragraphs'])==1
    ll_y=ocr[closest_block]['paragraphs'][0]['lines'][0]['box'][1 if below else 3]
    for l in range(1,len(ocr[closest_block]['paragraphs'][0]['lines'])):
        y = ocr[closest_block]['paragraphs'][0]['lines'][l]['box'][1 if below else 3]
        if (below and y<ll_y) or (not below and y>ll_y):
            lowest_line=l
            ll_y=y

    return (closest_block,0,lowest_line)



