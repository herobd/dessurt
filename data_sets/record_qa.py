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
from data_sets.qa import QADataset,collate
from data_sets.wiki_text import getWikiArticle

import utils.img_f as img_f



        

class RecordQA(QADataset):
    """
    Parent class for processing tabular records.
    Only used by CensusQA
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(RecordQA, self).__init__(dirPath,split,config,images)
        if self.train:
            self.q_types =         ['next-name','get-field','np']
            self.q_type_weights =  [1,          1,          0.05]
            self.q_types_single =         ['get-field','np']
            self.q_type_single_weights =  [1,          0.02]
        else:
            #these are what we'll use to actually score
            #(not actually looked at as it isn't sampling)
            self.q_types =         ['get-field']

        self.q_types_for_np = ['next-name','get-field']

       
        self.np_token = '№'
        self.blank_token = 'ø'
        self.end_token='‡' 

        self.punc_regex = re.compile('[%s]' % re.escape(string.punctuation))

    def makeQuestions(self,s,entries,households=None):

        all_ids=set()
        for entry in entries:
            for id_n in self.id_fields:
                if entry[id_n] is not None:
                    all_ids.add(self.punc_regex.sub('',entry[id_n].lower()))

        q_a_pairs = []
        if self.train:
            if len(entries)>1 or self.pretrain:
                q_types = random.choices(self.q_types,self.q_type_weights,k=self.questions*50)
            elif len(entries)>0:
                q_types = random.choices(self.q_types_single,self.q_type_single_weights,k=self.questions*50)
            else:
                q_types = ['np']*(self.questions*20)
        elif self.pretrain:
           q_types = [('whole-doc',None,None),
                      #('all-name',None,None),
                      #('all-age',None,None),
                      ] 
        else:
            q_types = []
            for entry in entries:
                for field_name in self.all_fields-set([self.main_id_field]):
                    q_types.append(('get-field',entry,field_name))

        

        for q_type in q_types:
            if not self.train:
                q_type,instance,switch = q_type
            else:
                switch=False
            if q_type == 'all-name':
                name_id = random.choice(self.next_name_ids)
                names=[]
                for entry in entries:
                    names.append(entry[name_id] if entry[name_id] is not None else self.blank_token)
                response = '\\'.join(names) + self.end_token if len(names)>0 else self.blank_token
                question='all-{}~'.format(name_id)
                self.qaAdd(q_a_pairs,question,response)
            elif q_type == 'all-age':
                ages=[]
                for entry in entries:
                    ages.append(entry['age'] if entry['age'] is not None else self.blank_token)
                response = '\\'.join(ages) + self.end_token if len(ages)>0 else self.blank_token
                question='all-age~'
                self.qaAdd(q_a_pairs,question,response)
            elif q_type == 'whole-doc':
                lines=[]
                for entry in entries:
                    line=[]
                    for field_id in self.ordered_ids:
                        line.append(entry[field_id] if entry[field_id]  is not None else self.blank_token)
                    lines.append('|'.join(line))
                response = '\\'.join(lines) + self.end_token if len(lines)>0 else self.blank_token
                question='record~'
                self.qaAdd(q_a_pairs,question,response)
            elif q_type == 'next-name':
                down = random.random()<0.5
                name_id = random.choice(self.next_name_ids)
                for i in range(20):
                    start = random.randrange(len(entries)-1)
                    if down:
                        question='d:{}~'.format(name_id)
                        step=start+1
                    else:
                        question='u:{}~'.format(name_id)
                        step=start
                        start=start+1
                    prompt = entries[start][name_id]
                    if prompt is not None and len(prompt)>0:
                        break

                response = entries[step][name_id]
                prompt = self.getFrontText(prompt,query=True)
                if response is not None:
                    response = self.getFrontText(response)
                else:
                    response = self.blank_token
                
                self.qaAdd(q_a_pairs,question+prompt,response)


            elif q_type == 'get-field':
                if self.train:
                    entry = random.choice(entries)
                    id_field = random.choice(tuple(self.id_fields))
                    target_field = random.choice(tuple(self.all_fields-set([id_field])))
                else:
                    entry = instance
                    id_field = self.main_id_field
                    target_field = switch


                id_prompt = entry[id_field]
                if id_prompt is None:
                    continue
                value = entry[target_field]

                question = 'f{}:{}~{}'.format(id_field[:2],target_field,id_prompt)
                question = self.getFrontText(question,query=True)
                if value is not None:
                    value = self.getFrontText(value)
                else:
                    value = self.blank_token
                self.qaAdd(q_a_pairs,question,value)


            elif q_type == 'np':
                sub_type = random.choice(self.q_types_for_np)

                match = True
                while match:
                    #check if this is indeed novel text for the given document
                    prompt_text = self.sampleText()
                    match = False
                    prompt_text_no_punc = self.punc_regex.sub('',prompt_text.lower())
                    for text in all_ids:
                        if prompt_text_no_punc in text:
                            match=True
                            break


                if sub_type == 'next-name':
                    if random.random()<0.5:
                        question='dn~'
                    else:
                        question='up~'
                elif sub_type == 'get-field':
                    id_field = random.choice(tuple(self.id_fields))
                    target_field = random.choice(tuple(self.all_fields-set([id_field])))

                    question = 'f:{}~{}'.format(target_field,prompt_text)
                    question = self.getFrontText(question,query=True)
                self.qaAdd(q_a_pairs,question+prompt_text,self.np_token) #Normally this has a blank out mask GT, but during training, we don't want this as it never is encrouraged to predict

            else:
                assert False
            
            if len(q_a_pairs)>10*self.questions and self.train:
                break #we have enough
        
        return q_a_pairs

    def selectPartText(self,text,length=None,ret_start=False):
        #Randomly select part of the text less than or equal to max_qa_len,
        #breaking on spaces (and newlines)
        if length is None:
            length = self.max_qa_len_in
        if len(text)>length:
            start = random.randrange(len(text)-length)
            end=start+length
            if start!=0 and text[start-1]!=' ' and text[start-1]!='\\':
                #search for nearest space
                before_start = start-1
                while before_start>0 and text[before_start-1]!=' ' and text[before_start-1]!='\\':
                    before_start -= 1
                after_start = start+1
                while after_start<end and text[after_start-1]!=' ' and text[after_start-1]!='\\':
                    after_start += 1
                if after_start==end:
                    after_start=None
            else:
                before_start=after_start=start

            if end!=len(text) and text[end]!=' ' and text[end]!='\\':
                #search for nearest space
                before_end = end-1
                while before_end>start and text[before_end]!=' ' and text[before_end]!='\\':
                    before_end -= 1
                if before_end==start:
                    before_end=None
                after_end = end+1
                while after_end<len(text) and text[after_end]!=' ' and text[after_end]!='\\':
                    after_end += 1
            else:
                after_end=before_end=end
            #get best combination
            len_b_b = before_end-before_start if before_end is not None else -1
            len_a_b = before_end-after_start if before_end is not None and after_start is not None else -1
            len_a_a = after_end-after_start if after_start is not None else -1

            best_len = max(len_b_b if len_b_b<=length else -1, len_a_b if len_a_b<=length else -1, len_a_a if len_a_a<=length else -1)

            if best_len==-1:
                ret = text[start:start+length] #failed to break on words
            else:
                if best_len==len_b_b:
                    ret = text[before_start:before_end]
                    start = before_start
                elif best_len==len_a_b:
                    ret = text[after_start:before_end]
                    start = after_start
                else:
                    ret = text[after_start:after_end]
                    start = after_start



        else:
            ret = text
            start = 0

        if ret_start:
            return ret, start
        else:
            return ret

    def getFrontText(self,text,list_split=False,term=None,query=False):
        #get the front part of the text, breaking on words
        if query:
            length = self.max_qa_len_in
        else:
            length = self.max_qa_len_out
        if len(text)>length:
            end = length
            while end>1 and text[end]!=' ' and text[end]!='\\' and (not list_split or (text[end]!='|' and text[end-1]!='|')):
                end-=1
            if end==1:
                #couldn't break
                return text[:length]
            else:
                return text[:end]
        else:
            if term is not None and len(text)+len(term)<=length:
                text+=term
            return text

    def convertBB(self,s,box):
        lX,tY,rX,bY = box
        #should return a list of [lX, tY, rX, tY, rX, bY, lX, bY, 
        bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
               s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/2.0, s*bY]  #we add these for conveince to crop BBs within window
        #??
        return bb

    def sampleText(self,length=None):
        if length is None:
            length = self.max_qa_len_in

        para = random.choice(getWikiArticle()) #select random paragraph

        para=re.sub(r' +',r' ',para)
        para=re.sub(r' ?\n ?',r'\n ',para)
        para = para.strip()
        words = para.split(' ')
        words = [w for w in words if len(w)>0]

        if len(words)==0:
            return self.sampleText(length)

        #choose starting word and build from their
        start=idx = random.randrange(len(words))
        num_words = random.randrange(len(words)-idx)
        
        last_text=None
        text = words[idx]
        while len(text)<self.max_qa_len_in and idx<start+num_words:
            last_text = text
            text += ' '+words[idx]
            idx+=1
            
        assert text!=''
        if len(text)>self.max_qa_len_in and last_text is not None:
            text = last_text
        elif len(text)>self.max_qa_len_in:
            text = self.selectPartText(text)
        if len(text)==0:
            return self.sampleText()
        return text
 
