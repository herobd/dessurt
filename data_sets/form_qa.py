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

import utils.img_f as img_f



class FormQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FormQA, self).__init__(dirPath,split,config,images)
        self.punc_regex = re.compile('[%s]' % re.escape(string.punctuation))

        self.only_types=None

        self.split_to_lines = config['split_to_lines']
        self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0.15
        self.do_words = config['do_words']
        self.char_qs = config['char_qs'] if 'char_qs' in config else False

    #entities =[  {class, box (whole), text, lines:{box,, bbid, text, ambiguous} ]
    #entity_adj =[(upper,lower)] either can be None
    #tables = obj. col/row_headers = [entity_id], cells = [[entity_id]]
    #
    def makeQuestions(self,entities,entity_link,tables):
        """
        Generates N questions from given docuemnt information:
         - entities: a list of Entity objects
         - entity_link
         """
        all_q_a=[] #question-answers go here

        questions_gs=set()
        answers_gs=set()
        headers_gs=set()
        others_gs=set()
        all_trans={}
        entity_count = len(entities)

        q_a_pairs = []
        all_qs = set() #just doulbe check we haven't select the same question twice
        q_types = random.choices(self.q_types,self.q_type_weights,k=self.questions*10)
        for q_type in q_types:
            if q_type=='class':
                e_i = random.randrange(len(entities))
                cls = entity['class']
                class_answer = '[ '+self.index_class_map[cls]+' ]'
                line = random.choice(entity['lines'])
                text=line['text']
                if self.max_qa_len is not None and len(text)>self.max_qa_len:
                    text = text[:self.max_qa_len]

                inmask=[]
                if random.random() <0.5 or line['ambiguous']:
                    #use query mask
                    question = 'c$~'
                    inmask.append(self.convertBB(line['box']))
                else:
                    question = 'cs~'
                self.qaAdd(q_a_pairs,question+text,class_answer,[bbid],inmask,[]) #This can be ambigous, although generally the same text has the same class
            
            elif q_type=='down-pair' or q_type=='up-pair':
                down = q_type=='down-pair'

                for i in range(min(10,len(entity_link))):
                    head_id, tail_id = random.choice(entity_link)
                    if down:
                        prompt_id = head_id
                        response_id = tail_id
                    else:
                        prompt_id = tail_id
                        response_id = head_id
                    if prompt_id is not None:
                        break
                if prompt_id is None:
                    continue

                #sample a span of text from prompt entity
                prompt_text = entities[prompt_id]['text']
                if len(prompt_text)>self.max_qa_len:
                    if random.random()<0.25: #take from end or random:
                        if down:
                            prompt_text = prompt_text[-self.max_qa_len:]
                        else:
                            prompt_text = prompt_text[:self.max_qa_len]
                    else:
                        #random
                        prompt_text = self.selectPartText(prompt_text)

                #get end of response text
                if response_id is not None:
                    response_text = entities[response_id]['text']
                    if not down:
                        response_text = response_text[::-1]
                    if len(response_text)>self.max_qa_len:
                        response_text = response_text[:self.max_qa_len]
                    elif len(response_text)+1<=self.max_qa_len:
                        response_text += self.end_token
                else:
                    response_text = self.blank_token

                inmask = []
                ambiguous = len(entities[prompt_id]['lines'])==1 and entities[prompt_id]['lines'][0]['ambiguous']
                if random.random()<0.5 or ambiguous:#should we use query mask
                    if entities[prompt_id]['class']=='question':
                        question = 'd0~' if down else 'v0~'
                    else:
                        assert entities[prompt_id]['class']=='header'
                        question = 'h0~' if down else 'q0~'

                    if random.random()<0.5: #should the mask be lines or whole entity
                        for line in entities[prompt_id]['lines']:
                            inmask.append(self.convertBB(line['box']))
                    else:
                        inmask.append(self.convertBB(entities[prompt_id]['box']))
                else:
                    if entities[prompt_id]['class']=='question':
                        question = 'l~' if down else 'v~'
                    else:
                        assert entities[prompt_id]['class']=='header'
                        question = 'hd~' if down else 'qu~'

                outmask = []
                bbids = []
                #outmask is lines, as we'd like the higher resolution
                for line in entities[response_id]['lines']:
                    outmask.append(self.convertBB(line['box']))
                    bbids.append(line['bbid'])

                
                bbids += [l['bbid'] for l in entities[prompt_id]['lines']]
                self.qaAdd(q_a_pairs,question+prompt_text,response_text,bbids,immask,outmask)

            elif q_type=='np':
                sub_type = random.choice(self.q_types_minus_np)

                match = True
                while match:
                    #check if this is indeed novel text for the given document
                    prompt_text = self.sample_text()
                    match = False
                    prompt_text_no_punc = self.punc_regex.sub('',prompt_text.lower())
                    for entity in entities:
                        if prompt_text_no_punc in  self.punc_regex.sub('',entity.text.lower()):
                            match=True
                            break

                if sub_type == 'class':
                    question = 'cs~'
                elif sub_type == 'down-pair':
                    question = 'l~' if random.random()<0.5 else 'hd~'
                elif sub_type == 'up-pair':
                    question = 'v~' if random.random()<0.5 else 'qu~'
                self.qaAdd(q_a_pairs,question+prompt_text,self.np_token,[],[],[])

            elif q_type=='read':
                TODO #line above, line below? ^^,^0,vv,v0
                #finish reading entity

            elif q_type=='cell':
                table = random.choice(tables)
                r,r_header = random.randrange(enumerate(table.row_headers))
                c,c_header = random.randrange(enumerate(table.col_headers))

                r_h_text = r_header['text']
                c_h_text = c_header['text']

                r_h_text = self.selectPartText(r_h_text,-1+self.max_qa_len//2)
                c_h_text = self.selectPartText(c_h_text,-1+self.max_qa_len//2)

                cell = table.cells[r][c]
                cell_text = cell['text']
                if len(cell_text) > 0:
                    cell_text = self.getFrontText(cell_text)
                elif len(cell_text)+1<=self.max_qa_len:
                    cell_text += self.end_token
                outmask = [self.convertBB(line['box']) for line in cell['lines']]
                
                if randon.random()<0.5:
                    question='t'
                    inmask=[]
                else:
                    question='t0'
                    inmask = [self.convertBB(line['box']) for line in r_header['lines']+c_header['lines']]

                ids = [line['bbid'] for line in r_header['lines']+c_header['lines']+cell['lines']]

                self.qaAdd(q_a_pairs,'{}~{}~~{}'.format(question,h2_text,h1_text),ids,cell_text,inmask,outmask)


            elif q_type=='row-header' or q_type=='col-header':
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='row-header':
                    row=True
                    r,header = random.choice(enumerate(table.row_headers))
                    c = random.randrange(len(table.col_headers))
                else:
                    row=False
                    c,header = random.choice(enumerate(table.col_headers))
                    r = random.randrange(len(table.row_headers))
                cell = table.cells[r][c]

                cell_text = self.selectPartText(cell['text'])

                header_text = header['text']
                if len(header_text) > self.max_qa_len:
                    header_text = header_text[:self.max_qa_len]
                elif len(header_text)+1<=self.max_qa_len:
                    header_text += self.end_token

                ids=[]
                inmask=[]
                if random.random()<0.5:
                    if row:
                        question = 'ri~'
                    else:
                        question = 'ci~'
                else:
                    if row:
                        question = 'r*~'
                    else:
                        question = 'c*~'
                    inmask = [self.convertBB(line['box']) for line in cell['lines']]

                ids = [line['bbid'] for line in cell['lines']]

                outmask = []
                for line in header['lines']:
                    outmask.append(self.convertBB(line['box']))
                    ids.append(line['bbid'])

                self.qaAdd(q_a_pairs,question+cell_text,header_text,ids,inmask,outmask)


            elif q_type=='all-row' or q_type=='all-col':
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='row-header':
                    row=True
                    r,header = random.choice(enumerate(table.row_headers))
                    all_cells = table.cells[r]
                else:
                    row=False
                    c,header = random.choice(enumerate(table.col_headers))
                    all_cells = [table.cells[r][c] for r in range(len(table.row_headers))]

                
                outmask = []
                if random.random()<0.5:
                    #just highligh the row/col
                    ids=[self.convertBB(line['bbid']) for line in header['lines']]
                    header_text = self.selectPartText(header['text'])
                    for cell in all_cells:
                        ids += [line['bbid'] for line in cell['lines']]
                        outmask += [self.convertBB(line['box']) for line in cell['lines']]

                    if random.random()<0.5:
                        inmask = [self.convertBB(line['box']) for line in header['lines']]
                        question = '#r~' if row else '#c~'
                    else:
                        inmask = []
                        question = '$r~' if row else '$c~'

                    self.qaAdd(q_a_pairs,question+header_text,'',ids,inmask,outmask)
                else:
                    #step, text-wise, through each entry
                    i = random.randrange(len(all_cells)+1)
                    cell = all_cells[i] if i<len(all_cells) else None
                    if i>0:
                        header = all_cells[i-1]
                        char = '}'
                    else:
                        char = '~'

                    ids=[self.convertBB(line['bbid']) for line in header['lines']]
                    header_text = self.selectPartText(header['text'])
                    if random.random()<0.5:
                        inmask = [self.convertBB(line['box']) for line in header['lines']]
                        question = '%r{}' if row else '%c{}'
                    else:
                        inmask = []
                        question = 'ar{}' if row else 'ac{}' 
                    question = question.format(char)

                    if cell is not None:
                        outmask =  [self.convertBB(line['box']) for line in cell['lines']]
                        ids += [line['bbid'] for line in cell['lines']]
                        cell_text = self.getFrontText(cell['text'],term='|' if i<len(all_cells)-1 else self.end_token)
                    else:
                        outmask = []
                        cell_text = self.end_token

                    self.qaAdd(q_a_pairs,question+header_text,cell_text,ids,inmask,outmask)

                        



            elif q_type=='list-row-headers' or q_type=='list-col-headers':
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='list-row-headers':
                    row=True
                    headers = table.row_headers
                else:
                    row=False
                    headers = table.col_headers

                
                outmask = []
                if random.random()<0.5:
                    #just highligh the headers
                    ids=[]
                    outmask=[]
                    for header in headers:
                        ids+=[self.convertBB(line['bbid']) for line in header['lines']]
                        outmask += [self.convertBB(line['box']) for line in header['lines']]
                    question = 'r*~' if row else 'c*~'

                    self.qaAdd(q_a_pairs,question+str(table_i),'',ids,[],outmask)
                else:
                    #step, text-wise, through each entry
                    i = random.randrange(len(headers)+1)
                    header = headers[i] if i<len(headers) else None
                    if i>0:
                        prev_header = headers[i-1]
                        prev_text = self.selectPartText(prev_header['text'])
                        ids=[self.convertBB(line['bbid']) for line in prev_header['lines']]
                        char = '}'
                    else:
                        char = '~'
                        prev_text = str(table_i)
                        ids=[]

                    if random.random()<0.5 and i>0:
                        inmask = [self.convertBB(line['box']) for line in prev_header['lines']]
                        question = 'r&{}' if row else 'c&{}'
                    else:
                        inmask = []
                        question = 'rh{}' if row else 'ch{}' 
                    question = question.format(char)

                    if header is not None:
                        outmask =  [self.convertBB(line['box']) for line in header['lines']]
                        ids += [line['bbid'] for line in header['lines']]
                        header_text = self.getFrontText(header['text'],term='|' if i<len(headers)-1 else self.end_token)
                    else:
                        outmask = []
                        header_text = self.end_token

                    self.qaAdd(q_a_pairs,question+prev_text,header_text,ids,inmask,outmask)

            elif q_type=='count-tables':
                table_ids=[]
                outmask=[]
                for table in tables:
                    for header in table.row_headers + table.col_headers:
                        for line in entities[header]['lines']:
                            outmask.append(self.convertBB(line['box']))
                            table_ids.append(line['bbid'])

                    for r in range(len(table.row_headers)):
                        for c in range(len(table.col_headers)):
                            cell = table.cells[r][c]
                            for line in cell['lines']:
                                outmask.append(self.convertBB(line['box']))
                                table_ids.append(line['bbid'])
                self.qaAdd(q_a_pairs,'t#>',str(len(tables)),tables_ids,[],outmask)

            elif q_type=='highlight-table':
                table_i = random.randrange(len(tables))
                table = tables[table_i]
                table_ids=[]
                outmask=[]
                for header in table.row_headers + table.col_headers:
                    for line in entities[header]['lines']:
                        outmask.append(self.convertBB(line['box']))
                        table_ids.append(line['bbid'])

                for r in range(len(table.row_headers)):
                    for c in range(len(table.col_headers)):
                        cell = table.cells[r][c]
                        for line in cell['lines']:
                            outmask.append(self.convertBB(line['box']))
                            table_ids.append(line['bbid'])


                self.qaAdd(q_a_pairs,'0t~{}'.format(table_i),'',table_ids,[],outmask)

        return q_a_pairs

    def selectPartText(self,text,length=None):
        #Randomly select part of the text less than or equal to max_qa_len,
        #breaking on spaces (and newlines)
        if length is None: 
            length = self.max_qa_len
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
            #get best combination
            len_b_b = before_end-before_start if before_end is not None else -1
            len_a_b = before_end-after_start if before_end is not None and after_start is not None else -1
            len_a_a = after_end-after_start if after_start is not None else -1

            best_len = max(len_b_b if len_b_b<=self.max_qa_len else -1, len_a_b if len_a_b<=self.max_qa_len else -1, len_a_a if len_a_a<=self.max_qa_len else -1)

            if best_len==-1:
                return text[start:start+length] #failed to break on words
            else:
                if best_len==len_b_b:
                    return text[before_start:before_end]
                elif best_len==len_a_b:
                    return text[after_start:before_end]
                else:
                    return text[after_start:after_end]


            #return text[start:start+length]
            #words = re.split(r'[\\ ]',text)
            #start = random.randrange(len(words))
            #end=start
            #ret = 
            #while True:
        else:
            return text

    def getFrontText(self,text,list_split=False,term=None):
        #get the front part of the text, breaking on words
        if len(text)>self.max_qa_len:
            end = self.max_qa_len
            while end>1 and text[end]!=' ' and text[end]!='\\' and (not list_split or (text[end]!='|' and text[end-1]!='|')):
                end-=1
            if end==1:
                #couldn't break
                return text[:self.max_qa_len]
            else:
                return text[:end]
        else:
            if term is not None and len(text)+len(term)<=self.max_qa_len:
                text+=term
            return text
