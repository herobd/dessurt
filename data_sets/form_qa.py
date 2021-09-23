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


class Table:
    def __init__(self,row_headers,col_headers):
        self.row_headers=row_headers
        self.col_headers=col_headers
        self.cells=[[None]*len(col_headers) for i in range(len(row_headers))]
    #def __getitem__(self,key):
    #    row,col = key
    #    r = self.row_headers.index(row)
    #    c = self.col_headers.index(col)
    #    return self.cells[r,c]
    #def __setitem__(self,key,value):
    #    row,col = key
    #    r = self.row_headers.index(row)
    #    c = self.col_headers.index(col)
    #    self.cells[r,c]=value
    def allEntities(self):
        ae = self.row_headers + self.col_headers
        for row in self.cells:
            ae += [c for c in row if c is not None]
        return ae

class Entity:
    #This represents a multi-line entity
    def __init__(self,cls,lines):
        self.cls=cls
        self.lines=lines

        self.text=''
        self.text_map = []
        lX=tY=99999999999
        rX=bY=-1
        full=False
        for li,line in enumerate(self.lines):
            self.text+=line.text+'\\'
            self.text_map+=[li]*(len(line.text)+1)
            if len(line.box)==4:
                lX = min(lX,line.box[0])
                tY = min(tY,line.box[1])
                rX = max(rX,line.box[2])
                bY = max(bY,line.box[3])
            else:
                full=True
                lX = min(lX,*line.box[::2])
                tY = min(tY,*line.box[1::2])
                rX = max(rX,*line.box[::2])
                bY = max(bY,*line.box[1::2])
        self.text=self.text[:-1]#removing trailing '\'
        if not full:
            self.box=[lX,tY,rX,bY]
        else:
            self.box=[lX,tY,rX,tY,rX,bY,lX,bY,
                    lX,(tY+bY)/2,rX,(tY+bY)/2,(lX+rX)/2,tY,(lX+rX)/2,bY]
        assert self.text != ''

    def __repr__(self):
        return 'Entity({} ({},{}) : {})'.format(self.cls,self.box[0],self.box[1],self.text)

    def append(self,entity):
        assert len(self.lines)==1 and len(entity.lines)==1
        self.text+=' '+entity.text
        if len(self.box)==4:
            raise NotImplementedError('assumed to only be needed in FUNSD')
        lX = self.box[0]
        tY = min(self.box[1],entity.box[1])
        rX = entity.box[2]
        bY = max(self.box[7],entity.box[7])
        self.box=[lX,tY,rX,tY,rX,bY,lX,bY,
                    lX,(tY+bY)/2,rX,(tY+bY)/2,(lX+rX)/2,tY,(lX+rX)/2,bY]
        self.lines=[Line(self.text,self.box)]

class Line:
    def __init__(self,text,box):
        self.box=box
        self.bbid=None
        self.text=text
        self.ambiguous=False
    def __repr__(self):
        return 'Line({} {})'.format(self.text,self.box)
        

class FormQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FormQA, self).__init__(dirPath,split,config,images)
        self.punc_regex = re.compile('[%s]' % re.escape(string.punctuation))
        self.do_masks=True

        #self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0.15
        #self.do_words = config['do_words']
        #self.char_qs = config['char_qs'] if 'char_qs' in config else False

        self.q_types =         ['np','all','class-link','class','down-pair','up-pair','read','cell','row-header','col-header','all-row','all-col', 'list-row-headers','list-col-headers','count-tables','highlight-table']
        self.q_type_weights = [0.2,  1.0,  1.3, 0.7,    1.0,        1.0,      1.0,   1.1,   1.1,         1.1,         1.1,       1.1,       1.1,              1.1,              0.9,           1.1]
        self.q_types_no_table =         ['np','all','class-link','class','down-pair','up-pair','read','count-tables']
        self.q_type_no_table_weights = [0.2,  1.0,  1.3, 0.7,    1.0,        1.0,      1.0,   0.05]
        self.q_types_only_table =        ['np','all','class-link','class','read','cell','row-header','col-header','all-row','all-col', 'list-row-headers','list-col-headers','count-tables','highlight-table']
        self.q_type_only_table_weights = [0.2, 1.0,  1.3, 0.7,    1.0,   1.1,   1.1,         1.1,         1.1,       1.1,       1.1,              1.1,              0.9,           1.1]

        self.q_types_for_np = ['class-link','class','down-pair','up-pair','read','cell','row-header','col-header','all-row', 'list-row-headers','list-col-headers']

       
        self.np_token = '№'
        self.blank_token = 'ø'
        self.end_token='‡' 

    #entities =[  {class, box (whole), text, lines:{box,, bbid, text, ambiguous} ]
    #entity_adj =[(upper,lower)] either can be None
    #tables = obj. col/row_headers = [entity_id], cells = [[entity_id]]
    #
    def makeQuestions(self,s,entities,entity_link,tables,full_entities,full_entity_dict):
        """
        Generates N questions from given docuemnt information:
         - entities: a list of Entity objects
         - entity_link: a list of (entity_id, entity_id) tuples where its (header,question)/(question,answer)
         - tables: a list of Table objects
         """

        all_of_cls=defaultdict(list)
        for entity in entities:
            all_of_cls[entity.cls].append(entity)

        q_a_pairs = []
        if len(tables)>0:
            if len(entity_link)>0:
                q_types = random.choices(self.q_types,self.q_type_weights,k=self.questions*50)
            else:
                q_types = random.choices(self.q_types_only_table,self.q_type_only_table_weights,k=self.questions*50)
        else:
            q_types = random.choices(self.q_types_no_table,self.q_type_no_table_weights,k=self.questions*50)
        
        #becuase a one-to-many is a single QA going down, but multiple QAs going up
        #this is created to sample the up links from
        unrolled_entity_link=[]
        for head,tail in entity_link:
            if isinstance(tail,(list,tuple)):
                for t in tail:
                    unrolled_entity_link.append((head,t))
            else:
                unrolled_entity_link.append((head,tail))

        #import pdb;pdb.set_trace()

        for q_type in q_types:
            if q_type == 'all':
                if random.random()<0.2:
                    cls = random.choice(list(all_of_cls.keys()))
                else:
                    cls = random.choice(entities).cls #pick class based on distrubition
                question = 'al~'+cls
                ids=[]
                outmask=[]
                response_text = str(len(all_of_cls[cls]))
                for entitiy in all_of_cls[cls]:
                    ids += [line.bbid for line in entitiy.lines]
                    outmask += [self.convertBB(s,line.box) for line in entitiy.lines]
                self.qaAdd(q_a_pairs,question,response_text,ids,[],outmask)
            elif q_type == 'class-link':
                ei = random.randrange(len(full_entities))
                entity = full_entities[ei]
                cls = entity.cls[0] #first character for compression

                #g0 get with str+mask
                #gs get with str
                #gm get with mask
                #z0 hl with str+mask
                #zs hl with str
                #zm highlight with mask

                if random.random()<0.5:
                    highlight=True
                    question='z'
                else:
                    highlight=False
                    question='g'

                if random.random()<0.333:
                    question+='0'
                    prompt_text = self.selectPartText(entity.text)
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                    mask=True
                elif random.random()<0.5:
                    question+='s'
                    prompt_text = self.selectPartText(entity.text)
                    inmask = []
                    mask=False
                else:
                    question+='m'
                    prompt_text=''
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                    mask=True

                if ei not in full_entity_dict:
                    question+='~'
                    response_text=self.blank_token
                    outmask=[]
                    ids = [line.bbid for line in entity.lines]
                else:
                    
                    if highlight:
                        question+='~'
                        response_text=str(len( full_entity_dict[ei]))
                        outmask = []
                        ids = [line.bbid for line in entity.lines]
                        for other_ei in full_entity_dict[ei]:
                            outmask += [self.convertBB(s,line.box) for line in full_entities[other_ei].lines]
                            ids += [line.bbid for line in full_entities[other_ei].lines]
                    else:
                        #step through
                        num = len( full_entity_dict[ei])
                        if num>1:
                            i = random.randrange(num+1)
                        elif random.random()<0.01:
                            i = 1
                        else:
                            i = 0
                        next_entity = full_entities[full_entity_dict[ei][i]] if i<num else None

                        if next_entity is not None:
                            response_text = self.getFrontText(next_entity.text,term='|' if i<num-1 else self.end_token)
                            outmask = [self.convertBB(s,line.box) for line in next_entity.lines]
                            ids= [line.bbid for line in next_entity.lines]
                        else:
                            response_text = self.blank_token
                            outmask = []
                            ids = []

                        if i==0:
                            question+='~'
                            response_text = '{}>'.format(num)+response_text
                            if len(response_text)<self.max_qa_len:
                                response_text = response_text[:self.max_qa_len]
                            ids = [line.bbid for line in entity.lines+next_entity.lines]
                        else:
                            prev_entity = full_entities[full_entity_dict[ei][i-1]]
                            if len(prompt_text)>-1+self.max_qa_len//2:
                                prompt_text = self.selectPartText(prompt_text,length=-1+self.max_qa_len//2)
                            next_part_len = self.max_qa_len-(1+len(prompt_text))
                            prompt_text = prompt_text+'>'+self.selectPartText(prev_entity.text,next_part_len)
                            question+='>'
                            ids = [line.bbid for line in prev_entity.lines]
                            if next_entity is not None:
                                ids += [line.bbid for line in next_entity.lines]
                            if mask:
                                inmask += [self.convertBB(s,line.box) for line in prev_entity.lines]

                response_text = '['+cls+']'+response_text
                self.qaAdd(q_a_pairs,question+prompt_text,response_text,ids,inmask,outmask)

                        

            elif q_type=='class':
                #e_i = random.randrange(len(entities))
                entity = random.choice(entities)
                cls = entity.cls
                class_answer = '[ '+cls+' ]'
                line = random.choice(entity.lines)
                text=line.text
                if self.max_qa_len is not None and len(text)>self.max_qa_len:
                    if random.random()<0.1:
                        text = text[:self.max_qa_len]
                    else:
                        text = self.selectPartText(text)

                inmask=[]
                if random.random() <0.5 or line.ambiguous:
                    #use query mask
                    question = 'c$~'
                    inmask.append(self.convertBB(s,line.box))
                else:
                    question = 'cs~'
                self.qaAdd(q_a_pairs,question+text,class_answer,[line.bbid],inmask,[]) #This can be ambigous, although generally the same text has the same class
            
            elif q_type in ('down-pair', 'up-pair'):
                down = q_type=='down-pair'

                for i in range(min(10,len(entity_link))):
                    #head_id, tail_id = random.choice(entity_link)
                    if down:
                        head_id, tail_id = random.choice(entity_link)
                        prompt_id = head_id
                        response_id = tail_id
                    else:
                        head_id, tail_id = random.choice(unrolled_entity_link)
                        #if isinstance(tail_id, list) or isinstance(tail_id, tuple):
                        #    prompt_id = random.choice(tail_id)
                        #else:
                        prompt_id = tail_id
                        response_id = head_id
                    if prompt_id is not None:
                        break
                if prompt_id is None:
                    continue

                prompt_cls = entities[prompt_id].cls


                #get end of response text
                if response_id is not None:
                    #this gets a little tricky. A down relationship has possibly multiple answers
                    #We use the same methodology as with listing table compenents
                    #It first returns the number of elements '#>' and 'the-first-element|'
                    #There can then be subsequent queries using 'QQ>last-element' which returns
                    #the next 'element|' until the 'last-element[endtoken]'
                    #Here there can't be blanks, so things are a little simpler
                    if isinstance(response_id,list) or isinstance(response_id,tuple):
                        assert down
                        i = random.randrange(len(response_id))
                        num_resp = len(response_id)
                        if i==0:
                            response_start = '{}>'.format(num_resp)
                            char='~'
                        else:
                            response_start = ''
                            char='>'
                            prompt_id = response_id[i-1]
                            #prev_text = entities[i-1].text
                            #prompt_text = self.selectPartText(prev_text)
                        response_id = response_id[i]
                    else:
                        i = 0
                        num_resp = 1
                        char='~'
                        response_start = '1>' if down else ''

                    response_text = entities[response_id].text
                    if not down:
                        response_text = response_text[::-1]
                    if len(response_text)>self.max_qa_len:
                        response_text = response_text[:self.max_qa_len]
                    elif len(response_text)+1<=self.max_qa_len:
                        response_text += self.end_token if i==num_resp-1 else '|'
                else:
                    response_text = self.blank_token
                    response_start = ''
                    char='~'

                #sample a span of text from prompt entity
                prompt_text = entities[prompt_id].text
                #if len(prompt_text)>self.max_qa_len:
                #    if random.random()<0.25: #take from end or random:
                #        if down:
                #            prompt_text = prompt_text[-self.max_qa_len:]
                #        else:
                #            prompt_text = prompt_text[:self.max_qa_len]
                #    else:
                #        #random
                prompt_text = self.selectPartText(prompt_text)

                inmask = []
                ambiguous = len(entities[prompt_id].lines)==1 and entities[prompt_id].lines[0].ambiguous
                mask =  random.random()<0.5 or ambiguous#should we use query mask
                if down and (entities[response_id].cls=='answer' if response_id is not None else entities[prompt_id].cls=='question'):
                    question = 'd0'+char if mask else 'l'+char
                elif not down and prompt_cls=='answer':
                    question = 'v0~' if mask else 'v~'
                elif down and prompt_cls=='header':
                    question = 'h0'+char if mask else 'hd'+char
                elif not down and entities[response_id].cls=='header' and prompt_cls=='header':
                    question = 'u1'+char if mask else 'uh'+char
                elif not down and prompt_cls=='question':
                    question = 'q0~' if mask else 'qu~'
                else:
                    assert False
                
                if mask:
                    if random.random()<0.5: #should the mask be lines or whole entity
                        for line in entities[prompt_id].lines:
                            inmask.append(self.convertBB(s,line.box))
                    else:
                        inmask.append(self.convertBB(s,entities[prompt_id].box))

                outmask = []
                bbids = []
                #outmask is lines, as we'd like the higher resolution
                if response_id is not None:
                    for line in entities[response_id].lines:
                        outmask.append(self.convertBB(s,line.box))
                        bbids.append(line.bbid)

                
                bbids += [l.bbid for l in entities[prompt_id].lines]
                self.qaAdd(q_a_pairs,question+prompt_text,response_start+response_text,bbids,inmask,outmask)

            elif q_type=='np':
                sub_type = random.choice(self.q_types_for_np)

                match = True
                while match:
                    #check if this is indeed novel text for the given document
                    prompt_text = self.sampleText()
                    match = False
                    prompt_text_no_punc = self.punc_regex.sub('',prompt_text.lower())
                    for entity in entities:
                        if prompt_text_no_punc in  self.punc_regex.sub('',entity.text.lower()):
                            match=True
                            break

                if sub_type == 'class':
                    question = 'cs~{}'
                elif sub_type == 'down-pair':
                    question = 'l~{}' if random.random()<0.5 else 'hd~'
                elif sub_type == 'up-pair':
                    question = 'v~{}' if random.random()<0.5 else 'qu~'
                elif sub_type == 'read':
                    question = 'fi~{}'
                elif sub_type == 'cell':
                    #we'll make this hard by selecting a real other header
                    if len(tables)>0:
                        table = random.choice(tables)
                        if random.random()>0.5:
                            header = random.choice(table.row_headers)
                        else:
                            header = random.choice(table.col_headers)
                        header_text = self.selectPartText(header.text,length=-1+self.max_qa_len//2)
                    else:
                        header_text = self.sampleText(-1+self.max_qa_len//2)
                    if random.random()>0.5:
                        question = 't~'+header_text+'~~{}'
                    else:
                        question = 't~{}~~'+header_text
                elif sub_type == 'row-header':
                    question = 'ri~{}'
                elif sub_type == 'col-header':
                    question = 'ci~{}'
                elif sub_type == 'all-row':
                    question = '$r~{}' if random.random()<0.5 else ('ar~{}' if random.random()<0.5 else 'ar>{}')
                elif sub_type == 'all-col':
                    question = '$c~{}' if random.random()<0.5 else ('ac~{}' if random.random()<0.5 else 'ac>{}')
                elif sub_type in ('list-row-headers','list-col-headers'):
                    if sub_type == 'list-row-headers':
                        char = 'r'
                    else:
                        char = 'c'
                    if random.random()<0.5:
                        question = '{}h~{}'.format(char,len(tables)+1)
                    else:
                        question = char+'h>{}'
                elif sub_type == 'class-link':
                    if random.random()<0.5:
                        question='zs~{}'
                    else:
                        question='gs~{}'

                self.qaAdd(q_a_pairs,question.format(prompt_text),self.np_token,[],[],[])

            elif q_type=='read':

                #finish reading entity
                for i in range(10):
                    entity = random.choice(entities)
                    text = entity.text
                    if len(text)>2:
                        break

                if random.random()<0.5:
                    forward=True
                else:
                    forward=False
                    text = text[::-1]

                last_space = max(text.rfind(' '),text.rfind('\\'))
                if last_space==-1: #no space
                    last_space = random.randrange(1,len(text)-1)
                query_part = text[:last_space] #so there's at least one word to predict
                query,q_start = self.selectPartText(query_part,ret_start=True)
                if len(query)==0:
                    continue
                q_end = q_start + len(query)


                if text[q_end]==' ' or text[q_end]=='\\':
                    r_start = q_end+1
                else:
                    r_start = q_end

                response = self.getFrontText(text[r_start:],term=self.end_token)

                ids = [line.bbid for line in entity.lines]

                if forward:
                    r_end = r_start+len(response)
                else:
                    #translate from reverse indices
                    r_end = -r_start-1
                    r_start = -(r_start+len(response))-1
                    q_t = -q_start-1
                    q_start = -q_end-1
                    q_end = q_t

                out_line_ids = set(entity.text_map[r_start:r_end])
                query_line_ids = set(entity.text_map[q_start:q_end])

                outmask = []
                for line_id in out_line_ids:
                    outmask.append(self.convertBB(s,entity.lines[line_id].box))
                
                ambiguous = all([entity.lines[li].ambiguous for li in query_line_ids])


                if random.random()<0.4 and not ambiguous and len(query)>6:
                    question='fi~' if forward else 'pr~' #for "finish" or "previous"
                    inmask=[]
                elif random.random()<0.5:
                    question='f0~' if forward else 'p0~'
                    #This uses the mask of the whol entity, as that is what other things will produce
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                else:
                    question='f1~' if forward else 'p1~'
                    #This uses the mask of only the query lines
                    inmask = [self.convertBB(s,entity.lines[li].box) for li in query_line_ids]

                self.qaAdd(q_a_pairs,question+query,response,ids,inmask,outmask)


            elif q_type=='cell':
                table = random.choice(tables)
                r,r_header = random.choice(list(enumerate(table.row_headers)))
                c,c_header = random.choice(list(enumerate(table.col_headers)))

                r_h_text = r_header.text if r_header is not None else self.blank_token
                c_h_text = c_header.text if c_header is not None else self.blank_token

                r_h_text = self.selectPartText(r_h_text,-1+self.max_qa_len//2)
                c_h_text = self.selectPartText(c_h_text,-1+self.max_qa_len//2)

                cell = table.cells[r][c]
                if cell is not None:
                    cell_text = cell.text
                    if len(cell_text) > 0:
                        cell_text = self.getFrontText(cell_text)
                    elif len(cell_text)+1<=self.max_qa_len:
                        cell_text += self.end_token
                    outmask = [self.convertBB(s,line.box) for line in cell.lines]
                    cell_lines = cell.lines
                else:
                    cell_text = self.blank_token
                    cell_lines=[]
                    outmask = []
                
                ambiguous = all(line.ambiguous for line in \
                        (r_header.lines if r_header is not None else [])+\
                        (c_header.lines if c_header is not None else [])) 

                if random.random()<0.5 and not ambiguous:
                    question='t'
                    inmask=[]
                else:
                    question='t0'
                    inmask = [self.convertBB(s,line.box) for line in \
                            (r_header.lines if r_header is not None else [])+\
                            (c_header.lines if c_header is not None else [])]

                ids = [line.bbid for line in \
                        (r_header.lines if r_header is not None else [])+\
                        (c_header.lines if c_header is not None else [])] 

                if random.random()<0.5:
                    h1_text = r_h_text
                    h2_text = c_h_text
                else:
                    h1_text = c_h_text
                    h2_text = r_h_text

                self.qaAdd(q_a_pairs,'{}~{}~~{}'.format(question,h2_text,h1_text),cell_text,ids,inmask,outmask)


            elif q_type in ('row-header', 'col-header'):
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='row-header':
                    row=True
                    r,header = random.choice(list(enumerate(table.row_headers)))
                    non_none_cells = [table.cells[r][c] for c in range(len(table.col_headers)) if table.cells[r][c] is not None]
                else:
                    row=False
                    c,header = random.choice(list(enumerate(table.col_headers)))
                    non_none_cells = [table.cells[r][c] for r in range(len(table.row_headers)) if table.cells[r][c] is not None]

                if len(non_none_cells)==0:
                    continue
                cell = random.choice(non_none_cells)

                cell_text = self.selectPartText(cell.text)

                if header is not None:
                    header_text = header.text
                    header_text = self.getFrontText(header_text,term=self.end_token)
                else:
                    header_text = self.blank_token
                #if len(header_text) > self.max_qa_len:
                #    header_text = header_text[:self.max_qa_len]
                #elif len(header_text)+1<=self.max_qa_len and header_text!=self.blank_token:
                #    header_text += self.end_token

                ambiguous = all([line.ambiguous for line in cell.lines])

                ids=[]
                inmask=[]
                if random.random()<0.5 and not ambiguous:
                    if row:
                        question = 'ri~'
                    else:
                        question = 'ci~'
                else:
                    if row:
                        question = 'r*~'
                    else:
                        question = 'c*~'
                    inmask = [self.convertBB(s,line.box) for line in cell.lines]

                ids = [line.bbid for line in cell.lines]

                outmask = []
                if header is not None:
                    for line in header.lines:
                        outmask.append(self.convertBB(s,line.box))
                        ids.append(line.bbid)

                self.qaAdd(q_a_pairs,question+cell_text,header_text,ids,inmask,outmask)


            elif q_type in ('all-row', 'all-col'):
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='all-row':
                    row=True
                    r,header = random.choice(list(enumerate(table.row_headers)))
                    if header is None:
                        continue
                    all_cells = table.cells[r]
                else:
                    row=False
                    c,header = random.choice(list(enumerate(table.col_headers)))
                    if header is None:
                        continue
                    all_cells = [table.cells[r][c] for r in range(len(table.row_headers))]

                ambiguous = all([line.ambiguous for line in header.lines])

                outmask = []
                if random.random()<0.5:
                    #just highligh the row/col
                    ids=[line.bbid for line in header.lines]
                    header_text = self.selectPartText(header.text)
                    for cell in all_cells:
                        if cell is not None:
                            ids += [line.bbid for line in cell.lines]
                            outmask += [self.convertBB(s,line.box) for line in cell.lines]

                    if random.random()<0.5 or ambiguous:
                        inmask = [self.convertBB(s,line.box) for line in header.lines]
                        question = '#r~' if row else '#c~'
                    else:
                        inmask = []
                        question = '$r~' if row else '$c~'

                    self.qaAdd(q_a_pairs,question+header_text,'',ids,inmask,outmask)
                else:
                    #step, text-wise, through each entry
                    #the text of a cell is the query for the next cell
                    #in the event of a blank cell, it just goes to the next cell, but adds 'ø|' to the front to show there is a blank cell before this
                    #when called with the final (filled) cell as the query, it produces the stop token '‡'
                    init = i = random.randrange(len(all_cells)+1)
                    cell = all_cells[i] if i<len(all_cells) else None

                    prepend=''
                    #go down the row/col till we reach a non-blank cell
                    while cell is None and i<len(all_cells):
                        prepend+=self.blank_token+'|'
                        i+=1
                        cell = all_cells[i] if i<len(all_cells) else None

                    #be sure we are including all cells before that are blank
                    while init>0 and all_cells[init-1] is None:
                        init-=1
                        prepend = self.blank_token+'|'+prepend

                    if init>0:
                        header = all_cells[init-1]
                        char = '>'
                        response_start = ''
                    else:
                        char = '~'
                        response_start = '{}>'.format(len(all_cells))
                    ids=[line.bbid for line in header.lines]

                    header_text = self.selectPartText(header.text)
                    if random.random()<0.5 or ambiguous:
                        inmask = [self.convertBB(s,line.box) for line in header.lines]
                        question = '%r{}' if row else '%c{}'
                    else:
                        inmask = []
                        question = 'ar{}' if row else 'ac{}' 
                    question = question.format(char)

                    if cell is not None:
                        outmask =  [self.convertBB(s,line.box) for line in cell.lines]
                        ids += [line.bbid for line in cell.lines]
                        cell_text = self.getFrontText(prepend+cell.text,term='|' if i<len(all_cells)-1 else self.end_token)
                    else:
                        outmask = []
                        cell_text = prepend[:-1]+self.end_token

                    self.qaAdd(q_a_pairs,question+header_text,response_start+cell_text,ids,inmask,outmask)
                        



            elif q_type in ('list-row-headers','list-col-headers'):
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='list-row-headers':
                    row=True
                    headers = table.row_headers
                else:
                    row=False
                    headers = table.col_headers
                #print('{} headers: {}'.format('row' if row else 'col',headers))

                
                outmask = []
                if random.random()<0.5:
                    #just highligh the headers
                    ids=[]
                    outmask=[]
                    for header in headers:
                        if header is not None:
                            ids+=[line.bbid for line in header.lines]
                            outmask += [self.convertBB(s,line.box) for line in header.lines]
                    question = 'r@~' if row else 'c@~'

                    self.qaAdd(q_a_pairs,question+str(table_i),'',ids,[],outmask)
                else:
                    #step, text-wise, through each entry
                    i = init = random.randrange(len(headers)+1)
                    header = headers[i] if i<len(headers) else None
                    prepend=''
                    #go down the row/col till we reach a non-blank header (FUNSD can have blank headers)
                    while header is None and i<len(headers):
                        prepend+=self.blank_token+'|'
                        i+=1
                        header = headers[i] if i<len(headers) else None

                    #be sure we are including all headers before that are blank
                    while init>0 and headers[init-1] is None:
                        init-=1
                        prepend = self.blank_token+'|'+prepend

                    if init>0:
                        prev_header = headers[init-1]
                        prev_text = self.selectPartText(prev_header.text)
                        ids=[line.bbid for line in prev_header.lines]
                        char = '>'
                        ambiguous = all([line.ambiguous for line in prev_header.lines])
                        response_start = ''
                    else:
                        char = '~'
                        prev_text = str(table_i)
                        ids=[]
                        ambiguous=False
                        response_start = '{}>'.format(len(headers))

                    if (random.random()<0.5 or ambiguous) and init>0:
                        inmask = [self.convertBB(s,line.box) for line in prev_header.lines]
                        question = 'r&{}' if row else 'c&{}'
                    else:
                        inmask = []
                        question = 'rh{}' if row else 'ch{}' 
                    question = question.format(char)

                    if header is not None:
                        outmask =  [self.convertBB(s,line.box) for line in header.lines]
                        ids += [line.bbid for line in header.lines]
                        header_text = self.getFrontText(prepend+header.text,term='|' if i<len(headers)-1 else self.end_token)
                    else:
                        outmask = []
                        header_text = self.end_token

                    self.qaAdd(q_a_pairs,question+prev_text,response_start+header_text,ids,inmask,outmask)

            elif q_type=='count-tables':
                table_ids=[]
                outmask=[]
                for table in tables:
                    for header in table.row_headers + table.col_headers:
                        if header is not None:
                            for line in header.lines:
                                outmask.append(self.convertBB(s,line.box))
                                table_ids.append(line.bbid)

                    for r in range(len(table.row_headers)):
                        for c in range(len(table.col_headers)):
                            cell = table.cells[r][c]
                            #print('{},{}: {}'.format(r,c,cell.text if cell is not None else '-'))
                            if cell is not None:
                                for line in cell.lines:
                                    outmask.append(self.convertBB(s,line.box))
                                    table_ids.append(line.bbid)
                self.qaAdd(q_a_pairs,'t#>',str(len(tables)),table_ids,[],outmask)

            elif q_type=='highlight-table':
                table_i = random.randrange(len(tables))
                table = tables[table_i]
                table_ids=[]
                outmask=[]
                for header in table.row_headers + table.col_headers:
                    if header is not None:
                        for line in header.lines:
                            outmask.append(self.convertBB(s,line.box))
                            table_ids.append(line.bbid)

                for r in range(len(table.row_headers)):
                    for c in range(len(table.col_headers)):
                        cell = table.cells[r][c]
                        #print('{},{}: {}'.format(r,c,cell.text if cell is not None else '-'))
                        if cell is not None:
                            for line in cell.lines:
                                outmask.append(self.convertBB(s,line.box))
                                table_ids.append(line.bbid)


                self.qaAdd(q_a_pairs,'0t~{}'.format(table_i),'',table_ids,[],outmask)
            else:
                assert False
            
            if len(q_a_pairs)>10*self.questions:
                break #we have enough

        return q_a_pairs

    def selectPartText(self,text,length=None,ret_start=False):
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

            best_len = max(len_b_b if len_b_b<=self.max_qa_len else -1, len_a_b if len_a_b<=self.max_qa_len else -1, len_a_a if len_a_a<=self.max_qa_len else -1)

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



            #return text[start:start+length]
            #words = re.split(r'[\\ ]',text)
            #start = random.randrange(len(words))
            #end=start
            #ret = 
            #while True:
        else:
            ret = text
            start = 0

        if ret_start:
            return ret, start
        else:
            return ret

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

    def convertBB(self,s,box):
        lX,tY,rX,bY = box
        #should return a list of [lX, tY, rX, tY, rX, bY, lX, bY, 
        bb = [lX*s, tY*s, rX*s, tY*s, rX*s, bY*s, lX*s, bY*s,
               s*lX, s*(tY+bY)/2.0, s*rX, s*(tY+bY)/2.0, s*(lX+rX)/2.0, s*tY, s*(rX+lX)/2.0, s*bY]  #we add these for conveince to crop BBs within window
        #??
        return bb

    def sampleText(self,length=None):
        if length is None:
            length = self.max_qa_len

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
        while len(text)<self.max_qa_len and idx<start+num_words:
            last_text = text
            text += ' '+words[idx]
            idx+=1
            
        assert text!=''
        if len(text)>self.max_qa_len and last_text is not None:
            text = last_text
        elif len(text)>self.max_qa_len:
            text = self.selectPartText(text)
        if len(text)==0:
            return self.sampleText()
        return text
 
    def sortLinkDict(self,entities,link_dict):
        new_link_dict={}
        #def readpos(a):
            #entity = entities[a]
        for e1,e2s in link_dict.items():
            #e2s.sort(key=readpos)
            #arrange into rows
            rows = []
            row_ys = []
            for e2 in e2s:
                top_y = entities[e2].box[1]
                hit=False
                for r,row in enumerate(rows):
                    mean_y = np.mean(row_ys[r])
                    if abs(top_y - mean_y)<30:
                        row.append(e2)
                        row_ys[r].append(top_y)
                        hit=True
                if not hit:
                    rows.append([e2])
                    row_ys.append([top_y])

            rows = [(row,np.mean(ys)) for row,ys in zip(rows,row_ys)]
            rows.sort(key=lambda a: a[1])
            sorted_e2s=[]
            for row,y in rows:
                row = [(e2,entities[e2].box[0]) for e2 in row]
                row.sort(key=lambda a:a[1])
                sorted_e2s += [a[0] for a in row]
            
            new_link_dict[e1]=e2s
        return new_link_dict
