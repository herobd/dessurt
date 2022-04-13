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
from data_sets.wiki_text import getWikiArticle,getWikiDataset

from utils.read_order import sortReadOrder

import utils.img_f as img_f
from transformers import BartTokenizer


class Table:
    def __init__(self,row_headers,col_headers):
        self.used=False
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
    def addRowHeader(self,entity):
        #find where they go
        entity_y = (entity.getBox()[1]+entity.getBox()[3])/2
        found=False
        for pos,header in enumerate(self.row_headers):
            if header is not None:
                header_y = (header.getBox()[1]+header.getBox()[3])/2
                if entity_y<header_y:
                    found=True
                    break
        if not found:
            pos = len(self.row_headers)

        self.row_headers.insert(pos,entity)
        self.cells.insert(pos,[None]*len(self.col_headers))

    def addColHeader(self,entity):
        #find where they go
        entity_x = (entity.getBox()[0]+entity.getBox()[2])/2
        found=False
        for pos,header in enumerate(self.col_headers):
            if header is not None:
                header_x = (header.getBox()[0]+header.getBox()[2])/2
                if entity_x<header_x:
                    found=True
                    break
        if not found:
            pos = len(self.col_headers)

        self.col_headers.insert(pos,entity)
        for row in self.cells:
            row.insert(pos,None)

    def addHeader(self,entity):
        row_mean_x = 0
        row_count = 0
        for header in self.row_headers:
            if header is not None:
                row_mean_x += sum(header.getBox()[0::2])
                row_count +=1
        row_mean_x/= 2*row_count
        col_mean_y = 0
        col_count = 0
        for header in self.col_headers:
            if headers is not None:
                col_mean_y += sum(header.getBox()[1::2])
                col_count += 1
        col_mean_y/= 2*col_count

        y_diff = abs(col_mean_y - sum(entity.getBox()[1::2])/2)
        x_diff = abs(row_mean_x - sum(entity.getBox()[::2])/2)

        if y_diff<x_diff:
            self.addRowHeader(entity)
        else:
            self.addColHeader(entity)


    def getBox(self):
        lx=ty=999999
        rx=by=-1
        all_entities = []
        if self.row_headers is not None:
            all_entities += self.row_headers
        if self.col_headers is not None:
            all_entities += self.col_headers
        for row in self.cells:
            all_entities += row

        for entity in all_entities:
            if entity is not None:
                x1,y1,x2,y2 = entity.getBox()
                lx = min(lx,x1)
                rx = max(rx,x2)
                ty = min(ty,y1)
                by = max(by,y2)
        return lx,ty,rx,by
    def getSortTopBot(self):
        return self.getBox()[1::2] #1,3=y1,y2
class FillInProse:
    #This is a paragraph/run of text where there are blanks to be filled in
    def __init__(self,entities):
        self.used = False
        self.entities = entities
    def getBox(self):
        lx=ty=999999
        rx=by=-1
        for entity in self.entities:
            x1,y1,x2,y2 = entity.getBox()
            lx = min(lx,x1)
            rx = max(rx,x2)
            ty = min(ty,y1)
            by = max(by,y2)
        return lx,ty,rx,by
    def __repr__(self):
        s = ''
        for e in self.entities:
            s+= 'Q' if e.cls=='question' else 'A'
            s+= ':'+e.text+', '
        return 'FillInProse({})'.format(s)
    def getSortTopBot(self):
        return self.getBox()[1::2] #1,3=y1,y2
class MinoredField:
    #This is a paragraph/run of text where there are blanks to be filled in
    def __init__(self,question,answers,minors):
        self.used=False
        self.question = question
        self.answers = answers
        self.minors = minors
    def getBox(self):
        lx,ty,rx,by = self.question.getBox() if self.question is not None else (99999,99999,-1,-1)
        for entity in self.answers:
            x1,y1,x2,y2 = entity.getBox()
            lx = min(lx,x1)
            rx = max(rx,x2)
            ty = min(ty,y1)
            by = max(by,y2)
        for entity in self.minors:
            x1,y1,x2,y2 = entity.getBox()
            lx = min(lx,x1)
            rx = max(rx,x2)
            ty = min(ty,y1)
            by = max(by,y2)
        return lx,ty,rx,by
    def __repr__(self):
        return 'Minored({}, {}, {})'.format(self.question,self.answers,self.minors)
    def getSortTopBot(self):
        if self.question is not None:
            return self.question.getSortTopBot()
        else:
            return self.answers[0].getSortTopBot()
class Entity:
    #This represents a multi-line entity
    def __init__(self,cls,lines=None):
        self.used = False #used for debuggin in JSON creation
        if isinstance(cls,Entity) and lines is None:
            #copy contstructor
            other = cls
            self.cls = other.cls
            self.lines = list(other.lines)
            self.text = other.text
            self.text_map= other.text_map
            self.box=other.box
        else:
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
    
    def getBox(self): #returns left-x,top-y,right-x,bottom-y regardless of type of self.box
        if len(self.box)==4:
            return self.box
        else:
            return min(self.box[::2]),min(self.box[1::2]),max(self.box[::2]),max(self.box[1::2])
    def getSortTopBot(self):
        return self.getBox()[1::2] #1,3=y1,y2

class Line:
    def __init__(self,text,box):
        if isinstance(box,np.ndarray) and len(box.shape)==2:
            self.box = box.flatten()
        else:
            self.box=box
        if len(self.box)==8:
            lX,tY,rX,tY,rX,bY,lX,bY = self.box
            self.box=[lX,tY,rX,tY,rX,bY,lX,bY,
                    lX,(tY+bY)/2,rX,(tY+bY)/2,(lX+rX)/2,tY,(lX+rX)/2,bY]
        self.bbid=None
        self.text=text
        self.ambiguous=False
    def __repr__(self):
        return 'Line({} {})'.format(self.text,self.box)
    def getBox(self): #returns left-x,top-y,right-x,bottom-y regardless of type of self.box
        if len(self.box)==4:
            return self.box
        else:
            return min(self.box[::2]),min(self.box[1::2]),max(self.box[::2]),max(self.box[1::2])
        

class FormQA(QADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FormQA, self).__init__(dirPath,split,config,images)
        self.wiki_dataset = getWikiDataset()
        self.punc_regex = re.compile('[%s]' % re.escape(string.punctuation))
        self.do_masks=True
        self.words = config.get('words',True)
        use_json = config.get('use_json',False)
        self.shorten_text_in_json = config.get('shorten_json',False)
        self.max_json_words=5

        self.max_q_tokens = config.get('max_q_tokens',20)
        self.max_a_tokens = config.get('max_a_tokens',800)

        if os.path.exists('./cache_huggingface/BART'):
            model_id = './cache_huggingface/BART'
        else:
            model_id = 'facebook/bart-base'
        self.tokenizer = BartTokenizer.from_pretrained(model_id)

        #self.corruption_p = config['text_corruption'] if 'text_corruption' in config else 0.15
        #self.do_words = config['do_words']
        #self.char_qs = config['char_qs'] if 'char_qs' in config else False
        if self.train:
            if use_json=='test':
                self.rel_vs_any_link_prob=0.001
                self.q_types = {
                        'read': 3,
                        }
                self.q_types_no_table = {
                        'read': 3,
                        }
                self.q_types_only_table = {
                        'read': 2,
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row', 'full-list-row-headers','full-list-col-headers']
            elif use_json=='only':
                self.rel_vs_any_link_prob=0.001
                self.q_types = {
                        'full_json': 3,
                        }
                self.q_types_no_table = {
                        'full_json': 3,
                        }
                self.q_types_only_table = {
                        'full_json': 2,
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row', 'full-list-row-headers','full-list-col-headers']
            elif use_json=='fine-tune':
                self.rel_vs_any_link_prob=0.001
                self.q_types = {
                        'full_json': 3,
                        'class-link-all': 1,
                        'class-linkdown-all': 1,
                        'class-linkup-all': 1,
                        'np':0.1,
                        'read':0.1,
                        'cell':1.1,
                        'row-header':1.1,
                        'col-header':1.1,
                        'full-all-row':1.1,
                        'full-all-col':1.1,
                        'full-list-row-headers':1.1,
                        'full-list-col-headers':1.1,
                        'count-tables':0.9,
                        'highlight-table':1.1
                        }
                self.q_types_no_table = {
                        'full_json': 3,
                        'class-link-all': 1,
                        'class-linkdown-all': 1,
                        'class-linkup-all': 1,
                        'np':0.1,
                        'read':0.1,
                        'count-tables':0.01
                        }
                self.q_types_only_table = {
                        'full_json': 2,
                        'class-link-all': 0.8,
                        'np':0.1,
                        'read':0.1,
                        'cell':1.1,
                        'row-header':1.1,
                        'col-header':1.1,
                        'full-all-row':1.1,
                        'full-all-col':1.1,
                         'full-list-row-headers':1.1,
                        'full-list-col-headers':1.1,
                        'count-tables':0.9,
                        'highlight-table':1.1
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row', 'full-list-row-headers','full-list-col-headers']
            elif use_json=='streamlined':
                self.rel_vs_any_link_prob=0.01
                self.q_types = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.01,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025
                        }
                self.q_types_no_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.01
                        }
                self.q_types_only_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.01,
                        'np':0.01,
                        'read':0.01,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025,
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row']
            elif use_json=='readtoo':
                self.rel_vs_any_link_prob=0.01
                self.q_types = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.025,
                        'readline':0.025,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025
                        }
                self.q_types_no_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.025,
                        'readline':0.025
                        }
                self.q_types_only_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.01,
                        'np':0.01,
                        'read':0.025,
                        'readline':0.025,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025,
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row']
            elif use_json=='readmore':
                self.rel_vs_any_link_prob=0.01
                self.q_types = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.1,
                        'readline':0.1,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025
                        }
                self.q_types_no_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.1,
                        'readline':0.1
                        }
                self.q_types_only_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.01,
                        'np':0.01,
                        'read':0.1,
                        'readline':0.1,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025,
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row']
            elif use_json=='readevenmore':
                self.rel_vs_any_link_prob=0.01
                self.q_types = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.4,
                        'readline':0.4,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025
                        }
                self.q_types_no_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.05,
                        'class-linkdown-all': 0.05,
                        'class-linkup-all': 0.05,
                        'np':0.01,
                        'read':0.4,
                        'readline':0.4
                        }
                self.q_types_only_table = {
                        'full_json': 0.68,
                        'class-link-all': 0.01,
                        'np':0.01,
                        'read':0.1,
                        'readline':0.1,
                        'cell':0.05,
                        'row-header':0.025,
                        'col-header':0.025,
                        'full-all-row':0.025,
                        'full-all-col':0.025,
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row']
            elif use_json:
                assert use_json is True
                self.rel_vs_any_link_prob=0.1
                self.q_types = {
                        'full_json': 12,
                        'class-link-all': 1,
                        'class-linkdown-all': 1,
                        'class-linkup-all': 1,
                        'np':0.1,
                        'read':0.1,
                        'cell':1.1,
                        'row-header':1.1,
                        'col-header':1.1,
                        'full-all-row':1.1,
                        'full-all-col':1.1,
                        'full-list-row-headers':1.1,
                        'full-list-col-headers':1.1,
                        'count-tables':0.9,
                        'highlight-table':1.1
                        }
                self.q_types_no_table = {
                        'full_json': 3.3,
                        'class-link-all': 1,
                        'class-linkdown-all': 1,
                        'class-linkup-all': 1,
                        'np':0.1,
                        'read':0.1,
                        'count-tables':0.01
                        }
                self.q_types_only_table = {
                        'full_json': 10.7,
                        'class-link-all': 0.8,
                        'np':0.1,
                        'read':0.1,
                        'cell':1.1,
                        'row-header':1.1,
                        'col-header':1.1,
                        'full-all-row':1.1,
                        'full-all-col':1.1,
                         'full-list-row-headers':1.1,
                        'full-list-col-headers':1.1,
                        'count-tables':0.9,
                        'highlight-table':1.1
                        }
                self.q_types_for_np = ['class-link-all','class-linkdown-all','class-linkup-all','read','cell','row-header','col-header','full-all-col','full-all-row', 'full-list-row-headers','full-list-col-headers']
            else:
                self.q_types = {
                        'np':0.2,
                        'all':1.0,
                        'class-link':1.3,
                        'class':0.7,
                        'down-pair':1.0,
                        'up-pair':1.0,
                        'read':1.0,
                        'cell':1.1,
                        'row-header':1.1,
                        'col-header':1.1,
                        'all-row':1.1,
                        'all-col':1.1,
                         'list-row-headers':1.1,
                        'list-col-headers':1.1,
                        'count-tables':0.9,
                        'highlight-table':1.1
                        }
                self.q_types_no_table = {
                        'np':0.2,
                        'all':1.0,
                        'class-link':1.3,
                        'class':0.7,
                        'down-pair':1.0,
                        'up-pair':1.0,
                        'read':1.0,
                        'count-tables':0.05
                        }
                self.q_types_only_table = {
                        'np':0.2,
                        'all':1.0,
                        'class-link':1.3,
                        'class':0.7,
                        'read':1.0,
                        'cell':1.1,
                        'row-header':1.1,
                        'col-header':1.1,
                        'all-row':1.1,
                        'all-col':1.1,
                         'list-row-headers':1.1,
                        'list-col-headers':1.1,
                        'count-tables':0.9,
                        'highlight-table':1.1
                        }
                self.q_types_for_np = ['class-link','class','down-pair','up-pair','read','cell','row-header','col-header','all-row', 'list-row-headers','list-col-headers']

        else:
            #these are what we'll use to actually score
            #(not actually looked at as it isn't sampling)
            if use_json:
                self.q_types =         ['full_json']
                self.q_types_no_table =        ['full_json']
                self.q_types_only_table =        ['full_json']
            else:
                self.q_types =         ['all','class-link','read']
                self.q_types_no_table =        ['all','class-link','read']
                self.q_types_only_table =        ['all','class-link','read']


       
        self.np_token = '№'
        self.blank_token = 'ø'
        self.end_token='‡' 

    #entities =[  {class, box (whole), text, lines:{box,, bbid, text, ambiguous} ]
    #entity_adj =[(upper,lower)] either can be None
    #tables = obj. col/row_headers = [entity_id], cells = [[entity_id]]
    #
    def makeQuestions(self,s,entities,entity_link,tables,raw_entities,raw_entity_dict,proses=None,minored_fields=None):
        """
        Generates N questions from given docuemnt information:
         - entities: a list of Entity objects
         - entity_link: a list of (entity_id, entity_id) tuples where its (header,question)/(question,answer) and value may be a list
         - tables: a list of Table objects
         """
        

        if len(entities)==0:
             return []


        #sort all entity_links and raw_entity_dict in read order
        new_entity_link=[]
        for head,tails in entity_link:
            if isinstance(tails,(list,tuple)):
                tails = sortReadOrder([(t,entities[t].lines[0].box) for t in tails])
            new_entity_link.append((head,tails))
        entity_link = new_entity_link

        for e_i in raw_entity_dict:
            items = sortReadOrder([(t,raw_entities[t].lines[0].box) for t in raw_entity_dict[e_i]])
            raw_entity_dict[e_i]=items
                

        all_of_cls=defaultdict(list)
        for entity in entities:
            all_of_cls[entity.cls].append(entity)

        q_a_pairs = []
        json_text=None
        if self.train:
            if len(tables)>0:
                if len(entity_link)>0:
                    probs = self.q_types
                else:
                    probs = self.q_types_only_table
            else:
                probs = self.q_types_no_table
            if 'full_json' in probs.keys():
                json_text = self.makeJsonText(entities,entity_link,tables,proses,minored_fields)
            q_types = random.choices(list(probs.keys()),probs.values(),k=self.questions*50)
        else:
            if 'full_json' in self.q_types:
                q_types = [('full_json',None,None)]
                json_text = self.makeJsonText(entities,entity_link,tables,proses,minored_fields)
            else:
                q_types = []
                for cls in all_of_cls:
                    q_types.append(('all',cls,None))
                for ei in range(len(raw_entities)):
                    q_types.append(('class-link', ei,True))
                    q_types.append(('class-link', ei,False))
                for entity in entities:
                    if len(entity.text)>=self.max_qa_len_out or len(entity.lines)>1:
                        q_types.append(('read',entity,True))
                        q_types.append(('read',entity,False))

        
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
        if json_text is not None:
            json_tokens = self.tokenizer(json_text,return_tensors="pt")['input_ids']
            tok_len = json_tokens.shape[1]
            #if tok_len>self.max_a_tokens:
            #    if tok_len-(self.max_q_tokens+self.max_a_tokens)>0:
            #        if self.train:
            #            if random.random()<0.1:
            #                r = random.randrange(tok_len-(self.max_q_tokens+self.max_a_tokens))
            #            else:
            #                r = random.randrange(tok_len-self.max_q_tokens-2)
            #        else:
            #            r = tok_len-(self.max_q_tokens+self.max_a_tokens)
            #    else:
            #        r=0
            #    q_json_tokens = json_tokens[0,r:r+self.max_q_tokens]
            #    json_tokens = json_tokens[0,r+self.max_q_tokens:]

            #    json_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(json_tokens,skip_special_tokens=True))
            #    q_json_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(q_json_tokens,skip_special_tokens=True))
            #else:
            #    q_json_text = None
            do_from_start = (tok_len+1<=self.max_a_tokens and not self.train) or (self.train and random.random()<0.5 and not (tok_len+1>self.max_a_tokens and not self.train))
            if do_from_start:
                q_json_text = None
            else:
                if not self.train:
                    r = max(tok_len-(self.max_q_tokens+self.max_a_tokens),0)
                elif tok_len>self.max_q_tokens+self.max_a_tokens and random.random()<0.1:
                    r = random.randrange(tok_len-(self.max_q_tokens+self.max_a_tokens))
                elif tok_len-self.max_q_tokens-2>0:
                    r = random.randrange(tok_len-self.max_q_tokens-2)
                else:
                    r = 0
                q_json_tokens = json_tokens[0,r:r+self.max_q_tokens]
                json_tokens = json_tokens[0,r+self.max_q_tokens:]

                json_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(json_tokens,skip_special_tokens=True))
                q_json_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(q_json_tokens,skip_special_tokens=True))





        for q_type in q_types:
            if not self.train:
                q_type,instance,switch = q_type
            else:
                switch=False

            if q_type == 'full_json':
                #if json_text is None:
                #    json_text = self.makeJsonText(entities,entity_link,tables)
                if q_json_text is None:
                    self.qaAdd(q_a_pairs,'json>',json_text)
                else:
                    self.qaAdd(q_a_pairs,'json~'+q_json_text,json_text)
            elif q_type == 'all':
                if self.train:
                    if len(entities)>0:
                        if random.random()<0.2:
                            cls = random.choice(list(all_of_cls.keys()))
                        else:
                            cls = random.choice(entities).cls #pick class based on distrubition
                    else:
                        cls = 'question' if random.random()<0.5 else 'answer'
                else:
                    cls=instance
                question = 'al~'+cls
                ids=[]
                outmask=[]
                response_text = str(len(all_of_cls[cls]))
                for entitiy in all_of_cls[cls]:
                    ids += [line.bbid for line in entitiy.lines]
                    outmask += [self.convertBB(s,line.box) for line in entitiy.lines]
                self.qaAdd(q_a_pairs,question,response_text,ids,[],outmask)


            elif q_type in ['class-link-all','class-linkdown-all','class-linkup-all']:
                if q_type == 'class-link-all':
                    if self.train:
                        if len(raw_entities)==0:
                            continue
                        ei = random.randrange(len(raw_entities))
                    else:
                        ei = instance
                    entity = raw_entities[ei]
                    question = 'link-'
                    linked = [raw_entities[lei] for lei in raw_entity_dict[ei]] if ei in raw_entity_dict else None
                else:
                    down = 'down' in q_type
                    prompt_id = None
                    if random.random()<self.rel_vs_any_link_prob and self.train:
                        #select valid relationship
                        for i in range(min(10,len(entity_link))):
                            if down:
                                head_id, tail_id = random.choice(entity_link)
                                prompt_id = head_id
                                response_id = tail_id
                            else:
                                head_id, tail_id = random.choice(unrolled_entity_link)
                                prompt_id = tail_id
                                response_id = head_id
                            if prompt_id is not None:
                                break
                    else:
                        #select random entitiy
                        if self.train:
                            if len(raw_entities)==0:
                                continue
                            prompt_id = random.randrange(len(entities))
                        else:
                            prompt_id = instance

                        response_id = None
                        if down:
                            for head,tail in entity_link:
                                if head==prompt_id:
                                    response_id = tail
                                    break
                        else:
                            for head,tail in unrolled_entity_link:
                                if tail==prompt_id:
                                    response_id = head
                                    break



                    if prompt_id is None:
                        continue
                    entity = entities[prompt_id]

                    question = 'linkdown-' if down else 'linkup-'
                    if response_id is None:
                        linked = None
                    else:
                        if not isinstance(response_id,(tuple,list)):
                            response_id = [response_id]

                        linked = [entities[lei] for lei in response_id]

                cls = entity.cls 
                

                if random.random()<0.333 or not self.train: #full info for eval
                    question+='both'
                    prompt_text = self.selectPartTextForInput(entity.text)
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                    mask=True
                elif random.random()<0.5:
                    question+='text'
                    prompt_text = self.selectPartTextForInput(entity.text)
                    inmask = []
                    mask=False
                else:
                    question+='box'
                    prompt_text=''
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                    mask=True

                question+='~'

                ids = [line.bbid for line in entity.lines]
                outmask = []
                if linked is None or len(linked)==0:
                    response_text=self.blank_token
                else:

                    linked_entities=[]
                    for next_entity in linked:
                        linked_entities.append(next_entity.text)
                        ids += [line.bbid for line in next_entity.lines]
                        outmask += [self.convertBB(s,line.box) for line in next_entity.lines]

                    response_text = '|'.join(linked_entities)

                response_text = '['+cls+']'+response_text
                self.qaAdd(q_a_pairs,question+prompt_text,response_text,ids,inmask,outmask)

            elif q_type == 'class-link':
                if self.train:
                    if len(raw_entities)==0:
                        continue
                    ei = random.randrange(len(raw_entities))
                else:
                    ei = instance
                entity = raw_entities[ei]
                cls = entity.cls[0] #first character for compression

                #g0 get with str+mask
                #gs get with str
                #gm get with mask
                #z0 hl with str+mask
                #zs hl with str
                #zm highlight with mask
                

                if (self.train and random.random()<0.5) or switch: 
                    highlight=True
                    question='z'
                else:
                    highlight=False
                    question='g'

                if random.random()<0.333 or not self.train: #full info for eval
                    question+='0'
                    prompt_text = self.selectPartTextForInput(entity.text)
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                    mask=True
                elif random.random()<0.5:
                    question+='s'
                    prompt_text = self.selectPartTextForInput(entity.text)
                    inmask = []
                    mask=False
                else:
                    question+='m'
                    prompt_text=''
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                    mask=True

                if ei not in raw_entity_dict:
                    question+='~'
                    response_text=self.blank_token
                    outmask=[]
                    ids = [line.bbid for line in entity.lines]
                else:
                    
                    if highlight:
                        question+='~'
                        response_text=str(len( raw_entity_dict[ei]))
                        outmask = []
                        ids = [line.bbid for line in entity.lines]
                        for other_ei in raw_entity_dict[ei]:
                            outmask += [self.convertBB(s,line.box) for line in raw_entities[other_ei].lines]
                            ids += [line.bbid for line in raw_entities[other_ei].lines]
                    else:
                        #step through
                        num = len( raw_entity_dict[ei])
                        if num>1:
                            i = random.randrange(num+1)
                        elif random.random()<0.01:
                            i = 1
                        else:
                            i = 0
                        next_entity = raw_entities[raw_entity_dict[ei][i]] if i<num else None

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
                            if len(response_text)<self.max_qa_len_out:
                                response_text = response_text[:self.max_qa_len_out]
                            ids = [line.bbid for line in entity.lines+next_entity.lines]
                        else:
                            prev_entity = raw_entities[raw_entity_dict[ei][i-1]]
                            if len(prompt_text)>-1+self.max_qa_len_in//2:
                                prompt_text = self.selectPartTextForInput(prompt_text,length=-1+self.max_qa_len_in//2)
                            next_part_len = self.max_qa_len_in-(1+len(prompt_text))
                            prompt_text = prompt_text+'>'+self.selectPartTextForInput(prev_entity.text,next_part_len)
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
                if len(entities)==0:
                    continue
                entity = random.choice(entities)
                cls = entity.cls
                class_answer = '[ '+cls+' ]'
                line = random.choice(entity.lines)
                text=line.text
                if self.max_qa_len_in is not None and len(text)>self.max_qa_len_in:
                    if random.random()<0.1:
                        text = text[:self.max_qa_len_in]
                    else:
                        text = self.selectPartTextForInput(text)

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
                prompt_id = None
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
                            #prompt_text = self.selectPartTextForInput(prev_text)
                        response_id = response_id[i]
                    else:
                        i = 0
                        num_resp = 1
                        char='~'
                        response_start = '1>' if down else ''

                    response_text = entities[response_id].text
                    if not down and not self.words:
                        response_text = response_text[::-1]
                    if len(response_text)>self.max_qa_len_out:
                        response_text = response_text[:self.max_qa_len_out]
                    elif len(response_text)+1<=self.max_qa_len_out:
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
                prompt_text = self.selectPartTextForInput(prompt_text)

                inmask = []
                ambiguous = len(entities[prompt_id].lines)==1 and entities[prompt_id].lines[0].ambiguous
                mask =  random.random()<0.5 or ambiguous#should we use query mask
                if down and (entities[response_id].cls=='answer' if response_id is not None else entities[prompt_id].cls=='question'):
                    question = 'l0'+char if mask else 'l'+char
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
                        if (random.random()>0.5 and len(table.row_headers)>0) or len(table.col_headers)==0:
                            if len(table.row_headers)==0:
                                continue
                            headers = table.row_headers
                        else:
                            headers = table.col_headers
                        for i in range(10):
                            header = random.choice(headers)
                            if header is not None:
                                break
                        if header is None:
                            continue
                        header_text = self.selectPartTextForInput(header.text,length=-1+self.max_qa_len_in//2)
                    else:
                        header_text = self.sampleText(-1+self.max_qa_len_in//2)
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
                elif sub_type == 'full-all-col':
                    question = 'full_col~{}' if random.random()<0.5 else 'full_col0~{}' 
                elif sub_type == 'full-all-row':
                    question = 'full_row~{}' if random.random()<0.5 else 'full_row0~{}' 
                elif sub_type == 'full-list-row-headers':
                    question = 'list_row_headers~{}'
                    prompt_text = len(tables)
                elif sub_type == 'full-list-col-headers':
                    question = 'list_column_headers~{}'
                    prompt_text = len(tables)
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
                elif sub_type == 'class-link-all':
                    question = 'link-text~{}'
                elif sub_type == 'class-linkdown-all':
                    question = 'linkdown-text~{}'
                elif sub_type == 'class-linkup-all':
                    question = 'linkup-text~{}'
                try:
                    self.qaAdd(q_a_pairs,question.format(prompt_text),self.np_token,[],[],[])
                except ValueError as er:
                    print(er)
                    print('question is {}'.format(question))
                    continue
                except KeyError as er:
                    print(er)
                    print('question is {}'.format(question))
                    print('prompt_text is {}'.format(prompt_text))

                    continue


            elif q_type=='read':
                #finish reading entity
                if self.train:
                    if len(entities)>0:
                        for i in range(10):
                            entity = random.choice(entities)
                            text = entity.text
                            if len(text)>2:
                                break
                    else:
                        continue
                else:
                    entity = instance
                    text = entity.text
                if len(text.split())<2: #force multiple words
                    continue

                if (self.train and random.random()<0.5) or switch or self.words:
                    forward=True
                else:
                    forward=False
                    text = text[::-1]

                if self.train:
                    last_space = max(text.rfind(' '),text.rfind('\\'))
                    if last_space==-1: #no space
                        last_space = random.randrange(1,len(text)-1)
                    query_part = text[:last_space] #so there's at least one word to predict
                    query,q_start = self.selectPartTextForInput(query_part,ret_start=True)
                else:
                    first_newline = text.find('\\')
                    if first_newline>-1:
                        query,q_start = self.getBackText(text[:first_newline],ret_start=True)
                    else:
                        last_space = text.rfind(' ')
                        if last_space==-1: #no space
                            last_space = random.randrange(1,len(text)-1)
                        query_part = text[:last_space] #so there's at least one word to predict
                        query = self.getFrontText(query_part,query=True)
                        q_start = 0
                if len(query)==0:
                    continue
                q_end = q_start + len(query)


                if text[q_end]==' ':
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


                if random.random()<0.7 and not ambiguous and len(query)>6 and self.train:
                    question='fi~' if forward else 'pr~' #for "finish" or "previous"
                    inmask=[]
                #elif random.random()<0.5 and self.train:
                else:
                    question='f0~' if forward else 'p0~'
                    #This uses the mask of the whol entity, as that is what other things will produce
                    inmask = [self.convertBB(s,line.box) for line in entity.lines]
                #else:
                #    question='f1~' if forward else 'p1~'
                #    #This uses the mask of only the query lines
                #    inmask = [self.convertBB(s,entity.lines[li].box) for li in query_line_ids]

                ####FIX for getting the right tokens
                query_tokens = self.tokenizer(query,return_tensors="pt")['input_ids']
                tok_len = query_tokens.shape[1] +1#for task token
                if tok_len>self.max_q_tokens:
                    over = tok_len-self.max_q_tokens
                    query_tokens = query_tokens[0,:-(over+1)] #+1 becuase trimming the end (SEP) token doesn't count
                    query = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(query_tokens,skip_special_tokens=True))
                #######
                self.qaAdd(q_a_pairs,question+query,response,ids,inmask,outmask)
            elif q_type=='readline':
                if self.train:
                    if len(entities)>0:
                        entity = random.choice(entities)
                    else:
                        continue
                else:
                    entity = instance

                line = random.choice(entity.lines)
                text = line.text

                ids = [line.bbid]
                inmask = [self.convertBB(s,line.box)]

                self.qaAdd(q_a_pairs,'w0>',text,ids,inmask)


            elif q_type=='cell':
                table = random.choice(tables)
                r,r_header = random.choice(list(enumerate(table.row_headers)))
                c,c_header = random.choice(list(enumerate(table.col_headers)))

                r_h_text = r_header.text if r_header is not None else self.blank_token
                c_h_text = c_header.text if c_header is not None else self.blank_token

                r_h_text = self.selectPartTextForInput(r_h_text,-1+self.max_qa_len_in//2)
                c_h_text = self.selectPartTextForInput(c_h_text,-1+self.max_qa_len_in//2)

                cell = table.cells[r][c]
                if cell is not None:
                    cell_text = cell.text
                    if len(cell_text) > 0:
                        cell_text = self.getFrontText(cell_text)
                    elif len(cell_text)+1<=self.max_qa_len_out:
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

                cell_text = self.selectPartTextForInput(cell.text)

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


            elif q_type in ('full-all-row', 'full-all-col'):
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='full-all-row':
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
                ids=[line.bbid for line in header.lines]
                header_text = header.text
                for cell in all_cells:
                    if cell is not None:
                        ids += [line.bbid for line in cell.lines]
                        outmask += [self.convertBB(s,line.box) for line in cell.lines]

                response = '|'.join((cell.text if cell is not None else self.blank_token) for cell in all_cells) + self.end_token

                if random.random()<0.5 or ambiguous:
                    inmask = [self.convertBB(s,line.box) for line in header.lines]
                    question = 'full_row0~' if row else 'full_col0~'
                else:
                    inmask = []
                    question = 'full_row~' if row else 'full_col~'


                self.qaAdd(q_a_pairs,question+header_text,response,ids,inmask,outmask)

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
                    header_text = self.selectPartTextForInput(header.text)
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

                    header_text = self.selectPartTextForInput(header.text)
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
                        



            elif q_type in ('full-list-row-headers','full-list-col-headers'):
                table_i = random.randrange(len(tables))
                table = tables[table_i]

                if q_type=='full-list-row-headers':
                    row=True
                    headers = table.row_headers
                else:
                    row=False
                    headers = table.col_headers
                
                ids=[]
                outmask = []
                for header in headers:
                    if header is not None:
                        ids+=[line.bbid for line in header.lines]
                        outmask += [self.convertBB(s,line.box) for line in header.lines]

                response = '|'.join((h.text if h is not None else self.blank_token) for h in headers) + self.end_token

                inmask = []
                question = 'list_row_headers~' if row else 'list_column_headers~'
                question += str(table_i)
            

                self.qaAdd(q_a_pairs,question,response,ids,inmask,outmask)

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
                        prev_text = self.selectPartTextForInput(prev_header.text)
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
                    if len(table.cells)>0:
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
                
                if len(table.cells)>0:
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
            
            if len(q_a_pairs)>10*self.questions and self.train:
                break #we have enough
        
        return q_a_pairs

    def selectPartTextForInput(self,text,length=None,ret_start=False):
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

    def getFrontText(self,text,list_split=False,term=None,query=False):
        #get the front part of the text, breaking on words
        if not query:
            length = self.max_qa_len_out
        else:
            length = self.max_qa_len_in
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
    def getBackText(self,text,ret_start):
        #get the back part of the text, breaking on words
        length = self.max_qa_len_in
        if len(text)>length:
            start = -length
            while start<-1 and text[start-1]!=' ' and text[start-1]!='\\':
                start+=1
            if start==-1:
                #couldn't break
                ret = text[-length:]
                start = len(text)-length
            else:
                ret = text[start:]
                start = len(text)+start

        else:
            ret = text
            start=0
        if ret_start:
            return ret,start
        else:
            return ret

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

        para = random.choice(getWikiArticle(dataset=self.wiki_dataset)) #select random paragraph

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
            text = self.selectPartTextForInput(text)
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
            
            new_link_dict[e1]=sorted_e2s
        return new_link_dict

    def makeJsonText(self,entities,entity_link,tables,proses=None,minored_fields=None):
        #spits out json with all structure


        #entities = [Entity(e) for e in entities]
        claimed_by={} #used later to find which entities aren;t claimed

        #First, sort entities into read order
        #  tables are going to become an entity.
        table_entities=[]
        table_map={}
        for table_id,table in enumerate(tables):

            all_table_entities = []
            if table.row_headers is not None:
                all_table_entities += table.row_headers
            if table.col_headers is not None:
                all_table_entities += table.col_headers
            for row in table.cells:
                all_table_entities += row
            table_entities+=all_table_entities
            for e in all_table_entities:
                for i,e2 in enumerate(entities):
                    if e==e2:
                        table_map[i]=table_id+len(entities)
                        break
        entities = entities+tables

        if proses is not None:
            for prose in proses:
                table_entities+=prose.entities
            entities+=proses
        if minored_fields is not None:
            for minored_field in minored_fields:
                if minored_field.question is not None:
                    table_entities.append(minored_field.question)
                table_entities+=minored_field.answers+minored_field.minors
            entities+=minored_fields

        
        old_entities = entities
        #table_ids = list(range(len(entities)-len(tables),len(entities)))
        
        entities = [(i,e) for i,e in enumerate(entities) if e not in table_entities] #remove entities in tabes (they are accounted for as we added the tables)
        #non_table_entities = []
        #for i,e in enumerate(entities):
        #    match=False
        #    if not isinstance(e,Table):
        #        for te in table_entities:
        #            if e==te or (e.text==te.text and e.getBox()[0]==te.getBox()[0] and e.getBox()[1]==te.getBox()[1]):
        #                match=True
        #                break
        #        #if (not match) and e.text=='<<Physical Characteristics>> Total Pressure Drop (encap.)':
        #            #import pdb;pdb.set_trace()
        #    if not match:
        #        non_table_entities.append((i,e))
        #entities = non_table_entities
        entities.sort(key=lambda a:a[1].getBox()[1]) #sort by y positions
        old_to_new={}
        new_entities=[]
        i=0
        while i<len(entities):
            gi,entity = entities[i]
            #We want to reorder things so evertthing on a (rougly) paralele line is processed left to right

            #first get the average line height
            if isinstance(entity,Entity):
                h_sum=0
                for line in entity.lines:
                    h_sum += line.getBox()[3]-line.getBox()[1]
                avg_line_h = h_sum/len(entity.lines)
            elif isinstance(entity,MinoredField):
                if entity.question is not None:
                    h_sum=entity.question.getBox()[3]-entity.question.getBox()[1]
                    h_count=1
                else:
                    h_sum=0
                    h_count=0
                for e in entity.answers+entity.minors:
                    for line in e.lines:
                        h_sum += line.getBox()[3]-line.getBox()[1]
                        h_count += 1
                avg_line_h = h_sum/h_count
            elif isinstance(entity,FillInProse):
                h_sum=0
                h_count=0
                for e in entity.entities:
                    for line in e.lines:
                        h_sum += line.getBox()[3]-line.getBox()[1]
                        h_count += 1
                avg_line_h = h_sum/h_count
            else: #Table
                h_sum = 0
                h_count = 0
                if entity.row_headers is not None:
                    for header in entity.row_headers:
                        if header is not None:
                            for line in header.lines:
                                h_sum += line.getBox()[3]-line.getBox()[1]
                                h_count +=1
                if entity.col_headers is not None:
                    for header in entity.col_headers:
                        if header is not None:
                            for line in header.lines:
                                h_sum += line.getBox()[3]-line.getBox()[1]
                                h_count +=1
                avg_line_h = h_sum/h_count if h_count>0 else 200

            #and use that to set the "rougly parallel line"
            sort_top,sort_bot = entities[i][1].getSortTopBot()
            y_min = sort_top-(avg_line_h*0.4)
            y_max = sort_bot+(avg_line_h*0.4)
            #print('y_min={}, y_max={},  entity={}'.format(y_min,y_max,entity.getBox()))

            #get all entities (following this one) that fall into that line
            j=i+1
            while j<len(entities) and entities[j][1].getSortTopBot()[0]>y_min and entities[j][1].getSortTopBot()[1]<y_max:
                j+=1
            do_this = True #no reorder, can add this entity
            if j>i+1:
                #Go through them and add any before this entity (horizontally) to the "before" ones
                before_entries=[]
                after_entries=[] #entities in range [i+1,j) that aren't moved before
                for sub_i in range(i+1,j):
                    other_entity = entities[sub_i][1]
                    if other_entity.getBox()[0]<entity.getBox()[0]:
                        before_entries.append(entities[sub_i])
                    else:
                        after_entries.append(entities[sub_i])
                
                if len(before_entries)>0: 
                    #Need to reorder
                    do_this = False #don't add this entity, wait and reevaluate the new first element
                    entities = before_entries+entities[i:i+1]+after_entries+entities[j:]
                    i=0 #reset iterator

            if do_this:
                old_to_new[gi]=len(new_entities)
                ###DEBUG
                #try:
                #    if entity.text == '6. COUNTRY OF\\CITIZENSHIP':
                #        import pdb; pdb.set_trace()
                #except:
                #    pass
                new_entities.append(entity)
                i+=1
     
        
        
        new_entity_link = []
        for head,tail in entity_link:
            if head not in old_to_new:
                continue
            head = old_to_new[head]
            if tail is None:
                pass
            elif isinstance(tail,(list,tuple)):
                #checny(k if any tail is in table
                #then there are two cases, header->subheader or a table header
                part_of_table = False
                not_in_table = []
                in_table = []
                for t in tail:
                    if t in table_map:
                        part_of_table = True
                        table_id = table_map[t]
                        in_table.append(t)
                        #break
                    else:
                        not_in_table.append(t)

                if not part_of_table:
                    tail = [old_to_new[t] for t in tail if t in old_to_new]
                    if len(tail)==0:
                        tail=None
                else:
                    table = old_entities[table_id]

                    #edit the table in include not-in-table entities
                    # should they be row or header?
                    is_row = any([old_entities[t] in table.row_headers for t in in_table])
                    is_col = any([old_entities[t] in table.col_headers for t in in_table])

                    assert is_row or is_col

                    if is_row and not is_col:
                        for t in not_in_table:
                            table.addRowHeader(old_entities[t])
                    elif not is_row and is_col:
                        for t in not_in_table:
                            table.addColHeader(old_entities[t])
                    else:
                        for t in not_in_table:
                            table.addHeader(old_entities[t])
                    table_entities += [old_entities[t] for t in tail]


                    #Is this a whole row/col/table super-header?
                    table_header = False
                    if len(tail)>=len(table.row_headers) and len(table.row_headers)>0:
                        table_header = True
                        tail_entities = [old_entities[t] for t in tail]
                        for r_h in table.row_headers:
                            if r_h not in tail_entities:
                                table_header = False
                                break

                    if not table_header and len(tail)>=len(table.col_headers):
                        table_header = True
                        tail_entities = [old_entities[t] for t in tail]
                        for c_h in table.col_headers:
                            if c_h not in tail_entities:
                                table_header = False
                                break

                    new_table_id = old_to_new[table_id]
                    
                    if table_header:
                        new_entity_link.append((head,new_table_id))
                    else:
                        #we assume this is a overheader over a couple col/row headers
                        #we'll just add extra text to each
                        head_text = new_entities[head].text
                        for t in tail:
                            #assert old_entities[t].cls == 'question'
                            old_entities[t].text = '<<'+head_text+'>> '+old_entities[t].text
                        claimed_by[head]=new_table_id #the table, which I don't have the index for
                        
                        
                    #not adding normal link
                    continue

            elif tail in old_to_new:
                tail = old_to_new[tail] if tail in old_to_new else None
            elif new_entities[head].cls=='header':
                #this actually is part of a table?
                possible_header = old_entities[tail]
                found=False
                for table in tables:
                    for header in table.row_headers+table.col_headers:
                        if header==possible_header:
                            header.text = '<<'+new_entities[head].text+'>> '+header.text
                            found=True
                            break
                    if found:
                        break
                if not found:
                    continue
            else:
                print('unhandeled case, probably {} {} is in a table'.format(tail,old_entities[tail]))
                import pdb;pdb.set_trace()
                print('ERROR')
            if tail is not None:
                new_entity_link.append((head,tail))
        #assert len(new_entity_link) == len(entity_link) or len(tables)>0 or len(proses)>0 or len(minored_fields)>0
        entity_link = new_entity_link
        entities = new_entities
        
        #display
        #entities = new_entities
        #test_img = np.zeros((1000,1000,3),dtype=np.uint8)
        #for i,entity in enumerate(entities):
        #    print('{} {}'.format(i,entity.text))
        #    color = (random.randrange(256),random.randrange(256),random.randrange(256))
        #    while sum(color)<150:
        #        color = (random.randrange(256),random.randrange(256),random.randrange(256))
        #    img_f.rectangle(test_img,entity.box[:2],entity.box[4:],color)
        #    #dots
        #    x= round(entity.box[0]+2)
        #    y= round(entity.box[1]+2)
        #    for j in range(i):
        #        if (j+1)%5==0:
        #            #cross
        #            img_f.line(test_img,(x-8,y),(x-2,y+4),color)
        #        else:
        #            test_img[y:y+5,x] = color
        #        x+=2
        #img_f.imshow('x',test_img)
        #img_f.show()

        link_dict = {k:v for k,v in entity_link}
        #link_dict = defaultdict(list)
        #for k,v in entity_link:
        #    link_dict[k].append(v)

        #First we need to identiy the "heads", which are unclained entities
        for ei,child in entity_link:
            if isinstance(child,(list,tuple)):
                for ch in child:
                    claimed_by[ch]=ei
            elif child is not None:
                claimed_by[child]=ei
            
        #full={}
        doc=[] #do list of tuples (list) to allow duplicate keys

        for ei,entity in enumerate(entities):

            if ei not in claimed_by and entity not in table_entities:
                children = self.getChildren(ei,entities,link_dict)
                #full[entities[ei]]=children
                if isinstance(entities[ei],Table) or isinstance(entities[ei],FillInProse) or isinstance(entities[ei],MinoredField):
                    doc.append(children)

                elif entities[ei].cls == 'header':
                    doc.append(formatHeader(entities[ei],children))
                    #if children is not None:
                    #    doc.append({'header':entities[ei],'content':children})
                    #else:
                    #    doc.append({'header':entities[ei]})
                else:
                    if children is not None:
                        assert entities[ei].cls=='question'
                        doc.append(formatQuestion(entities[ei],children))
                        #if len(children)>0:
                        #    doc.append({'question':entities[ei],'answers':children})
                        #else:
                        #    doc.append({'question':entities[ei]})
                    else:
                        #assert entities[ei].cls=='other'
                        doc.append(formatOther(entities[ei]))

        #if len(doc)==0:
        #    import pdb;pdb.set_trace()
        #TEST
        #found=False
        #for ele in doc:
        #    if old_entities[0].text in ele:
        #        found=True
        #        break
        #if not found:
        #    print('Missing entity')
        #    import pdb;pdb.set_trace()
        if self.shorten_text_in_json:
            doc = self.shortenElement(doc)
        return json.dumps(doc,ensure_ascii=False,default=lambda a:a.text)+self.end_token

    def shortenElement(self, ele):
        if isinstance(ele,str):
            new_lines=[]
            lines = ele.split('\\')
            w_count = 0
            new_lines=[]
            for line in lines:
                words = line.split(' ')
                new_words=words[:self.max_json_words-w_count]
                w_count+=len(new_words)
                new_lines.append(' '.join(new_words))
                if w_count>=self.max_json_words:
                    break


            return '\\'.join(new_lines)
        elif isinstance(ele,Entity):
            return self.shortenElement( ele.text)
        elif isinstance(ele,list):
            return [self.shortenElement(lv) for lv in ele]
        elif ele is None:
            return None

        assert isinstance(ele,dict)
        new_ele={}
        for k,v in ele.items():
            k=self.shortenElement(k)
            v = self.shortenElement(v)
            new_ele[k]=v
        return new_ele

    def getChildren(self,ei,entities,link_dict):
        if isinstance(entities[ei],Table):
            ret = formatTable(entities[ei])
        elif isinstance(entities[ei],FillInProse):
            ret = {}
            for e in entities[ei].entities:
                ret[e.text]=e.cls #dict preserves order
        elif isinstance(entities[ei],MinoredField):
            ret = {}
            if entities[ei].question is not None:
                ret[entities[ei].question.text]='question'
            if len(entities[ei].answers)>0:
                ret['answers']=entities[ei].answers
            if len(entities[ei].minors)>0:
                ret['subprompt']=entities[ei].minors

        elif ei in link_dict:
            children = link_dict[ei]

            if children is None:
                children = []
            if not isinstance(children,(list,tuple)):
                children = [children]
            new_children =[]
            for chd in children:
                if not entities[chd].used:
                    new_children.append(chd)
                #else:
                #    print("WARNING : prevented child which had already been used")
            children = new_children
            #assert isinstance(children,list)
            if entities[ei].cls=="header":
                #ret = {}
                ret = []
                for child in children:
                    if child is not None:
                        #assert entities[child].cls=='question' or 
                        next_children = self.getChildren(child,entities,link_dict)
                        if isinstance(entities[child],Table):
                            ret=next_children
                        elif entities[child].cls=='header':
                            ret.append(formatHeader(entities[child],next_children))
                            #if next_children is not None:
                            #    ret.append({'header':entities[child],'content':next_children})
                            #else:
                            #    ret.append({'header':entities[child]})
                        elif entities[child].cls=='question':
                            if not entities[child].used:
                                ret.append(formatQuestion(entities[child],next_children))
                            #else:
                            #    print("WARNING : prevented question entity from being added twice to JSON")
                        else:
                            ret.append(formatOther(entities[child]))
            elif entities[ei].cls=="question":

                ret = []
                for child in children:
                    if child is not None:
                        #assert entities[child].cls=='answer'
                        entities[child].used=True
                        ret.append(entities[child])
            else:
                assert children is None
                ret = entities[ei].text
        else:
            ret = None
        return ret
 
#This formatting is specifically choosen so the autoregressive predicts the text and than the class
def formatHeader(entity,children):
    assert not entity.used
    entity.used = True
    #return {'text':entity,
    #        'header content': children if children is not None else []
    #        }
    ret = {entity.text:'header'}
    if children is not None:
        ret['content']=children
    return ret
def formatQuestion(entity,children):
    if children is not None and len(children) == 1 and isinstance(children[0],Table):
        #unsusual circumstance with NAF dataset
        #make the question a header
        entity.cls='header'
        return formatHeader(entity,[formatTable(children[0])])
    assert not entity.used
    entity.used = True
    #return {'text':entity,
    #        'question answers': children if children is not None else []
    #        }
    ret = {entity.text:'question'}
    if children is not None and len(children)>0:
        ret['answers']=children
    return ret
def formatOther(entity):
    assert not entity.used
    entity.used = True
    #return {'text':entity}
    return {entity.text:entity.cls}
def formatTable(entity):
    ret = {
            'row headers': entity.row_headers,
            'column headers': entity.col_headers,
         }
    if len(entity.cells)>0: #because we forgot to transcribe table cells for NAF dataset
        ret['cells']= entity.cells
    return ret
