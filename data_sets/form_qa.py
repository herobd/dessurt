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
                        start = random.randrange(len(prompt_text)-self.max_qa_len)
                        prompt_text = prompt_text[start:start+self.max_qa_len]
                
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

            elif q_type=='cell':
                table = random.choice(tables)
                r = random.randrange(len(table.row_headers))

            elif q_type=='row-header' or q_type=='col-header':
                TODO

            elif q_type=='list-row-headers' or q_type=='list-col-headers':
                TODO
            elif q_type=='count-tables':
                TODO
                self.qaAdd(q_a_pairs,'t#>',str(len(tables)),tables_ids,[],outmask)
            elif q_type=='highlight-table':
                table_i = random.randrange(len(tables))
                table = tables[table_i]
                table_ids=[]
                outmask=[]
                for header in table.row_headers:
                    for line in entities[header]['lines']:


                self.qaAdd(q_a_pairs,'0t~{}'.format(table_i),'',table_ids,[],outmask)

            trans_bb.sort(key=lambda a:a[0] )
            trans=trans_bb[0][1]V
            for y,t in trans_bb[1:]:
                trans+='\\'+t
            all_trans[gi]=trans
            #print('c:{} {},{} full group trans: {}'.format(cls,bbs[group[0],0],bbs[group[0],1],trans))

            if self.index_class_map[cls] == 'question':
                questions_gs.add(gi)
            elif self.index_class_map[cls] == 'answer':
                answers_gs.add(gi)
            elif self.index_class_map[cls] == 'header':
                headers_gs.add(gi)
            else:
                others_gs.add(gi)

            if self.char_qs=='full' or self.char_qs=='sym':
                #classify all together
                text=trans
                if self.max_qa_len is not None and len(text)>self.max_qa_len:
                    text = text[:self.max_qa_len]
                class_qs.append(('cs~{}'.format(text),class_answer,[gi])) #This can be ambigous, although generally the same text has the same class

                all_q_a.append(random.choice(class_qs))

                #complete (read)
                self.addRead(all_q_a,trans)
                self.addRead(all_q_a,trans,backwards=True)


                    if self.max_qa_len is not None and len(chdr)>self.max_qa_len//2:
                        chdr = chdr[-self.max_qa_len//2:]
                    val = all_trans[v]
                    if self.char_qs=='sym':
                        if self.max_qa_len is not None and len(val)>self.max_qa_len:
                            val = val[:self.max_qa_len]
                        elif len(val)+1<=self.max_qa_len:
                            val += self.end_token
                    else:
                        if self.max_qa_len is not None and len(val)>self.max_qa_len:
                            val = val[:self.max_qa_len-2]+'>>'

                    all_q_a.append(('t~{}~~{}'.format(rhdr,chdr),val,[col_h,row_h,v]))
                    all_q_a.append(('t~{}~~{}'.format(chdr,rhdr),val,[col_h,row_h,v]))
                else:
                    all_q_a.append(('value of "{}" and "{}"?'.format(all_trans[row_h],all_trans[col_h]),all_trans[v],[col_h,row_h,v]))
                    all_q_a.append(('value of "{}" and "{}"?'.format(all_trans[col_h],all_trans[row_h]),all_trans[v],[col_h,row_h,v]))
            if all_trans[v] not in ambiguous:
                if row_h is not None:
                    if self.char_qs:
                        rhdr = all_trans[row_h]
                        if self.max_qa_len is not None and len(rhdr)>self.max_qa_len:
                            rhdr = rhdr[-self.max_qa_len:]
                        val = all_trans[v]
                        if self.char_qs=='sym':
                            if self.max_qa_len is not None and len(val)>self.max_qa_len:
                                val = val[:self.max_qa_len]
                            elif len(val)+1<=self.max_qa_len:
                                val += self.end_token
                        else:
                            if self.max_qa_len is not None and len(val)>self.max_qa_len:
                                val = val[:self.max_qa_len-2]+'>>'
                        all_q_a.append(('ri~{}'.format(val),rhdr,[v,row_h]))
                    else:
                        all_q_a.append(('row that "{}" is in?'.format(all_trans[v]),all_trans[row_h],[v,row_h]))
                if col_h is not None:
                    if self.char_qs:
                        chdr = all_trans[col_h]
                        if self.max_qa_len is not None and len(chdr)>self.max_qa_len:
                            chdr = chdr[-self.max_qa_len:]
                        val = all_trans[v]
                        if self.char_qs=='sym':
                            if self.max_qa_len is not None and len(val)>self.max_qa_len:
                                val = val[:self.max_qa_len]
                            elif len(val)+1<=self.max_qa_len:
                                val += self.end_token
                        else:
                            if self.max_qa_len is not None and len(val)>self.max_qa_len:
                                val = val[:self.max_qa_len-2]+'>>'
                        all_q_a.append(('ci~{}'.format(val),chdr,[v,col_h]))
                    else:
                        all_q_a.append(('column that "{}" is in?'.format(all_trans[v]),all_trans[col_h],[v,col_h]))

            x,y = bbs[groups[v][0],0:2]
            if col_h is not None:
                col_vs[col_h].append((v,y))
            if row_h is not None:
                row_vs[row_h].append((v,x))

        for row_h, vs in row_vs.items():
            trans_row_h = all_trans[row_h]
            if trans_row_h not in ambiguous:
                vs.sort(key=lambda a:a[1])
                a=all_trans[vs[0][0]]
                for v,x in vs[1:]:
                    a+='|'+all_trans[v]
                if self.char_qs:
                    if self.max_qa_len is not None and len(trans_row_h)>self.max_qa_len:
                        trans_row_h = trans_row_h[-self.max_qa_len:]
                    if self.max_qa_len is not None and len(a)>self.max_qa_len:
                        self.breakLong(all_q_a,a,'ar~{}'.format(trans_row_h),'ar>')
                    else:
                        all_q_a.append(('ar~{}'.format(trans_row_h),a,[row_h,vs[0][0]]))
                else:
                    all_q_a.append(('all values in row "{}"?'.format(trans_row_h),a,[row_h,vs[0][0]]))
        for col_h, vs in col_vs.items():
            trans_col_h = all_trans[col_h]
            if trans_col_h not in ambiguous:
                vs.sort(key=lambda a:a[1])
                a=all_trans[vs[0][0]]
                for v,y in vs[1:]:
                    a+='|'+all_trans[v]
                if self.char_qs: 
                    if self.max_qa_len is not None and len(trans_col_h)>self.max_qa_len:
                        trans_col_h = trans_col_h[-self.max_qa_len:]
                    if self.max_qa_len is not None and len(a)>self.max_qa_len:
                        self.breakLong(all_q_a,a,'ac~{}'.format(trans_col_h),'ac>')
                    else:
                        all_q_a.append(('ac~{}'.format(trans_col_h),a,[col_h,vs[0][0]]))
                else:
                    all_q_a.append(('all values in column "{}"?'.format(trans_col_h),a,[col_h,vs[0][0]]))

        if self.char_qs:
            all_q_a.append(('t#>',str(len(tables)),list(col_vs.keys())+list(row_vs.keys())))
            for i,(col_hs, row_hs) in enumerate(tables.values()):
                col_hs = [(h,bbs[groups[h][0]][0]) for h in col_hs]
                col_hs.sort(key=lambda a:a[1])
                col_hs = [h[0] for h in col_hs]
                col_h_strs = [all_trans[h] for h in col_hs]
                row_hs = [(h,bbs[groups[h][0]][1]) for h in row_hs]
                row_hs.sort(key=lambda a:a[1])
                row_hs = [h[0] for h in row_hs]
                row_h_strs = [all_trans[h] for h in row_hs]
                
                col_h_strs='|'.join(col_h_strs)
                row_h_strs='|'.join(row_h_strs)
                if self.max_qa_len is not None and len(col_h_strs)>self.max_qa_len:
                    self.breakLong(all_q_a,col_h_strs,'ch~{}'.format(i),'ch>')
                else:
                    all_q_a.append(('ch~{}'.format(i),col_h_strs,col_hs))
                if self.max_qa_len is not None and len(row_h_strs)>self.max_qa_len:
                    self.breakLong(all_q_a,row_h_strs,'rh~{}'.format(i),'rh>')
                else:
                    all_q_a.append(('rh~{}'.format(i),row_h_strs,row_hs))




        #Convert the group IDs on each QA pair to be BB IDs.
        #   This uses groups_id, which can be the word BB ids
        new_all_q_a =[]
        for q,a,group_ids in all_q_a:
            if group_ids is not None:
                bb_ids=[]
                for gid in group_ids:
                    bb_ids+=groups_id[gid]
            else:
                bb_ids=None
            self.qaAdd(new_all_q_a,q,a,bb_ids)
            if self.max_qa_len is not None:
                assert len(q)<self.max_qa_len+5
        return new_all_q_a


    def corrupt(self,s):
        new_s=''
        for c in s:
            r = random.random()
            if r<self.corruption_p/3:
                pass
            elif r<self.corruption_p*2/3:
                new_s+=random.choice(string.ascii_letters)
            elif r<self.corruption_p:
                if random.random()<0.5:
                    new_s+=c+random.choice(string.ascii_letters)
                else:
                    new_s+=random.choice(string.ascii_letters)+c
            else:
                new_s+=c
        return new_s

    def addRead(self,qa,text,np=False,backwards=False):
        if backwards:
            text = text[::-1]
            prompt = 'bk~{}'
        else:
            prompt = 're~{}'
        if len(text)<=2 or random.random()<0.05:
            start_point=len(text) #so we get [end]s with long texts
        elif len(text)>self.min_start_read+1:
            start_point = random.randrange(self.min_start_read,len(text)+1)
        else:
            start_point = random.randrange(len(text)//2,len(text)+1)
        start_text = text[:start_point].strip()
        finish_text = text[start_point:].strip()
        if len(finish_text)==0:
            finish_text=self.end_token
        if len(start_text)-self.min_start_read*2>0 and random.random()>0.33:
            real_start = random.randrange(0,len(start_text)-self.min_start_read*2)
            start_text = start_text[real_start:]

        if self.max_qa_len is not None:
            if len(start_text) > self.max_qa_len:
                start_text = start_text[-self.max_qa_len:]
            if len(finish_text) > self.max_qa_len:
                if self.char_qs=='sym':
                    finish_text = finish_text[:self.max_qa_len]
                    if len(finish_text)+1 > self.max_qa_len:
                        finish_text += self.end_token
                else:
                    finish_text = finish_text[:self.max_qa_len-2]+'>>'
        if np:
            qa.append((prompt.format(start_text),self.np_token,None))
        else:
            qa.append((prompt.format(start_text),finish_text,None))

    #break a long answer THAT ISN'T lines on the page into multiple QAs
    def breakLong(self,qa,full,initial_prompt,continue_prompt):
        if self.max_qa_len is not None and len(full)>self.max_qa_len:
            if self.do_masks:
                first_part = full[:self.max_qa_len]
                self.qaAdd(qa,initial_prompt,first_part)
                prev_part = first_part
                remainder = full[self.max_qa_len:]
                while len(remainder)>self.max_qa_len:
                    next_part = remainder[:self.max_qa_len]
                    self.qaAdd(qa,continue_prompt+prev_part,next_part)
                    prev_part = next_part
                    remainder = remainder[self.max_qa_len:]
                if len(remainder)+1 < self.max_qa_len:
                    self.qaAdd(qa,continue_prompt+prev_part,remainder+self.end_token)
                else:
                    self.qaAdd(qa,continue_prompt+prev_part,remainder)
                    self.qaAdd(qa,continue_prompt+remainder,self.end_token)
            else:
                first_part = full[:self.max_qa_len-2] + '>>' #mark to indicate not complete
                self.qaAdd(qa,initial_prompt,first_part)
                prev_part = first_part[:-2] #remove mark
                remainder = full[self.max_qa_len-2:]
                while len(remainder)>self.max_qa_len:
                    next_part = remainder[:self.max_qa_len-2] + '>>'
                    self.qaAdd(qa,continue_prompt+prev_part,next_part)
                    prev_part = next_part[:-2]
                    remainder = remainder[self.max_qa_len-2:]
                self.qaAdd(qa,continue_prompt+prev_part,remainder)
        elif len(full)+1<self.max_qa_len and self.do_masks:
          self.qaAdd(qa,initial_prompt,full+self.end_token)
        else:
          self.qaAdd(qa,initial_prompt,full)
          if self.max_qa_len:
              self.qaAdd(qa,continue_prompt+full,self.end_token)



def addTable(tables,table_map,groups,bbs,qi,ais,relationships_q_a,relationships_a_q):
    other_qis=[]
    for ai in ais:
        if len(relationships_a_q[ai])==2:
            q1,q2 = relationships_a_q[ai]
            if q1==qi:
                cls = bbs[groups[q2],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q2][0],0:2]
                other_qis.append((q2,x,y))
            else:
                cls = bbs[groups[q1],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q1][0],0:2]
                other_qis.append((q1,x,y))
        else:
            assert len(relationships_a_q[ai])==1 #blank row/column header. Skipping for now

    other_set = set(q[0] for q in other_qis)
    if len(other_set)<len(other_qis):
        import pdb;pdb.set_trace()
        return #label error
    
    my_qis=[]
    debug_hit=False
    for ai in relationships_q_a[other_qis[0][0]]:
        if len(relationships_a_q[ai])==2:
            q1,q2 = relationships_a_q[ai]
            if q1==other_qis[0][0]:
                if q2 in other_set:
                    return
                cls = bbs[groups[q2],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q2][0],0:2]
                my_qis.append((q2,x,y))
                if q2==qi:
                    debug_hit=True
            else:
                if q1 in other_set:
                    return
                cls = bbs[groups[q1],13:].argmax()
                assert cls == 1
                x,y = bbs[groups[q1][0],0:2]
                my_qis.append((q1,x,y))
                if q1==qi:
                    debug_hit=True
        else:
            assert len(relationships_a_q[ai])==1
    assert debug_hit


    #which are rows, which are cols?
    other_mean_x = np.mean([q[1] for q in other_qis])
    other_mean_y = np.mean([q[2] for q in other_qis])
    my_mean_x = np.mean([q[1] for q in my_qis])
    my_mean_y = np.mean([q[2] for q in my_qis])

    if my_mean_x<other_mean_x and my_mean_y>other_mean_y:
        #my is row headers
        my_qis.sort(key=lambda a:a[2]) #sort by y
        other_qis.sort(key=lambda a:a[1]) #sort by x
        row_hs = [q[0] for q in my_qis]
        col_hs = [q[0] for q in other_qis]
        
    elif my_mean_x>other_mean_x and my_mean_y<other_mean_y:
        #my is col headers
        my_qis.sort(key=lambda a:a[1]) #sort by x
        other_qis.sort(key=lambda a:a[2]) #sort by y
        col_hs = [q[0] for q in my_qis]
        row_hs = [q[0] for q in other_qis]
    else:
        assert False, 'unknown case'


    values={}
    for row_h in row_hs:
        vs = relationships_q_a[row_h]
        for v in vs:
            try:
                q1,q2 = relationships_a_q[v]
                if q1==row_h:
                    col_h=q2
                else:
                    col_h=q1
                values[(col_h,row_h)] = v
            except ValueError:
                pass

    table = {
            "row_headers": row_hs,
            "col_headers": col_hs,
            "values": values
            }
    for row_h in row_hs:
        #assert row_h not in table_map
        table_map[row_h]=len(tables)
    for col_h in col_hs:
        #assert col_h not in table_map
        table_map[col_h]=len(tables)
    for v in values.values():
        #assert v not in table_map
        table_map[v]=len(tables)
    tables.append(table)

def addTableElement(table_values,row_headers,col_headers,ai,qi1,qi2,groups,bbs,threshold=5):
    ele_x,ele_y = bbs[groups[ai][0],0:2]
    q1_x,q1_y = bbs[groups[qi1][0],0:2]
    x_diff_1 = abs(ele_x-q1_x)
    y_diff_1 = abs(ele_y-q1_y)
    if qi2 is not None:
        #which question is the row, which is the header?
        q2_x,q2_y = bbs[groups[qi2][0],0:2]
        x_diff_2 = abs(ele_x-q2_x)
        y_diff_2 = abs(ele_y-q2_y)

        if abs(q1_x-q2_x)<threshold or abs(q1_y-q2_y)<threshold:
            return False

        if (x_diff_1<y_diff_1 or y_diff_2<x_diff_2) and y_diff_1>threshold and x_diff_2>threshold:
            row_h = qi2
            col_h = qi1
        elif (x_diff_2<y_diff_2 or y_diff_1<x_diff_1) and y_diff_2>threshold and x_diff_1>threshold:
            row_h = qi1
            col_h = qi2
        else:
            #IDK
            #import pdb;pdb.set_trace()
            return False
        
        table_values[(col_h,row_h)]=ai
        row_headers.add(row_h)
        col_headers.add(col_h)
    else:
        if x_diff_1>y_diff_1:
            row_headers.add(qi1)
            table_values[(None,qi1)]=ai
        elif x_diff_1<y_diff_1:
            col_headers.add(qi1)
            table_values[(qi1,None)]=ai
    return True

