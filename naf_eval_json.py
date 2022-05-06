import os
import json
import logging
import argparse
import torch
import numpy as np
from model import *
from logger import Logger
from trainer import *
from data_loader import getDataLoader
import math
from collections import defaultdict
import pickle
#import requests
import warnings
from utils import img_f
import editdistance
import random
import re
from transformers import BartTokenizer
from funsd_eval_json import getFormData #fixLoadJSON
try:
    import easyocr
except:
    pass

end_token = '‡'
np_token = '№'
blank_token = 'ø'

def norm_ed(s1,s2):
    return editdistance.eval(s1.lower(),s2.lower())/max(len(s1),len(s2),1)

def breakIntoLines(entities,links):
    old_to_head={}
    old_to_tail={}
    new_entities=[]
    new_links=[]
    for old_id,entity in enumerate(entities):
        if len(entity.text_lines)==1:
            e_id = len(new_entities)
            new_entities.append(entity)
            old_to_head[old_id] = e_id
            old_to_tail[old_id] = e_id
        else:
            lines = entity.split()
            head_id = len(new_entities)
            new_entities.append(lines[0])
            prev_id = head_id
            for line in lines[1:]:
                next_id = len(new_entities)
                new_entities.append(line)
                new_links.append((prev_id,next_id))
                prev_id = next_id

            old_to_head[old_id] = head_id
            old_to_tail[old_id] = prev_id

    for a,b in links:
        #a_head = old_to_head[a]
        a_tail = old_to_tail[a]
        b_head = old_to_head[b]
        #b_tail = old_to_tail[b]
        new_links.append((a_tail,b_head))

    return new_entities,new_links




def derepeat(s):
    #hardcoded error the model makes a lot (for some reason)
    s = s.replace('{"*": "question"},','')
    s = s.replace(', {"*": "question"}','')
    #very rough
    while True:
        #m = re.search(r'(.......+)\1\1\1\1\1\1\1+',s) #8 chars, 7 repeat
        m = re.search(r'(.......+)\1\1\1\1\1+',s) #8 chars, 5 repeat
        if m is None:
            m = re.search(r'(..............+)\1\1\1\1+',s) #15 chars, 4 repeat
            if m is None:
                break

        start,end = m.span()
        #end-=len(m[1]) #keep one
        s = s[:start]+s[end:]
    return s

def findUnmatched(s):
    b_stack=[]
    c_stack=[]
    for i,c in enumerate(s):
        if c=='[':
            b_stack.append(i)
        elif c==']':
            b_stack.pop()
        elif c=='{':
            c_stack.append(i)
        elif c=='}':
            c_stack.pop()

    return b_stack[-1] if len(b_stack) > 0 else -1, c_stack[-1] if len(c_stack) > 0 else -1


class Entity():
    def __init__(self,text,cls,identity=None):
        #print('Created entitiy: {}'.format(text))
        self.text=text
        self.text_lines = text.split('\\')
        if cls=='header' or cls=='question' or cls=='other' or cls=='circle' or cls=='textGeneric':
            self.cls='textGeneric'
        elif cls=='answer' or cls=='fieldGeneric':
            self.cls='fieldGeneric'
        else:
            print('UNKNOWN PRED CLASS: '+cls)
            cls = 'other' # parsing error, likely
            self.cls = 'textGeneric'
        self.original_cls=cls
        self.id=identity

    def __repr__(self):
        return '({} :: {})'.format(self.text,self.cls)
    def split(self):
        ret=[]
        for line in self.text_lines:
            ret.append(Entity(line,self.original_cls))
        return ret


def parseDict(ent_dict,entities,links):
    if ent_dict=='':
        return []
    to_link=[]
    is_table=False
    row_headers = None
    col_headers = None
    cells = None
    my_text = None
    return_ids=[]
    prose=[]
    for text,value in ent_dict.items():
        if text=='content':
            if isinstance(value,list):
                for thing in reversed(value):
                    to_link+=parseDict(thing,entities,links)
            else:
                assert isinstance(value,dict)
                to_link+=parseDict(value,entities,links)
        elif text=='answers':
            if not isinstance(value,list):
                value=[value]
            for a in reversed(value):
                assert isinstance(a,str)
                a_id=len(entities)
                entities.append(Entity(a,'answer',a_id))
                to_link.append(a_id)
        elif text=='row headers':
            assert isinstance(value,list)
            row_headers = value
            is_table = True
        elif text=='column headers':
            assert isinstance(value,list)
            col_headers = value
            is_table = True
        else:
            if isinstance(value,str):
                if my_text is not None:
                    #merged entity?
                    prose.append((my_text,my_class,to_link))
                    #my_id=len(entities)
                    #entities.append(Entity(my_text,my_class,my_id))
                    #for other_id in to_link:
                    #    links.append((my_id,other_id))
                    #return_ids.append(my_id)
                    to_link = []
                my_text = text
                my_class = value
            elif isinstance(value,list) and text=='cells':
                is_table=True
                cells = value
    if not is_table:
        if my_text is not None:
            my_id=len(entities)
            entities.append(Entity(my_text,my_class,my_id))
            for other_id in to_link:
                links.append((my_id,other_id))
            return_ids.append(my_id)
        else:
            return_ids+=to_link
            my_id=None

        prev_id=my_id
        for my_text,my_class,to_link in reversed(prose):
            my_id=len(entities)
            entities.append(Entity(my_text,my_class,my_id))
            for other_id in to_link:
                links.append((my_id,other_id))
            if prev_id is not None:
                links.append((my_id,prev_id))
            return_ids.append(my_id)
            prev_id = my_id
    else:
        #a table
        #if cells is not None:
        #    cell_ids = defaultdict(dict)
        #    if not isinstance(cells[0],list):

        #    for r,row in reversed(list(enumerate(cells))):
        #        for c,cell in reversed(list(enumerate(row))):
        #            if cell is not None:
        #                c_id = len(entities)
        #                cell_ids[r][c]=c_id
        #                entities.append(Entity(cell,'answer',c_id))
        #                #if row_headers is not None and len(row_ids)>r:
        #                #    links.append((row_ids[r],c_id))
        #                #if col_headers is not None and len(col_ids)>c:
        #                #    links.append((col_ids[c],c_id))
        if row_headers is not None:
            subheaders=defaultdict(list)
            row_ids=[]
            #row_ids = list(range(len(entities),len(entities)+len(row_headers)))
            for rh in reversed(row_headers):
                if rh is not None:
                    if '<<' == rh[:2] and '>>' in rh:
                        #subent_dict
                        sub_end = rh.find('>>')
                        sub =  rh[2:sub_end]
                        rh=rh[sub_end+2:]
                        subheaders[sub].append(len(entities))
                    row_ids.append(len(entities))
                    entities.append(Entity(rh,'question',len(entities)))

            for subh,sub_links in subheaders.items():
                subi = len(entities)
                entities.append(Entity(subh,'header',len(entities)))
                for rhi in sub_links:
                    links.append((subi,rhi))
        else:
            row_ids = []
        if col_headers is not None:
            subheaders=defaultdict(list)
            #col_ids = list(range(len(entities),len(entities)+len(col_headers)))
            col_ids = []
            for ch in reversed(col_headers):
                if ch is not None:
                    if '<<' == ch[:2] and '>>' in ch:
                        #subent_dict
                        sub_end = ch.find('>>')
                        sub =  ch[2:sub_end]
                        ch=ch[sub_end+2:]
                        subheaders[sub].append(len(entities))
                    col_ids.append(len(entities))
                    entities.append(Entity(ch,'question',len(entities)))

            for subh,sub_links in subheaders.items():
                subi = len(entities)
                entities.append(Entity(subh,'header',len(entities)))
                for chi in sub_links:
                    links.append((subi,chi))
        else:
            col_ids = []
    
        #if cells is not None:
        #    for r,row in reversed(list(enumerate(cells))):
        #        for c,cell in reversed(list(enumerate(row))):
        #            if cell is not None:
        #                c_id = cell_ids[r][c]
        #                if row_headers is not None and len(row_ids)>r:
        #                    links.append((row_ids[r],c_id))
        #                if col_headers is not None and len(col_ids)>c:
        #                    links.append((col_ids[c],c_id))

        return_ids+=row_ids+col_ids
    


    return return_ids




def main(resume,config,addToConfig,gpu=False,do_pad=False,test=False,draw=False,max_qa_len=None,quiet=False,BROS=False,ENTITY_MATCH_THRESH=0.6,LINK_MATCH_THRESH=0.6,DEBUG=False,write=False):
    TRUER=True #False makes this do pair-first alignment, which is kind of cheating
    np.random.seed(1234)
    torch.manual_seed(1234)
    if DEBUG:
        print("DEBUG")
        print("EBUG")
        print("EBUG")
    
    
    #too_long_gen_thresh=10

    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded {} iteration {}'.format(checkpoint['config']['name'],checkpoint['iteration']))
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
        for key in config.keys():
            if 'pretrained' in key:
                config[key]=None
    else:
        checkpoint = None
        config = json.load(open(config))
    config['optimizer_type']="none"
    config['trainer']['use_learning_schedule']=False
    config['trainer']['swa']=False
    if not gpu:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu


    do_ocr=config['trainer']['do_ocr'] if 'do_ocr' in config['trainer'] else False
    if do_ocr and do_ocr!='no':
        ocr_reader = easyocr.Reader(['en'],gpu=config['cuda'])
    if addToConfig is not None:
        for add in addToConfig:
            addTo=config
            printM='added config['
            for i in range(len(add)-2):
                try:
                    indName = int(add[i])
                except ValueError:
                    indName = add[i]
                addTo = addTo[indName]
                printM+=add[i]+']['
            value = add[-1]
            if value=="":
                value=None
            elif value[0]=='[' and value[-1]==']':
                value = value[1:-1].split('-')
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value == 'None':
                            value=None
            addTo[add[-2]] = value
            printM+=add[-2]+']={}'.format(value)
            print(printM)
            #if (add[-2]=='useDetections' or add[-2]=='useDetect') and 'gt' not in value:
            #    addDATASET=True
    if max_qa_len is None:
        #max_qa_len=config['data_loader']['max_qa_len'] if 'max_qa_len' in config['data_loader'] else config['data_loader']['max_qa_len_out']
        max_qa_len_in = config['data_loader'].get('max_qa_len_in',640)
        max_qa_len_out = config['data_loader'].get('max_qa_len_out',2560,)

        
    if checkpoint is not None:
        if 'swa_state_dict' in checkpoint and checkpoint['iteration']>config['trainer']['swa_start']:
            state_dict = checkpoint['swa_state_dict']
            #SWA  leaves the state dict with 'module' in front of each name and adds extra params
            new_state_dict = {key[7:]:value for key,value in state_dict.items() if key.startswith('module.')}
            print('Loading SWA model')
        else:
            state_dict = checkpoint['state_dict']
            #DataParaellel leaves the state dict with 'module' in front of each name
            new_state_dict = {
                    (key[7:] if key.startswith('module.') else key):value for key,value in state_dict.items()
                    }

        model = eval(config['arch'])(config['model'])
        if 'query_special_start_token_embedder.emb.weight' in new_state_dict:
            loading_special = new_state_dict['query_special_start_token_embedder.emb.weight']
            model_special = model.state_dict()['query_special_start_token_embedder.emb.weight']

            if loading_special.size(0) != model_special.size(0):
                model_special[:loading_special.size(0)] = loading_special[:model_special.size(0)]
                new_state_dict['query_special_start_token_embedder.emb.weight'] = model_special
        if 'query_special_token_embedder.emb.weight' in new_state_dict:
            loading_special = new_state_dict['query_special_token_embedder.emb.weight']
            model_special = model.state_dict()['query_special_token_embedder.emb.weight']

            if loading_special.size(0) != model_special.size(0):
                model_special[:loading_special.size(0)] = loading_special[:model_special.size(0)]
                new_state_dict['query_special_token_embedder.emb.weight'] = model_special
        model.load_state_dict(new_state_dict)
    else:
        model = eval(config['arch'])(config['model'])

    model.eval()
    if gpu:
        model = model.cuda()

    if do_pad is not None:
        do_pad = do_pad.split(',')
        if len(do_pad)==1:
            do_pad+=do_pad
        do_pad = [int(p) for p in do_pad]
    else:
        do_pad = config['model']['image_size']
        if type(do_pad) is int:
            do_pad = (do_pad,do_pad)



    ##DAT##
    config['data_loader']['shuffle']=False
    config['validation']['shuffle']=False
    config['data_loader']['eval']=True
    config['validation']['eval']=True

    # change to graph pair dataset
    config['data_loader']['data_set_name']='FormsGraphPair'
    config['data_loader']['data_dir']='../data/forms'
    config['data_loader']['crop_params']=None
    config['data_loader']['batch_size']=1
    config['data_loader']['color']=False
    config['data_loader']['rescale_range']=[1,1]
    config['data_loader']['no_blanks']=True
    config['data_loader']['swap_circle']=True
    config['data_loader']['no_graphics']=True
    config['data_loader']['only_opposite_pairs']=False
    config['data_loader']['no_groups']=True
    config['data_loader']['rotation']=True
    

    if DEBUG:
        config['data_loader']['num_workers']=0

    config['validation']['data_set_name']='FormsGraphPair'
    config['validation']['data_dir']='../data/forms'
    config['validation']['crop_params']=None
    config['validation']['batch_size']=1
    config['validation']['color']=False
    config['validation']['rescale_range']=[1,1]
    config['validation']['no_blanks']=True
    config['validation']['swap_circle']=True
    config['validation']['no_graphics']=True
    config['validation']['only_opposite_pairs']=False
    config['validation']['no_groups']=True
    config['validation']['rotation']=True

    if not test:
        data_loader, valid_data_loader = getDataLoader(config,'train')
    else:
        valid_data_loader, data_loader = getDataLoader(config,'test')
        data_loader = valid_data_loader
    valid_iter = iter(valid_data_loader)

    num_classes = len(valid_data_loader.dataset.classMap)

    total_entity_true_pos =0
    total_entity_pred =0
    total_entity_gt =0
    total_rel_true_pos =0
    total_rel_pred =0
    total_rel_gt =0
    total_entity_true_pos2 =0
    total_entity_pred2 =0
    total_entity_gt2 =0
    total_rel_true_pos2 =0
    total_rel_pred2 =0
    total_rel_gt2 =0

    tokenizer = BartTokenizer.from_pretrained('./cache_huggingface/BART')
    to_write = {}

    going_DEBUG=False
    with torch.no_grad():
        for instance in valid_iter:
            groups = instance['gt_groups']
            classes_lines = instance['bb_gt'][0,:,-num_classes:]
            loc_lines = instance['bb_gt'][0,:,0:2] #x,y
            bb_lines = instance['bb_gt'][0,:,[5,10,7,12]].long()
            pairs = instance['gt_groups_adj']
            transcription_lines = instance['transcription']
            #transcription_lines = [s if cased else s for s in transcription_lines]
            img = instance['img'][0]
            if not quiet:
                print()
                print(instance['imgName'])

            if DEBUG and (not going_DEBUG and instance['imgName']!='007270209_00003'):
                continue
            going_DEBUG=True


            scale_height = do_pad[0]/img.shape[1]
            scale_width = do_pad[1]/img.shape[2]
            choosen_scale = min(scale_height, scale_width)
            img = img_f.resize(img[0].numpy(),fx=choosen_scale,fy=choosen_scale)
            img = torch.FloatTensor(img)[None,...]

            bb_lines = bb_lines*choosen_scale
            loc_lines = loc_lines*choosen_scale

            gt_line_to_group = instance['targetIndexToGroup']

            transcription_groups = []
            transcription_firstline = []
            pos_groups = []
            for group in groups:
                transcription_groups.append('\\'.join([transcription_lines[t] for t in group if transcription_lines[t] is not None]))
                transcription_firstline.append(transcription_lines[group[0]])
                pos_groups.append(loc_lines[group[0]])
            

            classes = [classes_lines[group[0]].argmax() for group in groups]
            gt_classes = [data_loader.dataset.index_class_map[c] for c in classes]

            if draw:
                draw_img = (128*(1-img.permute(1,2,0).expand(-1,-1,3).numpy())).astype(np.uint8)

            
            if do_pad and (img.shape[1]<do_pad[0] or img.shape[2]<do_pad[1]):
                diff_x = do_pad[1]-img.shape[2]
                diff_y = do_pad[0]-img.shape[1]
                p_img = torch.FloatTensor(img.size(0),do_pad[0],do_pad[1]).fill_(-1)#np.zeros(do_pad,dtype=img.dtype)
                pad_y = diff_y//2
                pad_x = diff_x//2
                if diff_x>=0 and diff_y>=0:
                    p_img[:,diff_y//2:p_img.shape[1]-(diff_y//2 + diff_y%2),diff_x//2:p_img.shape[2]-(diff_x//2 + diff_x%2)] = img
                elif diff_x<0 and diff_y>=0:
                    p_img[:,diff_y//2:p_img.shape[1]-(diff_y//2 + diff_y%2),:] = img[:,:,(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                elif diff_x>=0 and diff_y<0:
                    p_img[:,diff_x//2:p_img.shape[2]-(diff_x//2 + diff_x%2)] = img[:,(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),:]
                else:
                    p_img = img[:,(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                img=p_img

                loc_lines[:,0]+=pad_x
                loc_lines[:,1]+=pad_y
                #bb_lines[:,0]+=pad_x
                #bb_lines[:,1]+=pad_y
                #bb_lines[:,2]+=pad_x
                #bb_lines[:,3]+=pad_y

            img = img[None,...] #re add batch 
            img = torch.cat((img,torch.zeros_like(img)),dim=1) #add blank mask channel

            if gpu:
                img = img.cuda()
            
            pred_data, good_char_pred_ratio = getFormData(model,img,tokenizer,quiet)
            
            if not quiet:
                print('==Corrected==')
                print(json.dumps(pred_data,indent=2))
            if write:
                to_write[instance['imgName']]=pred_data
            
            #get entities and links
            pred_entities=[]
            pred_links=[]
            for thing in pred_data[::-1]: #build pred_entities in reverse
                if isinstance(thing,dict):
                    parseDict(thing,pred_entities,pred_links)
                #elif thing=='':
                #    pass
                #else:
                #    print('non-dict at document level: {}'.format(thing))
                #    assert False
                    #import pdb;pdb.set_trace()

            #New addition for NAF
            #Break all entities into individual ones, and redo linking
            pred_entities,pred_links = breakIntoLines(pred_entities,pred_links)

            if len(pred_entities)>0:
                #we're going to do a check for repeats of the last entity. This frequently happens
                last_entity = pred_entities[-1]
                remove = None
                entities_with_link = None
                for i in range(len(pred_entities)-2,0,-1):
                    if pred_entities[i].text==last_entity.text and pred_entities[i].cls==last_entity.cls:
                        if entities_with_link is None:
                            entities_with_link = set()
                            for a,b in pred_links:
                                entities_with_link.add(a)
                                entities_with_link.add(b)

                        if i not in entities_with_link:
                            remove=i+1
                        else:
                            break
                    else:
                        break
                if remove is not None:
                    if not quiet:
                        print('removing duplicate end entities: {}'.format(pred_entities[remove:]))
                        pred_entities = pred_entities[:remove]
                
            #align entities to GT ones
            #pred_to_gt={}
            #for g_i,gt in enumerate(transcription_groups):
            #    closest_dist=9999999
            #    closest_e_i=-1
            #    for e_i,entity in pred_entities:
            #        dist
            #should find pairs in GT with matching text and handle these seperately/after

            if TRUER:
                DIST_SCORE=600
                #order entities by y
                gt_entities = []
                for i,(text,text_firstline,(x,y),cls) in enumerate(zip(transcription_groups,transcription_firstline,pos_groups,gt_classes)):
                    gt_entities.append((i,x,y,text_firstline if BROS else text,cls))
                #gt_entities.sort(key=lambda a:a[2])

                pos_in_gt=0
                last_x=0
                last_y=0
                last_text=None
                gt_to_pred = defaultdict(list)
                all_scores = defaultdict(dict)
                for p_i,entity in reversed(list(enumerate(pred_entities))):
                    #if 'NAME' in entity.text:
                    #    import pdb; pdb.set_trace()
                    if BROS:
                        p_text = entity.text_lines[0]
                    else: 
                        p_text = entity.text
                    has_link=False
                    for a,b in pred_links:
                        if p_i==a or p_i==b:
                            has_link=True
                            break
                    #if 'cc' in p_text:
                    #    import pdb; pdb.set_trace()
                    if last_text == p_text:
                        #the model maybe double predicted? In anycase, there will be a 0 distance if we use the last_x/y, so we'll rewind one step back
                        last_x = last2_x
                        last_y = last2_y
                    best_score=9999999
                    for g_i,x,y,g_text,cls in gt_entities:

                        text_dist = norm_ed(p_text,g_text)
                        if text_dist<LINK_MATCH_THRESH:# and cls==entity.cls: cheating a little if we use class

                            #dist = abs(last_y-y) + 0.1*abs(last_x-x)#math.sqrt((last_y-y)**2)+((last_x-x)**2))
                            #asymetric, penalize up alignment more than down
                            dist = 0.15*abs(last_x-x)
                            if last_y<y:
                                dist+= y-last_y
                            else:
                                dist+= 1.2*(last_y-y)
                            score = text_dist + dist/DIST_SCORE

                            #if '532' in p_text:
                            #    import pdb;pdb.set_trace()

                            #adjust score if there are any links
                            #This obiviously doesn't effect these comprisons, but will help it ownership fights
                            if has_link:
                                score -= 0.03


                            all_scores[p_i][g_i]=score

                            if score<best_score:
                                align_g_i = g_i
                                align_x = x
                                align_y = y
                                best_score = score
                    if best_score<9999999:
                        gt_to_pred[align_g_i].append((p_i,best_score))
                        last2_x = last_x
                        last2_y = last_y
                        last_x=align_x
                        last_y=align_y
                        last_text = p_text

                #Now, we potentially aligned multiple pred entities to gt entities
                #We need to resolve these by finding the best alternate match for the worse match
                new_gt_to_pred={}#[None]*len(groups)
                pred_to_gt={}
                new_gt_to_pred_scores={}
                to_realign=[]
                for g_i,p_is in gt_to_pred.items():
                    if len(p_is)>1:
                        #import pdb;pdb.set_trace()
                        #we need to check if any links to the p_is have already been aligned.
                        #if so, we'll want to keep that consistant
                        new_pis=[]
                        for p_i,score in p_is:
                            aligned_link = False
                            for a,b in pred_links:
                                other_i = None
                                if a==p_i:
                                    other_i = b
                                elif b==p_i:
                                    other_i = a
                                if other_i in pred_to_gt:
                                    #The link has been aligned to a gt
                                    aligned_link=True
                                    break
                            if aligned_link:
                                new_pis.append((p_i,score-0.05))
                            else:
                                new_pis.append((p_i,score))
                        p_is = new_pis
                        p_is.sort(key=lambda a:a[1])

                    new_gt_to_pred[g_i]=p_is[0][0] #best score gets it
                    new_gt_to_pred_scores[g_i]=p_is[0][1]
                    pred_to_gt[p_is[0][0]]=g_i
                    for p_i,_ in p_is[1:]:
                        to_realign.append(p_i)
                debug_count=0
                while len(to_realign)>0:
                    if debug_count>50:
                        print('infinite loop')
                        assert False
                        #import pdb;pdb.set_trace()
                    debug_count+=1

                    doing = to_realign
                    to_realign = []
                    for p_i in doing:
                        scores = [(g_i,score) for g_i,score in all_scores[p_i].items()]
                        best_score=9999999
                        for g_i,score in all_scores[p_i].items():
                            if score<best_score:
                                can_match = g_i not in new_gt_to_pred or score<new_gt_to_pred_scores[g_i]
                                if can_match:
                                    align_g_i=g_i
                                    best_score=score
                        if best_score<9999999:
                            if align_g_i in new_gt_to_pred:
                                to_realign.append(new_gt_to_pred[align_g_i])
                            new_gt_to_pred[align_g_i]=p_i
                            new_gt_to_pred_scores[align_g_i]=best_score
                            pred_to_gt[p_i]=align_g_i
                        #else:
                            #unmatched

                
                entities_truepos = 0
                for g_i,p_i in new_gt_to_pred.items():
                    if gt_classes[g_i]==pred_entities[p_i].cls:
                        if not BROS or norm_ed(pred_entities[p_i].text,transcription_groups[g_i])<ENTITY_MATCH_THRESH:
                            entities_truepos+=1
                        #print('A hit G:{} <> P:{}'.format(transcription_groups[g_i],pred_entities[p_i].text))

                rel_truepos = 0
                good_pred_pairs = set()
                for g_i1,g_i2 in pairs:
                    #if (g_i1==18 and g_i2==55) or (g_i2==18 and g_i1==55):
                    #    import pdb;pdb.set_trace()
                    if g_i1 in new_gt_to_pred and g_i2 in new_gt_to_pred:
                        p_i1 = new_gt_to_pred[g_i1]
                        p_i2 = new_gt_to_pred[g_i2]
                        if gt_classes[g_i1]==pred_entities[p_i1].cls and gt_classes[g_i2]==pred_entities[p_i2].cls:
                            if (p_i1,p_i2) in pred_links or (p_i2,p_i1) in pred_links:
                                rel_truepos+=1
                                if draw:
                                    if (p_i1,p_i2) in pred_links:
                                        good_pred_pairs.add((p_i1,p_i2))
                                    else:
                                        good_pred_pairs.add((p_i2,p_i1))
                    
                        
                if draw:
                    #pred_to_gt = {p_i:g_i for g_i,p_i in new_gt_to_pred.items()}
                    bad_pred_pairs=set(pred_links)-good_pred_pairs

                ############
            else:
                assert False

            #######End cheating       

                    
            entity_recall = entities_truepos/len(transcription_groups) if len(transcription_groups)>0 else 1
            entity_prec = entities_truepos/len(pred_entities) if len(pred_entities)>0 else 1
            rel_recall = rel_truepos/len(pairs) if len(pairs)>0 else 1
            rel_prec = rel_truepos/len(pred_links) if len(pred_links)>0 else 1

            total_entity_true_pos += entities_truepos
            total_entity_pred += len(pred_entities)
            total_entity_gt += len(groups)
            assert entities_truepos<=len(pred_entities)
            assert entities_truepos<=len(groups)
            if not quiet:
                print('Entity precision: {}'.format(entity_prec))
                print('Entity recall:    {}'.format(entity_recall))
                print('Entity Fm:        {}'.format(2*entity_recall*entity_prec/(entity_recall+entity_prec) if entity_recall+entity_prec>0 else 0))
                print('Rel precision: {}'.format(rel_prec))
                print('Rel recall:    {}'.format(rel_recall))
                print('Rel Fm:        {}'.format(2*rel_recall*rel_prec/(rel_recall+rel_prec) if rel_recall+rel_prec>0 else 0))
            else:
                print('{} (calls:{}, goodChar:{}) EntityFm: {},  RelFm: {}'.format(instance['imgName'],
                    -1,
                    good_char_pred_ratio,
                    2*entity_recall*entity_prec/(entity_recall+entity_prec) if entity_recall+entity_prec>0 else 0,2*rel_recall*rel_prec/(rel_recall+rel_prec) if rel_recall+rel_prec>0 else 0))

            total_rel_true_pos += rel_truepos
            total_rel_pred += len(pred_links)
            total_rel_gt += len(pairs)



            if draw:
                mid_points=[]
                for p_i,entity in enumerate(pred_entities):
                    if p_i in pred_to_gt:
                        g_i = pred_to_gt[p_i]
                        print('{} [[matched to ]] {}'.format(pred_entities[p_i].text,transcription_groups[g_i]))
                        cls = entity.original_cls
                        if cls=='header':
                            color=(0,0,255) #header
                        elif cls=='question':
                            color=(0,255,255) #question
                        elif cls=='answer':
                            color=(255,255,0) #answer
                        elif cls=='other':
                            color=(255,0,255) #other 
                        elif cls=='circle':
                            color=(155,255,155) #answer

                        x1,y1,x2,y2 = bb_lines[groups[g_i][0]]
                        for l_i in groups[g_i][1:]:
                            x1_,y1_,x2_,y2_ = bb_lines[l_i]
                            x1=min(x1,x1_)
                            y1=min(y1,y1_)
                            x2=max(x2,x2_)
                            y2=max(y2,y2_)
                        mid_points.append(((x1+x2)//2,(y1+y2)//2))
                        if cls == gt_classes[g_i]:
                            img_f.rectangle(draw_img,(x1,y1),(x2,y2),color,2)
                        else:
                            x,y = mid_points[-1]
                            x=int(x)
                            y=int(y)
                            draw_img[y-3:y+3,x-3:x+3]=color

                    else:
                        if not quiet:
                            print('unmatched entity: {}'.format(entity.text))
                        best=9999999999
                        for g_i,g_text in enumerate(transcription_groups):
                            s = norm_ed(g_text,entity.text)
                            if s<best:
                                best=s
                                best_g = g_i

                        x1,y1,x2,y2 = bb_lines[groups[best_g][0]]
                        mid_points.append((x1,(y1+y2)//2))

                for p_a,p_b in bad_pred_pairs:
                    x1,y1 = mid_points[p_a]
                    x2,y2 = mid_points[p_b]
                    img_f.line(draw_img,(x1,y1+1),(x2,y2+1),(255,0,0),2)
                for p_a,p_b in good_pred_pairs:
                    x1,y1 = mid_points[p_a]
                    x2,y2 = mid_points[p_b]
                    img_f.line(draw_img,(x1,y1),(x2,y2),(0,255,0),2)

                img_f.imshow('f',draw_img)
                img_f.show()
        print('======================')


        total_entity_prec = total_entity_true_pos/total_entity_pred
        total_entity_recall = total_entity_true_pos/total_entity_gt
        total_entity_F = 2*total_entity_prec*total_entity_recall/(total_entity_recall+total_entity_prec) if total_entity_recall+total_entity_prec>0 else 0

        print('Total entity recall, prec, Fm:\t{:.3}\t{:.3}\t{:.3}'.format(total_entity_recall,total_entity_prec,total_entity_F))


        total_rel_prec = total_rel_true_pos/total_rel_pred
        total_rel_recall = total_rel_true_pos/total_rel_gt
        total_rel_F = 2*total_rel_prec*total_rel_recall/(total_rel_recall+total_rel_prec) if total_rel_recall+total_rel_prec>0 else 0
        print('Total rel recall, prec, Fm:\t{:.3}\t{:.3}\t{:.3}'.format(total_rel_recall,total_rel_prec,total_rel_F))

        if write:
            with open(write, 'w') as f:
                json.dump(to_write,f)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='run QA model on image(s)')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-p', '--pad', default=None, type=str,
                        help='pad image to this size (square)')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn".  You can nest keys with k1=k2=k3=v')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='run test set (default: False)')
    parser.add_argument('-B', '--BROS', default=False, action='store_const', const=True,
                        help='evaluate matching using only first line of entities (default: False)')
    parser.add_argument('-q', '--quiet', default=False, action='store_const', const=True,
                        help='prevent pred prints (default: False)')
    parser.add_argument('-m', '--max-qa-len', default=None, type=int,
                        help='max len for questions')
    parser.add_argument('-d', '--draw', default=False, action='store_const', const=True,
                        help='display image with pred annotated (default: False)')
    parser.add_argument('-D', '--DEBUG', default=False, action='store_const', const=True,
                        help='d')
    parser.add_argument('-E', '--ENTITY_MATCH_THRESH', default=0.6, type=float,
                        help='Edit distance required to have pred entity match a GT one for entity detection')
    parser.add_argument('-L', '--LINK_MATCH_THRESH', default=0.6, type=float,
                        help='Edit distance required to have pred entity match a GT one for linking')
    parser.add_argument('-w', '--write', default=False, type=str,
                        help='path to write all jsons to')

    args = parser.parse_args()

    addtoconfig=[]
    if args.addtoconfig is not None:
        split = args.addtoconfig.split(',')
        for kv in split:
            split2=kv.split('=')
            addtoconfig.append(split2)

    config = None
    if args.checkpoint is None and args.config is None:
        print('Must provide checkpoint (with -c)')
        exit()
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint,args.config,addtoconfig,True,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw, quiet=args.quiet,BROS=args.BROS,ENTITY_MATCH_THRESH=args.ENTITY_MATCH_THRESH,LINK_MATCH_THRESH=args.LINK_MATCH_THRESH,DEBUG=args.DEBUG,write=args.write)
    else:
        main(args.checkpoint,args.config, addtoconfig,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw,quiet=args.quiet,BROS=args.BROS,ENTITY_MATCH_THRESH=args.ENTITY_MATCH_THRESH,LINK_MATCH_THRESH=args.LINK_MATCH_THRESH,DEBUG=args.DEBUG,write=args.write)
