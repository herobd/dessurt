import os
import json
import logging
import argparse
import torch
from model import *
from logger import Logger
from trainer import *
from data_loader import getDataLoader
from evaluators import *
import math
from collections import defaultdict
import pickle
#import requests
import warnings
from utils.saliency import SimpleFullGradMod
from utils.debug_graph import GraphChecker
from utils import img_f
from utils.util import pointDistance
import editdistance
import random
try:
    import easyocr
except:
    pass

end_token = '‡'
np_token = '№'
blank_token = 'ø'

def norm_ed(s1,s2):
    return editdistance.eval(s1.lower(),s2.lower())/max(len(s1),len(s2),1)

def derepeat(s):
    #very rough
    test_len=8
    for start in range(0,len(s),test_len):
        test_str = s[start:start+test_len]


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

def fixLoadJSON(pred):
    pred_data = None
    start_len = len(pred)
    end_token_loc = pred.find(end_token)
    if end_token_loc != -1:
        pred = pred[:end_token_loc]
    counter=2000
    while pred_data is None:
        counter -=1
        if len(pred)>start_len+20 or counter==0:
            assert False
        pred = pred.replace(',,',',')
        pred = pred.replace('{{','{')
        try:
            pred_data = json.loads(pred)
        except json.decoder.JSONDecodeError as e:
            sections = '{}'.format(e)
            print(sections)
            sections=sections.replace("':'","';'")
            sections = sections.split(':')
            #if len(sections)==3:
            #    err,typ,loc =sections
            #else:
            typ,loc = sections

            assert 'line 1' in loc
            loc_char = loc.find('char ')
            loc_char_end = loc.rfind(')')
            char = int(loc[loc_char+5:loc_char_end])
            
            if "Expecting ',' delimiter" in typ:
                if char==len(pred):
                    #closing ] or }?
                    #bracket = pred.rfind('[')
                    #curley = pred.rfind('{')
                    bracket,curley = findUnmatched(pred)
                    assert bracket!=-1 or curley!=-1
                    if bracket>curley:
                        pred+=']'
                    else:
                        pred+='}'
                elif pred[char]==':':
                    #it didn't close a list
                    assert pred[:char-1].rfind('[')>pred[:char-1].rfind('{')
                    assert pred[char-1]=='"'
                    open_quote = pred[:char-1].rfind('"')
                    assert open_quote!=-1
                    comma = pred[:open_quote].rfind(',')
                    bracket = pred[:open_quote].rfind('[')
                    #assert comma != -1
                    if comma>bracket:
                        pred = pred[:comma]+']},{'+pred[comma+1:]
                    else:
                        pred = pred[:bracket+1]+'],'+pred[bracket+1]
                elif pred[char]==']' and pred[char-1]=='"':
                    assert pred[:char-1].rfind('[')<pred[:char-1].rfind('{')
                    pred = pred[:char]+'}'+pred[char:]
                else:
                    #pred+=','
                    assert False
            elif 'Unterminated string starting at' in typ:
                pred+='"'
            elif 'Expecting value' in typ:
                if char==len(pred) and pred[char-1]==':':
                    #We'll just remove this incomplete prediction
                    bracket = pred.rfind('{')
                    assert bracket > pred.rfind('}')
                    comma = pred[:bracket].rfind(',')
                    pred = pred[:comma]
                elif char==len(pred) and pred[char-1]!='"':
                    pred+='""'
                elif char==len(pred)-1 and pred[char]!='"':
                    pred+='""'
                elif pred[char]=='}' and pred[:char].rfind('{')<pred[:char].rfind('}'):
                    #random extra close curelybrace
                    pred = pred[:char]+pred[char+1:]
                else:
                    assert False
            elif "Expecting ';' delimiter" in typ:
                if char==len(pred):
                    pred+=':'
                else:
                    bracket = pred[:char-1].rfind('{')
                    colon = pred[:char-1].rfind(':')
                    if bracket>colon:
                        #this is missing the class prediction
                        pred = pred[:char]+':"other"'+pred[char:]
                    else:
                        #extra data?
                        open_quote= pred[colon:].find('"')
                        assert open_quote!=-1
                        open_quote += colon
                        close_quote= pred[open_quote+1:].find('"')
                        assert close_quote!=-1
                        close_quote += open_quote+1

                        pred =pred[:close_quote+1]+pred[char:] #REMOVE
                        
            elif 'Expecting property name enclosed in double quotes' in typ:
                if char==len(pred) or char==len(pred)-1:
                    if pred[-1]=='"':
                        pred = pred[:-1]
                        bracket = pred.rfind('{')
                        if bracket>pred.rfind('"'):
                            pred = pred[:bracket]
                    else:
                        if pred[-1]==',':
                            pred=pred[:-1]
                        pred+='}'
                else:
                    assert False
            elif 'Expecting value' in typ:
                if pred[-1]==',':
                    pred=pred[:-1]
                else:
                    assert False
            elif 'Extra data' in typ :
                if len(pred)==char:
                    assert pred[-1]==','
                    pred = pred[:-1]
                elif pred[char-1]==']':
                    #closed bracket too early?
                    pred = pred[:char-1]+','+pred[char:]
            else:
                assert False

            print('corrected pred: '+pred)
    return pred_data

class Entity():
    def __init__(self,text,cls,identity):
        print('Created entitiy: {}'.format(text))
        self.text=text
        self.text_lines = text.split('\\')
        self.cls=cls
        self.id=identity
def parseDict(header,entities,links):
    to_link=[]
    is_table=False
    row_headers = None
    col_headers = None
    cells = None
    for text,value in header.items():
        if text=='content':
            if isinstance(value,list):
                for thing in value:
                    to_link+=parseDict(thing,entities,links)
            else:
                assert isinstance(value,dict)
                to_link+=parseDict(value,entities,links)
        elif text=='answers':
            if not isinstance(value,list):
                value=[value]
            for a in value:
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
                my_text = text
                my_class = value
            elif isinstance(value,list) and text=='cells':
                is_table=True
                cells = value
    if not is_table:
        my_id=len(entities)
        entities.append(Entity(my_text,my_class,my_id))
        for other_id in to_link:
            links.append((my_id,other_id))
        return [my_id]
    else:
        #a table
        if row_headers is not None:
            row_ids = list(range(len(entities),len(entities)+len(row_headers)))
            for rh in row_headers:
                entities.append(Entity(rh,'question',len(entities)))
        if col_headers is not None:
            col_ids = list(range(len(entities),len(entities)+len(col_headers)))
            for ch in col_headers:
                entities.append(Entity(ch,'question',len(entities)))
    
        if cells is not None:
            for r,row in enumerate(cells):
                for c,cell in enumerate(row):
                    if cell is not None:
                        c_id = len(entities)
                        entities.append(Entity(cell,'answer',c_id))
                        if row_headers is not None and len(row_ids)>r:
                            links.append((row_ids[r],c_id))
                        if col_headers is not None and len(col_ids)>c:
                            links.append((col_ids[c],c_id))

        return row_ids+col_ids




def main(resume,config,img_path,addToConfig,gpu=False,do_pad=False,test=False,draw=False,max_qa_len=None,quiet=False,BROS=False):
    np.random.seed(1234)
    torch.manual_seed(1234)
    DEBUG=True
    print("DEBUG")
    
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
    config['data_loader']['data_set_name']='FUNSDGraphPair'
    config['data_loader']['data_dir']='../data/FUNSD'
    config['data_loader']['crop_params']=None
    config['data_loader']['batch_size']=1
    config['data_loader']['split_to_lines']=True
    config['data_loader']['color']=False
    config['data_loader']['rescale_range']=[1,1]
    if DEBUG:
        config['data_loader']['num_workers']=0

    config['validation']['data_set_name']='FUNSDGraphPair'
    config['validation']['data_dir']='../data/FUNSD'
    config['validation']['crop_params']=None
    config['validation']['batch_size']=1
    config['validation']['split_to_lines']=True
    config['validation']['color']=False
    config['validation']['rescale_range']=[1,1]

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

            if not going_DEBUG and instance['imgName']!='92314414':
                continue
            going_DEBUG=True

            gt_line_to_group = instance['targetIndexToGroup']

            transcription_groups = []
            transcription_firstline = []
            for group in groups:
                transcription_groups.append('\\'.join([transcription_lines[t] for t in group]))
                transcription_firstline.append(transcription_lines[group[0]])


            classes = [classes_lines[group[0]].argmax() for group in groups]
            gt_classes = [data_loader.dataset.index_class_map[c] for c in classes]

            if draw:
                draw_img = (255*(1-img.permute(1,2,0).expand(-1,-1,3).numpy())).astype(np.uint8)
            
            if do_pad and (img.shape[1]<do_pad[0] or img.shape[2]<do_pad[1]):
                diff_x = do_pad[1]-img.shape[2]
                diff_y = do_pad[0]-img.shape[1]
                p_img = torch.FloatTensor(img.size(0),do_pad[0],do_pad[1]).fill_(-1)#np.zeros(do_pad,dtype=img.dtype)
                pad_y = diff_y//2
                pad_x = diff_x//2
                if diff_x>=0 and diff_y>=0:
                    p_img[:,diff_y//2:-(diff_y//2 + diff_y%2),diff_x//2:-(diff_x//2 + diff_x%2)] = img
                elif diff_x<0 and diff_y>=0:
                    p_img[:,diff_y//2:-(diff_y//2 + diff_y%2),:] = img[:,:,(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                elif diff_x>=0 and diff_y<0:
                    p_img[:,diff_x//2:-(diff_x//2 + diff_x%2)] = img[:,(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),:]
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
            
            #First find tables, as those are done seperately (they should do multiline things alread)
            #TODO multi-step pred for long forms

            #print GT
            print('==GT form==')
            for ga,gb in pairs:
                print('{} [{}] <=> {} [{}]'.format(transcription_groups[ga],gt_classes[ga],transcription_groups[gb],gt_classes[gb]))
            print()

            question='json>'
            answer,out_mask = model(img,None,[[question]],RUN=True)
            print(answer)
            answer = derepeat(answer)
            total_answer = answer
            for i in range(3): #shouldn't need to be more than 4 calls for test set
                if end_token in total_answer:
                    break
                
                #how much of a lead? Need it to fit tokenwise in the 20 limit
                tokens = tokenizer.encode(tokens)
                tokens = tokens[-20:]
                prompt = tokenizer.decode(tokens,skip_special_tokens=True)
                question = 'json~'+prompt
                answer,out_mask = model(img,None,[[question]],RUN=True)
                answer = derepeat(answer)

                total_answer+=answer
            
            pred_data = fixLoadJSON(total_answer)

            print('==Corrected==')
            print(json.dumps(pred_data,indent=2))
            
            #get entities and links
            pred_entities=[]
            pred_links=[]
            for thing in pred_data:
                if isinstance(thing,dict):
                    parseDict(thing,pred_entities,pred_links)
                else:
                    print('non-dict at document level: {}'.format(thing))
                    import pdb;pdb.set_trace()
                
            #align entities to GT ones
            #pred_to_gt={}
            #for g_i,gt in enumerate(transcription_groups):
            #    closest_dist=9999999
            #    closest_e_i=-1
            #    for e_i,entity in pred_entities:
            #        dist
            #TODO should find pairs in GT with matching text and handle these seperately/after
            match_thresh = 0.6
            gt_pair_hit=[False]*len(pairs)
            rel_truepos=0
            pred_to_gt=defaultdict(list)
            good_pred_pairs = []
            bad_pred_pairs = []
            for p_a,p_b in pred_links:
                e_a = pred_entities[p_a]
                e_b = pred_entities[p_b]

                a_aligned = pred_to_gt.get(p_a,-1)
                b_aligned = pred_to_gt.get(p_b,-1)

                best_score = 99999
                best_gt_pair = -1
                for pairs_i,(g_a,g_b) in enumerate(pairs):
                    #can't match to a gt pair twice
                    if gt_pair_hit[pairs_i]:
                        continue

                    if a_aligned==-1 and b_aligned==-1:

                        if BROS:
                            dist_aa = norm_ed(transcription_firstline[g_a],e_a.text_lines[0]) if e_a.cls==gt_classes[g_a] else 99
                            dist_bb = norm_ed(transcription_firstline[g_b],e_b.text_lines[0]) if e_b.cls==gt_classes[g_b] else 99
                            dist_ab = norm_ed(transcription_firstline[g_a],e_b.text_lines[0]) if e_b.cls==gt_classes[g_a] else 99
                            dist_ba = norm_ed(transcription_firstline[g_b],e_a.text_lines[0]) if e_a.cls==gt_classes[g_b] else 99
                        else:
                            dist_aa = norm_ed(transcription_groups[g_a],e_a.text) if e_a.cls==gt_classes[g_a] else 99
                            dist_bb = norm_ed(transcription_groups[g_b],e_b.text) if e_b.cls==gt_classes[g_b] else 99
                            dist_ab = norm_ed(transcription_groups[g_a],e_b.text) if e_b.cls==gt_classes[g_a] else 99
                            dist_ba = norm_ed(transcription_groups[g_b],e_a.text) if e_a.cls==gt_classes[g_b] else 99
                        
                        if dist_aa+dist_bb < dist_ab+dist_ba and dist_aa<match_thresh and dist_bb<match_thresh:
                            score = dist_aa+dist_bb
                            if score<best_score:
                                best_score = score
                                best_gt_pair = pairs_i
                                matching = (g_a,g_b)
                        elif dist_ab<match_thresh and dist_ba<match_thresh:
                            score = dist_ab+dist_ba
                            if score<best_score:
                                best_score = score
                                best_gt_pair = pairs_i
                                matching = (g_b,g_a)
                    elif a_aligned!=-1 and b_aligned!=-1:
                        if g_a == a_aligned and g_b == b_aligned:
                            matching = (g_a,g_b)
                            best_gt_pair = pairs_i
                            break #can't get better than this if restricting alignment
                        elif g_a == b_aligned and g_b == a_aligned:
                            matching = (g_b,g_a)
                            best_gt_pair = pairs_i
                            break #can't get better than this if restricting alignment
                    else:
                        #only one is aligned
                        if a_aligned!=-1:
                            p_loose = p_b
                            e_loose = e_b
                            if g_a == a_aligned:
                                g_have = g_a
                                g_other = g_b
                            elif g_b == a_aligned:
                                g_have = g_b
                                g_other = g_a
                            else:
                                continue #not match for aligned
                        else:
                            p_loose = p_a
                            e_loose = e_a
                            if g_a == b_aligned:
                                g_have = g_a
                                g_other = g_b
                            elif g_b == b_aligned:
                                g_have = g_b
                                g_other = g_a
                            else:
                                continue

                        if BROS:
                            score = norm_ed(transcription_firstline[g_other],e_loose.text_lines[0]) if e_loose.cls==gt_classes[g_other] else 99
                        else:
                            score = norm_ed(transcription_groups[g_other],e_loose.text) if e_loose.cls==gt_classes[g_other] else 99
                        if score<best_score and score<match_thresh:
                            matching = (g_have,g_other) if a_aligned!=-1 else (g_other,g_have)
                            best_gt_pair = pairs_i


                if best_gt_pair!=-1:
                    gt_pair_hit[best_gt_pair]=True
                    pred_to_gt[p_a] = matching[0]
                    pred_to_gt[p_b] = matching[1]
                    rel_truepos+=1
                    good_pred_pairs.append((p_a,p_b))
                else:
                    bad_pred_pairs.append((p_a,p_b))

                #    rel_FP+=1
            assert rel_truepos==sum(gt_pair_hit)
            rel_recall = sum(gt_pair_hit)/len(pairs) if len(pairs)>0 else 1
            rel_prec = rel_truepos/len(pred_links) if len(pred_links)>0 else 1

            #Now look at the entities. We have some aligned already, do the rest
            gt_entities_hit = [[] for i in range(len(transcription_groups))]
            to_align = []
            for p_i in range(len(pred_entities)):
                if p_i in pred_to_gt:
                    gt_entities_hit[pred_to_gt[p_i]].append(p_i)
                else:

                    to_align.append(p_i)

            #resolve ambiguiotity
            for p_is in gt_entities_hit:
                if len(p_is)>1:
                    #can only align one
                    cls= pred_entities[p_is[0]].cls
                    for e_i in p_is[1:]:
                        assert pred_entities[e_i].cls == cls

            for p_i in to_align:
                e_i = pred_entities[p_i]
                best_score = 999999999
                match = None
                for g_i,p_is in enumerate(gt_entities_hit):
                    if len(p_is)==0:
                        if BROS:
                            score = norm_ed(transcription_firstline[g_i],e_i.text_lines[0]) if e_i.cls==gt_classes[g_i] else 99
                        else:
                            score = norm_ed(transcription_groups[g_i],e_i.text) if e_i.cls==gt_classes[g_i] else 99
                        if score<match_thresh and score<best_score:
                            best_score = score
                            match = g_i

                if match is None:
                    #false positive? Split entity?
                    print('No match found for pred entitiy: {}'.format(e_i.text))
                    #import pdb;pdb.set_trace()
                    pass
                else:
                    gt_entities_hit[match].append(p_i)
                    pred_to_gt[p_i]=match

            #check completion of entities (pred have all the lines)
            entities_truepos=0
            for g_i,p_i in enumerate(gt_entities_hit):
                if len(p_i)>0:
                    p_i=p_i[0]
                    p_lines = pred_entities[p_i].text_lines
                    g_lines = transcription_groups[g_i].split('\\')

                    if len(p_lines)==len(g_lines):
                        entities_truepos+=1
                    else:
                        print('Incomplete entity')
                        print('    GT:{}'.format(g_lines))
                        print('  pred:{}'.format(p_lines))
                    

                    
            entity_recall = entities_truepos/len(transcription_groups)
            entity_prec = entities_truepos/len(pred_entities)

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

            total_rel_true_pos += rel_truepos
            total_rel_pred += len(pred_links)
            total_rel_gt += len(pairs)



            if draw:
                mid_points=[]
                for p_i,entity in enumerate(pred_entities):
                    if p_i in pred_to_gt:
                        g_i = pred_to_gt[p_i]
                        cls = entity.cls
                        if cls=='header':
                            color=(0,0,255) #header
                        elif cls=='question':
                            color=(0,255,255) #question
                        elif cls=='answer':
                            color=(255,255,0) #answer
                        elif cls=='other':
                            color=(255,0,255) #other 

                        x1,y1,x2,y2 = bb_lines[groups[g_i][0]]
                        for l_i in groups[g_i][1:]:
                            x1_,y1_,x2_,y2_ = bb_lines[l_i]
                            x1=min(x1,x1_)
                            y1=min(y1,y1_)
                            x2=max(x2,x2_)
                            y2=max(y2,y2_)

                        img_f.rectangle(draw_img,(x1,y1),(x2,y2),color,2)
                        mid_points.append(((x1+x2)//2,(y1+y2)//2))
                    else:
                        print('unmatched entity: {}'.format(entity.text))
                        best=9999999999
                        for g_i,g_text in enumerate(transcription_groups):
                            s = norm_ed(g_text,entity.text)
                            if s<best:
                                best=s
                                best_g = g_i

                        x1,y1,x2,y2 = bb_lines[groups[best_g][0]]
                        mid_points.append((x1,(y1+y2)//2))

                for p_a,p_b in good_pred_pairs:
                    x1,y1 = mid_points[p_a]
                    x2,y2 = mid_points[p_b]
                    img_f.line(draw_img,(x1,y1),(x2,y2),(0,255,0),2)
                for p_a,p_b in bad_pred_pairs:
                    x1,y1 = mid_points[p_a]
                    x2,y2 = mid_points[p_b]
                    img_f.line(draw_img,(x1,y1),(x2,y2),(255,0,0),2)

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


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='run QA model on image(s)')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--image',default=None, type=str,
            help='path to image (default: prompt)')
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
            main(args.checkpoint,args.config,args.image,addtoconfig,True,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw, quiet=args.quiet)
    else:
        main(args.checkpoint,args.config, args.image,addtoconfig,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw,quiet=args.quiet)
