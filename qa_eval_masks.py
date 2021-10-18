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
    return editdistance.eval(s1,s2)/max(len(s1),len(s2),1)

def unrollList(model,img,ocr,prev_answer,query,count=10):
    if prev_answer[-1]!=end_token and prev_answer[-1]!=np_token and prev_answer[-1]!=blank_token:
        ret = [prev_answer]
        i=0
        while prev_answer[-1]!=end_token and prev_answer[-1]!=blank_token and prev_answer[-1]!=np_token and i<count*2:
            if prev_answer[-1]=='|':
                ready_answer = prev_answer[:-1]
            else:
                ready_answer = prev_answer

            prev_answer, outmask = model(img,ocr,[[query+ready_answer]],RUN=True)
            print(query+ready_answer+' {:} '+prev_answer)
            if prev_answer[0]=='[' and prev_answer[2]==']':
                prev_answer = prev_answer[3:] #strip off class prediciton
            if prev_answer != end_token and prev_answer != blank_token and prev_answer != np_token:
                ret.append(prev_answer)
            i+=1
        return ret
    else:
        return []
def readLongText(model,img,ocr,answer,reverse=False):
    full_answer=''
    prev_answers=[]
    while len(answer)>0 and answer[-1]!=end_token and answer[-1]!=np_token and answer[-1]!=blank_token:
        full_answer += answer #add to full text
        if reverse:
            new_question='pr~'+answer #form new question from last part
        else:
            new_question='fi~'+answer #form new question from last part
        #TODO masks
        answer,outmask = model(img,ocr,[[new_question]],RUN=True)  #new response
        repeat=False
        for pa in prev_answers:
            if answer==pa:
                repeat=True
                break
        if repeat:
            print('repeat...')
            break
        print(' cont>> {}'.format(answer))
        prev_answers.append(answer)
    if answer != np_token:
        full_answer += answer #finish text
    if len(full_answer)>0 and full_answer[-1]==end_token:
        full_answer = full_answer[:-1]

    if reverse:
        return full_answer[::-1]
    else:
        return full_answer

def main(resume,config,img_path,addToConfig,gpu=False,do_pad=False,test=False,draw=False,max_qa_len=None):
    np.random.seed(1234)
    torch.manual_seed(1234)
    PREVENT_MULTILINE=True
    
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
        max_qa_len=config['data_loader']['max_qa_len']
        
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
        model.load_state_dict(new_state_dict)
    else:
        model = eval(config['arch'])(config['model'])

    model.eval()
    model.max_pred_len=40
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
    with torch.no_grad():
        for instance in valid_iter:
            groups = instance['gt_groups']
            classes_lines = instance['bb_gt'][0,:,-num_classes:]
            loc_lines = instance['bb_gt'][0,:,0:2] #x,y
            bb_lines = instance['bb_gt'][0,:,[5,10,7,12]].long()
            pairs = instance['gt_groups_adj']
            transcription_lines = instance['transcription']
            transcription_lines = [s.lower() for s in transcription_lines]
            img = instance['img'][0]
            print(instance['imgName'])

            gt_line_to_group = instance['targetIndexToGroup']

            transcription_groups = []
            for group in groups:
                transcription_groups.append('\\'.join([transcription_lines[t] for t in group]))


            classes = [classes_lines[group[0]].argmax() for group in groups]
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
                bb_lines[:,0]+=pad_x
                bb_lines[:,1]+=pad_y
                bb_lines[:,2]+=pad_x
                bb_lines[:,3]+=pad_y

            img = img[None,...] #re add batch 
            img = torch.cat((img,torch.zeros_like(img)),dim=1) #add blank mask channel

            if do_ocr=='no':
                ocr = [[]]
            elif do_ocr:
                np_img = (255*(1-img[0])/2).numpy().astype(np.uint64)
                ocr_res = ocr_reader.readtext(np_img,decoder='greedy+softmax')
                print('OCR:')
                for res in ocr_res:
                    print(res[1][0])
                ocr = [ocr_res]
            else:
                ocrBoxes=[[]]
                ocr=[[]]
                ocr=(ocrBoxes,ocr)
            if len(img.shape)==2:
                img=img[...,None] #add color channel

            if gpu:
                img = img.cuda()
            
            #First find tables, as those are done seperately (they should do multiline things alread)
            rel_tables=[]
            used=[]
            question='t#>'
            answer,out_mask = model(img,ocr,[[question]],RUN=True)
            print(question+' {:} '+answer)
            if answer=='yes':
                answer=1
            pred_cells=[]
            pred_cell_classes=[]
            for table_i in range(int(answer)):
                index_start = len(pred_cells)
                #get column and row headers
                question='ch~'+str(table_i)
                answer,out_mask = model(img,ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                if answer==blank_token or answer==np_token:
                    col_headers = []
                else:
                    count_stop = answer.find('>')
                    count = int(answer[:count_stop])
                    answer = answer[count_stop+1:]
                    if count==1:
                        col_headers = [answer]
                    else:
                        col_headers = unrollList(model,img,ocr,answer,'ch>',count)
                    col_headers = [readLongText(model,img,ocr,ans) if (ans[-1]!='|' and ans[-1]!=end_token) else ans[:-1] for ans in col_headers]
                
                question='rh~'+str(table_i)
                answer, out_mask = model(img,ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                if answer==blank_token or answer==np_token:
                    row_headers = []
                else:
                    count_stop = answer.find('>')
                    count = int(answer[:count_stop])
                    answer = answer[count_stop+1:]
                    if count==1:
                        row_headers = [answer]
                    else:
                        row_headers = unrollList(model,img,ocr,answer,'rh>',count)
                    row_headers = [
                            readLongText(model,img,ocr,ans) \
                                    if (ans[-1]!='|' and ans[-1]!=end_token) else ans[:-1] \
                            for ans in row_headers]

                #align headers to gt
                h_to_g = [None]*(len(col_headers)+len(row_headers))
                g_to_h = {}
                c_to_g = {}
                g_to_c = {}
                matchings = [None]*(len(col_headers)+len(row_headers))
                for i,h in enumerate(col_headers+row_headers):
                    matching=[]
                    for gi,text in enumerate(transcription_groups):
                        matching.append((gi,norm_ed(answer,text)))
                    matching.sort(key=lambda a:a[1])
                    matchings[i]=matching
                    best_gi, best_score = matching[0]
                    if best_score<0.7:
                        if best_gi not in g_to_h:
                            h_to_g[i] = best_gi
                            g_to_h[best_gi] = (i,best_score)
                        elif g_to_h[best_gi][1]>best_score:#We're better
                            other = g_to_h[best_gi][0]
                            h_to_g[i] = best_gi
                            g_to_h[best_gi] = (i,best_score)
                            other_place=1
                            while matchings[other][other_place][0] in g_to_h:
                                other_place+=1
                            if matchings[other][other_place][1]<0.7:
                                h_to_g[other] = matchings[other][other_place][0]
                                g_to_h[matchings[other][other_place][0]] = (other,matchings[other][other_place][1])
                used += h_to_g
                ch_to_g = h_to_g[:len(col_headers)]
                rh_to_g = h_to_g[len(col_headers):]

                pred_cells += [groups[g] if g is not None else [-1] for g in h_to_g] #is this cheating?
                pred_cell_classes += [1]*len(pred_cells)
                
                i=len(col_headers)+len(row_headers)
                for ch_i,ch in enumerate(col_headers):
                    for rh_i,rh in enumerate(row_headers):
                        question='t~{}~~{}'.format(ch,rh)
                        answer, out_mask = model(img,ocr,[[question]],RUN=True)
                        print(question+' {:} '+answer)
                        answer = readLongText(model,img,ocr,answer)

                        matching=[]
                        for gi,text in enumerate(transcription_groups):
                            score = norm_ed(answer,text)
                            if score<0.6:
                                matching.append((gi,score))
                        if len(matching)>0:
                            matching.sort(key=lambda a:a[1])
                            assert i == len(matchings)
                            matchings.append(matching)
                            #it's possible there are several texts that are the same
                            #so we'll take distance into account
                            top_matching=matching[0:1]
                            for gi,score in matching[1:]:
                                if score-top_matching[0][1]<0.15:
                                    top_matching.append((gi,score))
                                else:
                                    break
                            if len(top_matching)>1:
                                if ch_to_g[ch_i] is not None and rh_to_g[rh_i] is not None:
                                    table_point = (loc_lines[groups[ch_to_g[ch_i]][0]][0], loc_lines[groups[rh_to_g[rh_i]][0]][1])
                                    matching = [(gi,pointDistance(loc_lines[groups[gi][0]],table_point)) for gi,score in top_matching]
                                else:
                                    matching = [(gi,score) for gi,score in top_matching]
                                matching.sort(key=lambda a:a[1])
                            best_gi, best_score = matching[0]
                            if best_gi not in g_to_c:
                                c_to_g[i] = best_gi
                                g_to_c[best_gi] = (i,best_score)
                            elif g_to_c[best_gi][1]>best_score:#We're better
                                other = g_to_c[best_gi][0]
                                c_to_g[i] = best_gi
                                g_to_c[best_gi] = (i,best_score)
                                other_place=1
                                while matchings[other][other_place][0] in g_to_c:
                                    other_place+=1
                                if matchings[other][other_place][1]<0.7:
                                    c_to_g[other] = matchings[other][other_place][0]
                                    g_to_c[matchings[other][other_place][0]] = (other,matchings[other][other_place][1])
                        else:
                            matchings.append(matching)
                        i+=1
                
                i=len(col_headers)+len(row_headers)
                for ci,ch in enumerate(col_headers):
                    for ri,rh in enumerate(col_headers):
                        assert i == len(pred_cells)
                        if i in c_to_g:
                            g = c_to_g[i]
                            pred_cells.append(groups[g])
                        else:
                            pred_cells.append([-1])
                        pred_cell_classes.append(2)
                        rel_tables.append((index_start+ci,i))
                        rel_tables.append((index_start+ri+len(col_headers),i))
                        i+=1
                used.extend(c_to_g.values())

                #TODO use 'ac~' and 'ar~' as a second check?

                used = [li for g in used if g is not None for li in groups[g]]
                #we purge the table from being processed latter
                #used.sort(reverse=True)
                #for ti in used:
                #    del transcription_lines[ti]
                


            #Next find groups using read 're~' prompt
            read_error=0
            pred_chain = {}
            claimed = {}
            new_id = len(transcription_lines)
            new_transcription_lines=[]
            for ti,textline in enumerate(transcription_lines):
                if ti in used:
                    continue
                if len(textline)>max_qa_len:
                    textline=textline[-max_qa_len:]
                question='f0~'+textline
                if PREVENT_MULTILINE:
                    final_text=textline
                else:
                    tlX,tlY,brX,brY = bb_lines[ti]
                    mask = torch.zeros_like(img[:,1])
                    mask[:,tlY:brY+1,tlX:brX+1] = 1
                    answer, out_mask = model(torch.stack((img[:,0],mask),dim=1),ocr,[[question]],RUN=True)
                    print(question+' {:} '+answer)
                    if answer==np_token:
                        disp_img = (torch.cat((1-2*img[:,0],1-2*img[:,0],mask),dim=0)*255).cpu().permute(1,2,0).numpy().astype(np.uint8)
                        img_f.imshow('s',disp_img)
                        img_f.show()
                    answer = readLongText(model,img,ocr,answer)
                    if answer==np_token or answer == '':
                        final_text = textline
                    else:
                        final_text = textline+(' ' if answer[0]!='\\' else '')+answer
                #now break it into lines
                answer_lines = final_text.split('\\')[1:]
                if len(answer_lines)>0:
                    #rebuilt_answer = ''
                    last_ti = ti
                    for ali,answer_line in enumerate(answer_lines):
                        if answer_line == ' ' or len(answer_line)==0:
                            continue
                        print('Trying to match: {}'.format(answer_line))
                        not_last = ali<len(answer_lines)-1
                        matching=[]
                        for ti2,textline2 in enumerate(transcription_lines):
                            if ti!=ti2 and ti2 not in used:
                                matching.append((ti2,norm_ed(answer_line,textline2)))
                        matching.sort(key=lambda a:a[1])
                        #it's possible there are several texts that are the same
                        #so we'll take distance into account
                        top_matching=matching[0:1]
                        for ti2,score in matching[1:]:
                            if score-top_matching[0][1]<0.15:
                                top_matching.append((ti2,score))
                            else:
                                break
                        if len(top_matching)>1:
                            matching = [(ti2,pointDistance(loc_lines[last_ti],loc_lines[ti2])) for ti2,score in top_matching]
                            matching.sort(key=lambda a:a[1])
                        best_ti2, best_score = matching[0]
                        if best_score<0.7:
                            if last_ti in pred_chain:
                                if best_ti2 != pred_chain[last_ti]: #hopefully we're consistent...
                                    print('Warning: Inconsistent chaining. Matched to [{}], but should be [{}]'.format(best_ti2,pred_chain[last_ti]))

                            print('matched [{}] to [{}]  {}'.format(answer_line,transcription_lines[best_ti2],best_score))
                            #rebuilt_answer+='\\'+transcription_lines[best_ti2]
                            if best_ti2 not in claimed:
                                pred_chain[last_ti] = best_ti2
                                claimed[best_ti2]=(last_ti,best_score)
                                last_ti=best_ti2
                            elif claimed[best_ti2][1]>best_score:#We're better
                                pred_chain[last_ti] = best_ti2
                                del pred_chain[claimed[best_ti2][0]] #they don't claim anymore
                                claimed[best_ti2]=(last_ti,best_score)
                                last_ti=best_ti2
                            else:
                                #We can't claim that instance, so we'll make a new one
                                new_transcription_lines.append(answer_line)
                                pred_chain[last_ti] = new_id
                                claimed[new_id]=(last_ti,None)
                                new_id+=1
                                print('Made new line (alread claimed): {}'.format(answer_line))
                            if best_ti2 not in groups[gt_line_to_group[ti]]:
                                read_error+=1
                        elif len(answer_line)>3 or not_last:
                            #we'll just make a new instance for this line
                            new_transcription_lines.append(answer_line)
                            pred_chain[last_ti] = new_id
                            claimed[new_id]=(last_ti,None)
                            new_id+=1
                            print('Made new line (no match): {}'.format(answer_line))
                    else:
                        if len(groups[gt_line_to_group[ti]])>1:
                            read_error+=1
            print('read accuracy {}'.format((len(transcription_lines)-read_error)/len(transcription_lines)))
            pred_inst = []
            pred_first = []
            pred_groups = []
            transcription_lines += new_transcription_lines
            num_lines = len(transcription_lines)
            for ti in range(num_lines):
                if ti not in claimed and ti not in used: #I'm not the middle of a chain
                    group=[ti]
                    full_text = transcription_lines[ti]
                    pred_first.append(full_text)
                    ti2 = ti
                    history=set([ti])
                    while ti2 in pred_chain:
                        ti2 = pred_chain[ti2]
                        if ti2 in history:
                            break
                            #import pdb;pdb.set_trace() #loop
                        history.add(ti2)
                        group.append(ti2)
                        full_text+=' '+transcription_lines[ti2]
                    pred_inst.append(full_text)
                    pred_groups.append(group)


            #Now get their class
            pred_classes = []
            for text,pred_group in zip(pred_inst,pred_groups):
                text = text[:max_qa_len] #if it's really long, that probably won't help
                question='c$~'+text
                mask = torch.zeros_like(img[:,1])
                for ti in pred_group:
                    if ti < len(bb_lines):
                        tlX,tlY,brX,brY = bb_lines[ti]
                        mask[:,tlY:brY+1,tlX:brX+1] = 1
                answer,out_mask = model(torch.stack((img[:,0],mask),dim=1),ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                pcls = answer[2:-2] #remove '[ ' & ' ]'
                if pcls in valid_data_loader.dataset.classMap:
                    icls = valid_data_loader.dataset.classMap[pcls] - 16
                else:
                    print('Odd class output')
                    icls=len(valid_data_loader.dataset.classMap)-1
                pred_classes.append(icls)


            #We now can calculate the entity scores
            #We'll align the pred_groups to gt ones
            true_pos=0
            group_claimed = [False]*len(groups)
            alignment = {}
            alignment_class = {}
            loc_pgroup=[]
            for pgi,(pgroup,pclass) in enumerate(zip(pred_groups+pred_cells,pred_classes+pred_cell_classes)):
                for ggi,(ggroup,gclass) in enumerate(zip(groups,classes)):
                    if pclass==gclass and len(pgroup)==len(ggroup) and all(x==y for x,y in zip(pgroup,ggroup)) and not group_claimed[ggi]:
                        group_claimed[ggi]=True
                        true_pos+=1
                    #if len(pgroup)==len(ggroup) and all(x==y for x,y in zip(pgroup,ggroup)) and not group_claimed[ggi]:
                    #    group_claimed[ggi]=True
                    #    true_pos+=1
                    if pgroup[0] == ggroup[0]:
                        alignment[pgi]=ggi
                        if pclass==gclass:
                            alignment_class[pgi]=ggi
                pgroup = [li for li in pgroup if li<len(loc_lines)] #filter out unmatched lines
                x = sum(loc_lines[li][0].item() for li in pgroup)/len(pgroup)
                y = sum(loc_lines[li][1].item() for li in pgroup)/len(pgroup)
                loc_pgroup.append((x-pad_x,y-pad_y))

            #we added the table groups to the end, so we'll bump their alignemtn
            rel_tables = [(a+len(pred_groups),b+len(groups)) for a,b in rel_tables]

            entity_recall = true_pos/len(groups) if len(groups)>0 else 1
            entity_prec = true_pos/len(pred_inst) if len(pred_inst)>0 else 1
            total_entity_true_pos += true_pos
            total_entity_pred += len(pred_inst)
            total_entity_gt += len(groups)
            print('Entity precision: {}'.format(entity_prec))
            print('Entity recall:    {}'.format(entity_recall))
            print('Entity Fm:        {}'.format(2*entity_recall*entity_prec/(entity_recall+entity_prec) if entity_recall+entity_prec>0 else 0))

            
            #Now predict linking/pairing
            rel_score=defaultdict(int)
            inconsistent_class_count=0
            for pgi,text in enumerate(pred_inst):
                short_text_front = text[:max_qa_len]
                short_text_back = text[-max_qa_len:]

                mask = torch.zeros_like(img[:,1])
                for ti in pred_groups[pgi]:
                    if ti < len(bb_lines):
                        tlX,tlY,brX,brY = bb_lines[ti]
                        mask[:,tlY:brY+1,tlX:brX+1] = 1
                q_img = torch.stack((img[:,0],mask),dim=1)
                
                if pred_classes[pgi]==0: #header
                    qs=[('h0~',short_text_back,False)]
                elif pred_classes[pgi]==1: #question
                    qs=[('q0~',short_text_front,True),('l0~',short_text_back,False)]
                    #qs=['l~']
                elif pred_classes[pgi]==2: #answer
                    qs=[('v0~',short_text_front,True)]

                for q,t,reverse in qs:
                    question=q+t
                    answer, out_mask = model(q_img,ocr,[[question]],RUN=True)
                    print(question+' {:} '+answer)
                    if answer!=blank_token and answer!=np_token:
                        answer = readLongText(model,img,ocr,answer,reverse=reverse)

                        matching=[]
                        for pgi2,text2 in enumerate(pred_inst):
                            if pgi!=pgi2:
                                score = norm_ed(answer,text2)
                                if score<0.6:
                                    matching.append((pgi2,score))
                        for pgi2,text2 in enumerate(pred_first):
                            if pgi!=pgi2:
                                score = norm_ed(answer,text2)
                                if score<0.6:
                                    matching.append((pgi2,score))
                        #import pdb;pdb.set_trace()
                        if len(matching)>0:
                            matching.sort(key=lambda a:a[1])
                            #it's possible there are several texts that are the same
                            #so we'll take distance into account
                            top_matching=matching[0:1]
                            for pgi2,score in matching[1:]:
                                if score-top_matching[0][1]<0.15:
                                    top_matching.append((pgi2,score))
                                else:
                                    break
                            if len(top_matching)>1:
                                matching = [(pgi2,pointDistance(loc_pgroup[pgi],loc_pgroup[pgi2])) for pgi2,score in top_matching]
                                matching.sort(key=lambda a:a[1])
                            best_pgi2, best_score = matching[0]
                            #if 'stores' in pred_inst[pgi] or 'stores' in pred_inst[best_pgi2]:
                            #    import pdb;pdb.set_trace()
                            rel = (min(pgi,best_pgi2),max(pgi,best_pgi2))
                            rel_score[rel]+=1
                            if (pred_classes[pgi]==0 and pred_classes[best_pgi2]!=1) or (pred_classes[pgi]==2 and pred_classes[best_pgi2]!=1) or (pred_classes[pgi]==1 and pred_classes[best_pgi2]==3):
                                inconsistent_class_count+=1


            #This will be noisy, we'll try and make it a little more consistent
            solid_paired=set()
            for rel,score in rel_score.items():
                if score>1:
                    solid_paired.add(rel[0])
                    solid_paired.add(rel[1])
            chopping_block=[]
            for rel,score in rel_score.items():
                if score==1:
                    if rel[0] in solid_paired:
                        chopping_block.append((rel,rel[1]))
                    if rel[1] in solid_paired:
                        chopping_block.append((rel,rel[0]))

            for examine,test in chopping_block:
                for rel in rel_score:
                    if rel!=examine and test in rel:
                        #The other instance has a relationship, this can probably to safely pruned
                        try:
                            del rel_score[examine]
                        except KeyError:
                            #double del?
                            pass
                        break

            pred_rel = list(rel_score.keys())

            
            #Finally, score the relationships
            true_pos=0
            true_pos_noclass=0
            claimed=set()
            claimed_noclass=set()
            for rel in pred_rel+rel_tables:
                try:
                    a0 = alignment[rel[0]]
                    a1 = alignment[rel[1]]
                    rel_a = (min(a0,a1),max(a0,a1))
                    if rel_a not in claimed and rel_a in pairs:
                        true_pos_noclass+=1
                        claimed_noclass.add(rel_a)
                except KeyError:
                    pass
                try:
                    a0 = alignment_class[rel[0]]
                    a1 = alignment_class[rel[1]]
                    rel_a = (min(a0,a1),max(a0,a1))
                    if rel_a not in claimed and rel_a in pairs:
                        true_pos+=1
                        claimed.add(rel_a)
                except KeyError:
                    pass
                if draw:
                    img_f.line(draw_img,loc_pgroup[rel[0]],loc_pgroup[rel[1]],(0,255,0),2)

            rel_prec = true_pos/len(pred_rel+rel_tables) if len(pred_rel+rel_tables) > 0 else 1
            rel_recall = true_pos/len(pairs) if len(pairs)>0 else 1
            rel_noclass_prec = true_pos_noclass/len(pred_rel+rel_tables) if len(pred_rel+rel_tables)>0 else 1
            rel_noclass_recall = true_pos_noclass/len(pairs) if len(pairs)>0 else 1

            #import pdb;pdb.set_trace()
            total_rel_true_pos += true_pos
            total_rel_pred += len(pred_rel+rel_tables)
            total_rel_gt += len(pairs)
            print('Rel precision: {}'.format(rel_prec))
            print('Rel recall:    {}'.format(rel_recall))
            print('Rel Fm:        {}'.format(2*rel_recall*rel_prec/(rel_recall+rel_prec) if rel_recall+rel_prec> 0 else 0))
            print('Rel_noclass precision: {}'.format(rel_noclass_prec))
            print('Rel_noclass recall:    {}'.format(rel_noclass_recall))
            print('Rel_noclass Fm:        {}'.format(2*rel_noclass_recall*rel_noclass_prec/(rel_noclass_recall+rel_noclass_prec) if rel_noclass_recall+rel_noclass_prec>0 else 0))
            print('inconsistent_class_count={}'.format(inconsistent_class_count))



            ######################
            #New prediction method
            ######################
            pred_classes2 = []
            pred_rel2 = []
            pred_links=[]
            for gi,(text,pred_group) in enumerate(zip(pred_inst,pred_groups)):
                text = text[:max_qa_len] #if it's really long, that probably won't help
                question='g0~'+text
                mask = torch.zeros_like(img[:,1])
                for ti in pred_group:
                    if ti < len(bb_lines):
                        tlX,tlY,brX,brY = bb_lines[ti]
                        mask[:,tlY:brY+1,tlX:brX+1] = 1
                answer, out_mask = model(torch.stack((img[:,0],mask),dim=1),ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                assert answer[0]=='['
                assert answer[2]==']'
                pcls = answer[1] #remove '[ ' & ' ]'
                #expand from single letter, and get class index
                for cls,icls in valid_data_loader.dataset.classMap.items():
                    if cls[0]==pcls:
                        pcls=cls
                        icls-=16
                        break
                pred_classes2.append(icls)
                
                answer = answer[3:]
                if answer==blank_token or answer==np_token:
                    links = []
                else:
                    count_stop = answer.find('>')
                    count = int(answer[:count_stop])
                    answer = answer[count_stop+1:]
                    if count==1:
                        linked_pred = [answer]
                    else:
                        linked_pred = unrollList(model,img,ocr,answer,'gs>',count)
                    #linked_pred = [readLongText(model,img,ocr,ans) for ans in linked_pred]
                    #shouldn't need full text, just enough to match
                    #remove ending characters
                    linked_pred = [ans[:-1] if (ans[-1]=='|' or ans[-1]==end_token) else ans for ans in linked_pred]
                    links = []
                    for pred in linked_pred:
                        best_score=0.6
                        best_match=None
                        for other_gi,text in enumerate(pred_inst):
                            text = text[:len(pred)] #shorten text
                            score = norm_ed(text,pred)
                            if score<best_score:
                                best_score=score
                                best_match=other_gi
                        if best_match is not None:
                            links.append(best_match)
                            pred_rel2.append((gi,best_match))

                pred_links.append(links)
            
            #Now settle the class predictions using the links
            #for gi,pred_class,links in enumerate(zip(pred_classes2,pred_links)):
            #    votes = defaultdict(int)
            #    votes[pred_class]+=1
            #    for link_i in links:
            #        other_class = pred_classes2[link_i]
            #        if other_class=='answer':
            #            votes['question']

            #We now can calculate the entity scores
            #We'll align the pred_groups to gt ones
            true_pos=0
            group_claimed = [False]*len(groups)
            alignment = {}
            alignment_class = {}
            loc_pgroup=[]
            for pgi,(pgroup,pclass) in enumerate(zip(pred_groups+pred_cells,pred_classes2+pred_cell_classes)):
                for ggi,(ggroup,gclass) in enumerate(zip(groups,classes)):
                    if pclass==gclass and len(pgroup)==len(ggroup) and all(x==y for x,y in zip(pgroup,ggroup)) and not group_claimed[ggi]:
                        group_claimed[ggi]=True
                        true_pos+=1
                    if pgroup[0] == ggroup[0]:
                        alignment[pgi]=ggi
                        if pclass==gclass:
                            alignment_class[pgi]=ggi
                pgroup = [li for li in pgroup if li<len(loc_lines)] #filter out unmatched lines
                x = sum(loc_lines[li][0].item() for li in pgroup)/len(pgroup)
                y = sum(loc_lines[li][1].item() for li in pgroup)/len(pgroup)
                loc_pgroup.append((x-pad_x,y-pad_y))

            #we added the table groups to the end, so we'll bump their alignemtn
            rel_tables = [(a+len(pred_groups),b+len(groups)) for a,b in rel_tables]

            entity_recall = true_pos/len(groups) if len(groups)>0 else 1
            entity_prec = true_pos/len(pred_inst) if len(pred_inst)>0 else 1
            total_entity_true_pos2 += true_pos
            total_entity_pred2 += len(pred_inst)
            total_entity_gt2 += len(groups)
            print('New Entity precision: {}'.format(entity_prec))
            print('New Entity recall:    {}'.format(entity_recall))
            print('New Entity Fm:        {}'.format(2*entity_recall*entity_prec/(entity_recall+entity_prec) if entity_recall+entity_prec>0 else 0))

            #Finally, score the relationships
            true_pos=0
            true_pos_noclass=0
            claimed=set()
            claimed_noclass=set()
            for rel in pred_rel2+rel_tables:
                try:
                    a0 = alignment[rel[0]]
                    a1 = alignment[rel[1]]
                    rel_a = (min(a0,a1),max(a0,a1))
                    if rel_a not in claimed and rel_a in pairs:
                        true_pos_noclass+=1
                        claimed_noclass.add(rel_a)
                except KeyError:
                    pass
                try:
                    a0 = alignment_class[rel[0]]
                    a1 = alignment_class[rel[1]]
                    rel_a = (min(a0,a1),max(a0,a1))
                    if rel_a not in claimed and rel_a in pairs:
                        true_pos+=1
                        claimed.add(rel_a)
                except KeyError:
                    pass
                #if draw:
                #    img_f.line(draw_img,loc_pgroup[rel[0]],loc_pgroup[rel[1]],(0,255,0),2)

            rel_prec = true_pos/len(pred_rel2+rel_tables) if len(pred_rel2+rel_tables) > 0 else 1
            rel_recall = true_pos/len(pairs) if len(pairs)>0 else 1
            rel_noclass_prec = true_pos_noclass/len(pred_rel2+rel_tables) if len(pred_rel2+rel_tables)>0 else 1
            rel_noclass_recall = true_pos_noclass/len(pairs) if len(pairs)>0 else 1

            #import pdb;pdb.set_trace()
            total_rel_true_pos2 += true_pos
            total_rel_pred2 += len(pred_rel2+rel_tables)
            total_rel_gt2 += len(pairs)
            print('New Rel precision: {}'.format(rel_prec))
            print('New Rel recall:    {}'.format(rel_recall))
            print('New Rel Fm:        {}'.format(2*rel_recall*rel_prec/(rel_recall+rel_prec) if rel_recall+rel_prec> 0 else 0))
            print('New Rel_noclass precision: {}'.format(rel_noclass_prec))
            print('New Rel_noclass recall:    {}'.format(rel_noclass_recall))
            print('New Rel_noclass Fm:        {}'.format(2*rel_noclass_recall*rel_noclass_prec/(rel_noclass_recall+rel_noclass_prec) if rel_noclass_recall+rel_noclass_prec>0 else 0))

            if draw:
                assert len(pred_classes+pred_cell_classes) == len(loc_pgroup)
                for cls,loc,pgroup in zip(pred_classes+pred_cell_classes,loc_pgroup,pred_groups+pred_cells):
                    if cls==0:
                        color=(0,0,255) #header
                    elif cls==1:
                        color=(0,255,255) #question
                    elif cls==2:
                        color=(255,255,0) #answer
                    elif cls==3:
                        color=(255,0,255) #other 
                    #min_x = draw_img.shape[1]
                    #max_x = 0
                    #min_y = draw_img.shape[0]
                    #max_y = 0

                    group_color = (random.randrange(200),random.randrange(200),random.randrange(200))
                    for li in pgroup:
                        if li<len(loc_lines):
                            x1,y1,x2,y2 = bb_lines[li]
                            x1 = (x1-pad_x).item()
                            y1 = (y1-pad_y).item()
                            x2 = (x2-pad_x).item()
                            y2 = (y2-pad_y).item()
                            img_f.rectangle(draw_img,(x1,y1),(x2,y2),group_color,2)
                            #x=int(loc_lines[li,0].item())
                            #y=int(loc_lines[li,1].item())
                            #draw_img[round(y-3):round(y+3),round(x-3):round(x+3)]=color
                            #min_x = min(x,min_x)
                            #max_x = max(x,max_x)
                            #min_y = min(y,min_y)
                            #max_y = max(y,max_y)
                    #img_f.line(draw_img,(min_x,min_y),(max_x,min_y),group_color,2)
                    #img_f.line(draw_img,(max_x,min_y),(max_x,max_y),group_color,2)
                    #img_f.line(draw_img,(max_x,max_y),(min_x,max_y),group_color,2)
                    #img_f.line(draw_img,(min_x,max_y),(min_x,min_y),group_color,2)
                    draw_img[round(loc[1]-4):round(loc[1]+4),round(loc[0]-4):round(loc[0]+4)]=color

                img_f.imshow('f',draw_img)
                img_f.show()

        total_entity_prec = total_entity_true_pos/total_entity_pred
        total_entity_recall = total_entity_true_pos/total_entity_gt
        total_entity_F = 2*total_entity_prec*total_entity_recall/(total_entity_recall+total_entity_prec) if total_entity_recall+total_entity_prec>0 else 0

        print('old Total entity recall, prec, Fm:\t{}\t{}\t{}'.format(total_entity_recall,total_entity_prec,total_entity_F))

        total_entity_prec = total_entity_true_pos2/total_entity_pred2
        total_entity_recall = total_entity_true_pos2/total_entity_gt
        total_entity_F = 2*total_entity_prec*total_entity_recall/(total_entity_recall+total_entity_prec) if total_entity_recall+total_entity_prec>0 else 0

        print('New Total entity recall, prec, Fm:\t{}\t{}\t{}'.format(total_entity_recall,total_entity_prec,total_entity_F))

        total_rel_prec = total_rel_true_pos/total_rel_pred
        total_rel_recall = total_rel_true_pos/total_rel_gt
        total_rel_F = 2*total_rel_prec*total_rel_recall/(total_rel_recall+total_rel_prec) if total_rel_recall+total_rel_prec>0 else 0
        print('old Total rel recall, prec, Fm:\t{}\t{}\t{}'.format(total_rel_recall,total_rel_prec,total_rel_F))

        total_rel_prec = total_rel_true_pos2/total_rel_pred2
        total_rel_recall = total_rel_true_pos2/total_rel_gt
        total_rel_F = 2*total_rel_prec*total_rel_recall/(total_rel_recall+total_rel_prec) if total_rel_recall+total_rel_prec>0 else 0
        print('New Total rel recall, prec, Fm:\t{}\t{}\t{}'.format(total_rel_recall,total_rel_prec,total_rel_F))

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
            main(args.checkpoint,args.config,args.image,addtoconfig,True,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw)
    else:
        main(args.checkpoint,args.config, args.image,addtoconfig,do_pad=args.pad,test=args.test,max_qa_len=args.max_qa_len, draw=args.draw)
