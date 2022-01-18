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
def fixLoadJSON(pred):
    pred_data = None
    while pred_data is None:
        try:
            pred_data = json.loads(pred)
        except json.decoder.JSONDecodeError as e:
            print(e)
            err,typ,loc = '{}'.format(e).split(':')
            assert 'line 1' in loc
            loc_char = loc.find('char ')
            loc_char_end = loc.rfind(')')
            char = loc[loc_char+1:loc_char_end]
            
            if "Expecting ',' delimiter" in typ:
                if char==len(pred):
                    #closing ] or }?
                    bracket = pred.rfind('[')
                    curley = pred.rfind('{')
                    assert bracket!=-1 or curley!=-1
                    if bracket>curley:
                        pred+=']'
                    else:
                        pred+='}'
                else:
                    #pred+=','
                    assert False
            elif 'Unterminated string starting at' in typ:
                pred+='"'
            elif 'Expecting value' in typ:
                pred+='""'
            elif 'Expecting ':' delimiter' in typ:
                pred+=':'
            elif 'Expecting property name enclosed in double quotes' in typ:
                if pred[-1]==',':
                    pred=pred[-1]
                pred+='}'
            elif 'Expecting value' in typ:
                if pred[-1]==',':
                    pred=pred[-1]
                else:
                    assert False
            else:
                assert False
    return pred_data

class Entity():
    def __init__(self,text,cls,idetity):
        self.text=text
        self.cls=cls
        self.id=identity
def parseDict(header,entities,links):
    to_link=[]
    is_table=False
    for text,value in header:
        if text=='content' or text=='answers':
            if isinstance(value,list):
                for thing in value:
                    to_link+=parseDict(thing,entities,links)
            else:
                assert isinstance(value,dict)
                to_link+=parseDict(value,entities,links)
        elif text=='row headers':
            assert isinstance(value,list)
            row_headers = value
            is_table = True
        elif text=='col headers':
            assert isinstance(value,list)
            col_headers = value
            is_table = True
        else:
            if isinstance(value,str)
                my_text = text
                my_class = value
            elif isinstance(value,list) and text=='cell':
                is_table=True
                cells = value
    if not is_table:
        my_id=len(entities)
        entities.append(Entity(text,cls,my_id))
        for other_id in to_link:
            links.append((my_id,other_id))
        return [my_id]
    else:
        #a table
        row_ids = list(range(len(entities),len(entities)+len(row_headers)))
        for rh in row_headers:
            entities.append(Entity(rh,'question',len(entities)))
        col_ids = list(range(len(entities),len(entities)+len(col_headers)))
        for ch in col_headers:
            entities.append(Entity(ch,'question',len(entities)))

        for r,row in enumerate(cells):
            for c,cell in enumerate(row):
                c_id = len(entities)
                entities.append(Entity(cell,'answer',c_id))
                links.append((row_ids[r],c_id))
                links.append((col_ids[c],c_id))

        return row_ids+col_ids




def main(resume,config,img_path,addToConfig,gpu=False,do_pad=False,test=False,draw=False,max_qa_len=None,quiet=False):
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
            transcription_lines = [s if cased else s for s in transcription_lines]
            img = instance['img'][0]
            if not quiet:
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

            if gpu:
                img = img.cuda()
            
            #First find tables, as those are done seperately (they should do multiline things alread)
            question='json>'
            answer,out_mask = model(img,ocr,[[question]],RUN=True)
            pred_data = fixLoadJSON(answer)
            
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
            match_thresh = 0.6
            gt_pair_hit=[False]*len(pairs)
            pred_to_gt=defaultdict(list)
            for p_a,p_b in pred_links:
                e_a = pred_entities[p_a]
                e_b = pred_entities[p_b]

                best_score = 99999
                best_gt_pair = -1
                for pairs_i,(g_a,g_b) in enumerate(pairs):
                    #can't match to a gt pair twice
                    if gt_pair_hit[pairs_i]:
                        continue
                    dist_aa = norm_ed(transcription_groups[g_a],e_a.text)
                    dist_bb = norm_ed(transcription_groups[g_b],e_b.text)
                    dist_ab = norm_ed(transcription_groups[g_a],e_b.text)
                    dist_ba = norm_ed(transcription_groups[g_b],e_a.text)
                    
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
                            matching = (g_b,g_b)

                if best_gt_pair!=-1:
                    gt_pair_hit[best_gt_pair]=True
                    pred_to_gt[p_a] = matching[0]
                    pred_to_gt[p_b] = matching[1]
                    rel_truepos+=1
                #else:
                #    rel_FP+=1
            
            rel_recall = sum(gt_pair_hit)/len(pairs)
            rel_prec = rel_truepos/len(pred_links)

            #Now look at the entities. We have some aligned already, do the rest
            for p_i in range(len(pred_entities)):
                if pred_to_gt
            #TODO

            entity_recall = true_pos/len(groups) if len(groups)>0 else 1
            entity_prec = true_pos/len(pred_inst) if len(pred_inst)>0 else 1
            total_entity_true_pos += true_pos
            total_entity_pred += len(pred_inst)
            total_entity_gt += len(groups)
            if not quiet:
                print('Entity precision: {}'.format(entity_prec))
                print('Entity recall:    {}'.format(entity_recall))
                print('Entity Fm:        {}'.format(2*entity_recall*entity_prec/(entity_recall+entity_prec) if entity_recall+entity_prec>0 else 0))

            total_rel_true_pos += true_pos
            total_rel_pred += len(pred_rel+rel_tables)
            total_rel_gt += len(pairs)
            if not quiet:
                print('Rel precision: {}'.format(rel_prec))
                print('Rel recall:    {}'.format(rel_recall))
                print('Rel Fm:        {}'.format(2*rel_recall*rel_prec/(rel_recall+rel_prec) if rel_recall+rel_prec> 0 else 0))
                print('Rel_noclass precision: {}'.format(rel_noclass_prec))
                print('Rel_noclass recall:    {}'.format(rel_noclass_recall))
                print('Rel_noclass Fm:        {}'.format(2*rel_noclass_recall*rel_noclass_prec/(rel_noclass_recall+rel_noclass_prec) if rel_noclass_recall+rel_noclass_prec>0 else 0))
                print('inconsistent_class_count={}'.format(inconsistent_class_count))



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
