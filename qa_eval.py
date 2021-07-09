import os
import json
import logging
import argparse
import torch
from model import *
from model.metric import *
from model.loss import *
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
try:
    import easyocr
except:
    pass

def norm_ed(s1,s2):
    return editdistance.eval(answer,textline2)/max(len(s1),len(s2))

def main(resume,config,img_path,addToConfig,gpu=False,do_pad=False):
    np.random.seed(1234)
    torch.manual_seed(1234)
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
    if do_ocr:
        ocr_reader = easyocr.Reader(['en'],gpu=config['cuda'])
    addDATASET=False
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
        
    if checkpoint is not None:
        if 'swa_state_dict' in checkpoint and checkpoint['iteration']>config['trainer']['swa_start']:
            model = eval(config['arch'])(config['model'])
            if 'style' in config['model'] and 'lookup' in config['model']['style']:
                model.style_extractor.add_authors(data_loader.dataset.authors) ##HERE
            #just strip off the 'module.' tag. I DON'T KNOW IF THIS WILL WORK PROPERLY WITH BATCHNORM
            new_state_dict = {key[7:]:value for key,value in checkpoint['swa_state_dict'].items() if key.startswith('module.')}
            model.load_state_dict(new_state_dict)
            print('Successfully loaded SWA model')
        elif 'state_dict' in checkpoint:
            model = eval(config['arch'])(config['model'])
            if 'style' in config['model'] and 'lookup' in config['model']['style']:
                model.style_extractor.add_authors(data_loader.dataset.authors) ##HERE
            model.load_state_dict(checkpoint['state_dict'])
        elif 'swa_model' in checkpoint:
            model = checkpoint['swa_model']
        else:
            model = checkpoint['model']
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
    config['data_loader']['shuffle']=shuffle
    #config['data_loader']['rot']=False
    config['validation']['shuffle']=shuffle
    config['data_loader']['eval']=True
    config['validation']['eval']=True
    #TODO change to graph pair dataset
    if not test:
        data_loader, valid_data_loader = getDataLoader(config,'train')
    else:
        valid_data_loader, data_loader = getDataLoader(config,'test')
        data_loader = valid_data_loader
    valid_iter = iter(valid_data_loader)

    with torch.no_grad():
        for instance in valid_iter:
            groups = instance['gt_groups']
            classes_lines = instance['bb_gt'][0,:,-num_classes:]
            pairs = instance['gt_groups_adj']
            transcription_lines = instance['transcription']
            img = instance['img']

            classes = [classes_lines[group[0]].argmax() for group in groups]
            
            if do_pad and (img.shape[0]<do_pad[0] or img.shape[1]<do_pad[1]):
                diff_x = do_pad[1]-img.shape[1]
                diff_y = do_pad[0]-img.shape[0]
                p_img = torch.FloatTensor(1,do_pad[0],do_pad[1]).fill_(-1)#np.zeros(do_pad,dtype=img.dtype)
                if diff_x>=0 and diff_y>=0:
                    p_img[diff_y//2:-(diff_y//2 + diff_y%2),diff_x//2:-(diff_x//2 + diff_x%2)] = img
                elif diff_x<0 and diff_y>=0:
                    p_img[diff_y//2:-(diff_y//2 + diff_y%2),:] = img[:,(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                elif diff_x>=0 and diff_y<0:
                    p_img[:,diff_x//2:-(diff_x//2 + diff_x%2)] = img[(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),:]
                else:
                    p_img = img[(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                img=p_img

            if do_ocr:
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
            question='t#>'+textline
            answer = model(img,ocr,[[question]],RUN=True)
            print(question+' {:} '+answer)
            pred_cells=[]
            for table_i in range(int(answer)):
                question='ch~'+str(table_i)
                answer = model(img,ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                column_headers = [h.strip() for h in answer.split(',')]
                
                question='rh~'+str(table_i)
                answer = model(img,ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                row_headers = [h.strip() for h in answer.split(',')]

                h_to_g = [None]*(len(column_headers)+len(row_headers))
                g_to_h = {}
                matchings = [None]*(len(column_headers)+len(row_headers))
                for i,h in enumerate(column_headers+row_headers):
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
                to_remove = h_to_g
                ch_to_g = h_to_g[:len(column_headers)]
                rh_to_g = h_to_g[len(column_headers):]

                pred_cells = [put in the group line ids?]
                pred_cell_classes = [1]*len(pred_cells)

                for ch in column_headers:
                    for rh in column_headers:
                        question='t~{}~~{}'.format(ch,rh)
                        answer = model(img,ocr,[[question]],RUN=True)
                        print(question+' {:} '+answer)

                        for gi,text in enumerate(transcription_groups):
                            matching.append((gi,norm_ed(answer,text)))
                        matching.sort(key=lambda a:a[1])
                        matchings[i]=matching
                        best_gi, best_score = matching[0]
                        if best_score<0.7:
                            pred_cells.append(group line ids)
                            pred_cell_classes.append(2)#answer
                            rel_table.append(?)
                            rel_table.append(?)
                        


                


            #Next find groups using read 're~' prompt
            pred_chain = {}
            claimed = {}
            for ti,textline in enumerate(transcription_lines):
                question='re~'+textline
                answer = model(img,ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                if len(answer)>0 and answer!=' ':
                    for ti2,textline2 in enumerate(transcription_lines):
                        if ti!=t2:
                            matching.append((ti2,norm_ed(answer,textline2)))
                    matching.sort(key=lambda a:a[1])
                    best_ti2, best_score = matching[0]
                    if best_score<0.7:
                        if best_ti2 not in claimed:
                            pred_chain[ti] = best_ti2
                            claimed[ti2]=(ti,best_score)
                        elif claimed[ti2][1]>best_score:#We're better
                            pred_chain[ti] = best_ti2
                            del pred_chain[claimed[ti2][0]] #they don't claim anymore
                            claimed[ti2]=(ti,best_score)

            pred_inst = []
            pred_first = []
            pred_groups = []
            num_lines = len(transcription_lines)
            for ti in range(num_lines):
                if ti not in claimed: #I'm not the middle of a chain
                    group=[ti]
                    full_text = transcription_lines[ti]
                    pred_first.append(full_text)
                    ti2 = ti
                    while ti2 in pred_chain:
                        ti2 = pred_chain[ti]
                        group.append(ti2)
                        full_text+=' '+transcription_lines[ti2]
                    pred_inst.append(full_text)
                    pred_groups.append(group)


            #Now get their class
            pred_classes = []
            for text in pred_inst:
                text = text[:100] #if it's really long, that probably won't help
                question='cs~'+textline
                answer = model(img,ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                pcls = answer[2:-2] #remove '[ ' & ' ]'
                if pcls in valid_data_loader.dataset.classMap:
                    icls = valid_data_loader.dataset.classMap[pcls] 
                else:
                    print('Odd class output')
                    icls=len(valid_data_loader.dataset.classMap)-1
                pred_classes.append(icls)

            #We now can calculate the entity scores
            true_pos=0
            group_claimed = [False]*len(groups)
            alignment = {}
            for pgroup,pclass in zip(pred_groups+pred_cells,pred_classes+pred_cell_classes):
                for ggi,(ggroup,gclass) in enumerate(zip(groups,classes)):
                    if pclass==gclass and len(pgroup)==len(ggroup) and all(x==y for x,y in zip(pgroup,ggroup)) and not group_claimed[ggi]:
                        group_claimed[ggi]=True
                        true_pos+=1
                    if len(pgroup)==len(ggroup) and all(x==y for x,y in zip(pgroup,ggroup)) and not group_claimed[ggi]:
                        group_claimed[ggi]=True
                        true_pos+=1


            entity_recall = true_pos/len(groups)
            entity_prec = true_pos/len(pred_inst)

            
            #Now predict linking/pairing
            rel_score=defaultdict(int)
            inconsistent_class_count=0
            for pgi,text in enumerate(pred_inst):
                short_text = text[:100]
                
                if pred_classes[pgi]==0: #header
                    qs=['hd~']
                elif pred_classes[pgi]==1: #question
                    qs=['qu~','l~']
                elif pred_classes[pgi]==2: #answer
                    qs=['v~']
                for q in qs:
                    question=q+text
                    answer = model(img,ocr,[[question]],RUN=True)
                    print(question+' {:} '+answer)
                    if len(answer)>0 and answer!='[ blank ]' and answer!='[ np ]':
                        for pgi2,text2 in enumerate(pred_inst):
                            if pgi!=pgi2:
                                matching.append((pgii2,norm_ed(answer,text2)))
                        for pgi2,text2 in enumerate(pred_first):
                            if pgi!=pgi2:
                                matching.append((pgii2,norm_ed(answer,text2)))
                        matching.sort(key=lambda a:a[1])
                        best_ti2, best_score = matching[0]
                        if best_score<0.8:
                            rel = (min(pgi,pgi2),max(pgi,pgi2))
                            rel_score[rel]+=1
                            if (pred_classes[pgi]==0 and pred_classes[pgi2]!=1) or (pred_classes[pgi]==2 and pred_classes[pgi2]!=1) or (pred_classes[pgi]==1 and pred_classes[pgi2]==3):
                                inconsistent_class_count+=1
            #This will be noisy, we'll try and make it a little more consistent
            solid_paired=set()
            for rel,score in rel_score.items():
                if score>1:
                    sold_paired.add(rel[0])
                    sold_paired.add(rel[1])

            for rel,score in rel_score.items():
                if score==1:
                    if rel[0] in solid_paired:
                        chopping_block.append((rel,rel[1]))
                    if rel[1] in solid_paired:
                        chopping_block.append((rel,rel[0]))

            for examine,test in chopping_block:
                for rel in rel_score:
                    if rel!=examine and test in rel:





            question = input('Question: ')
            while question!='q':
                if question.startswith('[nr]'):
                    run=False
                    question=question[4:]
                else:
                    run=True
                answer = model(img,ocr,[[question]],RUN=run)
                print('Answer: '+answer)
                #answer = model(img,ocr,[[question]],[['ok']])
                #print(answer[-1])

                question = input('Question ("q" to stop): ')
            if loop:
                img_path = input('Image path ("q" to stop): ')
            else:
                img_path = 'q'


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
            main(args.checkpoint,args.config,args.image,addtoconfig,True,do_pad=args.pad)
    else:
        main(args.checkpoint,args.config, args.image,addtoconfig,do_pad=args.pad)
