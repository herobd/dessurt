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
from utils.util import pointDistance
import editdistance
from utils.parseIAM import getWordAndLineBoundaries
try:
    import easyocr
except:
    pass

end_token = '‡'
np_token = '№'
blank_token = 'ø'

def readLongText(model,img,ocr,answer):
    full_answer=''
    all_answers=set()
    while len(answer)>0 and answer[-1]!=end_token and '\\' not in answer:
        full_answer += answer #add to full text
        new_question='re~'+answer #form new question from last part
        #TODO masks
        answer,outmask = model(img,ocr,[[new_question]],RUN=True)  #new response
        if answer in all_answers:
            break #prevent infinite loop
        all_answers.add(answer)
        print(' cont>> {}'.format(answer))
    if answer != np_token:
        full_answer += answer #finish text
    if full_answer[-1]==end_token:
        full_answer = full_answer[:-1]
    idx = full_answer.find('\\')
    if idx >= 0:
        full_answer = full_answer[:idx]
    return full_answer

def main(resume,config,addToConfig,gpu=False,crop_size=False,test=False,draw=False,max_qa_len=None,short=False):
    np.random.seed(1234)
    torch.manual_seed(1234)
    dirPath = '../data/IAM'
    
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
    if max_qa_len is None and 'max_qa_len' in config['data_loader']:
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

    if crop_size is not None:
        crop_size = crop_size.split(',')
        if len(crop_size)==1:
            crop_size+=crop_size
        crop_size = [int(p) for p in crop_size]
    else:
        crop_size = config['model']['image_size']
        if type(crop_size) is int:
            crop_size = (crop_size,crop_size)

    ocr = None

    split_by = 'rwth'
    split = 'test' if test else 'valid'
    split_file = os.path.join(dirPath,'ne_annotations','iam',split_by,'iam_{}_{}_6_all.txt'.format(split,split_by))
    doc_set = set()
    with open(split_file) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split('-')
            if len(parts)>1:
              name = '-'.join(parts[:2])
              doc_set.add(name)
    rescale=1.0

    sum_cer=0
    count_lines=0
    total_ed=0
    total_len=0
    with torch.no_grad():
        for name in doc_set:
            xml_path = os.path.join(dirPath,'xmls',name+'.xml')
            image_path = os.path.join(dirPath,'forms',name+'.png')
            W_lines,lines, writer,image_h,image_w = getWordAndLineBoundaries(xml_path)
            maxX=0
            maxY=0
            minX=image_w
            minY=image_h
            for words in W_lines:
                ocr_words=[]
                for word in words:
                    minX = min(minX,word[0][2])
                    minY = min(minY,word[0][0])
                    maxX = max(maxX,word[0][3])
                    maxY = max(maxY,word[0][1])
                    #print(word)
            crop_x1 = max(0,round(minX-40))
            crop_y1 = max(0,round(minY-40))
            crop_x2 = min(image_h,round(maxX+40))
            crop_y2 = min(image_w,round(maxY+40))


            np_img = img_f.imread(image_path,0)
            if np_img.max()<=1:
                np_img*=255
            #print('{} being cropped to {}:{}, {}:{}'.format(np_img.shape,crop_y1,crop_y2,crop_x1,crop_x2))
            np_img = np_img[crop_y1:crop_y2,crop_x1:crop_x2]
            #img_f.imshow('',np_img)
            #img_f.show()

            scale_height = crop_size[0]/np_img.shape[0]
            scale_width = crop_size[1]/np_img.shape[1]
            scale = min(scale_height, scale_width)
            if scale!=1:
                np_img = img_f.resize(np_img,(0,0),
                        fx=scale,
                        fy=scale,
                        )

                
            if len(np_img.shape)==2:
                np_img=np_img[...,None] #add color channel

            img = np_img.transpose([2,0,1]) #from [row,col,color] to [color,row,  col]
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
            img = 1.0 - img / 128.0 #ideally the median value would be 0

            if crop_size and (img.shape[1]<crop_size[0] or img.shape[2]<crop_size[1]):
                diff_x = crop_size[1]-img.shape[2]
                diff_y = crop_size[0]-img.shape[1]
                p_img = torch.FloatTensor(img.size(0),crop_size[0],crop_size[1]).fill_(-1)#np.zeros(crop_size,dtype=img.dtype)
                pad_y = diff_y//2
                pad_x = diff_x//2
                if diff_x>=0 and diff_y>=0:
                    p_img[:,diff_y//2:p_img.shape[1]-(diff_y//2 + diff_y%2),diff_x//2:p_img.shape[2]-(diff_x//2 + diff_x%2)] = img
                elif diff_x<0 and diff_y>=0:
                    p_img[:,diff_y//2:p_img.shape[1]-(diff_y//2 + diff_y%2),:] = img[:,:,(-diff_x)//2:img.shape[2]-((-diff_x)//2 + (-diff_x)%2)]
                elif diff_x>=0 and diff_y<0:
                    p_img[:,diff_x//2:p_img.shape[2]-(diff_x//2 + diff_x%2)] = img[:,(-diff_y)//2:img.shape[1]-((-diff_y)//2 + (-diff_y)%2),:]
                else:
                    p_img = img[:,(-diff_y)//2:img.shape[1]-((-diff_y)//2 + (-diff_y)%2),(-diff_x)//2:img.shape[2]-((-diff_x)//2 + (-diff_x)%2)]
                img=p_img

            img = img[None,...] #add batch 
            img = torch.cat((img,torch.zeros_like(img)),dim=1) #add blank mask channel


            if gpu:
                img = img.cuda()
            if short:
                lines = lines[1:2]
            for line in lines:
                y1,y2,x1,x2 = line[0]
                y1=round((y1-crop_y1)*scale)+pad_y
                x1=round((x1-crop_x1)*scale)+pad_x
                y2=round((y2-crop_y1)*scale)+pad_y
                x2=round((x2-crop_x1)*scale)+pad_x
                gt_text = line[1]

                question=';0>' #read highlighted line
                mask = torch.zeros_like(img[:,1])
                #print(gt_text)
                #print(x1,x2,y1,y2)
                mask[:,y1:y2,x1:x2]=1
                masked_img = torch.stack((img[:,0],mask),dim=1)
                ###
                #show_im = torch.cat((masked_img,masked_img[:,:1]),dim=1)
                #show_im = ((show_im[0].cpu().permute(1,2,0).numpy()+1)*127).astype(np.uint8)
                #img_f.imshow('x',show_im)
                #img_f.show()

                answer,out_mask = model(masked_img,ocr,[[question]],RUN=True)
                print(question+' {:} '+answer)
                if '\\' in answer:
                    idx = answer.find('//')
                    answer = answer[:idx]
                elif answer[-1]!=end_token:
                    #masked_img = torch.stack((img[:,0],mask),dim=1)
                    answer = readLongText(model,masked_img,ocr,answer)
                print('{}\tgt:   {}'.format(count_lines,gt_text))
                print(' \tpred: {}'.format(answer))
                assert '\\' not in answer

                ed = editdistance.eval(answer,gt_text)
                cer = ed/len(gt_text)
                sum_cer += cer
                count_lines += 1
                total_ed += ed
                total_len += len(gt_text)
    
    final_cer = sum_cer/count_lines
    all_cer = total_ed/total_len

    print('CER per line: {}'.format(final_cer))
    print('CER over all: {}'.format(all_cer))

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
    parser.add_argument('-T', '--test', default=False, type=bool,
                        help='run test set (default: False)')
    parser.add_argument('-m', '--max-qa-len', default=None, type=int,
                        help='max len for questions')
    parser.add_argument('-s', '--short', default=False, type=bool,
                        help='only do one line per image (default: False)')

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
            main(args.checkpoint,args.config,addtoconfig,True,crop_size=args.pad,test=args.test,max_qa_len=args.max_qa_len,short=args.short)
    else:
        main(args.checkpoint,args.config,addtoconfig,crop_size=args.pad,test=args.test,max_qa_len=args.max_qa_len,short=args.short)
