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
from skimage import future
try:
    import easyocr
except:
    pass


def main(resume,config,img_path,addToConfig,gpu=False,do_pad=False,scale=None):
    np.random.seed(1234)
    torch.manual_seed(1234)
    no_mask_qs = ['fli:','fna:','re~','l~','v~']
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

        #if 'swa_state_dict' in checkpoint and checkpoint['iteration']>config['trainer']['swa_start']:
        #    model = eval(config['arch'])(config['model'])
        #    if 'style' in config['model'] and 'lookup' in config['model']['style']:
        #        model.style_extractor.add_authors(data_loader.dataset.authors) ##HERE
        #    #just strip off the 'module.' tag. I DON'T KNOW IF THIS WILL WORK PROPERLY WITH BATCHNORM
        #    new_state_dict = {key[7:]:value for key,value in checkpoint['swa_state_dict'].items() if key.startswith('module.')}
        #    model.load_state_dict(new_state_dict)
        #    print('Successfully loaded SWA model')
        #elif 'state_dict' in checkpoint:
        #    model = eval(config['arch'])(config['model'])
        #    if 'style' in config['model'] and 'lookup' in config['model']['style']:
        #        model.style_extractor.add_authors(data_loader.dataset.authors) ##HERE
        #    model.load_state_dict(checkpoint['state_dict'])
        #elif 'swa_model' in checkpoint:
        #    model = checkpoint['swa_model']
        #else:
        #    model = checkpoint['model']
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

    with torch.no_grad():
        if img_path is None:
            loop=True
            img_path=input('Image path: ')
        else:
            loop=False
        while img_path!='q':
            img = img_f.imread(img_path,False)
            if img.max()<=1:
                img*=255

            if 'rescale_to_crop_size_first' in config['data_loader'] and  config['data_loader']['rescale_to_crop_size_first']:
                scale_height = do_pad[0]/img.shape[0]
                scale_width = do_pad[1]/img.shape[1]
                choosen_scale = min(scale_height, scale_width)
                if scale:
                    scale*=choosen_scale
                else:
                    scale=choosen_scale



            if scale:
                img = img_f.resize(img,fx=scale,fy=scale)
            
            if do_pad and (img.shape[0]!=do_pad[0] or img.shape[1]!=do_pad[1]):
                diff_x = do_pad[1]-img.shape[1]
                diff_y = do_pad[0]-img.shape[0]
                p_img = np.zeros(do_pad,dtype=img.dtype)
                if diff_x>=0 and diff_y>=0:
                    p_img[diff_y//2:p_img.shape[0]-(diff_y//2 + diff_y%2),diff_x//2:p_img.shape[1]-(diff_x//2 + diff_x%2)] = img
                elif diff_x<0 and diff_y>=0:
                    p_img[diff_y//2:p_img.shape[0]-(diff_y//2 + diff_y%2),:] = img[:,(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                elif diff_x>=0 and diff_y<0:
                    p_img[:,diff_x//2:p_img.shape[1]-(diff_x//2 + diff_x%2)] = img[(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),:]
                else:
                    p_img = img[(-diff_y)//2:-((-diff_y)//2 + (-diff_y)%2),(-diff_x)//2:-((-diff_x)//2 + (-diff_x)%2)]
                img=p_img

            if do_ocr=='no':
                ocr_res=[]
            elif do_ocr:
                ocr_res = ocr_reader.readtext(img,decoder='greedy+softmax')
                print('OCR:')
                for res in ocr_res:
                    print(res[1][0])
            if len(img.shape)==2:
                img=img[...,None] #add color channel
            np_img=img
            img = img.transpose([2,0,1])[None,...]
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
            img = 1.0 - img / 128.0

            if gpu:
                img = img.cuda()

            question = input('Question: ')
            while question!='q':
                if question.startswith('[nr]'):
                    run=False
                    question=question[4:]
                else:
                    run=True
                if do_ocr:
                    ocr = [ocr_res]
                else:
                    ocrBoxes=[[]]
                    ocr=[[]]
                    ocr=(ocrBoxes,ocr)

                needs_input_mask=True
                for q in no_mask_qs:
                    if question.startswith(q):
                        needs_input_mask=False
                        break
                if needs_input_mask:
                    # get input mask
                    mask = future.manual_lasso_segmentation(np_img)
                    if mask.sum()==0:
                        mask = np.zeros_like(mask)
                    mask = torch.from_numpy(mask)[None,None,...].to(img.device) #add batch and color channel
                else:
                    mask = torch.zeros_like(img)
                in_img = torch.cat((img,mask),dim=1)

                answer,pred_mask = model(in_img,ocr,[[question]],RUN=run)
                #pred_a, target_a, answer, pred_mask = model(img,ocr,[[question]],[['number']])
                print('Answer: '+answer+'      max mask={}'.format(pred_mask.max()))
                #show_mask = torch.cat((pred_mask,pred_mask>0.5).float()
                draw_img = 0.5*(1-img)
                threshed = torch.where(pred_mask>0.5,1-draw_img,draw_img)
                #high_score = 2*(pred_mask-0.5)/pred_mask.max()
                #import pdb;pdb.set_trace()
                #high = pred_mask/pred_mask.max()
                #high = torch.where(pred_mask>0.5,high_score,draw_img)
                show_im = torch.cat((draw_img,draw_img-pred_mask,threshed),dim=1)
                #show_im = torch.cat((1-high,draw_img-pred_mask,threshed),dim=1)
                #show_im = torch.cat((high,draw_img,draw_img),dim=1)
                show_im = (show_im[0]*255).cpu().permute(1,2,0).numpy().astype(np.uint8)
                img_f.imshow('x',show_im)
                img_f.show()

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
    parser.add_argument('-s', '--scale', default=None, type=float,
                        help='scale image by this amount')
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
            main(args.checkpoint,args.config,args.image,addtoconfig,True,do_pad=args.pad,scale=args.scale)
    else:
        main(args.checkpoint,args.config, args.image,addtoconfig,do_pad=args.pad,scale=args.scale)
