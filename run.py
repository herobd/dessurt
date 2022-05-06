import os
import json
import logging
import argparse
import torch
from model import *
from model.loss import *
from logger import Logger
from trainer import *
import math
from collections import defaultdict
import pickle
#import requests
import warnings
from utils.saliency_qa import InputGradModel
from utils import img_f
from skimage import future


def main(resume,config,img_path,addToConfig=None,gpu=False,do_pad=None,scale=None,do_saliency=False,default_task_token=None,dont_output_mask=False):
    np.random.seed(1234)
    torch.manual_seed(1234)
    no_mask_qs = ['fli:','fna:','re~','l~','v~', 'mm~','mk>','natural_q~','json>','json~','linkdown-text~', 'read_block>']
    remove_qs = ['rm>','mlm>','mm~','mk>']
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
        config['gpu']=None
    else:
        config['cuda']=True
        config['gpu']=gpu

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
        config['model']['init_from_pretrained']=False #don't need to load in weights that will be overrwritten
        model = eval(config['arch'])(config['model'])

        #ugh
        if 'answer_decode.0.weight' in new_state_dict and config['arch']=='MmSwin':
            new_state_dict['answer_decode.weight'] = new_state_dict['answer_decode.0.weight']
            del new_state_dict['answer_decode.0.weight'  ]

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
    if gpu:
        model = model.cuda()

    if do_saliency:
        s_model = InputGradModel(model)

    if do_pad is not None:
        do_pad = do_pad.split(',')
        if len(do_pad)==1:
            do_pad+=do_pad
        do_pad = [int(p) for p in do_pad]
    else:
        do_pad = config['model']['image_size']
        if type(do_pad) is int:
            do_pad = (do_pad,do_pad)

    if default_task_token is not None:
        print('Using default task token: {}'.format(default_task_token))
        print(' (if another token is entered with the query, the default is overridden)')

    with torch.no_grad():
        if img_path is None:
            loop=True
            img_path=input('Image path: ')
        else:
            loop=False
        while img_path!='q':
            img = img_f.imread(img_path,False)
            #import pdb;pdb.set_trace()
            if img.max()<=1:
                img*=255

            if 'rescale_to_crop_size_first' in config['data_loader'] and  config['data_loader']['rescale_to_crop_size_first']:
                scale_height = do_pad[0]/img.shape[0]
                scale_width = do_pad[1]/img.shape[1]
                choosen_scale = min(scale_height, scale_width)
                if scale:
                    new_scale=scale*choosen_scale
                else:
                    new_scale=choosen_scale
            else:
                new_scale=scale




            if new_scale:
                img = img_f.resize(img,fx=new_scale,fy=new_scale)
            
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

            if len(img.shape)==2:
                img=img[...,None] #add color channel
            np_img=img
            img = img.transpose([2,0,1])[None,...]
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
            img = 1.0 - img / 128.0

            if gpu:
                img = img.cuda()

            question = input('Query: ')
            while question!='q':
                if question.startswith('[nr]'):
                    run=False
                    question=question[4:]
                else:
                    run=True

                if default_task_token is not None and '~' not in question and '>' not in question:
                    question = default_task_token+question

                needs_input_mask=True
                for q in no_mask_qs:
                    if question.startswith(q):
                        needs_input_mask=False
                        break
                needs_remove_mask=False
                for q in remove_qs:
                    if question.startswith(q):
                        needs_remove_mask=True
                        break
                if needs_input_mask:
                    # get input mask
                    print('Select input mask')
                    mask = future.manual_lasso_segmentation(np_img)
                    if mask.sum()==0:
                        mask = np.zeros_like(mask)
                    mask = torch.from_numpy(mask)[None,None,...].float().to(img.device) #add batch and color channel
                else:
                    mask = torch.zeros_like(img)
                if needs_remove_mask:
                    # get remove mask
                    print('Select remove mask')
                    rm_mask = future.manual_lasso_segmentation(np_img)
                    rm_mask = torch.from_numpy(rm_mask)[None,None,...].to(img.device) #add batch and color channel
                    #mask[rm_mask]=-1
                    mask = torch.where(rm_mask==1,torch.FloatTensor(*mask.size()).fill_(-1).to(img.device),mask)
                    rm_img = img*(1-rm_mask)
                else:
                    rm_img = img
                in_img = torch.cat((rm_img,mask.to(img.device)),dim=1)
                
                if do_saliency:
                    answer,pred_mask = s_model.saliency(in_img,[[question]])
                else:
                    answer,pred_mask = model(in_img,[[question]],RUN=run)
                    #pred_a, target_a, answer, pred_mask = model(in_img,[[question]],[['number']])
                #print('Answer: {}      max mask={}'.format(answer,pred_mask.max()))
                print('Answer: {}'.format(answer))
                #show_mask = torch.cat((pred_mask,pred_mask>0.5).float()
                if not dont_output_mask:
                    draw_img = 0.5*(1-img)
                    threshed = torch.where(pred_mask>0.5,1-draw_img,draw_img)
                    #high_score = 2*(pred_mask-0.5)/pred_mask.max()
                    #import pdb;pdb.set_trace()
                    #high = pred_mask/pred_mask.max()
                    #high = torch.where(pred_mask>0.5,high_score,draw_img)
                    show_im = torch.cat((draw_img,draw_img*(1-pred_mask),threshed),dim=1)
                    #show_im = torch.cat((1-high,draw_img-pred_mask,threshed),dim=1)
                    #show_im = torch.cat((high,draw_img,draw_img),dim=1)
                    show_im = (show_im[0]*255).cpu().permute(1,2,0).numpy().astype(np.uint8)
                    img_f.imshow('x',show_im)
                    img_f.show()

                question = input('Query ("q" to stop): ')
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
    parser.add_argument('-S', '--saliency', default=False, action='store_const', const=True,
                        help='Run to get saliency map')
    parser.add_argument('-D', '--dont_output_mask', default=False, action='store_const', const=True,
                        help='Don\'t show output mask')
    parser.add_argument('-t', '--task_token', default=None, type=str,
                        help='set a default task token that gets apppended if no other task token is in query')

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
            main(args.checkpoint,args.config,args.image,addtoconfig,True,do_pad=args.pad,scale=args.scale,do_saliency=args.saliency,default_task_token=args.task_token,dont_output_mask=args.dont_output_mask)
    else:
        main(args.checkpoint,args.config, args.image,addtoconfig,do_pad=args.pad,scale=args.scale,do_saliency=args.saliency,default_task_token=args.task_token,dont_output_mask=args.dont_output_mask)
