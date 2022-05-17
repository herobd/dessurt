import os
import json
import logging
import argparse
import torch
import numpy as np
from model import *
from data_loader import getDataLoader
import math
from collections import defaultdict
import random
from trainer import QATrainer
from funsd_eval_json import derepeat,fixLoadJSON
import editdistance
import re
from transformers import BartTokenizer

logging.basicConfig(level=logging.INFO, format='')

############
#This script will run evaluation of all of the datasets except FUNSD and NAF, which have their own special scripts
#
#It takes the training snapshot (weights) and dataset, and computes the validation (or test) metrics for that dataset



# OCR METRICS: https://github.com/FactoDeepLearning/VerticalAttentionOCR/blob/master/basic/utils.py


def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit


def format_string_for_wer(string):
    string = string.replace('\\',' ')#my own newline predictions
    string = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', string)
    string = re.sub('([ \n])+', " ", string).strip()
    return string


def edit_wer_from_list(truth, pred):
    edit = 0
    for pred, gt in zip(pred, truth):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
    return edit


def nb_words_from_list(list_gt):
    len_ = 0
    for gt in list_gt:
        gt = format_string_for_wer(gt)
        gt = gt.split(" ")
        len_ += len(gt)
    return len_


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])


def cer_from_list_str(str_gt, str_pred):
        len_ = 0
        edit = 0
        for pred, gt in zip(str_pred, str_gt):
            edit += editdistance.eval(gt, pred)
            len_ += len(gt)
        cer = edit / len_
        return cer


def wer_from_list_str(str_gt, str_pred):
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return cer


def main(resume,data_set_name,gpu=None,  config=None, addToConfig=None, test=False,verbose=1,run=False,smaller_set=False,eval_full=None,ner_do_before=False):
    assert run
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded iteration {}'.format(checkpoint['iteration']))
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
    else:
        checkpoint = None
        config = json.load(open(config))
    if run:
        config['validation']['batch_size']=1

    if gpu is None:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu
    if addToConfig is not None:
        for add in addToConfig:
            addTo=config
            if verbose:
                printM='added config['
            for i in range(len(add)-2):
                addTo = addTo[add[i]]
                if verbose:
                    printM+=add[i]+']['
            value = add[-1]
            if value=="":
                value=None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            addTo[add[-2]] = value
            if verbose:
                printM+=add[-2]+']={}'.format(value)
                print(printM)
            if (add[-2]=='useDetections' or add[-2]=='useDetect') and value!='gt':
                addDATASET=True

    #Set up the dataset   
    image_h,image_w = config['model']['image_size']
    if data_set_name is None:
        data_set_name = config['data_loader']['data_set_name']
    get=None
    if data_set_name=='SynthParaQA':
        data_config={
                "data_loader": {
                    "data_set_name": "SynthParaQA",
                    "data_dir": "../data/fonts",
                    "mode": "mk_only",
                    "cased": True,
                    "batch_size": config['data_loader']['batch_size']*2,
                    "num_workers": 4,
                    "rescale_range": [0.9,1.1],
                    "crop_params": {
                            "crop_size": [
                                image_h,image_w
                            ],
                            "pad": 0,
                            "rot_degree_std_dev": 1
                        },
                        "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 2560,
                        "image_size": [
                                image_h-4,image_w-4
                        ],
                    "shuffle": False,
                    "num_batches": 1000
                        },
                "validation":{}
                }
    elif data_set_name=='SQuAD':
        data_config={
                "data_loader": {
                    "data_set_name": "SQuAD",
                    "data_dir": "../data/SQuAD",
                    "batch_size": config['data_loader']['batch_size']*3 if not run else 1,
                    "half": smaller_set,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "random": False
                    },
                    "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 2560,
                    "cased": True,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
                    "shuffle": False
                        },
                "validation":{}
                }
    elif data_set_name=='NAFRead':
        data_config={
                "data_loader": {
                    "data_set_name": "NAFRead",
                    "data_dir": "../data/forms",
                    "batch_size": config['data_loader']['batch_size']*3 if not run else 1,
                    "rescale_to_crop_size_first": True,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "pad": 0,
                        "random": False
                    },
                    "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 256000,
                    "cased": True,
                    'crop_to_q': True,
                    'min_text_height': 21,
                    "shuffle": False
                        },
                "validation":{}
                }
    elif data_set_name=='DocVQA':
        data_config={
                "data_loader": {
                    "data_set_name": "DocVQA",
                    "data_dir": "../data/DocVQA",
                    "batch_size": config['data_loader']['batch_size']*3 if not run else 1,
                    "rescale_to_crop_size_first": True,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "random": False
                    },
                    "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 2560,
                    "cased": True,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
                    "shuffle": False
                        },
                "validation":{}
                }
        if test:
            get=['pred']
    elif data_set_name=='HWSQuAD':
        data_config={
                "data_loader": {
                    "data_set_name": "HWSQuAD",
                    "data_dir": "../data/HW-SQuAD",
                    "batch_size": config['data_loader']['batch_size']*3 if not run else 1,
                    "rescale_to_crop_size_first": True,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "random": False
                    },
                    "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 2560,
                    "cased": True,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
                    "shuffle": False
                        },
                "validation":{}
                }
    elif data_set_name=='SROIE':
        data_config={
                "data_loader": {
                    "data_set_name": "SROIE",
                    "data_dir": "../data/SROIE",
                    "batch_size": config['data_loader']['batch_size']*3 if not run else 1,
                    "rescale_to_crop_size_first": True,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "random": False
                    },
                    "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 2560,
                    "cased": True,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
                    "shuffle": False
                        },
                "validation":{}
                }
        get=['pred']
    elif data_set_name=='RVL':
        data_config={
                "data_loader": {
                    "data_set_name": "RVLCDIPClass",
                    "data_dir": "../data/rvl-cdip",
                    "batch_size": config['validation']['batch_size'] if not run else 1,
                    "rescale_to_crop_size_first": True,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "random": False
                    },
                    "questions": 1,
                    "max_qa_len_in": 640,
                    "max_qa_len_out": 99999999999,
                    "cased": True,
                    "shuffle": False
                        },
                "validation":{}
                }
    elif data_set_name=='IAMNER':
        data_config={
                "data_loader": {
                    "data_set_name": "IAMNER",
                    "data_dir": "../data/IAM",
                    "batch_size": config['data_loader']['batch_size']*(3 if 'full' in config['data_loader'] else 2) if not run else 1,
                    "full": config['data_loader'].get('full',False),
                    "cased": config['data_loader'].get('cased',False),
                    "task": config['data_loader'].get('task',6),
                    "data_split": config['data_loader'].get('data_split','rwth'),
                    "eval_class_before": ner_do_before,
                    "eval_full": not test if eval_full is None else eval_full,
                    "rescale_to_crop_size_first": True,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "random": False
                    },
                    "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 2560,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
                    "shuffle": False
                        },
                "validation":{}
                }
    elif data_set_name=='IAMQA':
        data_config={
                "data_loader": {
                    "data_set_name": "IAMQA",
                    "data_dir": "../data/IAM",
                    "batch_size": config['data_loader']['batch_size']*(3 if 'full' in config['data_loader'] else 2) if not run else 1,
                    "cased": config['data_loader'].get('cased',True),
                    "mode": "IAM_para",
                    "data_split": config['data_loader']['data_split'] if 'data_split' in config['data_loader'] else "Coquenet",
                    "rescale_to_crop_size_first": True,
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            image_h,image_w
                        ],
                        "random": False
                    },
                    "questions": 1,
                        "max_qa_len_in": 640,
                        "max_qa_len_out": 25600000,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
                    "shuffle": False
                        },
                "validation":{}
                }
        get=['pred','gt']
        if os.path.exists('./cache_huggingface/BART'):
            model_id = './cache_huggingface/BART'
        else:
            model_id = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(model_id)
    else:
        print('unspecified dataset: {}'.format(data_set_name))
        data_config = config
        #print('Implemented datasets: DocVQA, RVL, IAMNER, IAMQA (page recognition), HWSQuAD, SROIE, SynthParaQA (evaluate word infilling), SQuAD (my synthetic version), NAFRead (recognition on NAF)')
        #exit(1)
    
    print('getting data ready')
    data_loader, valid_data_loader = getDataLoader(data_config,'train' if not test else 'test')

    if test:
        valid_data_loader = data_loader

    if resume is not None:
        config['model']['init_from_pretrained']=False #don't need to load in weights that will be overrwritten
    model = eval(config['arch'])(config['model'])
    model.eval()
    if verbose==2:
        model.summary()

    if gpu is not None:
        model = model.to(gpu)
    else:
        model = model.cpu()

    if 'multiprocess' in config:
        del config['multiprocess']
    if 'distributed' in config:
        del config['distributed']
    trainer = QATrainer(model,{},None,resume,config,data_loader,valid_data_loader)
    print('go!')

    #data_iter = iter(data_loader)
    metrics = defaultdict(list)
    collected_preds=[]
    gts=[]
    preds=[]
    with torch.no_grad():

        index=0
        for instance in valid_data_loader:
            if verbose:
                print('batch index: {}/{}'.format(index,len(valid_data_loader)),end='\r')

            #Run model on data instance
            _,res,out = trainer.run(instance,valid=True,run=run,get=get)

            for name,value in res.items():
                if 'oss' not in name:#name.startswith('mk_firsttoken_top'):
                    if isinstance(value,list):
                        metrics[name]+=value
                    else:
                        metrics[name].append(value)
            if data_set_name=='DocVQA' and test:
                collected_preds.append({
                    'questionId': int(instance['id'][0]),
                    'answer': out['pred']
                    })
            elif data_set_name=='SROIE':
                name = instance['imgName'][0][instance['imgName'][0].rfind('/')+1:]
                try:
                    data =fixLoadJSON(out['pred']) 
                except:
                    data = {}
                with open('sroie_results{}/{}.txt'.format(run if type(run) is str else '',name),'w') as f:
                    json.dump(data,f)
            elif data_set_name=='IAMQA':
                gts.append(out['gt'])
                init_len = len(out['pred'])
                fixed = derepeat(out['pred'])
                if verbose>1:
                    print('==================')
                    print('GT:   '+gts[-1])
                    print('PRED: '+fixed)
                if init_len>len(fixed):
                    #query again to try and recover lost text
                    #only do this once
                    tokens = tokenizer.encode(fixed)
                    tokens = tokens[-12:]
                    prompt = 're~'+tokenizer.decode(tokens,skip_special_tokens=True)
                    instance['questions'] = [[prompt]]
                    _,res,out = trainer.run(instance,valid=True,run=run,get=get)
                    response = derepeat(out['pred'])
                    fixed += response
                    if verbose>1:
                        print('FINAL PRED: '+fixed)
                if fixed[-1]=='‡':
                    fixed = fixed[:-1]
                preds.append(fixed)
            

            index+=1

    #Print the metrics
    F_measure_prec={}
    F_measure_recall={}
    for name,values in metrics.items():
        print('{} mean:{},  std:{}'.format(name,np.mean(values),np.std(values)))
        if 'F_prec' in name:
            last_underscore = name.rfind('_')
            var_name = name[last_underscore+1:]
            if var_name != 'o':
                F_measure_prec[var_name]=np.mean(values)
        if 'F_recall' in name:
            last_underscore = name.rfind('_')
            var_name = name[last_underscore+1:]
            if var_name != 'o':
                F_measure_recall[var_name]=np.mean(values)
                print('Class count [{}] = {}'.format(var_name,len(values)))
    names = set(F_measure_prec.keys())
    names.update(F_measure_recall.keys())
    if len(names)>0:
            total_Fms=0
            for name in names:
                p = F_measure_prec[name] if name in F_measure_prec else 1
                r = F_measure_recall[name] if name in F_measure_recall else 1
                f = 2*(p*r)/(p+r) if (p+r)>0 else 0
                total_Fms+=f
                print('{} Fm={},  P={},  R={}'.format(name,f,p,r))
            print('Macro Fm = {}'.format(total_Fms/len(names)))

    if data_set_name=='IAMQA':
        cer = cer_from_list_str(gts,preds)
        wer = wer_from_list_str(gts,preds)
        print('CER: {},  WER: {}'.format(cer,wer))

    if len(collected_preds)>0:
        #For submitting to evaluation server
        with open('DocVQA_OUT.json','w') as f:
            json.dump(collected_preds,f)
        print(' wrote DocVQA_OUT.json')

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Evaluator for Dessurt on non-form datasets')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (required)')
    parser.add_argument('-d', '--data_set_name', default=None, type=str,
                        help='name of dataset to eval (required)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='override config with supplied json')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn"')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set (default is validation set). For DocVQA outputs submission file DocVQA_OUT.json')
    parser.add_argument('-b', '--beam_search', default=False, type=int,
                        help='Do beam search using this number of beams')
    parser.add_argument('-1', '--teacherforcing', default=False, action='store_const', const=True,
                        help='Run with teacher forcing')
    parser.add_argument('-S', '--smaller_set', default=False, action='store_const', const=True,
                        help='Use less of val set, ONLY FOR SQUAD')
    parser.add_argument('-v', '--verbosity', default=1, type=int,
                        help='How much stuff to print [0,1,2] (default: 1)')
    parser.add_argument('-F', '--eval_full', default=None, type=bool,
                        help='for iamNER, whether to do whole doc (instead of lines)')
    parser.add_argument('-N', '--ner_do_before', default=False, action='store_const', const=True,
                        help='do NER evaluation class first (as opposed to word first like normal)')

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

    if args.beam_search:
        run = 'beam{}'.format(args.beam_search)
    else:
        run = not args.teacherforcing

    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint, args.data_set_name, gpu=args.gpu, config=args.config, addToConfig=addtoconfig,test=args.test,verbose=args.verbosity,run=run,smaller_set=args.smaller_set,eval_full=args.eval_full,ner_do_before=args.ner_do_before)
    else:
        main(args.checkpoint, args.data_set_name, gpu=args.gpu, config=args.config, addToConfig=addtoconfig,test=args.test,verbose=args.verbosity,run=run,smaller_set=args.smaller_set,eval_full=args.eval_full,ner_do_before=args.ner_do_before)
