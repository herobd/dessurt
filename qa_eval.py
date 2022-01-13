import os
import json
import logging
import argparse
import torch
from model import *
from model.metric import *
from data_loader import getDataLoader
from evaluators import *
import math
from collections import defaultdict
import random
from trainer import QATrainer

logging.basicConfig(level=logging.INFO, format='')


def main(resume,saveDir,data_set_name,gpu=None, shuffle=False, setBatch=None, config=None, thresh=None, addToConfig=None, test=False,verbose=2,run=False):
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

    if gpu is None:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu
    if thresh is not None:
        config['THRESH'] = thresh
        if verbose:
            print('Threshold at {}'.format(thresh))
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

        
    #config['data_loader']['batch_size']=math.ceil(config['data_loader']['batch_size']/2)
    image_h,image_w = config['model']['image_size']
    if data_set_name is None:
        data_set_name = config['data_loader']['data_set_name']
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
                    "cased": True,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
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
                    "cased": True,
                    "image_size": [
                            image_h-4,image_w-4
                    ],
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
                    "image_size": [
                            image_h-4,image_w-4
                    ],
                    "shuffle": False
                        },
                "validation":{}
                }
    else:
        print('ERROR, unknown dataset: {}'.format(data_set_name))
        exit(1)
    
    data_loader, valid_data_loader = getDataLoader(data_config,'train')


    #if checkpoint is not None:
    #    if 'state_dict' in checkpoint:
    #        model = eval(config['arch'])(config['model'])
    #        model.load_state_dict(checkpoint['state_dict'])
    #    else:
    #        model = checkpoint['model']
    #else:
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

    #data_iter = iter(data_loader)
    metrics = defaultdict(list)
    with torch.no_grad():

        for index,instance in enumerate(valid_data_loader):
            #if index==10:
            #    break
            if verbose:
                print('batch index: {}/{}'.format(index,len(valid_data_loader)),end='\r')
            _,res,_ = trainer.run(instance,valid=True,run=run)

            for name,value in res.items():
                if 'oss' not in name:#name.startswith('mk_firsttoken_top'):
                    if isinstance(value,list):
                        metrics[name]+=value
                    else:
                        metrics[name].append(value)
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

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--data_set_name', default=None, type=str,
                        help='name of dataset to eval')
    parser.add_argument('-o', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-n', '--number', default=0, type=int,
                        help='number of images to save out (from each train and valid) (default: 0)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-b', '--batchsize', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-s', '--shuffle', default=False, type=bool,
                        help='shuffle data')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-m', '--imgname', default=None, type=str,
                        help='specify image')
    parser.add_argument('-t', '--thresh', default=None, type=float,
                        help='Confidence threshold for detections')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn"')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set')
    parser.add_argument('-A', '--autoregressive', default=False, action='store_const', const=True,
                        help='Run with autoregressive prediction')
    parser.add_argument('-v', '--verbosity', default=1, type=int,
                        help='How much stuff to print [0,1,2] (default: 2)')

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

    #if args.index is None and args.imgname is not None:
    #    index = args.imgname
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint, args.savedir, args.data_set_name, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, thresh=args.thresh, addToConfig=addtoconfig,test=args.test,verbose=args.verbosity,run=args.autoregressive)
    else:
        main(args.checkpoint, args.savedir, args.data_set_name, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, thresh=args.thresh, addToConfig=addtoconfig,test=args.test,verbose=args.verbosity,run=args.autoregressive)
