import os
import sys
import signal
import json
import logging
import argparse
import torch
from model import *
from model.loss import *
from model.metric import *
from data_loader import getDataLoader
from trainer import *
from logger import Logger
import requests, socket
import warnings

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

webhook_url = 'https://hooks.slack.com/services/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'

def update_status(name,message,supercomputer):
    if supercomputer:
        return
    name = '{}: {}'.format(socket.gethostname(),name)
    try:
        proxies = {
                  "http": None,
                    "https": None,
                    }
        #print('http://sensei-status.herokuapp.com/sensei-update/{}?message={}'.format(name,message))
        r = requests.get('http://sensei-status.herokuapp.com/sensei-update/{}?message={}'.format(name,message),proxies=proxies)
    except requests.exceptions.ConnectionError:
        pass
    pass

logging.basicConfig(level=logging.INFO, format='')
logging.getLogger('shapely.geos').setLevel(logging.ERROR)

def set_procname(newname):
        from ctypes import cdll, byref, create_string_buffer
        newname=os.fsencode(newname)
        libc = cdll.LoadLibrary('libc.so.6')    #Loading a 3rd party library C
        buff = create_string_buffer(len(newname)+1) #Note: One larger than the name (man prctl says that)
        buff.value = newname                 #Null terminated string as it should be
        libc.prctl(15, byref(buff), 0, 0, 0) #Refer to "#define" of "/usr/include/linux/prctl.h" for the misterious value 16 & arg[3..5] are zero as the man page says.

def main_wraper(rank,config,resume,world_size):
    if 'gpus' not in config:
        config['gpu']=rank
    else:
        config['gpu']=config['gpus'][rank]
    with torch.cuda.device(config['gpu']):
        main(rank,config,resume,world_size)

@slack_sender(webhook_url=webhook_url, channel="herding-neural-networks")
def main(rank,config, resume,world_size=None):
    if rank is not None: #multiprocessing
        #print('Process {} can see these GPUs:'.format(rank,os.environ['CUDA_VISIBLE_DEVICES']))
        if 'distributed' in config:
            os.environ['CUDA_VISIBLE_DEVICES']='0'
            dist.init_process_group(
                            "nccl",
                            init_method='file:///fslhome/brianld/job_comm/{}'.format(config['name']),
                            rank=rank,
                            world_size=world_size)
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

    #np.random.seed(1234) I don't have a way of restarting the DataLoader at the same place, so this makes it totaly random
    train_logger = Logger()

    split = config['split'] if 'split' in config else 'train'
    data_loader, valid_data_loader = getDataLoader(config,split,rank,world_size)
    #valid_data_loader = data_loader.split_validation()

    model = eval(config['arch'])(config['model'])
    model.summary()

    if type(config['loss'])==dict:
        loss={}#[eval(l) for l in config['loss']]
        for name,l in config['loss'].items():
            loss[name]=eval(l)
    else:
        loss = eval(config['loss'])
    if type(config['metrics'])==dict:
        metrics={}
        for name,m in config['metrics'].items():
            metrics[name]=[eval(metric) for metric in m]
    else:
        metrics = [eval(metric) for metric in config['metrics']]

    if 'class' in config['trainer']:
        trainerClass = eval(config['trainer']['class'])
    else:
        trainerClass = Trainer
    trainer = trainerClass(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)

    name=config['name']
    supercomputer = config['super_computer'] if 'super_computer' in config else False

    if rank is not None and rank!=0:
        trainer.side_process=True #this tells the trainer not to log or validate on this thread
    else:
        def handleSIGINT(sig, frame):
            trainer.save()
            update_status(name,'stopped!',supercomputer)
            sys.exit(0)
        signal.signal(signal.SIGINT, handleSIGINT)

    print("Begin training")
    update_status(name,'started',supercomputer)
    #warnings.filterwarnings("error")
    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to checkpoint (default: None)')
    parser.add_argument('-s', '--soft_resume', default=None, type=str,
                        help='path to checkpoint that may or may not exist (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu to use (overrides config) (default: None)')
    parser.add_argument('-R', '--rank', default=None, type=int,
                        help='Set rank for process in distributed training')
    parser.add_argument('-W', '--worldsize', default=None, type=int,
                        help='Set worldsize (num tasks) in distributed training')
    #parser.add_argument('-m', '--merged', default=False, action='store_const', const=True,
    #                    help='Use combine train and valid sets.')

    args = parser.parse_args()

    #warnings.filterwarnings("once")

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
    if  args.resume is None and  args.soft_resume is not None:
        if not os.path.exists(args.soft_resume):
            print('WARNING: resume path ({}) was not found, starting from scratch'.format(args.soft_resume))
        else:
            args.resume = args.soft_resume
    if args.resume is not None and (config is None or 'override' not in config or not config['override']):
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None and args.resume is None:
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if os.path.exists(path):
            directory = os.fsencode(path)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename!='config.json': 
                    assert False, "Path {} already used!".format(path)

    supercomputer = config['super_computer'] if 'super_computer' in config else False
    name=config['name']
    if args.config is not None:
        file_name = args.config[8+3:-5]
        if name!=file_name:
            raise Exception('ERROR, name and file name do not match, {} != {} ({})'.format(name,file_name,args.config))

    assert config is not None

    if args.gpu is not None:
        config['gpu']=args.gpu
        print('override gpu to '+str(config['gpu']))
    set_procname(config['name'])

    try:
        if args.rank is not None:
            config['distributed']=True
            with torch.cuda.device(config['gpu']):
                main(args.rank,config, args.resume, args.worldsize)
        elif 'multiprocess' in config:
            assert(config['cuda'])
            num_gpu_processes=config['multiprocess']
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '8888'
            mp.spawn(main_wraper,
                    args=(config,args.resume,num_gpu_processes),
                    nprocs=num_gpu_processes,
                    join=True)
        elif config['cuda']:
            with torch.cuda.device(config['gpu']):
                main(None,config, args.resume)
        else:
            main(None,config, args.resume)
    except Exception as er:

        #urllib.request.urlopen(url
        update_status(name,er,supercomputer)
        raise er
    else:
        update_status(name,'DONE!',supercomputer)
