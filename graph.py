import os
import sys
import signal
import json
import logging
import argparse
import torch
from collections import defaultdict
import numpy as np
import re

logging.basicConfig(level=logging.INFO, format='')


def graph(log,plot=True,substring=None,regex=None):
    graphs=defaultdict(lambda:{'iters':[], 'values':[]})
    for index, entry in log.entries.items():
        iteration = entry['iteration']
        for metric, value in entry.items():
            if metric!='iteration':
                graphs[metric]['iters'].append(iteration)
                graphs[metric]['values'].append(value)
    
    print('summed')
    skip=[]
    for metric, data in graphs.items():
        #print('{} max: {}, min {}'.format(metric,max(data['values']),min(data['values'])))
        ndata = np.array(data['values'])
        if ndata.dtype is not np.dtype(object):
            maxV = ndata.max(axis=0)
            minV = ndata.min(axis=0)
            meanV = ndata.mean(axis=0)
            print('{} max: {}, min: {}, mean: {}'.format(metric,maxV,minV,meanV))
        else:
            skip.append(metric)

    if plot:
        import matplotlib.pyplot as plt
        i=1
        for metric, data in graphs.items():
            if metric in skip:
                continue
            if (substring is None and regex is None and (metric[:3]=='avg' or metric[:3]=='val')) or (substring is not None and substring in metric) or (regex is not None and re.match(regex,metric)):
                #print('{} == {}? {}'.format(metric[:len(substring)],substring,metric[:len(substring)]==substring))
                plt.figure(i)
                i+=1
                plt.plot(data['iters'], data['values'], '.-')
                plt.xlabel('iterations')
                plt.ylabel(metric)
                plt.title(metric)

                if i>15:
                    print('WARNING, too many windows, stopping')
                    break
        plt.show()
    else:
        i=1
        for metric, data in graphs.items():
            if metric in skip:
                continue
            if (substring is None and regex is None and (metric[:3]=='avg' or metric[:3]=='val')) or (substring is not None and substring in metric) or (regex is not None and re.match(regex,metric)):
                print(metric)
                #print(data['values'])
                for i,v in zip(data['iters'], data['values']):
                    print('{}: {}'.format(i,v))





if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-p', '--plot', default=1, type=int,
                        help='plot (default: True)')
    parser.add_argument('-o', '--only', default=None, type=str,
                        help='only stats with all these substrings (default: None)')
    parser.add_argument('-r', '--regex', default=None, type=str,
                        help='only stats that match regex (default: None)')
    parser.add_argument('-e', '--extract', default=None, type=str,
                        help='instead of ploting, save a new file with only the log (default: None)')
    parser.add_argument('-C', '--printconfig', default=False, type=bool,
                        help='print config (defaut False')

    args = parser.parse_args()

    assert args.checkpoint is not None
    try:
        saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)
    except RuntimeError:
        saved = torch.jit.load(args.checkpoint)
    log = saved['logger']
    iteration = saved['iteration']
    print('loaded iteration {}'.format(iteration))

    if args.printconfig:
        print(saved['config'])
        exit()

    saved=None

    if args.extract is None:
        graph(log,args.plot,args.only,args.regex)
    else:
        new_save = {
                'iteration': iteration,
                'logger': log
                }
        new_file = args.extract #args.checkpoint+'.ex'
        torch.save(new_save,new_file)
        print('saved '+new_file)
