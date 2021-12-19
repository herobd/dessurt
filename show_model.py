import os
import sys
import signal
import json
import logging
import argparse
import torch
from model import *

logging.basicConfig(level=logging.INFO, format='')





if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Show model description')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')

    args = parser.parse_args()

    assert args.checkpoint is not None
    try:
        saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)
    except RuntimeError:
        saved = torch.jit.load(args.checkpoint)
    
    config = saved['config']
    print(config['name'])


    model = eval(config['arch'])(config['model'])
    model.summary()
