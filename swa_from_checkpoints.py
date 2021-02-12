import argparse
import torch
from torch.optim.swa_utils import AveragedModel
from model import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='average weights from snapshots')
    parser.add_argument('-o', '--out', type=str,
                        help='out file path')
    parser.add_argument('checkpoints', metavar='C', type=str, nargs='+', help='checkpoint')

    args = parser.parse_args()
    
    
    checkpoint = torch.load(args.checkpoints[0],map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    model = eval(config['arch'])(config['model'])
    model.load_state_dict(checkpoint['state_dict'])

    swa_model = AveragedModel(model)
    
    swa_model.update_parameters(model)
    print(checkpoint['iteration'])

    for checkp in args.checkpoints[1:]:
        checkpoint = torch.load(checkp,map_location=lambda storage, loc: storage)
        config = checkpoint['config']
        model = eval(config['arch'])(config['model'])
        model.load_state_dict(checkpoint['state_dict'])
        
        swa_model.update_parameters(model)
        print(checkpoint['iteration'])

    checkpoint['swa_state_dict'] = swa_model.state_dict()
    checkpoint['config']['name'] += '_SWA'

    torch.save(checkpoint,args.out)
    print('saved {}'.format(args.out))

