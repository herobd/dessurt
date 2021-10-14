import argparse
import torch
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='remove things from checkpoint')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-o', '--out', type=str,
                        help='out file path')
    parser.add_argument('-a', '--arrange_layers', type=str,default=None,
                        help='out file path')

    args = parser.parse_args()

    assert args.checkpoint is not None
    saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)

    del saved['optimizer']
    if 'swa_state_dict' in saved:
        del saved['swa_state_dict']
    print('was iteration {}'.format(saved['iteration']))
    saved['iteration']=0
    saved['logger']=None

    if args.arrange_layers is not None:
        arrange = args.arrange_layers.split(',')
        sd = saved['state_dict']
        new_sd={}
        for k,v in sd.items():
            if k.startswith('layers.'):
                ks = k.split('.')
                index = int(ks[1])
                if arrange[index]=='r':
                    print('Removing layer: {}'.format(k))
                else:
                    nk = '.'.join(ks[0:1]+[arrange[index]]+ks[2:])
                    new_sd[nk]=v
                    print('Renamed layer {} to {}'.format(k,nk))
            else:
                new_sd[k]=v
        saved['state_dict']=sd


    print(saved.keys())

    new_file = args.out
    torch.save(saved,new_file)
    print('saved '+new_file)
