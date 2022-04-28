import argparse
import torch
import json
import os

def readRemoveWrite(in_path,out_path,arrange_layers=None,remove_layers=None,rename_layers=None,new_layer_names=None):
    saved = torch.load(in_path,map_location=lambda storage, loc: storage)

    del saved['optimizer']
    if 'swa_state_dict' in saved:
        del saved['swa_state_dict']
    print('was iteration {}'.format(saved['iteration']))
    saved['iteration']=0
    saved['logger']=None

    if arrange_layers is not None:
        arrange = arrange_layers.split(',')
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
        saved['state_dict']=new_sd

    if remove_layers is not None:
        to_remove = remove_layers.split(',')
        sd = saved['state_dict']
        new_sd={}
        for k,v in sd.items():
            if all(rm not in k for rm in to_remove):
                new_sd[k]=v
            else:
                print('Removed '+k)
        saved['state_dict']=new_sd
    if rename_layers is not None:
        to_rename = rename_layers.split(',')
        new_names = new_layer_names.split(',')
        sd = saved['state_dict']
        new_sd={}
        for k,v in sd.items():
            for rename,new in zip(to_rename,new_names):
                if k.startswith(rename):
                    new_k = new+k[len(rename):]
                    print('Renaming {} to {}'.format(k,new_k))
                    k = new_k
            new_sd[k]=v
        saved['state_dict']=new_sd

    print(saved.keys())

    new_file = out_path
    if new_file.endswith('.pth'):
        if not new_file.endswith('checkpoint-latest.pth'):
            print('WARNING: out file is not "checkpoint-latest.pth"!!')
    else:
        if not os.path.exists(new_file):
            os.mkdir(new_file)
        new_file = os.path.join(new_file,'checkpoint-latest.pth')
    torch.save(saved,new_file)
    print('SAVED '+new_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='remove things from checkpoint')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-o', '--out', type=str,
                        help='out file path')
    parser.add_argument('-a', '--arrange_layers', type=str,default=None,
                        help='out file path')
    parser.add_argument('-R', '--rename_layers', type=str,default=None,
                        help='out file path')
    parser.add_argument('-n', '--new_layer_names', type=str,default=None,
                        help='out file path')
    parser.add_argument('-r', '--remove_layers', type=str,default=None,
                        help='remove these layers. comman seperated')

    args = parser.parse_args()

    assert args.checkpoint is not None
    readRemoveWrite(args.checkpoint,args.out,args.arrange_layers,args.remove_layers,args.rename_layers,args.new_layer_names)
