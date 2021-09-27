import argparse
import torch
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='remove things from checkpoint')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-o', '--out', type=str,default=None,
                        help='out file path')
    parser.add_argument('-r', '--replace', type=str,
                        help='replace this')
    parser.add_argument('-w', '--with_this', type=str,
                        help='with this')

    args = parser.parse_args()


    assert args.checkpoint is not None
    saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)

    print('replacing "{}" with "{}"'.format(args.replace,args.with_this))

    for state_dict_id in ['state_dict','swa_state_dict']:
        sd = saved[state_dict_id]
        new_sd={}
        for k,v in sd.items():
            if args.replace in k:
                #import pdb;pdb.set_trace()
                print('>>> {}'.format(k))
            k = k.replace(args.replace,args.with_this)
            if args.replace in k:
                print('<<< {}'.format(k))
            new_sd[k]=v
        saved[state_dict_id]=new_sd


    if args.out is None:
        new_file = args.checkpoint
    else:
        new_file = args.out
    torch.save(saved,new_file)
    print('saved '+new_file)
