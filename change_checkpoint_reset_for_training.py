import argparse
import torch
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='remove things from checkpoint')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-o', '--out', type=str,
                        help='out file path')

    args = parser.parse_args()

    assert args.checkpoint is not None
    saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)

    del saved['optimizer']
    if 'swa_state_dict' in saved:
        del saved['swa_state_dict']
    saved['iteration']=0
    saved['logger']=None
    print(saved.keys())

    new_file = args.out
    torch.save(saved,new_file)
    print('saved '+new_file)