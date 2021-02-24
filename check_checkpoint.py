import argparse
import torch
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')

    args = parser.parse_args()

    assert args.checkpoint is not None
    saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)

    print(saved.keys())
    #print(type(saved['logger'].entries))
    if 'swa_state_dict' in saved:
        print(saved['swa_state_dict']['n_averaged'])
    print(saved['iteration'])

