import argparse
import torch
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-f', '--file', type=str,
                        help='config file')
    parser.add_argument('-o', '--out', type=str,
                        help='out file path')

    args = parser.parse_args()

    assert args.checkpoint is not None
    saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)
    with open(args.file) as f:
        new_cf = json.load(f)
    saved['config']=new_cf


    new_file = args.out
    torch.save(saved,new_file)
    print('saved '+new_file)
