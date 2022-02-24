import argparse
import torch
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')

    args = parser.parse_args()

    assert args.checkpoint is not None
    cp = args.checkpoint
    if not cp.endswith('.pth'):
        cp = 'saved/'+cp+'/checkpoint-latest.pth'
    saved = torch.load(cp,map_location=lambda storage, loc: storage)

    #print('arch: {}'.format(saved['arch']))
    #Eprint('arch: {}'.format(saved['config']['arch']))
    #print(saved.keys())
    #print(type(saved['logger'].entries))
    if 'swa_state_dict' in saved:
        print(saved['swa_state_dict']['n_averaged'])

    print('{} / {}'.format(saved['iteration'],saved['config']['trainer']['iterations']))

    #for name in saved['state_dict'].keys():
    #    print(name)

