import argparse
import torch
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint', default='../..//Downloads/export.pth', type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-o', '--out', type=str,
                        help='out file path')

    args = parser.parse_args()

    assert args.checkpoint is not None
    saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)

    sd = saved['state_dict']
    nsd ={k:v for k,v in sd.items() if not k.startswith('merge_embedding_layer')}
    saved['state_dict']=nsd
    #print(saved['state_dict'].keys())
    if 'swa_state_dict' in saved:
        sd = saved['swa_state_dict']
        nsd ={k:v for k,v in sd.items() if not k.startswith('module.merge_embedding_layer')}
        saved['swa_state_dict']=nsd
        #print(saved['swa_state_dict'].keys())
    del saved['optimizer']
    print(saved.keys())

    new_file = args.out
    torch.save(saved,new_file)
    print('saved '+new_file)
