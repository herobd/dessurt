from data_sets.nobrain_qa import NobrainQA
from data_sets import nobrain_qa
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

hs=[]
ws=[]

def display(data):
    b=0
    bbs = data['bb_gt']
    for b in range(len(data['transcription'])):
        #print(' '.join(data['transcription'][b]))
        for bb,t in zip(bbs[b],data['transcription'][b]):
            print('{} @ {}, {}'.format(t,bb[0],bb[1]))
        print('====')

        for q,a in zip(data['questions'][b],data['answers'][b]):
            print(q+': '+a)
        print('===========================')

    return


if __name__ == "__main__":
    dirPath = sys.argv[1]
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=NobrainQA(dirPath=dirPath,split='train',config={
        'color':False,
        'questions':5,
        "rescale_range": 1.0,
        "crop_params": None,
        "additional_doc_len": 0,
        "shuffle_doc": "all",
        "difficulty": "hard",
        "textfile":"fifteenwords.txt",
        "batch_size": 2,

})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, num_workers=0, collate_fn=nobrain_qa.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    try:
        while True:
            #print('?')
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')

    hs=np.array(hs)
    ws=np.array(ws)
    print('mean: {},{}   min: {},{}   max: {},{}'.format(hs.mean(),ws.mean(),hs.min(),ws.min,hs.max(),ws.max()))
