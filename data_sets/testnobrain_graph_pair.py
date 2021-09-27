from data_sets.nobrain_graph_pair import NobrainGraphPair
from data_sets import nobrain_graph_pair
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

    print(' '.join(data['form_metadata']['word_trans']))
    print('====')

    for q,a in zip(data['questions'],data['answers']):
        print(q+' '+a)
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
    data=NobrainGraphPair(dirPath=dirPath,split='train',config={
        'color':False,
        'rescale_range':[0.8,1.2],
        'questions':40
})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=nobrain_graph_pair.collate)
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
