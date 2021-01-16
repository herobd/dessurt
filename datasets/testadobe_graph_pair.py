from datasets.adobe_graph_pair import AdobeGraphPair
from datasets import adobe_graph_pair
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

def display(data):
    b=0

    #print (data['img'].size())
    #img = (data['img'][0].permute(1,2,0)+1)/2.0
    img = (data['img'][b].permute(1,2,0)+1)/2.0
    #print(img.shape)
    #print(data['pixel_gt']['table_pixels'].shape)
    print(data['imgName'])

    #if data['imgName']!='p2_dv130s.4':
    #    return



    fig = plt.figure()

    ax_im = plt.subplot()
    ax_im.set_axis_off()
    ax_im.imshow(img[:,:,0])

    #print('num bb:{}'.format(data['bb_sizes'][b]))
    if data['bb_gt'] is not None:
        for i in range(data['bb_gt'].size(1)):
            xc=data['bb_gt'][b,i,0]
            yc=data['bb_gt'][b,i,1]
            rot=data['bb_gt'][b,i,2]
            h=data['bb_gt'][b,i,3]
            w=data['bb_gt'][b,i,4]
            text=data['bb_gt'][b,i,13]
            field=data['bb_gt'][b,i,14]
            if text:
                color = 'b-'
            elif field:
                color = 'r-'
            tr = (math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
            tl = (math.cos(rot)*-w-math.sin(rot)*h +xc, math.sin(rot)*-w+math.cos(rot)*h +yc)
            br = (math.cos(rot)*w-math.sin(rot)*-h +xc, math.sin(rot)*w+math.cos(rot)*-h +yc)
            bl = (math.cos(rot)*-w-math.sin(rot)*-h +xc, math.sin(rot)*-w+math.cos(rot)*-h +yc)
            #print([tr,tl,br,bl])

            ax_im.plot([tr[0],tl[0],bl[0],br[0],tr[0]],[tr[1],tl[1],bl[1],br[1],tr[1]],color)
        #for ind1,ind2 in data['adj']:
        #    x1=data['bb_gt'][b,ind1,0]
        #    y1=data['bb_gt'][b,ind1,1]
        #    x2=data['bb_gt'][b,ind2,0]
        #    y2=data['bb_gt'][b,ind2,1]

        #    ax_im.plot([x1,x2],[y1,y2],'m-')
        #    #print('{} to {}, {} - {}'.format(ind1,ind2,(x1,y1),(x2,y2)))

        groupCenters=[]
        show=[]
        for gi,group in enumerate(data['gt_groups']):
                
            if len(group)>1:
                print('{}['.format(gi))
                show.append(gi)

            maxX=maxY=0
            minX=minY=999999999
            for i in group:
                xc=data['bb_gt'][b,i,0]
                yc=data['bb_gt'][b,i,1]
                rot=data['bb_gt'][b,i,2]
                assert(rot==0)
                h=data['bb_gt'][b,i,3]
                w=data['bb_gt'][b,i,4]
                maxX=max(maxX,xc+w)
                maxY=max(maxY,yc+h)
                minX=min(minX,xc-w)
                minY=min(minY,yc-h)
                if len(group)>1:
                    print('{},{}'.format(xc,yc))
            if len(group)>1:
                print(']')
            ax_im.plot([minX,maxX,maxX,minX,minX],[minY,minY,maxY,maxY,minY],'c:')
            #groupCenters.append(((minX+maxX)//2, (minY+minY)//2) )
            groupCenters.append((xc,yc))

        for g1,g2 in data['gt_groups_adj']:
            x1,y1 = groupCenters[g1]
            x2,y2 = groupCenters[g2]
            ax_im.plot([x1,x2],[y1,y2],'c-')
            if g1 in show or g2 in show:
                print(g1,g2)

    plt.show()


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
    data=AdobeGraphPair(dirPath=dirPath,split='train',config={
        'color':False,
        'Xrescale_range':[0.8,1.2],
        'rescale_range':[0.4,0.65],
        'crop_params':{
            "crop_size":[600,800],
            "pad":70,
            "rot_degree_std_dev": 0.7}, 
})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=adobe_graph_pair.collate)
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
