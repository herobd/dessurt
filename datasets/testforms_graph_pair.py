from datasets.forms_graph_pair import FormsGraphPair
from datasets import forms_graph_pair
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

only=''

def display(data):
    b=0

    #print (data['img'].size())
    #img = (data['img'][0].permute(1,2,0)+1)/2.0
    img = (data['img'][b].permute(1,2,0)+1)/2.0
    #print(img.shape)
    #print(data['pixel_gt']['table_pixels'].shape)
    if len(only)>0:
        if only not in data['imgName']:
            return
    print(data['imgName'])



    fig = plt.figure()
    #gs = gridspec.GridSpec(1, 3)

    ax_im = plt.subplot()
    ax_im.set_axis_off()
    ax_im.imshow(img[:,:,0],cmap='gray')

    colors = {  'text_start_gt':'g-',
                'text_end_gt':'b-',
                'field_start_gt':'r-',
                'field_end_gt':'y-',
                'table_points':'co'
                }
    #print('num bb:{}'.format(data['bb_sizes'][b]))
    for i in range(data['bb_gt'].size(1)):
        xc=data['bb_gt'][b,i,0]
        yc=data['bb_gt'][b,i,1]
        rot=data['bb_gt'][b,i,2]
        h=data['bb_gt'][b,i,3]
        w=data['bb_gt'][b,i,4]
        text=data['bb_gt'][b,i,13]
        field=data['bb_gt'][b,i,14]
        if text>0:
            color = 'b-'
        else:
            color = 'r-'
        tr = (math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
        tl = (math.cos(rot)*-w-math.sin(rot)*h +xc, math.sin(rot)*-w+math.cos(rot)*h +yc)
        br = (math.cos(rot)*w-math.sin(rot)*-h +xc, math.sin(rot)*w+math.cos(rot)*-h +yc)
        bl = (math.cos(rot)*-w-math.sin(rot)*-h +xc, math.sin(rot)*-w+math.cos(rot)*-h +yc)
        #print([tr,tl,br,bl])

        ax_im.plot([tr[0],tl[0],bl[0],br[0],tr[0]],[tr[1],tl[1],bl[1],br[1],tr[1]],color)
    #groups=[]
    #groupMap={}
    for ind1,ind2 in data['adj']:
        x1=data['bb_gt'][b,ind1,0]
        y1=data['bb_gt'][b,ind1,1]
        x2=data['bb_gt'][b,ind2,0]
        y2=data['bb_gt'][b,ind2,1]

        ax_im.plot([x1,x2],[y1,y2],'g-')
        #print('{} to {}, {} - {}'.format(ind1,ind2,(x1,y1),(x2,y2)))

    #    if data['bb_gt'][b,ind1,13]==data['bb_gt'][b,ind2,13]:
    #        if ind1 not in groupMap and ind2 not in groupMap:
    #            groups.append([ind1,ind2])
    #            groupMap[ind1]=groups[-1]
    #            groupMap[ind2]=groups[-1]
    #        elif ind1 not in groupMap:
    #            groupMap[ind2].append(ind1)
    #            groupMap[ind1]=groupMap[ind2]
    #        elif ind2 not in groupMap:
    #            groupMap[ind1].append(ind2)
    #            groupMap[ind2]=groupMap[ind1]
    #        else:
    #            goneGroup = groupMap[ind2]
    #            groups.remove(groupMap[ind2])
    #            groupMap[ind1] += goneGroup
    #            for indx in goneGroup:
    #                groupMap[indx] = groupMap[ind1]

    #for group in groups:
    #    maxX=0
    #    maxY=0
    #    minX=9999999
    #    minY=9999999
    #    for i in group:
    #        xc=data['bb_gt'][b,i,0]
    #        yc=data['bb_gt'][b,i,1]
    #        rot=data['bb_gt'][b,i,2]
    #        h=data['bb_gt'][b,i,3]
    #        w=data['bb_gt'][b,i,4]
    #        text=data['bb_gt'][b,i,13]
    #        field=data['bb_gt'][b,i,14]
    #        if text>0:
    #            color = 'y:'
    #        else:
    #            color = 'm:'
    #        tr = (math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
    #        tl = (math.cos(rot)*-w-math.sin(rot)*h +xc, math.sin(rot)*-w+math.cos(rot)*h +yc)
    #        br = (math.cos(rot)*w-math.sin(rot)*-h +xc, math.sin(rot)*w+math.cos(rot)*-h +yc)
    #        bl = (math.cos(rot)*-w-math.sin(rot)*-h +xc, math.sin(rot)*-w+math.cos(rot)*-h +yc)
    #        maxX = max(maxX,tr[0],tl[0],br[0],bl[0])
    #        minX = min(minX,tr[0],tl[0],br[0],bl[0])
    #        maxY = max(maxY,tr[1],tl[1],br[1],bl[1])
    #        minY = min(minY,tr[1],tl[1],br[1],bl[1])
    #    ax_im.plot([minX,maxX,maxX,minX,minX],[minY,minY,maxY,maxY,minY],color)
    groupCenters=[]
    for group in data['gt_groups']:
        maxX=maxY=0
        minX=minY=999999999
        for i in group:
            xc=data['bb_gt'][b,i,0]
            yc=data['bb_gt'][b,i,1]
            rot=data['bb_gt'][b,i,2]
            h=data['bb_gt'][b,i,3]
            w=data['bb_gt'][b,i,4]
            tr = (math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
            tl = (math.cos(rot)*-w-math.sin(rot)*h +xc, math.sin(rot)*-w+math.cos(rot)*h +yc)
            br = (math.cos(rot)*w-math.sin(rot)*-h +xc, math.sin(rot)*w+math.cos(rot)*-h +yc)
            bl = (math.cos(rot)*-w-math.sin(rot)*-h +xc, math.sin(rot)*-w+math.cos(rot)*-h +yc)
            maxX=max(maxX,tr[0],tl[0],br[0],bl[0])
            minX=min(minX,tr[0],tl[0],br[0],bl[0])
            maxY=max(maxY,tr[1],tl[1],br[1],bl[1])
            minY=min(minY,tr[1],tl[1],br[1],bl[1])

        if len(group)>1:
            ax_im.plot([minX,maxX,maxX,minX,minX],[minY,minY,maxY,maxY,minY],'c:')
        groupCenters.append(((minX+maxX)//2, (minY+minY)//2,len(group)>1 ))

    for g1,g2 in data['gt_groups_adj']:
        x1,y1,big1 = groupCenters[g1]
        x2,y2,big2 = groupCenters[g2]
        if big1 or big2:
            ax_im.plot([x1,x2],[y1,y2],'c-')
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
    data=FormsGraphPair(dirPath=dirPath,split='test',config={
	"data_set_name": "FormsGraphPair",
        "special_dataset": None,
        "data_dir": "../data/forms",
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 2,
        "crop_to_page":False,
        "color":False,
        "rescale_range": [0.4,0.65],
        "crop_params": None,
        "#crop_params": {
            "crop_size":[600,1200],
            "pad":0
        },
        "no_blanks": False,
        "swap_circle":True,
        "no_graphics":True,
        "cache_resized_images": True,
        "rotation": True,
        "only_opposite_pairs": False,
        "no_groups": True
})

    print('dataset len: {}'.format(len(data)))
    exit()

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0, collate_fn=forms_graph_pair.collate)
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
