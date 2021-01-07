from datasets.forms_box_detect import FormsBoxDetect
from datasets import forms_box_detect
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
from utils.forms_annotations import calcCorners

from utils import img_f

def display(data):
    batchSize = data['img'].size(0)
    for b in range(batchSize):
        #print (data['img'].size())
        #img = (data['img'][0].permute(1,2,0)+1)/2.0
        img = (data['img'][b].permute(1,2,0)+1)/2.0
        #print(img.shape)
        #print(data['pixel_gt']['table_pixels'].shape)
        if 'pixel_gt' in data and data['pixel_gt'] is not None:
            img[:,:,1] = data['pixel_gt'][b,0,:,:]
        print(data['imgName'][b])



        #fig = plt.figure()
        #gs = gridspec.GridSpec(1, 3)

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)A
        if  img.shape[2]==1:
            img = img_f.gray2rgb(img)

        colors = {  'text_start_gt':'g-',
                    'text_end_gt':'b-',
                    'field_start_gt':'r-',
                    'field_end_gt':'y-',
                    'table_points':'co',
                    'start_of_line':'y-',
                    'end_of_line':'c-',
                    }
        print('num bb:{}'.format(data['bb_sizes'][b]))
        for i in range(data['bb_sizes'][b]):
            xc=data['bb_gt'][b,i,0]
            yc=data['bb_gt'][b,i,1]
            rot=data['bb_gt'][b,i,2]
            #print(rot)
            h=data['bb_gt'][b,i,3]
            w=data['bb_gt'][b,i,4]
            text=data['bb_gt'][b,i,13]
            field=data['bb_gt'][b,i,14]
            if text>0:
                color = (1,0,0)#'b-'
            else:
                color = (0,1,0)#'r-'
            #tr = (int(math.cos(rot)*w-math.sin(rot)*h +xc), int(-math.sin(rot)*w-math.cos(rot)*h +yc))
            #tl = (int(-math.cos(rot)*w-math.sin(rot)*h +xc),int( math.sin(rot)*w-math.cos(rot)*h +yc))
            #br = (int(math.cos(rot)*w+math.sin(rot)*h +xc), int(-math.sin(rot)*w+math.cos(rot)*h +yc))
            #bl = (int(-math.cos(rot)*w+math.sin(rot)*h +xc),int( math.sin(rot)*w+math.cos(rot)*h +yc))
            #print([tr,tl,br,bl])
            tl,tr,br,bl = calcCorners(xc,yc,rot,h,w)
            tl = [int(x) for x in tl]
            tr = [int(x) for x in tr]
            br = [int(x) for x in br]
            bl = [int(x) for x in bl]

            #ax_im.plot([tr[0],tl[0],bl[0],br[0],tr[0]],[tr[1],tl[1],bl[1],br[1],tr[1]],color)
            img_f.polylines(img,np.array([tr,tl,bl,br]),'transparent',color)
            
            if data['bb_gt'].shape[2]>15:
                blank = data['bb_gt'][b,i,15]
                if blank>0:
                    #ax_im.plot(tr[0],tr[1],'mo')
                    img[tr[1]-1:tr[1]+2,tr[0]-1:tr[0]+2,2]=1
                if  data['bb_gt'].size(2)>16:
                    paired = data['bb_gt'][b,i,16]
                    if paired>0:
                        #ax_im.plot(br[0],br[1],'go')
                        img[tr[1]-1:tr[1]+2,tr[0]-1:tr[0]+2,0:2]=1


        if 'line_gt' in data and data['line_gt'] is not None:
            for name, gt in data['line_gt'].items():
                if gt is not None: 
                    #print (gt.size())
                    for i in range(data['line_label_sizes'][name][b]):
                        x0=gt[b,i,0]
                        y0=gt[b,i,1]
                        x1=gt[b,i,2]
                        y1=gt[b,i,3]
                        #print(1,'{},{}   {},{}'.format(x0,y0,x1,y1))

                        ax_im.plot([x0,x1],[y0,y1],colors[name])


        if 'point_gt' in data and data['point_gt'] is not None:
            for name, gt in data['point_gt'].items():
                if gt is not None:
                    #print (gt.size())
                    #print(data)
                    for i in range(data['point_label_sizes'][name][b]):
                        x0=gt[b,i,0]
                        y0=gt[b,i,1]

                        ax_im.plot([x0],[y0],colors[name])
        #plt.show()
        img_f.imshow('page',img)
        img_f.show()
    print('batch complete')


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
    data=FormsBoxDetect(dirPath=dirPath,split='train',config={
	"data_set_name": "FormsBoxDetect",
        "data_dir": "../data/forms",
        "batch_size": 5,
        "shuffle": True,
        "num_workers": 3,
        "crop_to_page":False,
        "color":False,
        "rescale_range": [0.4,0.65],
        "crop_params": {
            "crop_size":[700,1100],
            "pad":0,
            "rot_degree_std_dev": 1,
            "rot_freq": 0.5
        },
        "no_blanks": False,
        "swap_circle":True,
        "no_graphics":True,
        "cache_resized_images": True,
        "only_types": {
            "boxes":True
        },
        "rotation": True
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=forms_box_detect.collate)
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
