from datasets.forms_feature_pair import FormsFeaturePair
from datasets import forms_feature_pair
import math, cv2
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

ex=1

def plotRect(img,color,xyrhw):
        xc=xyrhw[0]
        yc=xyrhw[1]
        rot=xyrhw[2]
        h=xyrhw[3]
        w=xyrhw[4]
        h = min(30000,h)
        w = min(30000,w)
        if h ==0:
            h=2
        tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(-w*math.sin(rot)-h*math.cos(rot) + yc) )
        tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(w*math.sin(rot)-h*math.cos(rot) + yc) )
        br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
        bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(w*math.sin(rot)+h*math.cos(rot) + yc) )

        #cv2.line(img,tl,tr,color,1)
        #cv2.line(img,tr,br,color,1)
        #cv2.line(img,br,bl,color,1)
        #cv2.line(img,bl,tl,color,1)
        cv2.line(img,tl,tr,(255,0,0),1)
        cv2.line(img,tr,br,(255,255,0),1)
        cv2.line(img,br,bl,(0,255,255),1)
        #cv2.line(img,bl,tl,(255,0,255),1)

def display(instance):
    imagePath = instance['imgPath']
    qXY = instance['qXY']
    iXY = instance['iXY']
    label = instance['label']
    relNodeIds = instance['nodeIds']
    gtNumNeighbors=instance['numNeighbors']+1
    missedRels = instance['missedRels']
    data = instance['data'][0]
    batchSize=data.size(0)

    #print (data['img'].size())
    #img = (data['img'][0].permute(1,2,0)+1)/2.0
    image = cv2.imread(imagePath) #read as color
    #print(img.shape)
    #print(data['pixel_gt']['table_pixels'].shape)
    print(instance['imgName'])




    #print('num bb:{}'.format(data['bb_sizes'][b]))
    for b in range(batchSize):
        x,y = qXY[b]
        r = data[b,2].item()*math.pi
        h = data[b,0].item()*50/2
        w = data[b,1].item()*400/2
        plotRect(image,(0,0,255),(x,y,r,h,w))
        x2,y2 = iXY[b]
        r = data[b,6+ex].item()*math.pi
        h = data[b,4+ex].item()*50/2
        w = data[b,5+ex].item()*400/2
        plotRect(image,(0,0,255),(x2,y2,r,h,w))

        #r = data[b,2].item()
        #h = data[b,0].item()
        #w = data[b,1].item()
        #plotRect(image,(1,0,0),(x,y,r,h,w))

        if label[b].item()> 0:
           cv2.line(image,(int(x),int(y)),(int(x2),int(y2)),(0,255,0),1)

    fig = plt.figure()
    #gs = gridspec.GridSpec(1, 3)

    ax_im = plt.subplot()
    ax_im.set_axis_off()
    ax_im.imshow(image)#[:,:,0])
    ax_im.text(5, 5, 'missed:{}'.format(missedRels))
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
    data=FormsFeaturePair(dirPath=dirPath,split='valid',config={
        "data_set_name": "FormsFeaturePair",
        "eval": True,
        "special_dataset": "simple",
        "alternate_json_dir": "out_json/Simple18_staggerLight_NN_new",
        "data_dir": "../data/forms",
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        "no_blanks": True,
        "swap_circle":True,
        "no_graphics":True,
        "cache_resized_images": False,
        "rotation": False,
        "balance": False,
        "only_opposite_pairs": True,
        "corners":True
}
)

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=forms_feature_pair.collate)
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
