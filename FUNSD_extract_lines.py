from datasets.funsd_graph_pair import FUNSDGraphPair
from datasets import forms_graph_pair
import math
import sys, os
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch,cv2
from utils.util import ensure_dir

def write(data,out_dir,out_text): #,out_handwriting,out_signature):
    b=0

    #print (data['img'].size())
    #img = (data['img'][0].permute(1,2,0)+1)/2.0
    img = (data['img'][b].permute(1,2,0)+1)/2.0
    img = (img.numpy()*255).astype(np.uint8)
    #print(img.shape)
    #print(data['pixel_gt']['table_pixels'].shape)
    print(data['imgName'])




    #print('num bb:{}'.format(data['bb_sizes'][b]))
    heights=[]
    widths=[]
    #with open(os.path.join(out_dir,'trans.txt'),'w') as out_text:
    for i in range(data['bb_gt'].size(1)):
        xc=data['bb_gt'][b,i,0]
        yc=data['bb_gt'][b,i,1]
        rot=data['bb_gt'][b,i,2]
        h=data['bb_gt'][b,i,3]
        w=data['bb_gt'][b,i,4]
        text=data['bb_gt'][b,i,13]
        field=data['bb_gt'][b,i,14]
        #typeBB= data['metadata'][i]['type']
        if len(data['transcription'])>0:
            trans = data['transcription'][i]
        else:
            trans = '$UNKNOWN$'
        #tr = (math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
        #tl = (math.cos(rot)*-w-math.sin(rot)*h +xc, math.sin(rot)*-w+math.cos(rot)*h +yc)
        #br = (math.cos(rot)*w-math.sin(rot)*-h +xc, math.sin(rot)*w+math.cos(rot)*-h +yc)
        #bl = (math.cos(rot)*-w-math.sin(rot)*-h +xc, math.sin(rot)*-w+math.cos(rot)*-h +yc)
        ##from https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
        #rect = cv2.minAreaRect(np.array([bl,br,tr,tl]))
        ##import pdb;pdb.set_trace()
        rect = ((xc,yc),(2*w,2*h),-180*rot/np.pi)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img[:,:,0], M, (width, height))
        heights.append(warped.shape[0])
        widths.append(warped.shape[1])
        #line_img
        #print([tr,tl,br,bl])
        line_name = "{}_{}.png".format(data['imgName'],i)
        path = os.path.join(out_dir,line_name)
        cv2.imwrite(path,warped)
        out_text.write('{}|{}\n'.format(line_name,trans if trans is not None else '$UNKOWN$'))
    return heights,widths


if __name__ == "__main__":
    dirPath = sys.argv[1]
    out_dir = sys.argv[2]
    data=FUNSDGraphPair(dirPath=dirPath,split='train',config={
        'color':False,
        'crop_to_page':False,
        'rescale_range':[1,1],
        "split_to_lines": True,
        #"only_types": ["text_start_gt"]
})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0, collate_fn=forms_graph_pair.collate)
    dataLoaderIter = iter(dataLoader)

    heights=[]
    widths=[]
    ensure_dir(os.path.join(out_dir))
    out_text = open(os.path.join(out_dir,'text.txt'),'w') 
    #test=10
    try:
        while True:
            h,w =write(dataLoaderIter.next(),out_dir,out_text)
            heights+=h
            widths+=w
            #test-=1
            #if test<0:
            #    quit()
    except StopIteration:
        print('done')
    out_text.close()
    print('height\tm:{}\tstd:{}'.format(np.mean(heights),np.std(heights)))
    print('width\tm:{}\tstd:{}'.format(np.mean(widths),np.std(widths)))
