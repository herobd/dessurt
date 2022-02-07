from data_sets import cdip_cloud_qa
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
import utils.img_f as cv2
from collections import defaultdict

widths=[]

def display(data):
    batchSize = data['img'].size(0)
    #mask = makeMask(data['image'])
    question_types=[]
    for b in range(batchSize):
        #print (data['img'].size())
        img = (1-data['img'][b,0:1].permute(1,2,0))/2.0
        #img[:,:,1][img[:,:,1]<1]=0
        #img = torch.cat((img,1-data['img'][b,1:2].permute(1,2,0),1-data['mask_label'][b].permute(1,2,0)),dim=2)
        img = torch.cat((img,img,img),dim=2)
        show = data['img'][b,1]>0
        mask = data['img'][b,1]<0
        img[:,:,0] *= ~mask
        img[:,:,1] *= ~show
        img[:,:,2] *= 1-data['mask_label'][b,0]
        #img[2,img[2]<1]=0

        #label = data['label']
        #gt = data['gt'][b]
        #print(label[:data['label_lengths'][b],b])
        print(data['imgName'][b])
        #if data['spaced_label'] is not None:
        #    print('spaced label:')
        #    print(data['spaced_label'][:,b])
        #for bb,text in zip(data['bb_gt'][b],data['transcription'][b]):
        #    print('ocr: {} {}'.format(text,bb))
        #print('questions: {}'.format(data['questions'][b]))
        #print('answers: {}'.format(data['answers'][b]))
        print('questions and answers')
        for q,a in zip(data['questions'][b],data['answers'][b]):
            print(q+' : '+a)
            
            loc = q.find('~')
            if loc ==-1:
                loc = q.find('>')
                if loc ==-1:
                    loc = len(q)
            question_types.append(q[:loc])

        #widths.append(img.size(1))
        
        draw=True#q.startswith('mm')
        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            #cv2.imwrite('out/fg_mask{}.png'.format(b),fg_mask.numpy()*255)
            #cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)
            #cv2.imwrite('out/changed_img{}.png'.format(b),changed_img.numpy()*255)
            #plt.imshow(img.numpy()[:,:,0], cmap='gray')
            #plt.show()
            cv2.imshow('x',(img*255).numpy().astype(np.uint8))
            cv2.show()

            #cv2.imwrite('testsinglesize_1024.png',img.numpy()[:,:,0])

        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
    print('batch complete')
    return question_types

if __name__ == "__main__":
    if len(sys.argv)>1:
        dirPath = sys.argv[1]
    else:
        dirPath = '../data/CDIPCloud'
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=cdip_cloud_qa.CDIPCloudQA(dirPath=dirPath,split='train',config={
        'super_computer': False,
        'rescale_range': [0.9,1.1],
        'rescale_to_crop_size_first': True,
        '#rescale_to_crop_width_first': True,
        'crop_params': {
            "#crop_size":[960,1280],
            "crop_size":[1152,768],
            "#crop_size":[768,768],
            "pad":0,
            "rot_degree_std_dev": 1
            },
        'questions':1,
        "max_qa_len_in": 640,
        "max_qa_len_out": 2560,
        "cased": True,
        "mode": "hard_word",
        'reuse_factor': 0.0001, #about 5 images

})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0, collate_fn=cdip_cloud_qa.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    try:
        question_types = defaultdict(int)
        while True:
            #print('?')
            q_t=display(dataLoaderIter.next())
            for q in q_t:
                question_types[q]+=1
            print('question_types:')
            print(question_types)
    except StopIteration:
        print('done')

    #print('width mean: {}'.format(np.mean(widths)))
    #print('width std: {}'.format(np.std(widths)))
    #print('width max: {}'.format(np.max(widths)))
