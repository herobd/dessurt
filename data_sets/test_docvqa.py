from data_sets import docvqa
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
import utils.img_f as cv2
from collections import defaultdict
from transformers import BartTokenizer
import os
widths=[]

def display(data,tokenizer=None):
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
        if data['mask_label'] is not None:
            img[:,:,2] *= 1-data['mask_label'][b,0]
        #img[2,img[2]<1]=0

        print('questions and answers')
        for q,a in zip(data['questions'][b],data['answers'][b]):
            print(q+' [:] '+a)
            
            loc = q.find('~')
            if loc ==-1:
                loc = q.find('>')
                if loc ==-1:
                    loc = len(q)
            question_types.append(q[:loc])
        
        if tokenizer is not None:
            tok_len = len(tokenizer.tokenize(a))+2
        #widths.append(img.size(1))
        
        draw=False#'0w' in q or 'w0' in q or 'l0' in q
        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            #cv2.imwrite('out/fg_mask{}.png'.format(b),fg_mask.numpy()*255)
            #cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)
            #cv2.imwrite('out/changed_img{}.png'.format(b),changed_img.numpy()*255)
            #plt.imshow(img.numpy()[:,:,0], cmap='gray')
            #plt.show()
            img = (img*255).numpy().astype(np.uint8)
            cv2.imshow('x',img)
            cv2.show()


        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
    print('batch complete')
    return tok_len

if __name__ == "__main__":
    if len(sys.argv)>1:
        dirPath = sys.argv[1]
    else:
        dirPath = '../data/DocVQA'
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=docvqa.DocVQA(dirPath=dirPath,split='train',config={
        'batch_size':1,
        #'gt_ocr': True,
        'rescale_range':[0.9,1.1],
        'rescale_to_crop_size_first': True,
        'crop_params': {
            "crop_size":[768,768],
            "pad":0,
            "rot_degree_std_dev": 1
            },
        'questions':1,
        "image_size":[768,768],
        "prefetch_factor": 5,
        "persistent_workers": True,
        "cased": True

})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0, collate_fn=docvqa.collate)
    dataLoaderIter = iter(dataLoader)
    if os.path.exists('./cache_huggingface/BART'):
        model_id = './cache_huggingface/BART'
    else:
        model_id = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_id)
        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    max_tok_len = 0
    try:
        question_types = defaultdict(int)
        while True:
            #print('?')
            tok_len=display(dataLoaderIter.next(),tokenizer)
            max_tok_len = max(tok_len,max_tok_len)
    except StopIteration:
        print('max tok len: {}'.format(max_tok_len))

    #print('width mean: {}'.format(np.mean(widths)))
    #print('width std: {}'.format(np.std(widths)))
    #print('width max: {}'.format(np.max(widths)))
