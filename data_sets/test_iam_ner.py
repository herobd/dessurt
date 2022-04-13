from data_sets import iam_ner
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
        
        if tokenizer is not None:
            toks = len(tokenizer.tokenize(a))+2
        if 'noise_token_mask' in data and data['noise_token_mask'] is not None:
            print(data['noise_token_mask'])

        #widths.append(img.size(1))
        
        #draw='WORK_OF_ART' in a
        draw=True
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
    return toks#question_types

if __name__ == "__main__":
    if len(sys.argv)>1:
        dirPath = sys.argv[1]
    else:
        dirPath = '../data/IAM'
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=iam_ner.IAMNER(dirPath=dirPath,split='test',config={
        'rescale_range': [0.75,1],
        '#rescale_range': [0.9,0.9],
        'rescale_to_crop_size_first': True,
        'crop_params': {
            "#crop_size":[960,1280],
            "crop_size":[768,768],
            "pad":0,
            "rot_degree_std_dev": 1,
            "#random": False
            },
        'questions':1,
        'full': True,
        'class_first': True,
        'cased': True,
        "task": 6,
        "eval_class_before": True,
        "use_noise": 0.05

})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=iam_ner.collate)
    print('dataset size: {}'.format(len(dataLoader)))
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
    max_tok_len=0
    try:
        question_types = defaultdict(int)
        while True:
            #print('?')
            tok_len=display(dataLoaderIter.next(),tokenizer)
            #for q in q_t:
            #    question_types[q]+=1
            #print('question_types:')
            #print(question_types)
            max_tok_len = max(tok_len,max_tok_len)
    except StopIteration:
        print('max token length {}'.format(max_tok_len))

    #print('width mean: {}'.format(np.mean(widths)))
    #print('width std: {}'.format(np.std(widths)))
    #print('width max: {}'.format(np.max(widths)).
