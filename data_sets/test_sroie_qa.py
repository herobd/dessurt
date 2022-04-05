from data_sets import sroie
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
import utils.img_f as cv2
import json
from transformers import BartTokenizer
import os

widths=[]

def display(data,tokenizer):
    batchSize = data['img'].size(0)
    #mask = makeMask(data['image'])
    for b in range(batchSize):
        #print (data['img'].size())
        img = (1-data['img'][b,0:1].permute(1,2,0))/2.0
        img = torch.cat((img,img,img),dim=2)
        #show = data['img'][b,1]>0
        #mask = data['img'][b,1]<0
        #img[:,:,0] *= ~mask
        #img[:,:,1] *= ~show
        #if data['mask_label'] is not None:
        #    img[:,:,2] *= 1-data['mask_label'][b,0]
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
            if q=='json>':
                
                a = json.loads(a[:-1])
                a = json.dumps(a,indent=3)
            print(q+' : '+a)

        tok_len=tokenizer(a,return_tensors="pt")['input_ids'].shape[1]

        #widths.append(img.size(1)       
        draw = False

        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            #cv2.imwrite('out/fg_mask{}.png'.format(b),fg_mask.numpy()*255)
            #cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)
            #cv2.imwrite('out/changed_img{}.png'.format(b),changed_img.numpy()*255)
            plt.imshow((img*255).numpy().astype(np.uint8))
            plt.show()

            #cv2.waitKey()

            #cv2.imwrite('testsinglesize_1024.png',img.numpy()[:,:,0])

        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
    return tok_len


if __name__ == "__main__":
    if len(sys.argv)>1:
        dirPath = sys.argv[1]
    else:
        dirPath = '../data/SROIE'
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=sroie.SROIE(dirPath=dirPath,split='test',config={
        '#rescale_range':[0.8,1.2],
        'rescale_range':[1,1],
        'rescale_to_crop_size_first': True,
        'crop_params': {
            "crop_size":[1152,768],
            "pad":0,
            "rot_degree_std_dev": 1
            },
        'questions':1,
        'max_qa_len': 200000,
        "cased": True,

})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=sroie.collate)
    dataLoaderIter = iter(dataLoader)

    if os.path.exists('./cache_huggingface/BART'):
        model_id = './cache_huggingface/BART'
    else:
        model_id = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_id)
    #add = ['"answer"',"question","other","header","},{",'"answers":','"content":']
    #tokenizer.add_tokens(add, special_tokens=True)
    #max_tok_len=0S
    tok_lens=[]
        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print('test {}'.format(i))
        dataLoaderIter.next()
        #display(data[i])
    try:
        while True:
            #print('?')
            tok_lens.append(display(dataLoaderIter.next(),tokenizer))
    except StopIteration:
        print(tok_lens)
        print('max tok len : {}'.format(np.max(tok_lens)))

    #print('width mean: {}'.format(np.mean(widths)))
    #print('width std: {}'.format(np.std(widths)))
    #print('width max: {}'.format(np.max(widths)))
