from data_sets import synth_qadoc_dataset
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
import utils.img_f as cv2

widths=[]

def display(data,write):
    batchSize = data['img'].size(0)
    #mask = makeMask(data['image'])
    for b in range(batchSize):
        #print (data['img'].size())
        img = (1-data['img'][b,0:1].permute(1,2,0))/2.0
        img = torch.cat((img,img,img),dim=2)
        show = data['img'][b,1]>0
        mask = data['img'][b,1]<0
        img[:,:,0] *= ~mask
        img[:,:,1] *= ~show
        img[:,:,2] *= 1-data['mask_label'][b,0]
        #label = data['label']
        #gt = data['gt'][b]
        #print(label[:data['label_lengths'][b],b])
        print(data['imgName'][b])
        print('{} - {}'.format(data['img'].min(),data['img'].max()))
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

        #widths.append(img.size(1))
        
        #draw='ar~' in q or 'ac~' in q or '%' in q or '&' in q or 'rh~' in q or 'ch~' in q
        #draw = 'l~' in q or 'v~' in q or 'd0~' in q or 'v0~' in q
        #draw = False
        #for x in ['al~']:#['g0','gs','gm','z0','zx','zm']:#['r@','c@','r&','c&','rh~','rh>','ch~','ch>']:#['#r~', '#c~','$r~','$c~',
        #    if x in q:
        #        draw = True
        #        break
        draw = True
        if write:
            cv2.imwrite('test_single_512.png',(img.numpy()*255)[:,:,0].astype(np.uint8))
        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            #cv2.imwrite('out/fg_mask{}.png'.format(b),fg_mask.numpy()*255)
            #cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)
            #cv2.imwrite('out/changed_img{}.png'.format(b),changed_img.numpy()*255)


            #plt.imshow(img.numpy()[:,:,0], cmap='gray')
            #plt.show()
            #cv2.imshow('fd',img.numpy()[:,:,0])
            cv2.imshow('x',(img*255).numpy().astype(np.uint8))
            cv2.show()

            #cv2.waitKey()

        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
    print('batch complete')


if __name__ == "__main__":
    if len(sys.argv)>1:
        dirPath = sys.argv[1]
    else:
        dirPath = '../data/english4line_fontslonger2'
    if len(sys.argv)>2:
        write = int(sys.argv[2])
    else:
        write=False
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=synth_qadoc_dataset.SynthQADocDataset(dirPath=dirPath,split='train',config={
   #     "data_set_name": "SynthQADocDataset",
   #     "pretrain": False,
   #     "fontdir": "../data/fonts",
   #     "textdir": "../data/",
   #     "header_dir": "../data/english4line_fonts",
   #     "include_ocr": False,
   #     "batch_size": 52,
   #     "num_workers": 4,
   #     "rescale_range": [1.0,1.0],
   #     "crop_params": {
   #             "crop_size":500,#[192,384],
   #             "pad":0,
   #             "rot_degree_std_dev": 1
   #         },
   # "questions": 1,
   # "max_qa_len": 26,
   #     "min_entries": None,
   #     "max_entries": 44,
   # "use_read": 1,
   # "multiline": 0.0,
   #     "tables": 0.0,
   #     "change_size": True,
   # "word_questions": "simple",
   # "do_masks": True,
   #     "text_height": 32,
   #     "image_size": 500,#[192,384],
   #     "max_chars": 10,
   #     "min_chars": 1,
   #     "use_before_refresh": 99999999999999999999,
   #     "set_size": 500000,
   #     "num_processes": -1,
   #     "gen_type": "veryclean",
   #     "char_file": "../data/english_char_set.json",
   #     "shuffle": True
	"data_set_name": "SynthQADocDataset",
        "fontdir": "../data/fonts/single_font",
        "textdir": "../data/a2z",
        "data_dir": "../data/english4line",
        "batch_size": 6,
        "#num_workers": 4,
        "num_workers": 0,
        "rescale_range": [1.0,1.0],
        "crop_params": None,
        "questions": 1,
        "max_qa_len": 100,
        "min_entries": None,
        "max_entries": 2,
        "use_read": 0.01,
        "multiline": 0.000001,
        "change_size": True,
        "word_questions": "simple",
        "do_masks": True,
        "min_text_height": 31,
        "max_text_height": 32,
        "image_size": [192,384],
        "max_chars": 8,
        "min_chars": 1,
        "use_before_refresh": 99999999999999999999,
        "set_size": 10000,
        "num_processes": -1,
        "gen_type": "veryclean",
        "char_file": "../data/english_char_set.json",
        "shuffle": True


})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=0, collate_fn=synth_qadoc_dataset.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    try:
        while True:
            #print('?')
            display(dataLoaderIter.next(),write)
    except StopIteration:
        print('done')

    #print('width mean: {}'.format(np.mean(widths)))
    #print('width std: {}'.format(np.std(widths)))
    #print('width max: {}'.format(np.max(widths)))
