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

def display(data):
    batchSize = data['img'].size(0)
    #mask = makeMask(data['image'])
    for b in range(batchSize):
        #print (data['img'].size())
        img = (data['img'][b].permute(1,2,0)+1)/2.0
        #label = data['label']
        #gt = data['gt'][b]
        #print(label[:data['label_lengths'][b],b])
        print(data['imgName'][b])
        #if data['spaced_label'] is not None:
        #    print('spaced label:')
        #    print(data['spaced_label'][:,b])
        print('questions: {}'.format(data['questions'][b]))
        print('answers: {}'.format(data['answers'][b]))

        #widths.append(img.size(1))
        
        draw=True
        if draw :
            #cv2.imshow('line',img.numpy())
            #cv2.imshow('mask',maskb.numpy())
            #cv2.imwrite('out/mask{}.png'.format(b),maskb.numpy()*255)
            #cv2.imwrite('out/fg_mask{}.png'.format(b),fg_mask.numpy()*255)
            #cv2.imwrite('out/img{}.png'.format(b),img.numpy()*255)
            #cv2.imwrite('out/changed_img{}.png'.format(b),changed_img.numpy()*255)
            plt.imshow(img.numpy()[:,:,0], cmap='gray')
            plt.show()

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
    dirPath = sys.argv[1]
    if len(sys.argv)>2:
        logged = int(sys.argv[2])
    else:
        logged=False
    data=synth_qadoc_dataset.SynthQADocDataset(dirPath=dirPath,split='train',config={
        "create": True,
	"fontdir": "../data/fonts/single_font",
        "textdir": "../data/randomletters",
        "num_workers": 0,
        "rescale_range": [1.0,1.0],
        "crop_params": None,
        "batch_size": 4,
        "questions": 4,
        "min_entries": 4,
        "max_entries": 4,
        "text_height": 32,
        "image_size": 512,
        "max_chars": 10,
        "min_chars": 1,
        "use_before_refresh": 99999999999999999999,
        "set_size": 100000,
        "num_processes": 9,
        "gen_type": "veryclean",
        "char_file": "../data/english_char_set.json"


})
    data.refresh_data(logged)

    dataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=0, collate_fn=synth_qadoc_dataset.collate)
    dataLoaderIter = iter(dataLoader)

    try:
        while True:
            #print('?')
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')

    #print('width mean: {}'.format(np.mean(widths)))
    #print('width std: {}'.format(np.std(widths)))
    #print('width max: {}'.format(np.max(widths)))
