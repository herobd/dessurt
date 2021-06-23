import easyocr
import torch
import numpy as np
from utils import img_f

reader = easyocr.Reader(['en'],gpu=False)#,quantize=False)
#results=reader.readtext('../pairing/test3.png',text_threshold=0.01,min_size=5,low_text=0.01)
img = img_f.imread('../pairing/test3.png',False)
img = (255*(img/img.max())).astype(np.uint8)
#img = img/img.max()
results=reader.readtext(img)#,decoder='softmax')
for i,r in enumerate(results):
    tlx,tly = r[0][0]
    brx,bry = r[0][2]
    img_window = img[tly:bry+1,tlx:brx+1]
    img_window = np.repeat(img_window,3,axis=2)
    char_pred = r[1].argmax(dim=1)

    per_pix = img_window.shape[1]/char_pred.size(0)
    x=0
    for char in char_pred:
        c=round(x)
        c2=round(x+per_pix)
        if char!=0:
            img_window[:,c:c2,:,0]=0
            img_window[:,c:c2,:,1]=255
        x+=per_pix
    img_f.imshow(str(i),img_window)
    img_f.wait()
    #print('{}: {}'.format(i,r[1]))
#import pdb;pdb.set_trace()
#print(results)
