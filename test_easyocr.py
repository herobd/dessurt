import easyocr
import torch
import numpy as np
from utils import img_f

reader = easyocr.Reader(['en'],gpu=True)#,quantize=False)
characters = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#results=reader.readtext('../pairing/test3.png',text_threshold=0.01,min_size=5,low_text=0.01)
img = img_f.imread('../test-0.png',False)
img = (255*(img/img.max())).astype(np.uint8)
#img = img/img.max()
results=reader.readtext(img,decoder='greedy+softmax')
for i,r in enumerate(results):
    #print(r[0])
    #tlx,tly = r[0][0]
    #brx,bry = r[0][2]
    #img_window = img[round(tly):round(bry)+1,round(tlx):round(brx)+1]
    #img_window = np.repeat(img_window[:,:,None],3,axis=2)
    str_pred,char_prob = r[1]
    print(str_pred)
    #char_pred = char_prob.argmax(dim=1)
    #char_loc = char_pred!=0
    #new_char_prob = char_prob[char_loc]
    #new_char_prob2=new_char_prob.cpu()
    #char_prob2=char_prob.cpu()


    #per_pix = img_window.shape[1]/char_pred.size(0)
    #x=0
    #first=''
    #second=''
    #for i,char in enumerate(char_pred):
    #    c=round(x)
    #    c2=round(x+per_pix)
    #    if char!=0:
    #        img_window[:,c:c2,0]=0
    #        img_window[:,c:c2,1]=255
    #        
    #        prob = char_prob2[i]
    #        prob[char]=0
    #        new_char = prob.argmax().item()
    #        second += characters[new_char-1]
    #        first += characters[char-1]
    #    x+=per_pix
    #print(str_pred)
    #print(len(str_pred))
    #print(first)
    #print(second)
    #print('===-')
    #first=''
    #second=''
    #for i in range(new_char_prob.size(0)):
    #    char = new_char_prob[i].argmax().item()
    #    new_char_prob2[i][char]=0
    #    nchar = new_char_prob2[i].argmax().item()
    #    second += characters[nchar-1]
    #    first += characters[char-1]
    #print(first)
    #print(second)
    #print('===-======')
    #img_f.imshow(str(i),img_window)
    #img_f.show()
