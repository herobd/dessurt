from utils import img_f
import numpy as np
import random

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def lineIntersection(lineA, lineB, threshA_low=10, threshA_high=10, threshB_low=10, threshB_high=10, both=False):
    a1=lineA[0]
    a2=lineA[1]
    b1=lineB[0]
    b2=lineB[1]
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    point = (num / denom.astype(float))*db + b1
    #check if it is on atleast one line segment
    vecA = da/np.linalg.norm(da)
    p_A = np.dot(point,vecA)
    a1_A = np.dot(a1,vecA)
    a2_A = np.dot(a2,vecA)

    vecB = db/np.linalg.norm(db)
    p_B = np.dot(point,vecB)
    b1_B = np.dot(b1,vecB)
    b2_B = np.dot(b2,vecB)
    
    ###rint('A:{},  B:{}, int p:{}'.format(lineA,lineB,point))
    ###rint('{:.0f}>{:.0f} and {:.0f}<{:.0f}  and/or  {:.0f}>{:.0f} and {:.0f}<{:.0f} = {} {} {}'.format((p_A+threshA_low),(min(a1_A,a2_A)),(p_A-threshA_high),(max(a1_A,a2_A)),(p_B+threshB_low),(min(b1_B,b2_B)),(p_B-threshB_high),(max(b1_B,b2_B)),(p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)),'and' if both else 'or',(p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B))))
    if both:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) and
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    else:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) or
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    return None

a= np.zeros((500,500),dtype=np.uint8)
lines=[]
for i in range(1000):
    x1=random.random()*500
    x2=random.random()*500
    y1=random.random()*500
    y2=random.random()*500
    new_line = np.array([[x1,y1],[x2,y2]])
    
    hit= False
    for line in lines:
        if lineIntersection(line,new_line) is not None:
            hit = True
            break
    if not hit:
        lines.append(new_line)
        img_f.line(a,[x1,y1],[x2,y2],255)

img_f.imshow('x',a)
img_f.show()

#from data_sets.gen_daemon import GenDaemon
#import numpy as np
#
#gen_daemon = GenDaemon('../data/fonts')
#
##texts=['\u265e [ ] ( ) { } ? \u2713 \u2714 \u2611 \u274c \u274e \u2716 \u2717 \u2718 \u2612']
#texts=['This (has) [brackets]']
#
#for i in range(120):
#    ws,font = gen_daemon.generate(texts[0],ret_font=True)
#    #print(font)
#    #assert len(ws)==1
#    length=sum(w[1].shape[1]+10 for w in ws)
#    height = max(w[1].shape[0] for w in ws)
#    img = np.zeros((height,length),dtype=np.uint8)
#    x=0
#    nt=''
#    for t,wimg in ws:
#        img[:wimg.shape[0],x:x+wimg.shape[1]]=wimg
#        x+=10+wimg.shape[1]
#        nt+=t+' '
#    #text,img = ws[0]
#    print(nt)
#    img_f.imshow('',img)
#    img_f.show()
#    for text in texts[1:]:
#        ws = gen_daemon.generate(text,font=font)
#        assert len(ws)==1
#        text,img = ws[0]
#        print(text)
#        img_f.imshow('',img)
#        img_f.show()


#import re
#from funsd_eval_json import derepeat

#a ='bkd dfkljrgt fkldsf something sdjff something dkf something sdkf something defsdfdsfds something sdfd something s something {bad: class}, {bad: class}, {bad: class}, {bad: class}, {bad: class}, {bad: class}, {bad: class}, {bad: class}, {bad: cl something dfl;jsdklgdfjklg {bad: class} something'

#print(derepeat(a))






#from transformers import BartTokenizer
#tokenizer = BartTokenizer.from_pretrained('./cache_huggingface/BART')
#
#print(tokenizer('This is a sentence[NE:O]'))
#
#tokens = ["[NE:{}]".format(cls) for cls in ['N', 'C', 'L', 'T', 'O', 'P', 'G','N  ORP', 'LAW', 'PER', 'QUANTITY', 'MONEY', 'CARDINAL', 'LOCATION', 'LANGUAGE', 'ORG', 'DATE',   'FAC', 'ORDINAL', 'TIME', 'WORK_OF_ART', 'PERCENT', 'GPE', 'EVENT', 'PRODUCT']]
#tokenizer.add_tokens(tokens, special_tokens=True)
#
#print(tokens)
#
#print(tokenizer('This is a sentence[NE:O]'))



#from utils import img_f
#import numpy as np
#import torch.nn as nn
#import torch
#import torch.nn.functional as F
#
#from data_sets.gen_daemon import GenDaemon
#a=GenDaemon('../data/fonts')
#a.generateLabelValuePairs()
