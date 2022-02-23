from zss import simple_distance, Node

A = (
    Node("f")
        .addkid(Node("abaa")
            .addkid(Node("h"))
            .addkid(Node("c")
                .addkid(Node("l"))))
        .addkid(Node("f"))
        .addkid(Node("g"))
        .addkid(Node("1")
            .addkid(Node("xx"))
            .addkid(Node("aaa"))
            )
        .addkid(Node("e"))
        .addkid(Node("2")
            .addkid(Node("3"))
            .addkid(Node("444"))
            )
    )
B = (
    Node("f")
        .addkid(Node("abar")
            .addkid(Node("d"))
            .addkid(Node("c")
                .addkid(Node("b"))))
        .addkid(Node("e"))
        .addkid(Node("2")
            .addkid(Node("3"))
            .addkid(Node("444"))
            )
        .addkid(Node("1")
            .addkid(Node("xx"))
            .addkid(Node("aaa"))
            )
    )
print(simple_distance(A, B))


#from utils import img_f
#import numpy as np
#import random
#
#from data_sets.naf_qa import putInReadOrder
#
#size=100
#offset=20
#ax1=[10,10,10,10,10]
#ax2=[10,10,40,30,30]
#ax3=[20,20,40,30,30]
#ax4=[20,20,10,10,10]
#ay1=[40,40,10,10,10]
#ay2=[10,10,10,10,14]
#ay3=[10,10,30,30,34]
#ay4=[40,40,30,30,30]
#xoff = [0,40,0,0,0]
#yoff = [-30,0,20,20,20]
#for i in range(3):
#    a= np.zeros((size,size,3),dtype=np.uint8)
#    x1=ax1[i]
#    x2=ax2[i]
#    x3=ax3[i]
#    x4=ax4[i]
#    y1=ay1[i]
#    y2=ay2[i]
#    y3=ay3[i]
#    y4=ay4[i]
#    #x1=random.random()*size
#    #x2=random.random()*size
#    #x3=random.random()*size
#    #x4=random.random()*size
#    #y1=random.random()*size
#    #y2=random.random()*size
#    #y3=random.random()*size
#    #y4=random.random()*size
#    img_f.line(a,(x1,y1),(x4,y4),[255,0,0])
#    img_f.line(a,(x2,y2),(x3,y3),[255,0,0])
#    poly1 = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
#    p1_1 = (poly1[0]+poly1[3])/2
#    p1_2 = (poly1[1]+poly1[2])/2
#    h1_1 = (poly1[3]+poly1[2])/2
#    h1_2 = (poly1[0]+poly1[1])/2
#    offset_x = xoff[i]#random.random()*offset - offset//2
#    offset_y = yoff[i]#random.random()*offset - offset//2
#    #offset_x = random.random()*offset - offset//2
#    #offset_y = random.random()*offset - offset//2
#    x1+=offset_x
#    x2+=offset_x
#    x3+=offset_x
#    x4+=offset_x
#    y1+=offset_y
#    y2+=offset_y
#    y3+=offset_y
#    y4+=offset_y
#    img_f.line(a,(x1,y1),(x4,y4),[0,255,0])
#    img_f.line(a,(x2,y2),(x3,y3),[0,255,0])
#    poly2 = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
#    p2_1 = (poly2[0]+poly2[3])/2
#    p2_2 = (poly2[1]+poly2[2])/2
#    h2_1 = (poly2[3]+poly2[2])/2
#    h2_2 = (poly2[0]+poly2[1])/2
#
#    order = putInReadOrder(1,poly1,2,poly2)
#    
#    img_f.line(a,p1_1,p1_2,[255,0,0])
#    a[round(p1_1[1]),round(p1_1[0])]=[255,255,0]
#    img_f.line(a,p2_1,p2_2,[0,255,0])
#    a[round(p2_1[1]),round(p2_1[0])]=[0,255,255]
#
#    img_f.line(a,h1_1,h1_2,[255,0,0])
#    a[round(h1_1[1]),round(h1_1[0])]=[255,0,255]
#    img_f.line(a,h2_1,h2_2,[0,255,0])
#    a[round(h2_1[1]),round(h2_1[0])]=[255,255,255]
#
#    if order[0]==1:
#        print('red on top')
#    else:
#        print('green on top')
#
#    img_f.imshow('x',a)
#    img_f.show()




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
