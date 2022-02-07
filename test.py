from data_sets.gen_daemon import GenDaemon
from utils import img_f
import numpy as np

gen_daemon = GenDaemon('../data/fonts')

#texts=['\u265e [ ] ( ) { } ? \u2713 \u2714 \u2611 \u274c \u274e \u2716 \u2717 \u2718 \u2612']
texts=['This (has) [brackets]']

for i in range(120):
    ws,font = gen_daemon.generate(texts[0],ret_font=True)
    #print(font)
    #assert len(ws)==1
    length=sum(w[1].shape[1]+10 for w in ws)
    height = max(w[1].shape[0] for w in ws)
    img = np.zeros((height,length),dtype=np.uint8)
    x=0
    nt=''
    for t,wimg in ws:
        img[:wimg.shape[0],x:x+wimg.shape[1]]=wimg
        x+=10+wimg.shape[1]
        nt+=t+' '
    #text,img = ws[0]
    print(nt)
    img_f.imshow('',img)
    img_f.show()
    for text in texts[1:]:
        ws = gen_daemon.generate(text,font=font)
        assert len(ws)==1
        text,img = ws[0]
        print(text)
        img_f.imshow('',img)
        img_f.show()


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
