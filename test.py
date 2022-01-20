from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('./cache_huggingface/BART')

print(tokenizer('This is a sentence[NE:O]'))

tokens = ["[NE:{}]".format(cls) for cls in ['N', 'C', 'L', 'T', 'O', 'P', 'G','N  ORP', 'LAW', 'PER', 'QUANTITY', 'MONEY', 'CARDINAL', 'LOCATION', 'LANGUAGE', 'ORG', 'DATE',   'FAC', 'ORDINAL', 'TIME', 'WORK_OF_ART', 'PERCENT', 'GPE', 'EVENT', 'PRODUCT']]
tokenizer.add_tokens(tokens, special_tokens=True)

print(tokens)

print(tokenizer('This is a sentence[NE:O]'))



#from utils import img_f
#import numpy as np
#import torch.nn as nn
#import torch
#import torch.nn.functional as F
#
#from data_sets.gen_daemon import GenDaemon
#a=GenDaemon('../data/fonts')
#a.generateLabelValuePairs()
