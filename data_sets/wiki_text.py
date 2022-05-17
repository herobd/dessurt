try:
    from datasets import load_dataset, load_from_disk
except:
    print('could not import datasets')
import random
import os
from utils.util import ensure_dir
import re
#import unicodedata


#if 'myVar' in locals():
#_text_data = []#load_dataset('wikipedia', '20200501.en', cache_dir=cache_path)['train']

_prune_headers = ["See also", "Gallery", "External media", "History", "Notes"]
_wiki_end_keywords = ['References','Sources','External links']
_wiki_end_keywords = ['\n'+k+'\n' for k in _wiki_end_keywords] + ['\n'+k+' \n' for k in _wiki_end_keywords] + ['\nCategory:']

def getWikiDataset():
    global _text_data
    #Returns a list of text paragraphs from a randome wikipedia article
    if '_text_data' not in globals():
        if os.path.exists('DIR'):
            with open('DIR') as f:
                cache_path = f.readline().strip()
        else:
            cache_path = '../data/wiki_cache' 
            ensure_dir(cache_path)

        if not os.path.exists(os.path.join(cache_path,'dataset_info.json')):
            _text_data = load_dataset('wikipedia', '20200501.en', cache_dir=cache_path)['train']
            _text_data.save_to_disk(cache_path)
        else:
            _text_data = load_from_disk(cache_path)
    return _text_data

def getWikiArticle(all_newline=False,dataset=None):
    global _text_data
    #Returns a list of text paragraphs from a randome wikipedia article

    if dataset is None:
        if '_text_data' not in globals():
            _text_data = getWikiDataset()
    else:
        _text_data = dataset

    instance_i = random.randrange(_text_data.num_rows)
    text = _text_data[instance_i]['text']
    #text = unicodedata.normalize(text,'NFC')#.decode('utf')


    #We first want to cut off the end of the wikipedia article, which has the references and stuff 
    for keyword in _wiki_end_keywords:
        cut_i = text.find(keyword)
        if cut_i>-1:
            break
    if cut_i>-1:
        text = text[:cut_i]

    #break by paragraph (double newline)
    text=re.sub(r' +',r' ',text)
    if all_newline:
        text=re.sub(r'\n+',r'\n',text)
        paras = text.split('\n')
    else:
        paras = text.split('\n\n')

    paras = [para for para in paras if para.strip() not in _prune_headers]
    
    if len(paras)>0:
        return paras
    else:
        print('blank article:')
        print(text)
        print('------------')
        print(_text_data[instance_i]['text'])
        return getWikiArticle(all_newline,dataset)
