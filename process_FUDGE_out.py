import json
import sys
import numpy as np
from collections import defaultdict

def sortByY(eis,entities):
    heights = []#[entities[ei]['corners'][3][1]-entities[ei]['corners'][0][1] for ei in eis]
    for ei in eis:
        for line in entities[ei]['lines']:
            heights.append(line['corners'][3][1]-line['corners'][0][1])
    avg_h = np.mean(heights)

    buckets=[]
    for ei in eis:
        added=False
        for other_eis in buckets:
            mean_y = np.mean([entities[oei]['lines'][0]['corners'][0][1] for oei in other_eis])
            if abs(entities[ei]['lines'][0]['corners'][0][1]-mean_y)<avg_h:
                other_eis.append(ei)
                added=True
                break

        if not added:
            buckets.append([ei])

    #sort the buckets
    bb=[(b,np.mean([entities[oei]['lines'][0]['corners'][0][1] for oei in b])) for b in buckets]
    bb.sort(key=lambda a:a[1])
    ret=[]
    for b in bb:
        ret+=b[0]
    return ret

def parse(this_i,entities,link_down_to,same_link,used,answers=False):
    used.add(this_i)
    if entities[this_i]['class'] == 'header':
        ele = {entities[this_i]['tesseract_text']:'header'}
        to_link = [ei for ei in link_down_to[this_i] + same_link[this_i] if ei not in used]
        if len(to_link)>0:
            to_link = sortByY(to_link,entities)
            children=[parse(ei,entities,link_down_to,same_link,used) for ei in to_link]
            ele['content']=children
    elif entities[this_i]['class'] == 'question':
        ele = {entities[this_i]['tesseract_text']:'question'}
        to_link = [ei for ei in link_down_to[this_i] + same_link[this_i] if ei not in used]
        if len(to_link)>0:
            to_link = sortByY(to_link,entities)
            children=[parse(ei,entities,link_down_to,same_link,used,True) for ei in to_link]
            ele['answers']=children
    elif answers:
        ele = entities[this_i]['tesseract_text']
    else:
        ele = {entities[this_i]['tesseract_text']:entities[this_i]['class']}

    return ele




inpath = sys.argv[1]
outpath = sys.argv[2]


with open(inpath) as f:
    preds = json.load(f)


processed={}

for img_name,data in preds.items():
    claimed_by = defaultdict(list)
    link_down_to = defaultdict(list)
    same_link = defaultdict(list)

    entities = data['entities']

    for entity in entities:
        #texts = [(line['tesseract_text'],line['corners'][0][1]) for line in entity['lines']]
        #texts.sort(key = lambda a:a[1])
        #entity['tesseract_text'] = '\\'.join([t[0] for t in texts])
        entity['lines'].sort(key = lambda a:a['corners'][0][1])
        entity['tesseract_text'] = '\\'.join([line['tesseract_text'] for line in entity['lines']])

    
    for a,b in data['links']:
        a_cls = entities[a]['class']
        b_cls = entities[b]['class']
        if a_cls == b_cls:
            same_link[a].append(b)
            same_link[b].append(a)
        elif (a_cls=='header') or (a_cls=='question' and b_cls!='header'):
            link_down_to[a].append(b)
            claimed_by[b].append(a)
        elif (b_cls=='header') or (b_cls=='question' and a_cls!='header'):
            link_down_to[b].append(a)
            claimed_by[a].append(b)
        else:
            print('Unknown class comb, {} {}'.format(a_cls,b_cls))
            assert False


    #top_headers = []
    top_everything = []
    for ei,entity in enumerate(entities):
        #if entity['class']=='header' and len(claimed_by[ei])==0:
        #    top_headers.append(ei)
        if len(claimed_by[ei])==0:
            top_everything.append(ei)

    used=set()
    structured=[]
    #for hi in sortByY(top_headers,entities):
    #    structured.append(parse(hi,entities,link_down_to,same_link,used))
    for ei in sortByY(top_everything,entities):
        structured.append(parse(ei,entities,link_down_to,same_link,used))

    processed[img_name]=structured

        
with open(outpath,'w') as f:
    json.dump(processed,f,indent=3)
print('wrote '+outpath)
