
import os
import json
import logging
import argparse
from data_loader import getDataLoader
import math
from collections import defaultdict
import random
import editdistance
import re
from utils.GAnTED import  GAnTED,nTED,shuffleTree
from utils.GAnTED import TableNode
from utils.GAnTED import FormNode as Node
import numpy as np


#This script calculates nTED and GAnTED on the output of Dessurt on FUNSD and NAF
#For NAF, this can take a few hours


#Take a JSON object and parse it into a node for the tree (recursive, so all belore nodes are obtained as well)
def parseDict(obj):
    if isinstance(obj, str):
        return [Node(obj)],[]
    elif isinstance(obj,int) or isinstance(obj,float):
        return [Node(str(obj))],[]
    elif isinstance(obj, list):
        to_ret=[]
        all_tables=[]
        for thing in obj:
            ret,tab = parseDict(thing)
            to_ret+=ret
            all_tables+=tab
        return to_ret,all_tables

    my_children=[]
    is_table=False
    row_headers = None
    col_headers = None
    cells = None
    my_text = None
    to_ret=[]
    all_tables=[]
    for text,value in obj.items():
        if text=='content':
            if isinstance(value,list):
                for thing in value:
                    children,tables = parseDict(thing)
                    my_children+=children
                    all_tables+=tables
            else:
                assert isinstance(value,dict)
                children,tables = parseDict(value)
                my_children+=children
                all_tables+=tables
        elif text=='answers':
            if not isinstance(value,list):
                value=[value]
            for a in value:
                if isinstance(a,str):
                    my_children.append(Node(a))
                else:
                    assert isinstance(a,dict)
                    children,tables = parseDict(a)
                    my_children+=children
                    all_tables+=tables
        elif text=='row headers':
            assert isinstance(value,list)
            row_headers = value
            is_table = True
        elif text=='column headers':
            assert isinstance(value,list)
            col_headers = value
            is_table = True
        else:
            if isinstance(value,str):
                if my_text is not None:
                    #merged entity?
                    node = Node(my_text)
                    for child in my_children:
                        node.addkid(child)
                    to_ret.append(node)
                    my_children = []
                my_text = text
                my_class = value
            elif isinstance(value,list) and text=='cells':
                is_table=True
                cells = value
            elif isinstance(value,list) and my_text is None:
                #potentially bad qa?
                my_text = text
                my_class = 'question'
                node = Node(my_text)
                for child in my_children:
                    node.addkid(child)
                for a in value:
                    if isinstance(a,str):
                        my_children.append(Node(a))
                    else:
                        children,tables = parseDict(a)
                        my_children+=children
                        all_tables+=tables


    if is_table:
        headers=[]
        if row_headers is not None:

            for rh in reversed(row_headers):
                if rh is not None:
                    if '<<' == rh[:2] and '>>' in rh:
                        #subent_dict
                        super_end = rh.find('>>')
                        super_h =  rh[2:super_end]
                        rh=rh[super_end+2:]
                        if len(headers)>0 and isinstance(headers[-1],tuple) and headers[-1][0]==super_h:
                            headers[-1][1].append(rh)
                        else:
                            headers.append((super_h,[rh]))
                    else:
                        headers.append(rh)
                else:
                    headers.append(rh)
        new_row_headers=headers
    
        headers=[]
        if col_headers is not None:
            subheaders=defaultdict(list)
            #col_ids = list(range(len(entities),len(entities)+len(col_headers)))
            col_ids = []
            for ch in reversed(col_headers):
                if ch is not None:
                    if '<<' == ch[:2] and '>>' in ch:
                        #subent_dict
                        #subent_dict
                        super_end = ch.find('>>')
                        super_h =  ch[2:super_end]
                        ch=ch[super_end+2:]
                        if len(headers)>0 and isinstance(headers[-1],tuple) and headers[-1][0]==super_h:
                            headers[-1][1].append(ch)
                        else:
                            headers.append((super_h,[ch]))
                    else:
                        headers.append(ch)
                else:
                    headers.append(ch)
        new_col_headers=headers
    
        table =  TableNode(new_row_headers,new_col_headers,cells)
        to_ret.append(table)
        all_tables.append(table)
        
    else:
        node = Node(my_text)
        for child in my_children:
            node.addkid(child)
        to_ret.append(node)

    

    return to_ret,all_tables

#Try different permutations of row/col major ordering at this level (and all lower levels recursively)
def getScore(scorer,pred,gt,tables):
    if len(tables) == 0:
        return [scorer(pred,gt)]
    ret = []
    tables[0].set_row_major(True)
    ret += getScore(scorer,pred,gt,tables[1:])
    tables[0].set_row_major(False)
    ret += getScore(scorer,pred,gt,tables[1:])
    return ret


def main(predictions,data_set_name,test=False,match_thresh=1,twice=False,shuffle=False,parallel=1):

    if data_set_name=='FUNSD':
        data_config={
                "data_loader": {
                    "data_set_name": "FUNSDQA",
                    "data_dir": "../data/FUNSD",
                    "use_json": "only",
                    "max_a_tokens": 2000000000000,
                    "cased": True,
                    "words": True,
                    "batch_size": 1,
                    "num_workers": 2,
                    "questions": 1,
                    "rescale_range": [1.0,1.0],
                    "shuffle": False,
                        },
                "validation":{}
                }
    elif data_set_name=='NAF':
        data_config={
                "data_loader": {
                    "data_set_name": "NAFQA",
                    "data_dir": "../data/forms",
                    "use_json": "only",
                    "max_a_tokens": 2000000000000,
                    "cased": True,
                    "batch_size": 1,
                    "num_workers": 2,
                    "questions": 1,
                    "rescale_range": [1.0,1.0],
                    "shuffle": False,
                        },
                "validation":{}
                }
    else:
        print('Unknown dataset: '+data_set_name)
        exit()

    name = predictions
    if '/' in name: 
        name = name[name.rfind('/')+1:]
    progress_file = 'progress_'+name

    with open(predictions) as f:
        predictions = json.load(f)
    

    #Becuase this takes so long, keep a log file so it can be resumed
    try:
        with open(progress_file) as f:
            already_done = json.load(f)
    except:
        already_done = {}
    
    data_loader, valid_data_loader = getDataLoader(data_config,'train' if not test else 'test')

    if test:
        valid_data_loader = data_loader

    scores=[]
    vanilla_scores=[]
    second_scores=[]
    for instance in valid_data_loader:
        if instance['imgName'][0] not in already_done:

            ans = instance['answers'][0][0]
            assert ans[-1]=='‡'
            gt = json.loads(ans[:-1])
            pred = predictions[instance['imgName'][0]]
            
            #Build GT tree
            tree_gt = Node('')
            gt_tables = []
            for ele in gt:
                nodes,tables = parseDict(ele)
                for node in nodes:
                    tree_gt.addkid(node)
                gt_tables+=tables

            #Build predicted tree
            tree_pred = Node('')
            pred_tables = []
            for ele in pred:
                nodes,tables = parseDict(ele)
                for node in nodes:
                    tree_pred.addkid(node)
                pred_tables+=tables

            all_tables = pred_tables+gt_tables


            if shuffle:
                shuffleTree(tree_pred)

            if (len(gt_tables)==0 and len(pred_tables)==0) or data_set_name=='NAF':
                #NAF has no cells 
                vanilla_score = nTED(tree_pred,tree_gt)
                score = GAnTED(tree_pred,tree_gt,match_thresh=match_thresh,num_processes=parallel)
                if twice:
                    second_score = GAnTED(tree_pred,tree_gt,num_processes=parallel)
            else:
                assert len(all_tables)<10 #need new method if too many tables
                tab_scores=getScore(nTED,tree_pred,tree_gt,all_tables)
                vanilla_score=min(tab_scores)

                #Try every combination of row/col major on tables
                tab_scores=getScore(lambda a,b:GAnTED(a,b,match_thresh,num_processes=parallel),tree_pred,tree_gt,all_tables)
                score=min(tab_scores)
                if twice:
                    tab_scores=getScore(lambda a,b:GAnTED(a,b,match_thresh,num_processes=parallel),tree_pred,tree_gt,all_tables)
                    second_score=min(tab_scores)

            #Update log
            if twice:
                already_done[instance['imgName'][0]] =(score,vanilla_score,second_score)
            else:
                already_done[instance['imgName'][0]] =(score,vanilla_score,None)

            with open(progress_file, 'w') as f:
                json.dump(already_done,f)
        else:
            score,vanilla_score,second_score = already_done[instance['imgName'][0]]
        if twice:
            print('{}: {}  v:{}, 2nd:{}'.format(instance['imgName'],score,vanilla_score,second_score))
        else:
            print('{}: {}  v:{}'.format(instance['imgName'],score,vanilla_score))
        scores.append(score)
        vanilla_scores.append(vanilla_score)
        if twice:
            second_scores.append(second_score)

    final_score = np.mean(scores)
    print('Overall   nTED: {}'.format(np.mean(vanilla_scores)))
    print('Overall GAnTED: {}'.format(final_score))
    if twice:
        print('Overall second: {}'.format(np.mean(second_scores)))






if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Evaluate the output of Dessurt on FUNSD and NAF with nTED related metrics')
    parser.add_argument('-p', '--predictions', default=None, type=str,
                        help='path to json output from funsd/naf_eval.py (using "-w" option)')
    parser.add_argument('-d', '--data_set_name', default=None, type=str,
                        help='name of dataset to eval')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set')
    parser.add_argument('-t', '--match_thresh', default=1, type=float,
                        help='nED to threshold for a matching string')
    parser.add_argument('-2', '--twice', default=False, action='store_const', const=True,
                        help='Run alignment twice')
    parser.add_argument('-s', '--shuffle', default=False, action='store_const', const=True,
                        help='Shuffle the order of the kids for each node')
    parser.add_argument('-P', '--parallel', default=1, type=int,
                        help='number of processes')

    args = parser.parse_args()

    assert args.predictions is not None and args.data_set_name is not None

    main(args.predictions,args.data_set_name,args.test,args.match_thresh,args.twice,args.shuffle,args.parallel)
