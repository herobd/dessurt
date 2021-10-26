import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate
import threading
import urllib
import csv

import utils.img_f as img_f


class CDIPQA(ParaQADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(CDIPQA, self).__init__(dirPath,split,config,images)

        self.cache_resized = False
        #NEW the document must have a block_score above thresh for anything useing blocks (this is newline following too)
        self.block_score_thresh = 0.73 #eye-balled this one

        self._lock = threading.Lock()
        self.loader_lock = threading.Lock()
        self.reuse_factor = config['reuse_factor'] if 'reuse_factor' in config else 1.0
        self.calls = 0

        assert images is None

        csv_path = os.path.join(dirPath,'download_urls.csv')
        with open(csv_path) as f:
            download_urls = csv.reader(f)
            self.download_urls = {name:url for name,url in download_urls}

        #check if any are downloaded:
        status = self.getStatus()
        min_calls = 999999999999999
        list_path = None
        self.using = None
        for i,s in enumerate(status):
            if s['downloaded'] and s['untared'] and s['calls']<min_calls:
                self.using = i
                min_calls = s['calls']
                list_path = s['list_path']
                self.calls = s['calls']

        if list_path is None:
            if status[0]['downloaded']:
                untar(self,status[0]['tar_name'])
            elif downloaded1:
                untar(self,tar_name1)
            else:
                download(self,tar_name0)
            #how to stall?
        else:
            self.updateImages(list_path)

        self.startWorker()





    def parseAnn(self,ocr,s):
        
        self.calls += 1
        if self.calls > len(self.images)*self.reuse_factor:
            self.switch()
        elif self.calls%100 == 0:
            self.updateStatus(self.using,calls=self.calls)
            

        image_h=ocr['height']
        image_w=ocr['width']
        ocr=ocr['blocks']

        block_score_sum=0
        line_count=0
        for block in ocr:
            t,l,b,r = block['box']
            h=b-t
            w=r-l
            if w==0 or h==0:
                continue
            squareness = min(0.4,h/w)
            area_whole = h*w
            area_covered = 0 #we'll assume lines don't overlap
            num_lines=0
            for para in block['paragraphs']:
                for line in para['lines']:
                    num_lines+=1
                    for word in line['words']:
                        top,left,bottom,right = word['box']
                        height = bottom-top
                        width = right-left
                        area_covered+=height*width
            if num_lines>1:
                area_score = area_covered/area_whole
            else:
                area_score = 0
            total_score = area_score+squareness
            block_score_sum += total_score*num_lines
            line_count += num_lines
        block_score = block_score_sum/line_count if line_count>0 else 0
        use_blocks = block_score>self.block_score_thresh
        #print('block_score: {} {}'.format(block_score,'good!' if use_blocks else 'bad'))
        qa, qa_bbs = self.makeQuestions(ocr,image_h,image_w,s,use_blocks)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, {}, {}, qa

    def switch(self):
        use_next = None

        status = self.getStatus()
        status_tars=set()
        for i,s in status:
            status_tars.add(s['name'])
            if i!=self.using and s['downloaded'] and s['untared']:
                    use_next=i

        if use_next is not None:

            self.updateImages(status[next]['list_path'])
            
            #pick next tar
            all_tars = set(self.download_urls.keys())
            new_tars = all_tars-status_tars
            todo_tar = random.choice(list(new_tars))
            #update status
            self.updateStatus(self.using,todo_tar,False,False,'',0)

            self.using = use_next
            self.calls=status[use_next]['calls']
            
            assert len(status)==2
            self.startWorker()



    def getStatus(self):
        ret = []
        with self._lock:
            if os.path.exists(self.status_path):
                with open(self.status_path) as f:
                    status = f.readlines()
                for l in status:
                    tar_name0,downloaded0,untared0,list_path0,calls0 = l.strip().split(',')

                    tar_info = {'name': tar_name0,
                                 'downloaded': downloaded0=='True',
                                 'untared': untared0=='True',
                                 'list_path': list_path0,
                                 'calls': int(calls0)
                                 }
                    ret.append(tar_info)
        return ret
    def updateStatus(self,status_i,tar_name=None,downloaded=None,untared=None,list_path=None,calls=None):
        with self._lock:
            with open(self.status_path) as f:
                status = [s.strip() for s in f.readlines()]

            status_i_data = status[status_i].split(',')
            if tar_name is None:
                tar_name = status_i_data[0]
            if downloaded is None:
                downloaded = status_i_data[0]
            if untared is None:
                untared = status_i_data[0]
            if list_path is None:
                list_path = status_i_data[0]
            if calls is None:
                calls = status_i_data[0]


            status[status_i] = '{},{},{},{},{}'.format(tar_name,downloaded,untared,list_path,calls)

            with open(self.status_path, 'w') as f:
                f.write('\n'.join(status))


    def updateImages(self,list_path):
        self.images = []
        with open(list_path) as lst:
            images = [path.strip() for path in lst.readlines()]
        for path in images:
            try:
                name = path[path.rindex('/')+1:]
            except ValueError:
                name = path
            image_path = os.path.join(dirPath,path+'.png')
            json_path = os.path.join(dirPath,path+'.layout.json')

            self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':json_path, 'rescaled':1.0 })

    def startWorker(self):
        x = threading.Thread(target=loader, args=(self,), daemon=True)
        x.start()


def loader(dataset):
    with dataset.loader_lock:
        did_something=True
        while did_something:
            status=dataset.getStatus()
            did_something = False
            for i,(tar_name,downloaded,untared,list_path,calls):
                if not downloaded:
                    download(dataset,tar_name)
                    downloaded=True
                    dataset.updateStatus(i,downloaded=True)
                if not untared:
                    untar(dataset,tar_name)
                    list_path = getListPath(dataset,tar_name)
                    untared=True
                    dataset.updateStatus(i,untared=True,list_path=list_path)
                    did_something = True

def download(dataset,tar_name):
    url = dataset.download_urls[tar_name]
    urllib.request.urlretrieve(url, outpath)
