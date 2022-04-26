#Dataset for CDIP dataset hosted on my Box
#Dowloads a chunk to use and starts downloading another
import numpy as np
import os
import shutil
import math, random, string, re
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from data_sets.para_qa_dataset import ParaQADataset, collate
import threading
import urllib
import csv
import tarfile
from utils.util import ensure_dir

import utils.img_f as img_f


class CDIPCloudQA(ParaQADataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super().__init__(dirPath,split,config,images)
        if split!='train':
            self.images=[]
            return

        self.cache_resized = False
        #NEW the document must have a block_score above thresh for anything useing blocks (this is newline following too)
        self.block_score_thresh = 0.73 #eye-balled this one

        if config['super_computer']:
            self.cache_dir = '/tmp/cache_'+config['super_computer']
            self.status_path = os.path.join(self.cache_dir,'status.csv')
            self.tar_dir = '/fslhome/brianld/compute/uploaded'
            self.rm_tar_when_done=False
        else:
            self.cache_dir = os.path.join(dirPath,'cache')
            self.status_path = os.path.join(dirPath,'status.csv')
            self.tar_dir = self.cache_dir
            self.rm_tar_when_done=True

        ensure_dir(self.cache_dir)

        self._lock = threading.Lock()
        self.loader_lock = threading.Lock()
        self.reuse_factor = config['reuse_factor'] if 'reuse_factor' in config else 1.0
        self.calls = 0
        self.num_load_queue = 2
        self.to_rm = []

        assert images is None

        csv_path = 'data_sets/cdip_download_urls.csv'#os.path.join(dirPath,'download_urls.csv')
        with open(csv_path) as f:
            download_urls = csv.reader(f)
            self.download_urls = dict(download_urls)

        #check if any are downloaded:
        if os.path.exists(self.status_path):
            status = self.getStatus()
        else:
            status = []
        queued_tars=set()
        for s in status:
            queued_tars.add(s['name'])
        all_tars = set(self.download_urls.keys())
        while len(status)<self.num_load_queue:
            #pick next tar
            left_tars = all_tars-queued_tars
            todo_tar = random.choice(list(left_tars))
            #update status
            self.updateStatus(len(status),todo_tar,False,False,'',0)
            status.append(      {'name': todo_tar,
                                 'downloaded': False,
                                 'untared': False,
                                 'list_path': '',
                                 'calls': 0
                                 })
            queued_tars.add(todo_tar)
        max_calls = -1 #We left off on whatever one has the most calls
        list_path = None
        self.using = None
        for i,s in enumerate(status):
            if s['downloaded'] and s['untared'] and s['calls']>max_calls:
                self.using = i
                max_calls = s['calls']
                list_path = s['list_path']
                self.calls = s['calls']

        if list_path is None:
            #do the download/untar in this thread to block
            print('CDIP needs to get dataset ready, will stall...')
            for i,s in enumerate(status):
                if s['downloaded']:
                    list_path = untar(self,status[i]['name'],i)
                    self.calls = 0
                    self.using = i
                    break
            if list_path is None:
                download(self,status[0]['name'],0)
                list_path = untar(self,status[0]['name'],0)
                self.calls = 0
                self.using = 0

        self.updateImages(list_path)

        self.startWorker()





    def parseAnn(self,ocr,s):
        
            

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


        self.calls += 1
        #print('calls {} / {}'.format(self.calls,len(self.images)*self.reuse_factor))
        if self.calls > len(self.images)*self.reuse_factor:
            self.switch()
        elif self.calls%100 == 0:
            self.updateStatus(self.using,calls=self.calls)


        return qa_bbs, list(range(qa_bbs.shape[0])), None, None, qa

    #switches to other downloaded chunk (if one is downloaded)
    # and starts downloading new replacement
    def switch(self):
        use_next = None

        status = self.getStatus()
        status_tars=set()
        for i,s in enumerate(status):
            status_tars.add(s['name'])
            if i!=self.using and s['downloaded'] and s['untared']:
                use_next=i
                break

        if use_next is not None:
            print('CDIP switching to '+status[use_next]['name'])

            self.updateImages(status[use_next]['list_path'])
            
            #pick next tar
            all_tars = set(self.download_urls.keys())
            new_tars = all_tars-status_tars
            todo_tar = random.choice(list(new_tars))
    
            self.to_rm.append(status[self.using]['name'])

            #update status
            self.updateStatus(self.using,todo_tar,False,False,'',0)

            self.using = use_next
            self.calls=status[use_next]['calls']
            
            assert len(status)==2
            self.startWorker()
        #else:
        #    print('cannot switch, not ready')


    #read the status file and return
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
            try:
                with open(self.status_path) as f:
                    status = [s.strip() for s in f.readlines()]
            except FileNotFoundError:
                status=[]

            status_i_data = status[status_i].split(',') if len(status)>status_i else None
            if tar_name is None:
                tar_name = status_i_data[0]
            if downloaded is None:
                downloaded = status_i_data[1]
            if untared is None:
                untared = status_i_data[2]
            if list_path is None:
                list_path = status_i_data[3]
            if calls is None:
                calls = status_i_data[4]


            data = '{},{},{},{},{}'.format(tar_name,downloaded,untared,list_path,calls)
            if len(status)>status_i:
                status[status_i] = data
            else:
                assert status_i==len(status)
                status.append(data)

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
            #if path.endswith('.layout.json':
            #        print('ERROR 
            image_path = os.path.join(self.cache_dir,path+'.png')
            json_path = os.path.join(self.cache_dir,path+'.layout.json')

            self.images.append({'id':name, 'imageName':name, 'imagePath':image_path, 'annotationPath':json_path, 'rescaled':1.0 })

    def startWorker(self):
        x = threading.Thread(target=loader, args=(self,), daemon=True)
        x.start()


def loader(dataset):
    with dataset.loader_lock:
        while len(dataset.to_rm)>0:
            tar_name = dataset.to_rm.pop()
            remove(dataset,tar_name)
        did_something=True
        while did_something:
            status=dataset.getStatus()
            did_something = False
            for i,s in enumerate(status):
                tar_name = s['name']
                downloaded = s['downloaded']
                untared = s['untared']
                if not downloaded:
                    assert not untared
                    download(dataset,tar_name,i)
                if not untared:
                    untar(dataset,tar_name,i)
                    did_something = True

def remove(dataset,tar_name):
    print('CDIP removing dir for '+tar_name)
    name = tar_name[:3]
    dir_path = os.path.join(dataset.cache_dir,name)
    shutil.rmtree(dir_path)

def download(dataset,tar_name,i):
    outpath = os.path.join(dataset.tar_dir,tar_name)
    if not os.path.exists(outpath):
        print('CDIP downloading '+tar_name)
        url = dataset.download_urls[tar_name]
        urllib.request.urlretrieve(url, outpath)
    dataset.updateStatus(i,downloaded=True)

def untar(dataset,tar_name,i):
    try:
        print('CDIP untarring '+tar_name)
        tar = tarfile.open(os.path.join(dataset.tar_dir,tar_name))
        tar.extractall(dataset.cache_dir)
        list_path = getListPath(dataset,tar_name)
        dataset.updateStatus(i,untared=True,list_path=list_path)
        if dataset.rm_tar_when_done:
            tar_path = os.path.join(dataset.tar_dir,tar_name)
            os.remove(tar_path)
    except tarfile.ReadError:
        try:
            download(dataset,tar_name,i)
            print('CDIP untarring again '+tar_name)
            tar = tarfile.open(os.path.join(dataset.tar_dir,tar_name))
            tar.extractall(dataset.cache_dir)
            list_path = getListPath(dataset,tar_name)
            dataset.updateStatus(i,untared=True,list_path=list_path)
        except:
            print('ERROR CDIP could not load tar: '+tar_name)
            remove(dataset,tar_name)
            #pick next tar
            all_tars = set(dataset.download_urls.keys())
            todo_tar = random.choice(list(all_tars))
            #update status
            dataset.updateStatus(i,todo_tar,False,False,'',0)
            return None

    print('CDIP ready '+tar_name)
    return list_path

def getListPath(dataset,tar_name):
    name = tar_name[:4]
    list_name = name+'list'
    possible_paths=[
            os.path.join(dataset.cache_dir,'compute','out'+name[0],list_name),
            os.path.join(dataset.cache_dir,'Data6/davis/CDIP_ready2/',list_name),
            os.path.join(dataset.cache_dir,'/fslhome/brianld/compute/out'+list_name[0],list_name),
            os.path.join(dataset.cache_dir,list_name),
            ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    print('Could not find list for '+tar_name)
    print(possible_paths)
    exit(1)
