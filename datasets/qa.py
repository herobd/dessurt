import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random
from utils.crop_transform import CropBoxTransform
from utils import augmentation
from collections import defaultdict, OrderedDict
from utils.forms_annotations import fixAnnotations, convertBBs, getBBWithPoints, getStartEndGT
import timeit

import utils.img_f as img_f


def collate(batch):
    assert(len(batch)==1)
    return batch[0]


class QADataset(torch.utils.data.Dataset):
    """
    Class for reading dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.questions = config['questions']
        self.color = config['color'] if 'color' in config else True
        self.rotate = config['rotation'] if 'rotation' in config else False
        #patchSize=config['patch_size']
        if 'crop_params' in config and config['crop_params'] is not None:
            self.transform = CropBoxTransform(config['crop_params'],self.rotate)
        else:
            self.transform = None
        self.rescale_range = config['rescale_range']
        if type(self.rescale_range) is float:
            self.rescale_range = [self.rescale_range,self.rescale_range]
        if 'cache_resized_images' in config:
            self.cache_resized = config['cache_resized_images']
            if self.cache_resized:
                self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
                if not os.path.exists(self.cache_path):
                    os.mkdir(self.cache_path)
        else:
            self.cache_resized = False
        self.aug_params = config['additional_aug_params'] if 'additional_aug_params' in config else {}


        self.pixel_count_thresh = config['pixel_count_thresh'] if 'pixel_count_thresh' in config else 10000000
        self.max_dim_thresh = config['max_dim_thresh'] if 'max_dim_thresh' in config else 2700

        #t#self.opt_history = defaultdict(list)#t#




    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,scaleP=None,cropPoint=None):
        #t#ticFull=timeit.default_timer()#t#
        imagePath = self.images[index]['imagePath']
        imageName = self.images[index]['imageName']
        annotationPath = self.images[index]['annotationPath']
        #print(annotationPath)
        rescaled = self.images[index]['rescaled']
        with open(annotationPath) as annFile:
            annotations = json.loads(annFile.read())

        #t#tic=timeit.default_timer()#t#
        #np_img = img_f.imread(imagePath, 1 if self.color else 0)#*255.0
        #if np_img.max()<=1:
        #    np_img*=255
        #if np_img is None or np_img.shape[0]==0:
        #    print("ERROR, could not open "+imagePath)
        #    return self.__getitem__((index+1)%self.__len__())
        #if scaleP is None:
        #    s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        #else:
        #    s = scaleP
        #partial_rescale = s/rescaled
        #if self.transform is None: #we're doing the whole image
        #    #this is a check to be sure we don't send too big images through
        #    pixel_count = partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]
        #    if pixel_count > self.pixel_count_thresh:
        #        partial_rescale = math.sqrt(partial_rescale*partial_rescale*self.pixel_count_thresh/pixel_count)
        #        print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,pixel_count,rescaled*partial_rescale,partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]))
        #        s = rescaled*partial_rescale


        #    max_dim = partial_rescale*max(np_img.shape[0],np_img.shape[1])
        #    if max_dim > self.max_dim_thresh:
        #        partial_rescale = partial_rescale*(self.max_dim_thresh/max_dim)
        #        print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,max_dim,rescaled*partial_rescale,partial_rescale*max(np_img.shape[0],np_img.shape[1])))
        #        s = rescaled*partial_rescale

        #
        ##np_img = img_f.resize(np_img,(target_dim1, target_dim0))
        #np_img = img_f.resize(np_img,(0,0),
        #        fx=partial_rescale,
        #        fy=partial_rescale,
        #)
        np_img = np.array([[0,0],[0,0]])
        s=1

        if len(np_img.shape)==2:
            np_img=np_img[...,None] #add 'color' channel
        if self.color and np_img.shape[2]==1:
            np_img = np.repeat(np_img,3,axis=2)
        ##print('resize: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))
        
        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['image read and setup'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        bbs,ids,trans, metadata, form_metadata, questions_and_answers = self.parseAnn(annotations,s)

        #t#time = timeit.default_timer()-tic#t#
        #t#self.opt_history['parseAnn'].append(time)#t#
        #t#tic=timeit.default_timer()#t#

        if self.transform is not None:
            if 'word_boxes' in form_metadata:
                word_bbs = form_metadata['word_boxes']
                dif_f = bbs.shape[2]-word_bbs.shape[1]
                blank = np.zeros([word_bbs.shape[0],dif_f])
                prep_word_bbs = np.concatenate([word_bbs,blank],axis=1)[None,...]
                crop_bbs = np.concatenate([bbs,prep_word_bbs],axis=1)
                crop_ids=ids+['word{}'.format(i) for i in range(word_bbs.shape[0])]
            else:
                crop_bbs = bbs
                crop_ids = ids
            out, cropPoint = self.transform({
                "img": np_img,
                "bb_gt": crop_bbs,
                'bb_auxs':crop_ids,
                #'word_bbs':form_metadata['word_boxes'] if 'word_boxes' in form_metadata else None
                #"line_gt": {
                #    "start_of_line": start_of_line,
                #    "end_of_line": end_of_line
                #    },
                #"point_gt": {
                #        "table_points": table_points
                #        },
                #"pixel_gt": pixel_gt,
                
            }, cropPoint)
            np_img = out['img']

            if 'word_boxes' in form_metadata:
                saw_word=False
                word_index=-1
                for i,ii in enumerate(out['bb_auxs']):
                    if not saw_word:
                        if type(ii) is str and 'word' in ii:
                            saw_word=True
                            word_index=i
                    else:
                        assert 'word' in ii
                bbs = out['bb_gt'][:,:word_index]
                ids= out['bb_auxs'][:word_index]
                form_metadata['word_boxes'] = out['bb_gt'][0,word_index:,:8]
                word_ids=out['bb_auxs'][word_index:]
                form_metadata['word_trans'] = [form_metadata['word_trans'][int(id[4:])] for id in word_ids]
            else:
                bbs = out['bb_gt']
                ids= out['bb_auxs'] 

            if questions_and_answers is not None:
                questions=[]
                answers=[]
                questions_and_answers = [(q,a,qids) for q,a,qids in questions_and_answers if all((i in ids) for i in qids)]
        if questions_and_answers is not None:
            if len(questions_and_answers) > self.questions:
                questions_and_answers = random.sample(questions_and_answers,k=self.questions)
            if len(questions_and_answers)>0:
                questions,answers,_ = zip(*questions_and_answers)
            else:
                return self.getitem((index+1)%len(self))
        else:
            questions=answers=None




            ##tic=timeit.default_timer()
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)
            else:
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)
            ##print('augmentation: {}'.format(timeit.default_timer()-tic))

        img = np_img.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0
        #if pixel_gt is not None:
        #    pixel_gt = pixel_gt.transpose([2,0,1])[None,...]
        #    pixel_gt = torch.from_numpy(pixel_gt)

        bbs = convertBBs(bbs,self.rotate,1)
        #if 'word_boxes' in form_metadata:
        #     form_metadata['word_boxes'] = convertBBs(form_metadata['word_boxes'][None,...],self.rotate,0)[0,...]

        
        transcription = [trans[id] for id in ids]

        #t#time = timeit.default_timer()#t#
        #t#self.opt_history['remainder'].append(time-tic)#t#
        #t#self.opt_history['Full get_item'].append(time-ticFull)#t#
        #t#self.print_opt_times()#t#


        return {
                "img": img,
                "bb_gt": bbs,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint,
                "transcription": transcription,
                "metadata": [metadata[id] for id in ids if id in metadata],
                "form_metadata": None,
                "questions": questions,
                "answers": answers
                }

    #t#def print_opt_times(self):#t#
        #t#for name,times in self.opt_history.items():#t#
            #t#print('time data {}({}): {}'.format(name,len(times),np.mean(times)))#t#
            #t#if len(times)>300: #t#
                #t#times.pop(0)   #t#
                #t#if len(times)>600:#t#
                    #t#times.pop(0)#t#
