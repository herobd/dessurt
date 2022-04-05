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


class GraphPairDataset(torch.utils.data.Dataset):
    """
    Class for reading dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        ##self.questions = config['questions'] if 'questions' in config else False
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

        self.semi_supervised = config['semi_supervised'] if 'semi_supervised' in config else False #Float




    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,scaleP=None,cropPoint=None):
        ##ticFull=timeit.default_timer()
        imagePath = self.images[index]['imagePath']
        imageName = self.images[index]['imageName']
        annotationPath = self.images[index]['annotationPath']
        #print(annotationPath)
        rescaled = self.images[index]['rescaled']
        with open(annotationPath) as annFile:
            annotations = json.loads(annFile.read())

        ##tic=timeit.default_timer()
        np_img = img_f.imread(imagePath, 1 if self.color else 0)#*255.0
        if np_img.max()<200:
            np_img*=255
        if np_img is None or np_img.shape[0]==0:
            print("ERROR, could not open "+imagePath)
            return self.__getitem__((index+1)%self.__len__())
        if scaleP is None:
            s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        else:
            s = scaleP
        partial_rescale = s/rescaled
        if self.transform is None: #we're doing the whole image
            #this is a check to be sure we don't send too big images through
            pixel_count = partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]
            if pixel_count > self.pixel_count_thresh:
                partial_rescale = math.sqrt(partial_rescale*partial_rescale*self.pixel_count_thresh/pixel_count)
                print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,pixel_count,rescaled*partial_rescale,partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]))
                s = rescaled*partial_rescale


            max_dim = partial_rescale*max(np_img.shape[0],np_img.shape[1])
            if max_dim > self.max_dim_thresh:
                partial_rescale = partial_rescale*(self.max_dim_thresh/max_dim)
                print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,max_dim,rescaled*partial_rescale,partial_rescale*max(np_img.shape[0],np_img.shape[1])))
                s = rescaled*partial_rescale

        
        
        ##tic=timeit.default_timer()
        #np_img = img_f.resize(np_img,(target_dim1, target_dim0))
        np_img = img_f.resize(np_img,(0,0),
                fx=partial_rescale,
                fy=partial_rescale,
                )
        if len(np_img.shape)==2:
            np_img=np_img[...,None] #add 'color' channel
        if self.color and np_img.shape[2]==1:
            np_img = np.repeat(np_img,3,axis=2)
        ##print('resize: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))
        
        ##tic=timeit.default_timer()

        bbs,ids,numClasses,trans, groups, metadata, form_metadata = self.parseAnn(annotations,s)
        #trans = {i:v for i,v in enumerate(trans)}
        #metadata = {i:v for i,v in enumerate(metadata)}

        #start_of_line, end_of_line = getStartEndGT(annotations['byId'].values(),s)
        #Try:
        #    table_points, table_pixels = self.getTables(
        #            fieldBBs,
        #            s, 
        #            np_img.shape[0], 
        #            np_img.shape[1],
        #            annotations['samePairs'])
        #Except Exception as inst:
        #    if imageName not in self.errors:
        #        table_points=None
        #        table_pixels=None
        #        print(inst)
        #        print('Table error on: '+imagePath)
        #        self.errors.append(imageName)


        #pixel_gt = table_pixels

        ##ticTr=timeit.default_timer()
        #if self.questions: #we need to do questions before crop to have full context
        #    #we have to relationships to get questions
        #    pairs=set()
        #    for index1,id in enumerate(ids): #updated
        #        responseBBIdList = self.getResponseBBIdList(id,annotations)
        #        for bbId in responseBBIdList:
        #            try:
        #                index2 = ids.index(bbId)
        #                pairs.add((min(index1,index2),max(index1,index2)))
        #            except ValueError:
        #                pass
        #    groups_adj = set()
        #    if groups is not None:
        #        for n0,n1 in pairs:
        #            g0=-1
        #            g1=-1
        #            for i,ns in enumerate(groups):
        #                if n0 in ns:
        #                    g0=i
        #                    if g1!=-1:
        #                        break
        #                if n1 in ns:
        #                    g1=i
        #                    if g0!=-1:
        #                        break
        #            if g0!=g1:
        #                groups_adj.add((min(g0,g1),max(g0,g1)))
        #    questions_and_answers = self.makeQuestions(bbs,trans,groups,groups_adj)
        #else:
        questions_and_answers=None

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
            ##tic=timeit.default_timer()
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)
            else:
                np_img = augmentation.apply_tensmeyer_brightness(np_img,**self.aug_params)
            ##print('augmentation: {}'.format(timeit.default_timer()-tic))
        if questions_and_answers is not None:
            if len(questions_and_answers) > self.questions:
                questions_and_answers = random.sample(questions_and_answers,k=self.questions)
            if len(questions_and_answers)>0:
                questions,answers,_ = zip(*questions_and_answers)
            else:
                return self.getitem((index+1)%len(self))
        else:
            questions=answers=None




        newGroups = []
        for group in groups:
            newGroup=[ids.index(bbId) for bbId in group if bbId in ids]
            if len(newGroup)>0:
                newGroups.append(newGroup)
                #print(len(newGroups)-1,newGroup)
        groups=newGroups
        ##print('transfrm: {}  [{}, {}]'.format(timeit.default_timer()-ticTr,org_img.shape[0],org_img.shape[1]))
        pairs=set()
        #import pdb;pdb.set_trace()
        numNeighbors=[0]*len(ids)
        for index1,id in enumerate(ids): #updated
            responseBBIdList = self.getResponseBBIdList(id,annotations)
            for bbId in responseBBIdList:
                try:
                    index2 = ids.index(bbId)
                    #adjMatrix[min(index1,index2),max(index1,index2)]=1
                    pairs.add((min(index1,index2),max(index1,index2)))
                    numNeighbors[index1]+=1
                except ValueError:
                    pass
        #ones = torch.ones(len(pairs))
        #if len(pairs)>0:
        #    pairs = torch.LongTensor(list(pairs)).t()
        #else:
        #    pairs = torch.LongTensor(pairs)
        #adjMatrix = torch.sparse.FloatTensor(pairs,ones,(len(ids),len(ids))) # This is an upper diagonal matrix as pairings are bi-directional

        #if len(np_img.shape)==2:
        #    img=np_img[None,None,:,:] #add "color" channel and batch
        #else:
        img = np_img.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0
        #if pixel_gt is not None:
        #    pixel_gt = pixel_gt.transpose([2,0,1])[None,...]
        #    pixel_gt = torch.from_numpy(pixel_gt)

        #start_of_line = None if start_of_line is None or start_of_line.shape[1] == 0 else torch.from_numpy(start_of_line)
        #end_of_line = None if end_of_line is None or end_of_line.shape[1] == 0 else torch.from_numpy(end_of_line)
        
        bbs = convertBBs(bbs,self.rotate,numClasses)
        if 'word_boxes' in form_metadata:
             form_metadata['word_boxes'] = convertBBs(form_metadata['word_boxes'][None,...],self.rotate,0)[0,...]
        if len(numNeighbors)>0:
            numNeighbors = torch.tensor(numNeighbors)[None,:] #add batch dim
        else:
            numNeighbors=None
        #if table_points is not None:
        #    table_points = None if table_points.shape[1] == 0 else torch.from_numpy(table_points)
        groups_adj = set()
        if groups is not None:
            for n0,n1 in pairs:
                g0=-1
                g1=-1
                for i,ns in enumerate(groups):
                    if n0 in ns:
                        g0=i
                        if g1!=-1:
                            break
                    if n1 in ns:
                        g1=i
                        if g0!=-1:
                            break
                if g0!=g1:
                    groups_adj.add((min(g0,g1),max(g0,g1)))
            for group in groups:
                for i in group:
                    assert(i<bbs.shape[1])
            targetIndexToGroup={}
            for groupId,bbIds in enumerate(groups):
                targetIndexToGroup.update({bbId:groupId for bbId in bbIds})
        
        transcription = [trans[id] for id in ids]

        if self.semi_supervised:
            groups_adj = set(pair for pair in groups_adj if random.random()<self.semi_supervised)
            only_groups=set()
            for g1,g2 in groups_adj:
                only_groups.add(g1)
                only_groups.add(g2)
        else:
            only_groups=None

        return {
                "img": img,
                "bb_gt": bbs,
                "num_neighbors": numNeighbors,
                "adj": None,#pairs,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint,
                "transcription": transcription,
                "metadata": [metadata[id] for id in ids if id in metadata],
                "form_metadata": form_metadata,
                "gt_groups": groups,
                "targetIndexToGroup":targetIndexToGroup,
                "gt_groups_adj": groups_adj,
                "questions": questions,
                "answers": answers,
                "only_groups": only_groups #semi supervised
                }


