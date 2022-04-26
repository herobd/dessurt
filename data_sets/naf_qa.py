import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string
from collections import defaultdict, OrderedDict
import timeit
from data_sets.form_qa import FormQA,collate, Line, Entity, Table, FillInProse, MinoredField
from utils.forms_annotations import fixAnnotations
from utils.read_order import getVertReadPosition,getHorzReadPosition,putInReadOrder,sortReadOrder, intersection,getHeight,sameLine

from utils import img_f


SKIP=['174']#bad for some reason I can't remember

class NAFQA(FormQA):
    """
    Class for reading NAF dataset and preping for FormQA format
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(NAFQA, self).__init__(dirPath,split,config,images)
        
        cant_do = ['cell','row-header', 'col-header','full-all-row', 'full-all-col','all-row', 'all-col'] #becuase the transcriptions of the table cells isn't done, no tasks involving cells should be used
        for cant in cant_do:
            if cant in self.q_types:
                del self.q_types[cant]
            if cant in self.q_types_no_table:
                del self.q_types_no_table[cant]
            if cant in self.q_types_only_table:
                del self.q_types_only_table[cant]

        self.min_start_read = 7
        self.cased=True

        self.crop_to_data = self.train
        self.warp_lines = None

        self.extra_np = 0.05

        with open('data_sets/long_naf_images.txt') as f:
            #These are images where the resulting JSON is long (can't be predicted in single output)
            #We'll do these twice as often to compinsate
            do_twice = f.read().split('\n')

        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                split_file = 'overfit_split.json' #tiny split for debugging purposes
            else:
                split_file = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,split_file)) as f:
                readFile = json.loads(f.read())
                if type(split) is str:
                    groups_to_use = readFile[split]
                elif type(split) is list:
                    groups_to_use = {}
                    for spstr in split:
                        newGroups = readFile[spstr]
                        groups_to_use.update(newGroups)
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
            self.images=[]
            group_names = list(groups_to_use.keys())
            group_names.sort()

            #Go through each group and get its images
            
            for groupName in group_names:
                imageNames=groups_to_use[groupName]
                
                if groupName in SKIP:
                    print('Skipped group {}'.format(groupName))
                    continue
                for imageName in imageNames:

                    org_path = os.path.join(dirPath,'groups',groupName,imageName)
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName)
                    else:
                        path = org_path
                    jsonPath = org_path[:org_path.rfind('.')]+'.json'
                    if os.path.exists(jsonPath): #not all images have annotations
                        rescale=1.0
                        if self.cache_resized:
                            rescale = self.rescale_range[1]
                            if not os.path.exists(path):
                                org_img = img_f.imread(org_path)
                                if org_img is None:
                                    print('WARNING, could not read {}'.format(org_img))
                                    continue
                                resized = img_f.resize(org_img,(0,0),
                                        fx=self.rescale_range[1], 
                                        fy=self.rescale_range[1], 
                                        )
                                img_f.imwrite(path,resized)
                        if self.train:
                            name = imageName[:imageName.rfind('.')]

                            #add image
                            self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':name})
                            if name in do_twice:
                                #easy way to do twice as frequently
                                self.images.append({'id':imageName+'2', 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':name})

                        else:
                            assert self.rescale_range[1]==self.rescale_range[0]
                            assert self.questions==1
                            #create questions for each image
                            with open(jsonPath) as f:
                                annotations = json.load(f)
                            all_entities,entity_link,tables,proses,minored_fields,bbs,link_dict = self.getEntitiesAndSuch(annotations,rescale)
                            qa = self.makeQuestions(self.rescale_range[1],all_entities,entity_link,tables,all_entities,link_dict,proses=proses,minored_fields=minored_fields)
                            for _qa in qa:
                                _qa['bb_ids']=None
                                self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')], 'qa':[_qa]})


    def parseAnn(self,annotations,s):

        all_entities,entity_link,tables,proses,minored_fields,bbs,link_dict = self.getEntitiesAndSuch(annotations,s)


        if self.train:
            qa = self.makeQuestions(s,all_entities,entity_link,tables,all_entities,link_dict,proses=proses,minored_fields=minored_fields)
        else:
            qa = None #This is pre-computed

        return bbs, list(range(bbs.shape[0])), None, None, qa


    def convertBB(self,s,box):
        #assert s==1
        if isinstance(box,list):
            return [v*s for v in box]
        return (s*box).tolist()

    #Converts NAF types to FUNSD classes
    def typeToClass(self,typ):
        if 'Circle' in typ:
            cls = 'circle'
        elif 'text' in typ:
            cls = 'question'
        elif 'field' in typ:
            assert 'Row' not in typ
            assert 'Col' not in typ
            cls = 'answer'
        elif 'comment' in typ:
            cls = 'other'
        else:
            assert False
        return cls

    #Processes the NAF annotations, grouping the lines into entities and getting the tables
    def getEntitiesAndSuch(self,annotations,image_scale):
        fixAnnotations(None,annotations) #fixes somethings

        all_bbs = annotations['byId']
        for bb in all_bbs.values():
            bb['poly_points'] = np.array(bb['poly_points'])
        all_pairs = annotations['pairs']#+annotations['samePairs']
        transcriptions = annotations['transcriptions']
        
        #group lines into entities
        #This is whenever a pair have the same type or are a fill-in-the-blank prose (para)
        groups = defaultdict(list)
        group_num=0
        id_to_group = defaultdict(lambda: None)
        same_row = []
        same_col = []
        all_pairs = sortReadOrder([(pair,all_bbs[pair[0]]['poly_points']) for pair in all_pairs]) #so we can assume read order
        for id1,id2 in all_pairs:
            if all_bbs[id1]['type']==all_bbs[id2]['type'] or ('P' in all_bbs[id1]['type'] and 'P' in all_bbs[id2]['type']):
                if 'Row' in all_bbs[id1]['type'] and 'Row' in all_bbs[id2]['type']:
                    #a single row annotated with two rectangles due to page warping
                    same_row.append((id1,id2))
                elif 'Col' in all_bbs[id1]['type'] and 'Col' in all_bbs[id2]['type']:
                    #single column
                    same_col.append((id1,id2))
                else:
                    assert 'Row' not in all_bbs[id1]['type'] and 'Row' not in all_bbs[id2]['type']
                    assert 'Col' not in all_bbs[id1]['type'] and 'Col' not in all_bbs[id2]['type']
                    #There's a specific case we don't want to group, which is things on the same textline connected to Minor labels
                    if sameLine(all_bbs[id1]['poly_points'],all_bbs[id2]['poly_points']):
                        #check Minor links
                        dont=False
                        for l1,l2 in all_pairs:
                            if (l1==id1 or l1==id2) and 'Minor' in all_bbs[l2]['type']:
                                dont=True
                                break
                            if (l2==id1 or l2==id2) and 'Minor' in all_bbs[l1]['type']:
                                dont=True
                                break
                        if dont:
                            continue
                    


                    group1 = id_to_group[id1]
                    group2 = id_to_group[id2]
                    if group1 is None and group2 is None:
                        groups[group_num] = [id1,id2]
                        id_to_group[id1]=group_num
                        id_to_group[id2]=group_num
                        group_num+=1
                    elif group1 is None:
                        groups[group2].append(id1)
                        id_to_group[id1]=group2
                    elif group2 is None:
                        groups[group1].append(id2)
                        id_to_group[id2]=group1
                    elif group1 != group2:
                        if group1>group2:
                            group1,group2 = group2,group1 #preserve read order
                        #merge
                        for idx in groups[group2]:
                            id_to_group[idx] = group1
                        #append in read order
                        groups[group1]+=groups[group2]
                        del groups[group2]
                    #else nothing needed

        used_ids = set()
        e_groups = []
        entities = []
        bb_to_e = {}
        proses = []
        prose_groups = []
        bb_to_prose = {}
        tables = []
        entity_link = defaultdict(list) #(head,tails)
        has_link=set()

        #Create the entities
        for group in groups.values():
            #sort the text lines
            group = sortReadOrder([(bb_id,all_bbs[bb_id]['poly_points']) for bb_id in group])
            if all(all_bbs[group[0]]['type'] == all_bbs[bb_id]['type'] for bb_id in group[1:]):
                #assert 'Row' not in all_bbs[id1]['type'] and 'Row' not in all_bbs[id2]['type']
                #assert 'Col' not in all_bbs[id1]['type'] and 'Col' not in all_bbs[id2]['type']
                #same class
                cls = self.typeToClass(all_bbs[group[0]]['type'])
                lines=[]
                for bb_id in group:
                    if bb_id not in transcriptions and 'Circle' in all_bbs[group[0]]['type']:
                        transcriptions[bb_id]='§'#'UNTRANSCRIBED CIRCLE'
                    elif all_bbs[bb_id]['isBlank']=='blank':
                        continue
                    elif bb_id not in transcriptions:
                        continue
                    elif transcriptions[bb_id]=='':
                        continue
                    lines.append(Line(transcriptions[bb_id],all_bbs[bb_id]['poly_points']))
                    bb_to_e[bb_id]=len(entities)

                if len(lines)>0:
                    entities.append(Entity(cls,lines))
                    e_groups.append(group)
            else:
                assert all('P' in all_bbs[bb_id]['type'] for bb_id in group)
                #A fill-in-the-blank prose
                p_entities = []
                p_group = []
                lines = []
                prev_type = all_bbs[group[0]]['type']
                prev_id = None
                cls = 'question' if 'text' in prev_type else 'answer'

                #go through text lines, mering text from adjacent lines with the same class
                for bb_id in group:
                    if all_bbs[bb_id]['isBlank']=='blank':
                        transcriptions[bb_id]=self.blank_token
                    elif (bb_id not in transcriptions or transcriptions[bb_id]=='') and 'answer'==cls:
                        transcriptions[bb_id]=self.blank_token
                    elif bb_id not in transcriptions or transcriptions[bb_id]=='':
                        continue
                    
                    l = Line(transcriptions[bb_id],all_bbs[bb_id]['poly_points'  ])
                    if all_bbs[bb_id]['type']==prev_type:
                        lines.append(l)
                        p_group.append(bb_id)
                    else:
                        if len(lines)>0:
                            p_entities.append(Entity(cls,lines))
                            if prev_id is not None:
                                entity_link[prev_id].append(len(entities))
                            for a_id in p_group:
                                bb_to_e[a_id]=len(entities)
                            prev_id = len(entities) #update
                            has_link.add(len(entities))
                            entities.append(p_entities[-1])
                            e_groups.append(p_group)
                        lines=[l]
                        p_group = [bb_id]
                        prev_type = all_bbs[bb_id]['type']
                        cls = 'question' if 'text' in prev_type else 'answer'

                if len(lines)>0:
                    p_entities.append(Entity(cls,lines))
                    if prev_id is not None:
                        entity_link[prev_id].append(len(entities))
                        has_link.add(len(entities))
                    entities.append(p_entities[-1])
                    e_groups.append(p_group)
                    for a_id in p_group:
                        bb_to_e[a_id]=len(entities)
                
                if len(p_entities)>1:
                    for bb_id in group:
                        bb_to_prose[bb_id]=len(proses)
                    proses.append(FillInProse(p_entities))
                    prose_groups.append(group)
        
            used_ids.update(group)

        #put all unsed bbs in their own groups (except table elements)
        unused_ids = set(all_bbs.keys())-used_ids
        for bb_id in unused_ids:
            bb= all_bbs[bb_id]
            if 'Row' not in bb['type'] and 'Col' not in bb['type']:
                if bb['isBlank']!='blank' and bb_id in transcriptions and transcriptions[bb_id]!='':
                    cls = self.typeToClass(bb['type'])

                    bb_to_e[bb_id]=len(e_groups)
                    entities.append(Entity(cls,[Line(transcriptions[bb_id],bb['poly_points'])]))
                    e_groups.append([bb_id])

        #build tables
        found_tables=[]
        for bb in all_bbs.values():
            if 'Row' in bb['type'] or 'Col' in bb['type']:
                intersections = []
                for ti,lines in enumerate(found_tables):
                    for line_bb in lines:
                        if intersection(bb,line_bb):
                            intersections.append(ti)
                            break

                if len(intersections) == 0:
                    found_tables.append([bb])
                elif len(intersections) == 1:
                    found_tables[ti].append(bb)
                else:
                    new_lines = [bb]
                    for ti in reversed(intersections):
                        new_lines+=found_tables[ti]
                        del found_tables[ti]
                    found_tables.append(new_lines)

        table_headers=set()
        for lines in found_tables:
            rows = []
            cols = []

            #ugh,sort out same_row
            dont=[]
            for rid1,rid2 in same_row:
                for bb in lines:
                    if bb['id']==rid2:
                        got_it=False
                        for id1,id2 in all_pairs:
                            if id1==bb['id'] and 'field' not in all_bbs[id2]['type']:
                                dont.append(rid1)
                                got_it=True
                                break
                            elif id2==bb['id'] and 'field' not in all_bbs[id1]['type']:
                                dont.append(rid1)
                                got_it=True
                                break
                        if not got_it:
                            dont.append(rid2)
                        break
            #same col
            for rid1,rid2 in same_col:
                for bb in lines:
                    if bb['id']==rid2:
                        got_it=False
                        for id1,id2 in all_pairs:
                            if id1==bb['id'] and 'field' not in all_bbs[id2]['type']:
                                dont.append(rid1)
                                got_it=True
                                break
                            elif id2==bb['id'] and 'field' not in all_bbs[id1]['type']:
                                dont.append(rid1)
                                got_it=True
                                break
                        if not got_it:
                            dont.append(rid2)
                        break
            
            #find things linked to the table (row and column headers)
            for bb in lines:
                links=[]
                if bb['id'] in dont:
                    continue
                for id1,id2 in all_pairs:
                    if id1==bb['id'] and 'field' not in all_bbs[id2]['type']:
                        links.append(id2)
                    elif id2==bb['id'] and 'field' not in all_bbs[id1]['type']:
                        links.append(id1)


                if 'Row' in bb['type']:
                    pos = getVertReadPosition(bb['poly_points'])

                    #generally, there is just one header
                    if len(links)>1:
                        if  len(links)==2:
                            #some special cases when there's 2
                            posR = getHorzReadPosition(bb['poly_points'])
                            pos0 = getHorzReadPosition(all_bbs[links[0]]['poly_points'])
                            pos1 = getHorzReadPosition(all_bbs[links[1]]['poly_points'])

                            if pos0<posR and pos<pos1:
                                linked=links[0]
                            elif pos1<posR and pos<pos0:
                                linked = links[1]
                            elif 'Number' in all_bbs[links[0]]['type']:
                                linked = links[1]
                            elif 'Number' in all_bbs[links[1]]['type']:
                                linked = links[0]
                            else:
                                assert False
                        else:
                            #if there's more, just take the one closest to the row
                            links = [l for l in links if 'Number' not in all_bbs[l]['type']]
                            if len(links)>0:
                                min_dist=999999
                                best_l=None
                                for bb_id in links:
                                    pos_l = getVertReadPosition(all_bbs[bb_id]['poly_points'])
                                    dist = abs(pos-pos_l)
                                    if dist<min_dist:
                                        min_dist=dist
                                        best_l = bb_id
                                links=[best_l]
                    else:
                        linked=links[0] if len(links)>0 else None

                    rows.append((pos,bb,linked))
                else:
                    pos = getHorzReadPosition(bb['poly_points'])
                    if len(links)>0:
                        links = [l for l in links if 'Number' not in all_bbs[l]['type']]
                        if len(links)>0:
                            min_dist=999999
                            best_l=None
                            for bb_id in links:
                                pos_l = getHorzReadPosition(all_bbs[bb_id]['poly_points'])
                                dist = abs(pos-pos_l)
                                if dist<min_dist:
                                    min_dist=dist
                                    best_l = bb_id
                            links=[best_l]
                    linked=links[0] if len(links)>0 else None
                    cols.append((pos,bb,linked))

            rows.sort(key=lambda a:a[0])
            cols.sort(key=lambda a:a[0])
            
            row_headers=[]
            for pos,line,header_id in rows:
                if header_id is not None and header_id in bb_to_e:
                    e_id = bb_to_e[header_id]
                    has_link.add(e_id)
                    table_headers.add(e_id)
                    row_headers.append(entities[e_id])
            col_headers=[]
            for pos,line,header_id in cols:
                if header_id is not None and header_id in bb_to_e:
                    e_id = bb_to_e[header_id]
                    has_link.add(e_id)
                    table_headers.add(e_id)
                    col_headers.append(entities[e_id])
                else:
                    col_headers.append(None)

            if all(r is None for r in row_headers):
                row_headers=[] #becuse we don't have transcription
            table = Table(row_headers,col_headers)
            table.cells=[] #we didn't transcribe table cells, WHOOPS

            tables.append(table)


        
        minor_groups = [] #to keep track of minor labels. We'll make special objects for handeling them
        all_minor = set() #so we can go back include same-same links in minor field
 
        #get the links between entities
        for id1,id2 in all_pairs:

            #We'll initially keep links with blanks so minor groups get processed correctly.
            # We'll remove then later
            try:
                if id1 not in bb_to_e and all_bbs[id1]['isBlank']=='blank':
                    bb1=None
                    type1 = all_bbs[id1]['type']
                else:
                    bb1 = all_bbs[id1]
                    type1 = bb1['type']
                if id2 not in bb_to_e and all_bbs[id2]['isBlank']=='blank':
                    bb2=None
                    type2 = all_bbs[id2]['type']
                else:
                    bb2 = all_bbs[id2]
                    type2 = bb2['type']
            except KeyError:
                continue

            if 'Row' not in type1 and 'Col' not in type2 and 'Row' not in type2 and 'Col' not in type2 and ('P' not in type1 or 'P' not in type2) and type1!=type2:
                try:
                    if bb1 is not None:
                        e_id1 = bb_to_e[id1]
                        e1_cls = entities[e_id1].cls
                    else:
                        e_id1 = id1
                        e1_cls=self.typeToClass(type1)
                    if bb2 is not None:
                        e_id2 = bb_to_e[id2]
                        e2_cls = entities[e_id2].cls
                    else:
                        e_id2 = id2
                        e2_cls=self.typeToClass(type2)
                except KeyError:
                    continue
                if 'question'==e1_cls and ('answer'==e2_cls or 'circle'==e2_cls):
                    entity_link[e_id1].append( e_id2)
                    has_link.add(e_id1)
                elif 'question'==e2_cls and ('answer'==e1_cls or 'circle'==e1_cls):
                    entity_link[e_id2].append( e_id1)
                    has_link.add(e_id2)
                elif 'Minor' in type1 and e_id1 in table_headers:
                    #link to super header
                    entity_link[e_id2].append( e_id1)
                    has_link.add(e_id2)
                elif 'Minor' in type2 and e_id2 in table_headers:
                    #link to super header
                    entity_link[e_id1].append( e_id2)
                    has_link.add(e_id1)

                if ('Minor' in type1 or 'Minor' in type2) and e_id1 not in table_headers and e_id2 not in table_headers:
                    #special handling for groups with minor labels
                    mg1=mg2=None
                    all_minor.update((e_id1,e_id2))
                    for mi,group in enumerate(minor_groups):
                        if e_id1 in group:
                            mg1 = mi
                        if e_id2 in group:
                            mg2 = mi
                        if mg1 is not None and mg2 is not None:
                            break

                    #adding to minor group or merging
                    if mg1 is None and mg2 is None:
                        minor_groups.append([e_id1,e_id2])
                    elif mg1 is None:
                        minor_groups[mg2].append(e_id1)
                    elif mg2 is None:
                        minor_groups[mg1].append(e_id2)
                    else:
                        #merge
                        keep = min(mg1,mg2)
                        remove = max(mg1,mg2)
                        minor_groups[keep]+=minor_groups[remove]
                        del minor_groups[remove]

        additional_links = defaultdict(list) #for minor fields
        for id1,id2 in all_pairs:
            id1 = bb_to_e[id1] if id1 in bb_to_e else id1
            id2 = bb_to_e[id2] if id2 in bb_to_e else id2
            if id1!=id2 and id1 in all_minor and id2 in all_minor:
                additional_links[id1].append(id2)
                additional_links[id2].append(id1)
                    
        link_ups = defaultdict(list)
        for up,downs in entity_link.items():
            for d in downs:
                link_ups[d].append(up)

        new_minor_groups=[]
        for group in minor_groups:
            #find all other attached things
            new_group = set(group)
            for e_id in group:
                new_group.update(entity_link[e_id])
                new_group.update(link_ups[e_id])
                new_group.update(additional_links[e_id])
            if len(new_group.intersection(table_headers))==0:
                new_minor_groups.append(new_group)
        

        #merge minor groups
        while True:
            new_gi=0
            start_len = len(new_minor_groups)
            while new_gi<len(new_minor_groups):
                to_combine=[]
                for i in range(new_gi+1,len(new_minor_groups)):
                    for mine in new_minor_groups[new_gi]:
                        if mine in new_minor_groups[i]:
                            to_combine.append(i)
                            break
                for i in reversed(to_combine):
                    new_minor_groups[new_gi].update(new_minor_groups[i])
                    del new_minor_groups[i]
                new_gi+=1
                

            if len(new_minor_groups) == start_len:
                break



        #now create minor group objects
        minored_fields=[]
        for new_group in new_minor_groups:

            #we expect one question(text) multiple answers and multiple minor
            #sort em out!
            question = None
            answers = []
            minors = []
            blank_answers=[]
            for e_id in new_group:
                if isinstance(e_id,str):
                    blank_answers.append(e_id)
                else:
                    typ = all_bbs[e_groups[e_id][0]]['type']
                    if 'Number' in typ or 'comment' in typ:
                        continue
                    if 'Minor' in typ:
                        minors.append(entities[e_id])
                    elif 'field' in typ:
                        answers.append(entities[e_id])
                    else:
                        assert 'text' in typ
                        if question is None:
                            question = entities[e_id]
                        else:
                            question = sortReadOrder([(question,question.lines[0].box),(entities[e_id],entities[e_id].lines[0].box)])[0]
            
            if len(minors)>len(answers) and len(answers)>0:
                #
                for bb_id in blank_answers:
                    bb = all_bbs[bb_id]
                    e_id = len(entities)
                    cls = self.typeToClass(bb['type'])
                    entities.append(Entity(cls,[Line(self.blank_token,bb['poly_points'])]))
                    answers.append(entities[e_id])
            
            if question is not None or ((len(answers)>1 or len(minors)>1) and (len(answers)>0 and len(minors)>0)):
                answers = sortReadOrder([(ans,ans.lines[0].box) for ans in answers])
                minors = sortReadOrder([(minor,minor.lines[0].box) for minor in minors])
                minored_fields.append(MinoredField(question,answers,minors))
            #otherwise treat these as normal entities
        



        #now, relying on has_link we assign "questions" with no answer to the proper class of other (as all texts that are questions should be linked to a blank field)
        no_link = set(range(len(entities)))-has_link
        for e_id in no_link:
            entity = entities[e_id]
            if entity.cls=='question':
                entity.cls = 'other'
        
        #Remove blank links
        new_entity_link=defaultdict(list)
        link_ups = defaultdict(list)
        for up,downs in entity_link.items():
            if isinstance(up,int):
                downs = [d for d in downs if isinstance(d,int)]
                if len(downs)>0:
                    new_entity_link[up]=downs
                    for d in downs:
                        link_ups[d].append(up)
        entity_link = new_entity_link

        #run through all entites to build bbs, assign bbid, and find ambiguity
        boxes = []
        text_line_counts = defaultdict(list)
        for ei,entity in enumerate(entities):
            for li,line in enumerate(entity.lines):
                text = self.punc_regex.sub('',line.text.lower())
                text_line_counts[text].append((ei,li))
                bbid = len(boxes)
                boxes.append(self.convertBB(image_scale,line.box))
                line.bbid = bbid

        bbs = np.array(boxes)

        #assign ambiguity
        for line_ids in text_line_counts.values():
            if len(line_ids)>1:
                for ei,li in line_ids:
                    entities[ei].lines[li].ambiguous = True

        link_dict=entity_link
        entity_link=[(e1,list(e2s) if e2s is not None else None) for e1,e2s in link_dict.items()]
        #now set up a full linking dictionary
        for e1,e2s in entity_link:
            if e2s is not None:
                for e2 in e2s:
                    if e2 is not None:
                        if link_dict[e2] is None:
                            link_dict[e2]=[]
                        link_dict[e2].append(e1)
            elif link_dict[e1] is None or len(link_dict[e1])==0:
                del link_dict[e1]

        #Add all the link for tables
        for table in tables:
            for r,r_header in enumerate(table.row_headers):
                if r_header is not None:
                    r_index = entities.index(r_header)
                for c,c_header in enumerate(table.col_headers):
                    if c_header is not None:
                        c_index = entities.index(c_header)
                    if len(table.cells)>0:
                        v=table.cells[r][c]
                        if v is not None:
                            v_index = entities.index(v)
                            link_dict[r_index].append(v_index)
                            link_dict[c_index].append(v_index)
                            link_dict[v_index].append(r_index)
                            link_dict[v_index].append(c_index)

        link_dict = self.sortLinkDict(entities,link_dict)
        return entities,entity_link,tables,proses,minored_fields,bbs,link_dict

    #This is going to see if the image is landscape and needs cut in half
    #It only will cut large images
    #It attempts to not split text lines, but will if it has to
    #Also 10% of the time it doesn't split, so the model sees full low-res images sometimes
    def getCropAndLines(self,annotations,shape):
        ratio_threshold=0.9
        width_threshold=2900
        force_half_thresh=3200
        sure_force_half_thresh=3600
        pad=80
        if shape[1]>width_threshold and shape[1]/shape[0]>ratio_threshold and random.random()>0.1:
            mid_x = shape[1]/2
            #see if we can

            #we'll group the bbs based on overlap horizontally
            #If there is a clear split on these groups, we're in the clear
            groups={}
        
            all_bbs = annotations['fieldBBs']+annotations['textBBs']
            min_x = shape[1]
            max_x = 0
            for bb in all_bbs:
                points = bb['poly_points']
                x1 = min(p[0] for p in points)
                x2 = max(p[0] for p in points)

                min_x = min(min_x,x1)
                max_x = max(max_x,x2)

                #check all previously existing groups for horizontal overlap
                added=False
                overlaps=[]
                for rang,bbs in groups.items():
                    if max(x1,rang[0]) < min(x2,rang[1]): #overlap
                        overlaps.append(rang)

                if len(overlaps)==0:
                    #None? make a new group
                    groups[(x1,x2)]=[bb]
                elif len(overlaps)==1:
                    #one, add me, and extend the range appropriately
                    rang = overlaps[0]
                    bbs = groups[rang]
                    if x1<rang[0] or x2<rang[1]:
                        #new range
                        new_rang = (min(x1,rang[0]),max(x2,rang[1]))
                        new_bbs = bbs+[bb]
                        del groups[rang]
                        groups[new_rang]=new_bbs
                    else:
                        bbs.append(bb)
                else:
                    #merge groups
                    new_rang = (min([x1]+[o[0] for o in overlaps]), max([x2]+[o[1] for o in overlaps]))
                    new_bbs = [bb]
                    for rang in overlaps:
                        new_bbs+=groups[rang]
                        del groups[rang]
                    groups[new_rang]=new_bbs

            if len(groups)>1: #something to work with

                ranges = list(groups.keys())
                ranges.sort(key=lambda a:a[0])
                
                #sort into definitely left, def right, and things in the middle
                left=[]
                mid=[]
                right=[]
                for rang in ranges:
                    if rang[1]<=mid_x:
                        left.append(rang)
                    elif rang[0]>=mid_x:
                        right.append(rang)
                    else:
                        mid.append(rang)
                
                #For those in the middle, are they primarily on one side or the other
                are_left=[(m[0]+m[1])/2 < mid_x for m in mid]
                are_right=[(m[0]+m[1])/2 > mid_x for m in mid]
                if all(are_left) or all(are_right) or shape[1]>force_half_thresh:
                    #sometimes forcing the split because the image is too big

                    #if they are, we'll include/disclude them in the split as a whole
                    if all(are_left):
                        left+=mid
                        mid=[]
                    if all(are_right):
                        right+=mid
                        mid=[]
                    if  (random.random()<0.5 and len(left)>0) or len(right)==0:
                        #left
                        X = max(l[1] for l in left)
                        crop = (0,0,X+pad,shape[0])
                        ranges = left
                    else:
                        #right
                        X = min(t[0] for t in right)
                        crop = (X-pad,0,shape[1],shape[0])
                        ranges = right
                    
                    middle_bbs=[]
                    for rang in mid:
                        middle_bbs+=groups[rang]
                    groups['mid bbs']=[]
                    for bb in middle_bbs: #if we couldn't split mid, add the ones that best match
                        points = bb['poly_points']
                        x1 = min(p[0] for p in points)
                        x2 = max(p[0] for p in points)
                        inside = min(x2,crop[2])-max(x1,crop[0])
                        ratio = inside/(x2-x1)
                        if ratio>0.25: #sort of here
                            new_points = [[max(crop[0],min(crop[2],x)),y] for x,y in points] #cropped bb
                            bb['poly_points'] = new_points
                            groups['mid bbs'].append(bb)



                    new_field_bbs = []
                    new_text_bbs = []
                    new_ids = []
                    #add the cut leftovers
                    for rang in ranges+['mid bbs']:
                        for bb in groups[rang]:
                            bb_id = bb['id']
                            new_ids.append(bb_id)
                            if bb_id.startswith('f'):
                                new_field_bbs.append(bb)
                            else:
                                new_text_bbs.append(bb)
                    
                    #rebuild links with remaining bbs
                    new_pairs = []
                    for (p1,p2) in annotations['pairs']:
                        if p1 in new_ids and p2 in new_ids:
                            new_pairs.append([p1,p2])
                    new_same_pairs = []
                    for (p1,p2) in annotations['samePairs']:
                        if p1 in new_ids and p2 in new_ids:
                            new_same_pairs.append([p1,p2])

                    annotations['fieldBBs']=new_field_bbs
                    annotations['textBBs']=new_text_bbs
                    annotations['pairs']=new_pairs
                    annotations['samePairs']=new_same_pairs
                    cropAnnotations(annotations,crop)
                    return crop, None
            elif shape[1]>force_half_thresh and (shape[1]>sure_force_half_thresh or random.random()<0.5):
                #sometimes just split, sometimes just not, unless it's really big
                left = random.random()<0.5
                right = not left

                #see which fall on our side based on midpoint (and not being super long)
                good_bbs=[]
                other_bbs=[]
                right_boundary=0 if left else shape[1]
                left_boundary=shape[1] if right else 0
                for bb in all_bbs:
                    points = bb['poly_points']
                    x1 = min(p[0] for p in points)
                    x2 = max(p[0] for p in points)
                    xm = (x1+x2)/2
                    if ((left and xm<mid_x) or (right and xm>mid_x)) and (x2-x1)<shape[1]/4 and bb['id'] in annotations['transcriptions']:
                        good_bbs.append(bb)
                        if left and x2>right_boundary:
                            right_boundary=x2
                        elif right and x1<left_boundary:
                            left_boundary=x1
                    else:
                        other_bbs.append(bb)
                if left_boundary>=right_boundary:
                    #print('error')
                    return (0,0,shape[1],shape[0]), None

                #define crop based on these "good" bbs
                crop = (left_boundary,0,right_boundary,shape[0])
                mid_x = right_boundary if left else left_boundary

                #everything else gets cut (if atleast 1/3 of it in on this side)
                for bb in other_bbs:
                    points = bb['poly_points']
                    x1 = min(p[0] for p in points)
                    x2 = max(p[0] for p in points)
                    if (left and x1<mid_x) or (right and x2>mid_x):
                        inside = min(x2,crop[2])-max(x1,crop[0])
                        ratio = inside/(x2-x1)
                        #try:
                        #    print('{} : {}'.format(annotations['transcriptions'][bb['id']],ratio))
                        #except KeyError:
                        #    print('[{}] : {}'.format(bb['id'],ratio))
                        if ratio>0.33:
                            new_points = [[max(crop[0],min(crop[2],x)),y] for x,y in points]
                            bb['poly_points'] = new_points
                            good_bbs.append(bb)

                #redo bbs
                new_field_bbs = []
                new_text_bbs = []
                new_ids = []
                for bb in good_bbs:
                    bb_id = bb['id']
                    new_ids.append(bb_id)
                    if bb_id.startswith('f'):
                        new_field_bbs.append(bb)
                    else:
                        new_text_bbs.append(bb)
                #redo linking
                new_pairs = []
                for (p1,p2) in annotations['pairs']:
                    if p1 in new_ids and p2 in new_ids:
                        new_pairs.append([p1,p2])
                new_same_pairs = []
                for (p1,p2) in annotations['samePairs']:
                    if p1 in new_ids and p2 in new_ids:
                        new_same_pairs.append([p1,p2])

                annotations['fieldBBs']=new_field_bbs
                annotations['textBBs']=new_text_bbs
                annotations['pairs']=new_pairs
                annotations['samePairs']=new_same_pairs
                cropAnnotations(annotations,crop)
                return crop, None

                
            #just crop in to the bbs
            crop = (max(0,min_x-pad),0,min(shape[1],max_x+pad),shape[0])
            cropAnnotations(annotations,crop)
            return crop,None



        #no crop
        return (0,0,shape[1],shape[1]),None

#adjust all points based on crop
def cropAnnotations(annotations,crop):
    for bb in annotations['fieldBBs']+annotations['textBBs']:
        points = bb['poly_points']
        new_points = [[x-crop[0],y-crop[1]] for x,y in points]
        bb['poly_points'] = new_points

##################################

