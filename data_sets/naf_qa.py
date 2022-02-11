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
from utils.read_order import getVertReadPosition,getHorzReadPosition,putInReadOrder,sortReadOrder, intersection

from utils import img_f


SKIP=['174']#['193','194','197','200']

class NAFQA(FormQA):
    """
    Class for reading forms dataset and preping for FormQA format
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(NAFQA, self).__init__(dirPath,split,config,images)
        
        cant_do = ['cell','row-header', 'col-header','full-all-row', 'full-all-col','all-row', 'all-col']
        for cant in cant_do:
            if cant in self.q_types:
                del self.q_types[cant]
            if cant in self.q_types_no_table:
                del self.q_types_no_table[cant]
            if cant in self.q_types_only_table:
                del self.q_types_only_table[cant]

        self.min_start_read = 7
        self.cased=True

        self.extra_np = 0.05

        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                split_file = 'overfit_split.json'
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
            
            for groupName in group_names:
                imageNames=groups_to_use[groupName]
                
                if groupName in SKIP:
                    print('Skipped group {}'.format(groupName))
                    continue
                for imageName in imageNames:
                    ###DEBUG(')
                    imageName = '005025348_00007.jpg'

                    org_path = os.path.join(dirPath,'groups',groupName,imageName)
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName)
                    else:
                        path = org_path
                    jsonPath = org_path[:org_path.rfind('.')]+'.json'
                    if os.path.exists(jsonPath):
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
                            self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')]})
                        else:
                            assert self.rescale_range[1]==self.rescale_range[0]
                            assert self.questions==1
                            #create questions for each image
                            with open(jsonPath) as f:
                                annotations = json.load(f)
                            all_entities,entity_link,table,proses,minored_fields,bbs,link_dict = self.getEntitiesAndSuch(annotations,rescale)
                            qa = self.makeQuestions(self.rescale_range[1],all_entities,entity_link,tables,full_entities,link_dict,proses=proses,minored_fields=minored_fields)
                            #import pdb;pdb.set_trace()
                            for _qa in qa:
                                _qa['bb_ids']=None
                                self.images.append({'id':image_name, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':image_name[:image_name.rfind('.')], 'qa':[_qa]})


    def parseAnn(self,annotations,s):

        all_entities,entity_link,tables,proses,minored_fields,bbs,link_dict = self.getEntitiesAndSuch(annotations,s)


        if self.train:
            qa = self.makeQuestions(s,all_entities,entity_link,tables,all_entities,link_dict,proses=proses,minored_fields=minored_fields)
        else:
            qa = None #This is pre-computed

        ocr=None
        #ocr = [self.corrupt(text) for text in ocr]
        return bbs, list(range(bbs.shape[0])), ocr, {}, {}, qa


    def convertBB(self,s,box):
        #assert s==1
        if isinstance(box,list):
            return [v*s for v in box]
        return (s*box).tolist()

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


    def getEntitiesAndSuch(self,annotations,image_scale):
        fixAnnotations(None,annotations) #fixes somethings

        all_bbs = annotations['byId']#annotations['fieldBBs']+annotations['textBBs']
        #all_bbs = {bb['id']:bb for bb in all_bbs}
        for bb in all_bbs.values():
            bb['poly_points'] = np.array(bb['poly_points'])
        all_pairs = annotations['pairs']#+annotations['samePairs']
        transcriptions = annotations['transcriptions']
        
        #group paragraphs / multiline
        #I think this is whenever they have the same type
        #also para
        groups = defaultdict(list)
        group_num=0
        id_to_group = defaultdict(lambda: None)
        same_row = []
        for id1,id2 in all_pairs:
            if all_bbs[id1]['type']==all_bbs[id2]['type'] or ('P' in all_bbs[id1]['type'] and 'P' in all_bbs[id2]['type']):
                if 'Row' in all_bbs[id1]['type'] and 'Row' in all_bbs[id2]['type']:
                    same_row.append((id1,id2))
                else:
                    assert 'Row' not in all_bbs[id1]['type'] and 'Row' not in all_bbs[id2]['type']
                    assert 'Col' not in all_bbs[id1]['type'] and 'Col' not in all_bbs[id2]['type']
                    group1 = id_to_group[id1]
                    group2 = id_to_group[id2]
                    if group1 is None and group2 is None:
                        groups[group_num] = [id1,id2]#putInReadOrder(id1,all_bbs[id1]['poly_points'],id2,all_bbs[id2]['poly_points'])
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
                        #merge
                        for idx in groups[group2]:
                            id_to_group[idx] = group1
                        #append in read order
                        #ordered_combine_groups = sortReadOrder([(bb_id,all_bbs[bb_id]['poly_points']) for bb_id in groups[group2]+groups[group1]])
                        #groups[group1]=ordered_combine_groups
                        groups[group1]+=groups[group2]
                        del groups[group2]
                    #else nothing needed

        #put all unsed bbs in their own groups
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


        for group in groups.values():
            group = sortReadOrder([(bb_id,all_bbs[bb_id]['poly_points']) for bb_id in group])
            if all(all_bbs[group[0]]['type'] == all_bbs[bb_id]['type'] for bb_id in group[1:]):
                #assert 'Row' not in all_bbs[id1]['type'] and 'Row' not in all_bbs[id2]['type']
                #assert 'Col' not in all_bbs[id1]['type'] and 'Col' not in all_bbs[id2]['type']
                #same class
                cls = self.typeToClass(all_bbs[group[0]]['type'])
                lines=[]
                for bb_id in group:
                    if bb_id not in transcriptions and 'Circle' in all_bbs[group[0]]['type']:
                        transcriptions[bb_id]='UNTRANSCRIBED CIRCLE'
                    elif all_bbs[bb_id]['isBlank']=='blank':
                        continue
                    lines.append(Line(transcriptions[bb_id],all_bbs[bb_id]['poly_points']))
                    bb_to_e[bb_id]=len(entities)

                if len(lines)>0:
                    entities.append(Entity(cls,lines))
                    e_groups.append(group)
            else:
                #should be para
                assert all('P' in all_bbs[bb_id]['type'] for bb_id in group)
                p_entities = []
                p_group = []
                lines = []
                prev_type = all_bbs[group[0]]['type']
                prev_id = None
                cls = 'question' if 'text' in prev_type else 'answer'
                for bb_id in group:
                    if all_bbs[bb_id]['isBlank']=='blank':
                        transcriptions[bb_id]=self.blank_token
                    l = Line(transcriptions[bb_id],all_bbs[bb_id]['poly_points'  ])
                    if all_bbs[bb_id]['type']==prev_type:
                        lines.append(l)
                        p_group.append(bb_id)
                    else:
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
                p_entities.append(Entity(cls,lines))
                if prev_id is not None:
                    entity_link[prev_id].append(len(entities))
                    has_link.add(len(entities))
                for a_id in p_group:
                    bb_to_e[a_id]=len(entities)
                entities.append(p_entities[-1])
                e_groups.append(p_group)
                
                if len(p_entities)>1:
                    for bb_id in group:
                        bb_to_prose[bb_id]=len(proses)
                    proses.append(FillInProse(p_entities))
                    prose_groups.append(group)
        
            used_ids.update(group)

        unused_ids = set(all_bbs.keys())-used_ids
        for bb_id in unused_ids:
            bb= all_bbs[bb_id]
            if 'Row' not in bb['type'] and 'Col' not in bb['type']:
                if transcriptions[bb_id]=='':
                    assert bb['isBlank']=='blank'
                    #for id1,id2 in all_pairs:
                    #    if id1==bb_id:
                    #        annotations['isBlankQuestion'].append(id2)
                    #    elif id2==bb_id:
                    #        annotations['isBlankQuestion'].append(id1)
                else:
                    cls = self.typeToClass(bb['type'])

                    bb_to_e[bb_id]=len(e_groups)
                    entities.append(Entity(cls,[Line(transcriptions[bb_id],bb['poly_points'])]))
                    e_groups.append([bb_id])


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

        table_headers=[]
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
                            if id1==bb['id'] and 'field' not in all_pairs[id2]['type']:
                                dont.append(rid1)
                                got_it=True
                                break
                            elif id2==bb['id'] and 'field' not in all_pairs[id1]['type']:
                                dont.append(rid1)
                                got_it=True
                                break
                        if not got_it:
                            dont.append(rid2)
                        break
            
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
                    if len(links)>1:
                        assert len(links)==2
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
                        linked=links[0] if len(links)>0 else None

                    rows.append((pos,bb,linked))
                else:
                    pos = getHorzReadPosition(bb['poly_points'])
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
                if header_id is not None:
                    e_id = bb_to_e[header_id]
                    has_link.add(e_id)
                    table_headers.append(e_id)
                    row_headers.append(entities[e_id])
                else:
                    row_headers.append(None)
            col_headers=[]
            for pos,line,header_id in cols:
                if header_id is not None:
                    e_id = bb_to_e[header_id]
                    has_link.add(e_id)
                    table_headers.append(e_id)
                    col_headers.append(entities[e_id])
                else:
                    col_headers.append(None)

            if all(r is None for r in row_headers):
                row_headers=[] #becuse we don't have transcription
            table = Table(row_headers,col_headers)
            table.cells=[] #we didn't transcribe tables, WHOOPS

            tables.append(table)

            #for ri,row in enumerate(rows):
            #    row_bb = row[1]
            #    for ci,col in enumerate(cols):
            #        col_bb = col[1]

        
        minor_groups = [] #to keep track of minor labels. We'll make special objects for handeling them
        
        
        for id1,id2 in all_pairs:
            #if id1 not in bb_to_e or id2 not in bb_to_e:
            #    continue
            #if id2=='f16':
            #    import pdb;pdb.set_trace()

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
                #if 'P' in type1 and id1 in bb_to_prose:
                #    if type2 == 'comment' or type2==:
                #        continue
                #    print('Para linked to something else...')
                #    import pdb; pdb.set_trace()
                #elif 'P' in type2 and id2 in bb_to_prose:
                #    if type1 == 'comment':
                #        continue
                #    print('Para linked to something else...')
                #    import pdb; pdb.set_trace()
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
                if 'question'==e1_cls and ('answer'==e2_cls or 'circle'==e2_cls):
                    entity_link[e_id1].append( e_id2)
                    has_link.add(e_id1)
                elif 'question'==e2_cls and ('answer'==e1_cls or 'circle'==e1_cls):
                    entity_link[e_id2].append( e_id1)
                    has_link.add(e_id2)
                elif 'Minor' in type1 and e_id1 in table_headers:
                    entity_link[e_id2].append( e_id1)
                    has_link.add(e_id2)
                elif 'Minor' in type2 and e_id2 in table_headers:
                    entity_link[e_id1].append( e_id2)
                    has_link.add(e_id1)
                else:
                    print(f'WARNING unhandled class pairing (skipping): {e1.cls}:{type1}<->{e2.cls}:{type2}')

                if ('Minor' in type1 or 'Minor' in type2) and e_id1 not in table_headers and e_id2 not in table_headers:
                    mg1=mg2=None
                    for mi,group in enumerate(minor_groups):
                        if e_id1 in group:
                            mg1 = mi
                        if e_id2 in group:
                            mg2 = mi
                        if mg1 is not None and mg2 is not None:
                            break
                        
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
            new_minor_groups.append(new_group)

        #merge minor groups
        new_gi=0
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



        #now create objects
        minored_fields=[]
        for new_group in new_minor_groups:

            #we expect one question(text) multiple answers and multiple minor
            question = None
            answers = []
            minors = []
            for e_id in new_group:
                if isinstance(e_id,str):
                    continue #a blank
                typ = all_bbs[e_groups[e_id][0]]['type']
                if 'Minor' in typ:
                    minors.append(entities[e_id])
                elif 'field' in typ:
                    answers.append(entities[e_id])
                else:
                    assert 'text' in typ
                    assert question is None
                    question = entities[e_id]

            if question is not None or (len(answers)>0 or len(minors)>0):
                answers = sortReadOrder([(ans,ans.lines[0].box) for ans in answers])
                minors = sortReadOrder([(minor,minor.lines[0].box) for minor in minors])
                minored_fields.append(MinoredField(question,answers,minors))


        #update has_link with questions with blank answers
        #for bb_id in annotations['isBlankQuestion']:
        #    try:
        #        e_id = bb_to_e[bb_id]
        #        has_link.add(e_id)
        #    except KeyError:
        #        pass


        #now, relying on has_link we assign "questions" with no answer to the proper class of other
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

        #return entities,entity_link,tables,proses
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
        #actually prose will already have links in entity_link
        #for prose in proses:
        #    prev_idx = all_entities.index(prose.entities[0])
        #    for entity in prose.entities[1:]:
        #        this_idx = all_entities.index(entity)
        #        link_dict[prev_idx] = this_idx
        #        link_dict[this_idx] = prev_idx
                

        link_dict = self.sortLinkDict(entities,link_dict)
        return entities,entity_link,tables,proses,minored_fields,bbs,link_dict

##################################

