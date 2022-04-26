import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines, fixFUNSD
import timeit
from data_sets.form_qa import FormQA,collate, Line, Entity, Table

from utils import img_f



class FUNSDQA(FormQA):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FUNSDQA, self).__init__(dirPath,split,config,images)


        self.classMap={
                'header':16,
                'question':17,
                'answer': 18,
                'other': 19
                }
        self.index_class_map=[
                'header',
                'question',
                'answer',
                'other'
                ]

        self.min_start_read = 7

        self.extra_np = 0.05

        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                split_file = 'overfit_split.json'
            else:
                split_file = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,split_file)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                read_file = json.loads(f.read())
                if type(split) is str:
                    to_use = read_file[split]
                    imagesAndAnn = []
                    imageDir = os.path.join(dirPath,to_use['root'],'images')
                    annDir = os.path.join(dirPath,to_use['root'],'annotations')
                    for name in to_use['images']:
                        imagesAndAnn.append( (name+'.png',os.path.join(imageDir,name+'.png'),os.path.join(annDir,name+'.json')) )
                elif type(split) is list:
                    imagesAndAnn = []
                    for spstr in split:
                        to_use = read_file[spstr]
                        imageDir = os.path.join(dirPath,to_use['root'],'images')
                        annDir = os.path.join(dirPath,to_use['root'],'annotations')
                        for name in to_use['images']:
                            imagesAndAnn.append( 
                                (name+'.png',os.path.join(imageDir,name+'.png'),
                                os.path.join(annDir,name+'.json')) 
                                )
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
            self.images=[]
            for image_name,imagePath,jsonPath in imagesAndAnn:
                ##DEBUG
                #if '0011973451' not in image_name:
                #    continue
                #if len(self.images)==0 and '01073843' not in image_name:
                #    continue
                ##DEBUG

                org_path = imagePath
                if self.cache_resized:
                    path = os.path.join(self.cache_path,image_name)
                else:
                    path = org_path
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
                        self.images.append({'id':image_name, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':image_name[:image_name.rfind('.')]})
                    else:
                        assert self.rescale_range[1]==self.rescale_range[0]
                        assert self.questions==1
                        #create questions for each image
                        with open(jsonPath) as f:
                            annotations = json.load(f)
                        s=1
                        bbs, numNeighbors, trans, groups = createLines(annotations,self.classMap,s)
                        bbs = bbs[0]

                        groups_adj=set()
                        for group_id,entity in enumerate(annotations['form']):
                            for linkIds in entity['linking']:
                                groups_adj.add((min(linkIds),max(linkIds)))
                                
                        entities,entity_link,tables = self.prepareForm(bbs,trans,groups,groups_adj)
                        full_entities,full_entity_link_dict = self.prepareFormRaw(bbs,trans,groups,groups_adj)
                        qa = self.makeQuestions(self.rescale_range[1],entities,entity_link,tables,full_entities,full_entity_link_dict)
                        #import pdb;pdb.set_trace()
                        for _qa in qa:
                            _qa['bb_ids']=None
                            self.images.append({'id':image_name, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':image_name[:image_name.rfind('.')], 'qa':[_qa]})
        self.errors=[]



    def parseAnn(self,annotations,s):
        #if useBlankClass:
        #    numClasses+=1
        #if usePairedClass:
        #    numClasses+=1

        fixFUNSD(annotations)

        numClasses=len(self.classMap)
        #if self.split_to_lines:
        bbs, numNeighbors, trans, groups = createLines(annotations,self.classMap,s)
        bbs = bbs[0]

        word_boxes=[]
        word_trans=[]
        word_groups=[]
        groups_adj=set()
        for group_id,entity in enumerate(annotations['form']):
            for linkIds in entity['linking']:
                groups_adj.add((min(linkIds),max(linkIds)))
                

        ocr = trans

        entities,entity_link,tables = self.prepareForm(bbs,trans,groups,groups_adj)
        full_entities,full_entity_link_dict = self.prepareFormRaw(bbs,trans,groups,groups_adj)

        #run through all entites to build bbs, assign bbid, and find ambiguity
        boxes = []
        text_line_counts = defaultdict(list)
        for ei,entity in enumerate(entities):
            for li,line in enumerate(entity.lines):
                text = self.punc_regex.sub('',line.text.lower())
                text_line_counts[text].append((ei,li))
                bbid = len(boxes)
                boxes.append(self.convertBB(1,line.box))
                line.bbid = bbid

        for ei,entity in enumerate(full_entities):
            for li,line in enumerate(entity.lines):
                #text = self.punc_regex.sub('',line.text.lower())
                #text_line_counts[text].append((ei,li))
                bbid = len(boxes)
                boxes.append(self.convertBB(1,line.box))
                line.bbid = bbid

        bbs = np.array(boxes)

        #assign ambiguity
        for line_ids in text_line_counts.values():
            if len(line_ids)>1:
                for ei,li in line_ids:
                    entities[ei].lines[li].ambiguous = True
        if self.train:
            qa = self.makeQuestions(1.0,entities,entity_link,tables,full_entities,full_entity_link_dict)
        else:
            qa = None #This is pre-computed

        return bbs, list(range(bbs.shape[0])), None,None,qa


    def convertBB(self,s,box):
        assert s==1
        if isinstance(box,list):
            return box
        return box.tolist()

    def prepareForm(self,bbs,transcription,groups,groups_adj):

        entities=[]


        for gi,group in enumerate(groups):
            cls = bbs[group[0],16:].argmax()
            cls_name = self.index_class_map[cls]
            lines = []
            for bbid in group:
                #box = calcXYWH(bbs[bbid,:8])[:4]
                if transcription[bbid]!='':
                    lines.append(Line(transcription[bbid],bbs[bbid,:16]))
            if len(lines)>0:
                entities.append(Entity(cls_name,lines))
                #if entities[-1].text=='$250':
                #    BBB=len(entities)-1
                #    print('BBB {}'.format(BBB))
            else:
                entities.append(None)
                #removed.add(gi)


        #import pdb;pdb.set_trace()
        entities,groups_adj = cleanUp(entities,groups_adj)
        group_count = len(entities)

        #for i,entity in enumerate(entities):
        #    if entity.text=='Dollar Cost':
        #        DDD=i


        questions_gs=set()
        answers_gs=set()
        headers_gs=set()
        others_gs=set()
        #removed=set()
        for gi,entity in enumerate(entities):
            cls_name = entity.cls
            if cls_name == 'question':
                questions_gs.add(gi)
            elif cls_name == 'answer':
                answers_gs.add(gi)
            elif cls_name == 'header':
                headers_gs.add(gi)
            else:
                others_gs.add(gi)


        relationships_h_q=defaultdict(list)
        relationships_h_h=defaultdict(list)
        for h_gi in headers_gs:
            for gi1,gi2 in groups_adj:
                if gi1 == h_gi and gi2 in questions_gs:
                    relationships_h_q[h_gi].append(gi2)
                elif gi2 == h_gi and gi1 in questions_gs:
                    relationships_h_q[h_gi].append(gi1)
                elif gi1 == h_gi and  gi2 in headers_gs:
                    relationships_h_h[h_gi].append(gi2)
                elif gi2 == h_gi and  gi1 in headers_gs:
                    relationships_h_h[h_gi].append(gi1)

        relationships_q_a=defaultdict(list)
        relationships_a_q=defaultdict(list)
        for q_gi in questions_gs:
            found=False
            for gi1,gi2 in groups_adj:
                if gi1 == q_gi and gi2 in answers_gs:
                    #q_a_pairs.append((gi1,gi2))
                    assert entities[q_gi].cls=='question'
                    relationships_q_a[q_gi].append(gi2)
                    relationships_a_q[gi2].append(q_gi)
                    found=True
                elif gi2 == q_gi and gi1 in answers_gs:
                    #q_a_pairs.append((gi2,gi1))
                    assert entities[q_gi].cls=='question'
                    relationships_q_a[q_gi].append(gi1)
                    relationships_a_q[gi1].append(q_gi)
                    found=True
            if not found:
                #q_a_pairs.append((q_gi,None))
                assert entities[q_gi].cls=='question'
                relationships_q_a[q_gi].append(None)

        for a_gi in answers_gs:
            if a_gi not in relationships_a_q:
                relationships_a_q[a_gi].append(None)



        q_a_pairs=defaultdict(list)
        #table_map = {}
        #tables=[]
        table_values=defaultdict(set)
        row_headers=set()
        col_headers=set()
        skip=set()
        
        merged_groups = {}
        for qi,ais in relationships_q_a.items():
            #if DDD in ais:
            #    import pdb;pdb.set_trace()
            if qi in skip:
                continue
            if len(ais)==1:
                ai=ais[0]
                if ai is None or len(relationships_a_q[ai])==1:
                    assert entities[qi].cls=='question'
                    q_a_pairs[qi].append(ai)
                else:
                    #this is probably part of a table with single row/col
                    if len(relationships_a_q[ai])>2:
                        continue #too many question links, label error
                    q1,q2 = relationships_a_q[ai]
                    if q1==qi:
                        other_qi = q2
                    else:
                        other_qi = q1
                    success = addTableElement(table_values,row_headers,col_headers,ai,q1,q2,entities)
                    #if len(relationships_q_a[other_qi])>1: #be sure this is the non-single index
                    #    #assert ai not in table_map
                    #    #addTable(tables,table_map,goups,bbs,qi,ais,relationships_q_a,relationships_a_q)
                    #    addTableElement(table_values,row_headers,col_headers,ai,q1,q2,groups,bbs)
                    if not success and len(relationships_q_a[other_qi])==1:
                        #broken label?
                        gi = group_count
                        group_count+=1
                        bb_ids = groups[qi]+groups[other_qi]
                        groups = groups+[bb_ids]

                        entity = entities[qi]
                        other_entity = entities[other_qi]
                        #which is first?
                        if entity.box[1]<other_entity.box[1]:
                            first_entity = entity
                            second_entity = other_entity
                        else:
                            first_entity = other_entity
                            second_entity = entity
                        new_entity = Entity(entities[qi].cls,first_entity.lines+second_entity.lines)
                        entities.append(new_entity)
                        entities[qi]=None
                        entities[other_qi]=None

                        assert entities[gi].cls=='question'
                        q_a_pairs[gi].append(ai)
                        skip.add(other_qi)
                        merged_groups[qi]=gi
                        merged_groups[other_qi]=gi
            else:
                #is this a misslabled multiline answer or a table?
                if all(len(relationships_a_q[ai])==1 for ai in ais):
                    #just put as multiple answers
                    #may be incorrectly labeled multiline answer...
                    answers = []
                    for ai in ais:
                        #assert len(groups[ai])==1 this can be a list, in which  case we lose new lines
                        #bbi = groups[ai][0]
                        #answers.append((bbs[bbi,1],ai))
                        answers.append((entities[ai].box[9],ai))
                    answers.sort(key=lambda a:a[0])
                    answers = [a[1] for a in answers]
                    assert entities[qi].cls=='question'
                    q_a_pairs[qi]+=answers
                else:
                    #assert qi not in table_map
                    #addTable(tables,table_map,groups,bbs,qi,ais,relationships_q_a,relationships_a_q)
                    for ai in ais:
                        if len(relationships_a_q[ai])==2:
                            q1,q2 = relationships_a_q[ai]
                        elif len(relationships_a_q[ai])==1:
                            q1 = relationships_a_q[ai][0]
                            q2 = None
                        addTableElement(table_values,row_headers,col_headers,ai,q1,q2,entities)

        entity_link=[]
        new_entities=[]
        old_to_new_e_map={}
        for oi,entity in enumerate(entities):
            if entity is not None:
                old_to_new_e_map[oi]=len(new_entities)
                new_entities.append(entity)
        
        for oi,gi in merged_groups.items():
            old_to_new_e_map[oi]=old_to_new_e_map[gi]

        for qi,ais in q_a_pairs.items():
            head = old_to_new_e_map[qi]
            tail = [old_to_new_e_map[ai] for ai in ais if ai is not None]
            if len(tail) == 0:
                tail = None
            else:
                assert head not in tail
                assert all(new_entities[t]!='header' for t in tail)
            assert new_entities[head].cls!='answer'
            entity_link.append((head,tail))





        #Do headers

        #first sort out header->subheader relationships
        possible_sub_rel={}
        for h1,h2s in relationships_h_h.items():
            if len(h2s)>1:
                #should be main header
                assert h1 not in relationships_h_q
                relationships_h_q[h1]=h2s
            else:
                possible_sub_rel[h1]=h2s
        for h1,h2s in possible_sub_rel.items():
            if h2s[0] not in relationships_h_q:
                relationships_h_q[h1]=h2s

        for hi,qis in relationships_h_q.items():
            hi = old_to_new_e_map[hi]
            qis = [old_to_new_e_map[qi] for qi in qis]

            pos_qs=[]
            for qi in qis:
                if len(qis)>1:
                    x=y=h=0
                    #for bbi in groups[qi]:
                    #    x += bbs[bbi,0]
                    #    x += bbs[bbi,4]
                    #    y += bbs[bbi,1]
                    #    y += bbs[bbi,5]
                    #    h += bbs[bbi,5]-bbs[bbi,1]+1
                    #xc = x/(2*len(groups[qi]))
                    #yc = y/(2*len(groups[qi]))
                    #h = h/len(groups[qi])
                    xc = new_entities[qi].box[15]
                    yc = new_entities[qi].box[9]
                    h = 0
                    for line in new_entities[qi].lines:
                        h += line.box[5]-line.box[1]
                    h /= len(new_entities[qi].lines)
                else:
                    xc=yc=h=-1
                pos_qs.append((qi,xc,yc,h))

            #Now we need to put all the questions into read order
            if len(pos_qs)>1:
                rows=[]
                rows_mean_y=[]
                for qi,x,y,h in pos_qs:
                    row_i=None
                    for r,mean_y in enumerate(rows_mean_y):
                        if abs(mean_y-y)<h*0.6:
                            row_i=r
                            break
                    if row_i is None:
                        rows.append([(qi,x,y)])
                        rows_mean_y.append(y)
                    else:
                        rows[row_i].append((qi,x,y))
                        mean_y=0
                        for (qi2,x2,y2) in rows[r]:
                            mean_y+=y2
                        rows_mean_y[row_i] = mean_y/len(rows[row_i])
                ordered_qs = []
                rows = list(zip(rows,rows_mean_y))
                rows.sort(key=lambda a:a[1])
                for row,mean_y in rows:
                    row.sort(key=lambda a:a[1])
                    for qi,x,y in row:
                        ordered_qs.append(qi)
            else:
                ordered_qs = pos_qs[0][0]
            
            entity_link.append((hi,ordered_qs))

        #Aggregate the table information
        col_vs=defaultdict(list)
        row_vs=defaultdict(list)
        tables={}
        header_to_table_id={}
        cur_table_id=0
        new_table_values=[]

        #seperate into different tables
        for (col_h,row_h) in table_values.keys():
            col_h = old_to_new_e_map[col_h] if col_h is not None else None
            row_h = old_to_new_e_map[row_h] if row_h is not None else None
            #v = old_to_new_e_map[v]

            if col_h is not None and col_h in header_to_table_id:
                col_tab = header_to_table_id[col_h]
            else:
                col_tab = None
            if row_h is not None and row_h in header_to_table_id:
                row_tab = header_to_table_id[row_h]
            else:
                row_tab = None


            if col_tab is None and row_tab is None:
                #if col_h is not None and row_h is not None:
                tables[cur_table_id] = [[col_h],[row_h]]
                #elif col_h is not None:
                #    tables[cur_table_id] = [[col_h],[]]
                #elif row_h is not None:
                #    tables[cur_table_id] = [[],[row_h]]
                #else:
                #    assert False
                    #tables[cur_table_id] = [[],[]]
                if col_h is not None:
                    header_to_table_id[col_h]=cur_table_id
                if row_h is not None:
                    header_to_table_id[row_h]=cur_table_id
                tab_id = cur_table_id
                cur_table_id+=1
            elif col_tab is None:
                tables[row_tab][0].append(col_h)
                tab_id = row_tab
                if col_h is not None:
                    header_to_table_id[col_h]=row_tab
            elif row_tab is None:
                tables[col_tab][1].append(row_h)
                tab_id = col_tab
                if row_h is not None:
                    header_to_table_id[row_h]=col_tab
            else:
                if col_tab!=row_tab:
                    #merge tables
                    tables[col_tab][0]+=tables[row_tab][0]
                    tables[col_tab][1]+=tables[row_tab][1]
                    for h in tables[row_tab][0]+tables[row_tab][1]:
                        header_to_table_id[h]=col_tab
                    del tables[row_tab]
                    tab_id = col_tab

        #Rebuild table_values with table ids
        blank_col = defaultdict(lambda:defaultdict(list))
        blank_row = defaultdict(lambda:defaultdict(list))
        for (col_h,row_h),vs in table_values.items():
            col_h = old_to_new_e_map[col_h] if col_h is not None else None
            row_h = old_to_new_e_map[row_h] if row_h is not None else None
            vs = [old_to_new_e_map[v] for v in vs]
            
            if col_h is not None:
                tab_id = header_to_table_id[col_h]
                if row_h is not None:
                    assert tab_id == header_to_table_id[row_h]
            else:
                tab_id = header_to_table_id[row_h]
            
            for v in vs:
                new_table_values.append((tab_id,col_h,row_h,v))
            if col_h is None:
                blank_col[tab_id][row_h]+=vs
            elif row_h is None:
                blank_row[tab_id][col_h]+=vs
            #x,y = bbs[groups[v][0],0:2]
            #if col_h is not None:
            #    col_vs[col_h].append((v,y))
            #if row_h is not None:
            #    row_vs[row_h].append((v,x))

        real_tables=[]
        all_col_entity_ids=[]
        all_row_entity_ids=[]
        tab_id_to_pos={}
        blank_col_pos={}
        blank_row_pos={}
        for tab_id,(col_hs,row_hs) in tables.items():

            #find possible conflict (header of row and column)
            bad = set(col_hs).intersection(set(row_hs))
            if len(bad)>0:
                print('Found overlapped headers {}'.format(bad))
            for bad_header in bad:
                if bad_header is not None:
                    #determin if col or row has most votes
                    votes_col=0
                    votes_row=0
                    for tab_id,col_h,row_h,v in new_table_values:
                        if bad_header==col_h:
                            votes_col+=1
                        if bad_header==row_h:
                            votes_row+=1
                    if votes_col>votes_row:
                        row_hs.remove(bad_header)
                    elif votes_row>votes_col:
                        col_hs.remove(bad_header)
                    else:
                        assert False #what?

            ##Get columns sorted

            #first make a list with the headers and their x m=position
            pos_col_hs = [(new_entities[ch].box[0],ch) for ch in col_hs if ch is not None]
            centerx_col_headers = [((new_entities[ch].box[0]+new_entities[ch].box[4])/2,ch) for ch in col_hs if ch is not None]
            if None in col_hs:
                #We also need to find where the columns without headers go
                #Also, often cells may be mislabeled, so we should check if they have a good header candidiate
                #how many blank columns are there?
                #We will seperate cells with no header by their x position
                #first compute the average width
                sum_len=0
                count=0
                for row_h,values in blank_col[tab_id].items():
                    for value in values:
                        cell = new_entities[value]
                        sum_len += cell.box[4]-cell.box[0]
                        count+=1

                avg_len = sum_len/count

                #Now seperate them into columns, assuming if they are more than the
                #average width, they are a different column
                columns_xs=[]
                blank_cols_items=[]
                for row_h,values in blank_col[tab_id].items():
                    for value in values:
                        cell = new_entities[value]
                        x_center = (cell.box[0]+cell.box[4])/2
                        matched_col_id=None
                        #for each no-header column
                        for col_id,col in enumerate(columns_xs):
                            if abs(x_center-np.mean(col))<avg_len:
                                #I belong in this column
                                col.append(x_center)
                                blank_cols_items[col_id].append(value)
                                matched_col_id=col_id
                                break
                        if matched_col_id is None:
                            #I belong in a new column
                            blank_cols_items.append([value])
                            columns_xs.append([x_center])


                for blank_col_items,positions in zip(blank_cols_items,columns_xs):
                    #Now add these columns to list with positions, we use the column cell ids instead of the header id
                    x_center = np.mean(positions)
                    add_to=None
                    for h_x_center,col_h in centerx_col_headers:
                        if abs(x_center - h_x_center) < avg_len*0.6:
                            #add to this column instead
                            add_to = col_h
                            break

                    if add_to is None:
                        pos_col_hs.append((x_center,blank_col_items))
                    else:
                        #we need to update the new_table_values entries
                        for v in blank_col_items:
                            added=False
                            for i in range(len(new_table_values)):
                                tab_id2,col_h,row_h,v2 = new_table_values[i]
                                if v==v2:
                                    assert col_h is None
                                    new_table_values[i] = (tab_id,add_to,row_h,v)
                                    added=True
                                    break
                            assert added

            pos_col_hs.sort(key=lambda x:x[0])
            col_entities = []#ordered list used to create Table object
            these_col_entity_ids = []#for looking up position of cells
            for pos,(x,ch) in enumerate(pos_col_hs):
                if isinstance(ch,int):
                    #normal header
                    col_entities.append(new_entities[ch])
                    these_col_entity_ids.append(ch)
                else:
                    #empty header
                    col_entities.append(None)
                    these_col_entity_ids.append(None)
                    for v in ch:
                        blank_col_pos[v]=pos #as these can't use the col_entity_ids to lookup position
            all_col_entity_ids.append(these_col_entity_ids)


            ##Get rows sorted
            #same as col
            pos_row_hs = [(new_entities[rh].box[1],rh) for rh in row_hs if rh is not None]
            centery_row_headers = [((new_entities[rh].box[1]+new_entities[rh].box[5])/2,rh) for rh in row_hs if rh is not None]
            if None in row_hs:
                #how many blank rows are there?
                sum_height=0
                count=0
                for row_h,values in blank_row[tab_id].items():
                    for value in values:
                        cell = new_entities[value]
                        sum_height += cell.box[5]-cell.box[1]
                        count+=1

                avg_height = sum_height/count

                rows_ys=[]
                blank_rows_items=[]
                for row_h,values in blank_row[tab_id].items():
                    for value in values:
                        cell = new_entities[value]
                        y_center = (cell.box[1]+cell.box[5])/2
                        matched_row_id=None
                        for row_id,row in enumerate(rows_ys):
                            if abs(y_center-np.mean(row))<avg_height:
                                row.append(y_center)
                                blank_rows_items[row_id].append(value)
                                matched_row_id=row_id
                                break
                        if matched_row_id is None:
                            blank_rows_items.append([value])
                            rows_ys.append([y_center])


                for blank_row_items,positions in zip(blank_rows_items,rows_ys):
                    y_center = np.mean(positions)

                    add_to=None
                    for h_y_center,row_h in centery_row_headers:
                        if abs(y_center - h_y_center) < avg_height*0.6:
                            #add to this row instead
                            add_to = row_h
                            break

                    if add_to is None:
                        pos_row_hs.append((y_center,blank_row_items))
                    else:
                        #we need to update the new_table_values entries
                        for v in blank_row_items:
                            added=False
                            for i in range(len(new_table_values)):
                                tab_id2,col_h,row_h,v2 = new_table_values[i]
                                if v==v2:
                                    assert row_h is None
                                    new_table_values[i] = (tab_id,col_h,add_to,v)
                                    added=True
                                    break
                            assert added


            pos_row_hs.sort(key=lambda y:y[0])
            row_entities = []
            these_row_entity_ids = []
            for pos,(x,rh) in enumerate(pos_row_hs):
                if isinstance(rh,int):
                    row_entities.append(new_entities[rh])
                    these_row_entity_ids.append(rh)
                else:
                    row_entities.append(None)
                    these_row_entity_ids.append(None)
                    for v in rh:
                        blank_row_pos[v]=pos
            all_row_entity_ids.append(these_row_entity_ids)
            #row_entities = [new_entities[rh[1]] for rh in pos_row_hs]
            #all_row_entity_ids.append( [rh[1] for rh in pos_row_hs] )

            tab_id_to_pos[tab_id] = len(real_tables)
            real_tables.append(Table(row_entities,col_entities))

        double_assigned=defaultdict(list)
        reverse_row_index=defaultdict(set)
        reverse_col_index=defaultdict(set)
        for tab_id, col_h, row_h, v in new_table_values:
            tab_pos = tab_id_to_pos[tab_id]
            #Lookup poition of v in Table
            try:
                r = all_row_entity_ids[tab_pos].index(row_h) if row_h is not None else blank_row_pos[v]
                c = all_col_entity_ids[tab_pos].index(col_h) if col_h is not None else blank_col_pos[v]
                reverse_row_index[r].add(row_h)
                reverse_col_index[c].add(col_h)
                if real_tables[tab_pos].cells[r][c] is None: #shouldn't write twice
                    real_tables[tab_pos].cells[r][c]=new_entities[v]
                else:
                    if len(double_assigned[(tab_pos,r,c)])==0:
                        double_assigned[(tab_pos,r,c)].append(real_tables[tab_pos].cells[r][c])
                    double_assigned[(tab_pos,r,c)].append(new_entities[v])

            except ValueError:
                pass

        #resolve double assignment
        row_h_split_candidates=set()
        col_h_split_candidates=set()
        for (tab_pos,r,c),cells in double_assigned.items():
            assert len(cells)==2 #I'm not sure what to do with more...
            assert len(reverse_row_index[r])==1
            assert len(reverse_col_index[c])==1

            row_h = next(iter(reverse_row_index[r]))
            col_h = next(iter(reverse_col_index[c]))

            #are the cells verticle or horizontally stacked?
            min_x=max_x=(cells[0].box[0]+cells[0].box[4])/2
            min_y=max_y=(cells[0].box[1]+cells[0].box[5])/2
            for cell in cells[1:]:
                xc = (cell.box[0]+cell.box[4])/2
                yc = (cell.box[1]+cell.box[5])/2
                min_x = min(min_x,xc)
                max_x = max(max_x,xc)
                min_y = min(min_y,yc)
                max_y = max(max_y,yc)

            y_diff = max_y - min_y
            x_diff = max_x - min_x

            if y_diff>x_diff:
                #row_h_split_candidates[row_h]+=cells
                row_h_split_candidates.add((tab_pos,row_h,r))
            else:
                col_h_split_candidates.add((tab_pos,col_h,c))

            #if '\\' in new_entities[row_h].text:
                #row_h_split_candidates.append((row_h,r))
        
        if len(double_assigned)>0:
            #Can split rows?
            can_split_rows= len(col_h_split_candidates)==0
            if can_split_rows:
                for tab_pos,split_row_h,r in row_h_split_candidates:
                    if len(new_entities[row_h].lines)!=2:
                        can_split_rows=False
                        break

            if can_split_rows:
                #Okay, get those rows split!
                for tab_pos,split_row_h,r in row_h_split_candidates:
                    #if len(new_entities[row_h].lines)==2:
                    #we'll split these lines
                    top_header = Entity(new_entities[split_row_h].cls,new_entities[split_row_h].lines[0:1])
                    top_yc = (top_header.box[1]+top_header.box[5])/2
                    bot_header = Entity(new_entities[split_row_h].cls,new_entities[split_row_h].lines[1:])
                    bot_yc = (bot_header.box[1]+bot_header.box[5])/2

                    real_tables[tab_pos].cells[r]=[None]*len(real_tables[tab_pos].col_headers)
                    real_tables[tab_pos].cells.insert(r+1,[None]*len(real_tables[tab_pos].col_headers))

                    real_tables[tab_pos].row_headers[r]=top_header
                    real_tables[tab_pos].row_headers.insert(r+1,bot_header)

                    new_entities[split_row_h]=top_header
                    bot_row_h = len(new_entities)
                    new_entities.append(bot_header)

                    for i in range(len(new_table_values)):
                        tab_id,col_h,row_h,v = new_table_values[i]
                        if split_row_h == row_h:
                            #which one is it assigned to?
                            yc = (new_entities[v].box[1]+new_entities[v].box[5])/2
                            dist_top = abs(top_yc-yc)
                            dist_bot = abs(bot_yc-yc)
                            c = all_col_entity_ids[tab_pos].index(col_h) if col_h is not None else blank_col_pos[v]
                            if dist_bot<dist_top:
                                new_table_values[i] = (tab_id,col_h,bot_row_h,v)
                                real_tables[tab_pos].cells[r+1][c] = new_entities[v]
                            else:
                                real_tables[tab_pos].cells[r][c] = new_entities[v]
            else:
                #Just double assign each cell 
                newer_to_new_entities=list(range(len(new_entities)))
                for (tab_pos,r,c),cells in double_assigned.items():
                    #read-order the entities
                    cells.sort(key=lambda a:a.box[1])
                    lines=[]
                    cls = cells[0].cls
                    i=0
                    while i<len(cells):
                        assert cells[i].cls==cls
                        j=i+1
                        while j<len(cells) and cells[j].box[1]-cells[i].box[1]<5:
                            j+=1

                        row = cells[i:j]
                        row.sort(key=lambda a:a.box[0])
                        for entity in row:
                            lines+=entity.lines
                        i=j
                    new_e = Entity(cls,lines)
                    real_tables[tab_pos].cells[r][c]=new_e

                    to_remove = []
                    for i,e in enumerate(new_entities):
                        if e in cells:
                            to_remove.append(i)
                    to_remove.sort(reverse=True)
                    for i in to_remove:
                        del new_entities[i]
                        del newer_to_new_entities[i]
                    new_entities.append(new_e)

                new_to_newer = {j:i for i,j in enumerate(newer_to_new_entities)}
                new_entity_link=[]
                for head,tail in entity_link:
                    head = new_to_newer[head]
                    if tail is not None:
                        if isinstance(tail,list):
                            tail = [new_to_newer[t] for t in tail]
                        else:
                            tail = new_to_newer[tail]
                        new_entity_link.append((head,tail))
                entity_link = new_entity_link


                

        return new_entities,entity_link,real_tables

    def prepareFormRaw(self,bbs,transcription,groups,groups_adj):


        entities=[]

        old_to_new={}
        for gi,group in enumerate(groups):
            cls = bbs[group[0],16:].argmax()
            cls_name = self.index_class_map[cls]
            lines = []
            for bbid in group:
                #box = calcXYWH(bbs[bbid,:8])[:4]
                if transcription[bbid]!='':
                    lines.append(Line(transcription[bbid],bbs[bbid,:16]))
            if len(lines)>0:
                entity = Entity(cls_name,lines)
                if len(entity.text)>0:
                    old_to_new[gi]=len(entities)
                    entities.append(entity)



        link_dict = defaultdict(list)
        for e1,e2 in groups_adj:
            if e1 in old_to_new and e2 in old_to_new:
                e1 = old_to_new[e1]
                e2 = old_to_new[e2]
                link_dict[e1].append(e2)
                link_dict[e2].append(e1)


        link_dict=self.sortLinkDict(entities,link_dict)
        return entities,link_dict





def addTableElement(table_values,row_headers,col_headers,ai,qi1,qi2,entities,threshold=5):
    ele_x = entities[ai].box[12]#bbs[groups[ai][0],0:2]
    ele_y = entities[ai].box[9]
    q1_x = entities[qi1].box[12]#bbs[groups[qi1][0],0:2]
    q1_y = entities[qi1].box[9]
    x_diff_1 = abs(ele_x-q1_x)
    y_diff_1 = abs(ele_y-q1_y)
    if qi2 is not None:
        #which question is the row, which is the header?
        #q2_x,q2_y = bbs[groups[qi2][0],0:2]
        q2_x = entities[qi2].box[12]
        q2_y = entities[qi2].box[9]
        x_diff_2 = abs(ele_x-q2_x)
        y_diff_2 = abs(ele_y-q2_y)

        if abs(q1_x-q2_x)<threshold or abs(q1_y-q2_y)<threshold:
            return False

        if (x_diff_1<y_diff_1 or y_diff_2<x_diff_2) and y_diff_1>threshold and x_diff_2>threshold:
            row_h = qi2
            col_h = qi1
        elif (x_diff_2<y_diff_2 or y_diff_1<x_diff_1) and y_diff_2>threshold and x_diff_1>threshold:
            row_h = qi1
            col_h = qi2
        else:
            #IDK
            #import pdb;pdb.set_trace()
            return False
        
        table_values[(col_h,row_h)].add(ai)
        row_headers.add(row_h)
        col_headers.add(col_h)
    else:
        if x_diff_1>y_diff_1:
            row_headers.add(qi1)
            table_values[(None,qi1)].add(ai)
        elif x_diff_1<y_diff_1:
            col_headers.add(qi1)
            table_values[(qi1,None)].add(ai)
    return True


def cleanUp(entities,entity_adj):
    if entity_adj is None:
        return entities,entity_adj
    #This is to check answers with two questions. These should be tables, but there are many
    #errors in the dataset labeling.

    #First change questions that should be headers to correct class
    for i,entity in enumerate(entities):
        if entity is None or len(entity.text.replace(' ',''))==0 or entity.cls!='question':
            continue
        links_to_answers=[]
        links_to_questions=[]
        for a,b in entity_adj:
            if a==i:
                other_i=b
            elif b==i:
                other_i=a
            else:
                other_i=None
            if other_i is not None and entities[other_i] is not None:
                cls = entities[other_i].cls
                if cls=='answer':
                    links_to_answers.append(other_i)
                elif cls=='question':
                    links_to_questions.append(other_i)
        if len(links_to_questions)>1 and len(links_to_answers)==0:
            entity.cls='header'

    entity_i_to_remove=set()
    adj_to_remove=set()
    for i,entity in enumerate(entities):
        if i in entity_i_to_remove:
            continue
        elif entity is None or len(entity.text.replace(' ',''))==0:
            entity_i_to_remove.add(i)
            continue
        links_to_answers=[]
        links_to_questions=[]
        links_to_headers=[]
        for a,b in entity_adj:
            if a==i:
                other_i=b
            elif b==i:
                other_i=a
            else:
                other_i=None
            if other_i is not None and entities[other_i] is not None:
                cls = entities[other_i].cls
                if cls=='answer':
                    links_to_answers.append(other_i)
                elif cls=='question':
                    links_to_questions.append(other_i)
                elif cls=='header':
                    links_to_headers.append(other_i)

        if entity.cls=='answer':
            if len(links_to_answers)>0:
                for other_i in links_to_answers:
                    #print('Error: {} linked to {}'.format(entity,entities[other_i]))
                    adj_to_remove.add((i,other_i))
                    adj_to_remove.add((other_i,i))
                #import pdb;pdb.set_trace()
            if len(links_to_headers)>0:
                for other_i in links_to_headers:
                    #print('{} linked to {}'.format(entity,entities[other_i]))
                    adj_to_remove.add((i,other_i))
                    adj_to_remove.add((other_i,i))
                #import pdb;pdb.set_trace()
            if len(links_to_questions)>2:
                for other_i in links_to_questions:
                    #print('{} linked to {}'.format(entity,entities[other_i]))
                    adj_to_remove.add((i,other_i))
                    adj_to_remove.add((other_i,i))
                #bad and weird table thing
                #import pdb;pdb.set_trace()
            elif len(links_to_questions)==2:
                #check we have a row and col header
                first = entities[links_to_questions[0]].box
                second= entities[links_to_questions[1]].box
                row=False
                col=False
                #x_diff1 = entity.box[12]-entities[links_to_questions[0]].box[12]#center x
                #y_diff1 = entity.box[9]-entities[links_to_questions[0]].box[9]#center y
                #x_diff2 = entity.box[12]-entities[links_to_questions[1]].box[12]#center x
                #y_diff2 = entity.box[9]-entities[links_to_questions[1]].box[9]#center y
                #these need written more robustly
                #first_left_of = max(first[::2])-5<min(entity.box[::2])
                #first_right_of = min(first[::2])+5>max(entity.box[::2])
                #first_top_of = max(first[1::2])-5<min(entity.box[1::2])
                #first_bottom_of = min(first[1::2])+5>max(entity.box[1::2])

                first_left_dist = min(entity.box[::2])-max(first[::2])
                first_right_dist = min(first[::2])-max(entity.box[::2])
                first_top_dist = min(entity.box[1::2])-max(first[1::2])
                first_bot_dist = min(first[1::2])-max(entity.box[1::2])

                first_left_of = first_left_dist > max(first_top_dist,first_bot_dist)
                first_right_of = first_right_dist > max(first_top_dist,first_bot_dist)
                first_top_of = first_top_dist > max(first_left_dist,first_right_dist)
                first_bottom_of = first_bot_dist > max(first_left_dist,first_right_dist)

                first_is_row_header = first_left_of or first_right_of
                first_is_col_header = first_top_of or first_bottom_of

                #second_left_of = max(second[::2])-5<min(entity.box[::2])
                #second_right_of = min(second[::2])+5>max(entity.box[::2])
                #second_top_of = max(second[1::2])-5<min(entity.box[1::2])
                #second_bottom_of = min(second[1::2])+5>max(entity.box[1::2])

                second_left_dist = min(entity.box[::2])-max(second[::2])
                second_right_dist = min(second[::2])-max(entity.box[::2])
                second_top_dist = min(entity.box[1::2])-max(second[1::2])
                second_bot_dist = min(second[1::2])-max(entity.box[1::2])

                second_left_of = second_left_dist > max(second_top_dist,second_bot_dist)
                second_right_of = second_right_dist > max(second_top_dist,second_bot_dist)
                second_top_of = second_top_dist > max(second_left_dist,second_right_dist)
                second_bottom_of = second_bot_dist > max(second_left_dist,second_right_dist)


                second_is_row_header = second_left_of or second_right_of
                second_is_col_header = second_top_of or second_bottom_of

                if first_is_row_header==second_is_row_header and first_is_col_header==second_is_col_header:
                    #This is probably an oversegmented line
                    y_diffs = abs(first[1]-second[1])+abs(first[5]-second[5])
                    if y_diffs<15:
                        if first[12]<second[12]:
                            first_i = links_to_questions[0]
                            second_i = links_to_questions[1]
                        else:
                            first_i = links_to_questions[1]
                            second_i = links_to_questions[0]

                        first = entities[first_i]
                        second = entities[second_i]
                        #confirm x position
                        x_diff = second.box[0]-first.box[4]
                        if x_diff>-4 and len(first.lines)==1 and len(second.lines)==1:
                            #indeed, these need merged
                            first.append(second)
                            entity_i_to_remove.add(second_i)



                bad=False
                if first_is_row_header==first_is_col_header:
                    #bad=True
                    #print('{} has ambiguous header {}'.format(entity,entities[links_to_questions[0]]))
                    #if i==35 or i==13:
                    #    import pdb;pdb.set_trace()
                    adj_to_remove.add((i,links_to_questions[0]))
                    adj_to_remove.add((links_to_questions[0],i))
                elif second_is_row_header==second_is_col_header:
                    #bad=True
                    #print('{} has ambiguous header {}'.format(entity,entities[links_to_questions[1]]))
                    #if i==35 or i==13:
                    #    import pdb;pdb.set_trace()
                    adj_to_remove.add((i,links_to_questions[1]))
                    adj_to_remove.add((links_to_questions[1],i))

                elif first_is_row_header==second_is_row_header:
                    #bad=True
                    #print('{} has conflicting headers \n0:{}, \n1:{}'.format(
                    #        entity,
                    #        entities[links_to_questions[0]],
                    #        entities[links_to_questions[1]]))
                    #if i==35 or i==13:
                    #    import pdb;pdb.set_trace()
                    r = int(random.random()<0.5)
                    adj_to_remove.add((i,links_to_questions[r]))
                    adj_to_remove.add((links_to_questions[r],i))

                #if bad:
                #    import pdb;pdb.set_trace()
        elif entity.cls=='question':
            if len(links_to_questions)>0:
                for other_i in links_to_questions:
                    #print('{} linked to {}'.format(entity,entities[other_i]))
                    adj_to_remove.add((i,other_i))
                    adj_to_remove.add((other_i,i))
                #bad, should be header
                #should be header, but some questions are labeled as answers
                #import pdb;pdb.set_trace()
            if len(links_to_headers)>1:
                for other_i in links_to_headers:
                    #print('{} linked to {}'.format(entity,entities[other_i]))
                    adj_to_remove.add((i,other_i))
                    adj_to_remove.add((other_i,i))
                #import pdb;pdb.set_trace()
        #elif entity.cls=='header':
        #    if len(links_to_headers)>0:
        #        for other_i in links_to_headers:
        #            print('{} linked to {}'.format(entity,entities[other_i]))
        #            #subheaders
        #        import pdb;pdb.set_trace()
    debug = len(entity_adj)
    #import pdb;pdb.set_trace()
    entity_adj -= adj_to_remove
    assert len(adj_to_remove)==0 or len(entity_adj)<debug
    if len(entity_i_to_remove)==0:
        new_entities=entities
        new_entity_adj=entity_adj
    else:
        new_entities=[]
        old_to_new={}
        for i,entity in enumerate(entities):
            if i not in entity_i_to_remove:
                old_to_new[i]=len(new_entities)
                new_entities.append(entity)
        new_entity_adj=[(old_to_new[e1],old_to_new[e2]) for e1,e2 in entity_adj if \
                e1 in old_to_new and e2 in old_to_new] 
    return new_entities,new_entity_adj
