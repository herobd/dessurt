import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math, random, string
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from data_sets.form_qa import FormQA,collate, Line, Entity, Table

from utils import img_f



class FUNSDQA(FormQA):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FUNSDQA, self).__init__(dirPath,split,config,images)



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
                    self.images.append({'id':image_name, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':image_name[:image_name.rfind('.')]})
        self.errors=[]

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


    def parseAnn(self,annotations,s):
        #if useBlankClass:
        #    numClasses+=1
        #if usePairedClass:
        #    numClasses+=1

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

        bbs = np.array(boxes)

        #assign ambiguity
        for line_ids in text_line_counts.values():
            if len(line_ids)>1:
                for ei,li in line_ids:
                    entities[ei].lines[li].ambiguous = True
        qa = self.makeQuestions(1.0,entities,entity_link,tables)

        ocr=None
        #ocr = [self.corrupt(text) for text in ocr]
        return bbs, list(range(bbs.shape[0])), ocr, {}, {}, qa


    def convertBB(self,s,box):
        assert s==1
        if isinstance(box,list):
            return box
        return box.tolist()

    def prepareForm(self,bbs,transcription,groups,groups_adj):


        entities=[]


        questions_gs=set()
        answers_gs=set()
        headers_gs=set()
        others_gs=set()
        removed=set()
        group_count = len(groups)
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
                if cls_name == 'question':
                    questions_gs.add(gi)
                elif cls_name == 'answer':
                    answers_gs.add(gi)
                elif cls_name == 'header':
                    headers_gs.add(gi)
                else:
                    others_gs.add(gi)
            else:
                entities.append(None)
                removed.add(gi)


        groups_adj = cleanUpAdj(entities,groups_adj)


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
                    relationships_q_a[q_gi].append(gi2)
                    relationships_a_q[gi2].append(q_gi)
                    found=True
                elif gi2 == q_gi and gi1 in answers_gs:
                    #q_a_pairs.append((gi2,gi1))
                    relationships_q_a[q_gi].append(gi1)
                    relationships_a_q[gi1].append(q_gi)
                    found=True
            if not found:
                #q_a_pairs.append((q_gi,None))
                relationships_q_a[q_gi].append(None)

        for a_gi in answers_gs:
            if a_gi not in relationships_a_q:
                relationships_a_q[a_gi].append(None)



        q_a_pairs=defaultdict(list)
        #table_map = {}
        #tables=[]
        table_values={}
        row_headers=set()
        col_headers=set()
        skip=set()

        merged_groups = {}
        for qi,ais in relationships_q_a.items():
            if qi in skip:
                continue
            if len(ais)==1:
                ai=ais[0]
                if ai is None or len(relationships_a_q[ai])==1:
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
                    success = addTableElement(table_values,row_headers,col_headers,ai,q1,q2,groups,bbs)
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
                        bbi = groups[ai][0]
                        answers.append((bbs[bbi,1],ai))
                    answers.sort(key=lambda a:a[0])
                    answers = [a[1] for a in answers]
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
                        addTableElement(table_values,row_headers,col_headers,ai,q1,q2,groups,bbs)

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
            tail = [old_to_new_e_map[ai] for ai in ais if ai is not None and ai not in removed]
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
                    for bbi in groups[qi]:
                        x += bbs[bbi,0]
                        x += bbs[bbi,4]
                        y += bbs[bbi,1]
                        y += bbs[bbi,5]
                        h += bbs[bbi,5]-bbs[bbi,1]+1
                    xc = x/(2*len(groups[qi]))
                    yc = y/(2*len(groups[qi]))
                    h = h/len(groups[qi])
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
        for (col_h,row_h),v in table_values.items():
            col_h = old_to_new_e_map[col_h] if col_h is not None else None
            row_h = old_to_new_e_map[row_h] if row_h is not None else None
            v = old_to_new_e_map[v]

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
        for (col_h,row_h),v in table_values.items():
            col_h = old_to_new_e_map[col_h] if col_h is not None else None
            row_h = old_to_new_e_map[row_h] if row_h is not None else None
            v = old_to_new_e_map[v]
            
            if col_h is not None:
                tab_id = header_to_table_id[col_h]
                if row_h is not None:
                    assert tab_id == header_to_table_id[row_h]
            else:
                tab_id = header_to_table_id[row_h]

            new_table_values.append((tab_id,col_h,row_h,v))
            if col_h is None:
                blank_col[tab_id][row_h].append(v)
            elif row_h is None:
                blank_row[tab_id][col_h].append(v)
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
            if None in col_hs:
                #We also need to find where the columns without headers go
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
                    pos_col_hs.append((x_center,blank_col_items))

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
                        y_center = (cell.box[0]+cell.box[4])/2
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
                    pos_row_hs.append((y_center,blank_row_items))

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

        for tab_id, col_h, row_h, v in new_table_values:
            tab_pos = tab_id_to_pos[tab_id]
            #Lookup poition of v in Table
            r = all_row_entity_ids[tab_pos].index(row_h) if row_h is not None else blank_row_pos[v]
            c = all_col_entity_ids[tab_pos].index(col_h) if col_h is not None else blank_col_pos[v]
            real_tables[tab_pos].cells[r][c]=new_entities[v]

        ##DEBUG
        for head,tail in entity_link:
            assert new_entities[head].cls!='answer'
            if tail is not None:
                if isinstance(tail,list):
                    for t in tail:
                        assert new_entities[head].text != new_entities[t].text
                else:
                    assert new_entities[head].text != new_entities[tail].text

        return new_entities,entity_link,real_tables

def addTableElement(table_values,row_headers,col_headers,ai,qi1,qi2,groups,bbs,threshold=5):
    ele_x,ele_y = bbs[groups[ai][0],0:2]
    q1_x,q1_y = bbs[groups[qi1][0],0:2]
    x_diff_1 = abs(ele_x-q1_x)
    y_diff_1 = abs(ele_y-q1_y)
    if qi2 is not None:
        #which question is the row, which is the header?
        q2_x,q2_y = bbs[groups[qi2][0],0:2]
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
        
        table_values[(col_h,row_h)]=ai
        row_headers.add(row_h)
        col_headers.add(col_h)
    else:
        if x_diff_1>y_diff_1:
            row_headers.add(qi1)
            table_values[(None,qi1)]=ai
        elif x_diff_1<y_diff_1:
            col_headers.add(qi1)
            table_values[(qi1,None)]=ai
    return True


def cleanUpAdj(entities,entity_adj):
    if entity_adj is None:
        return entity_adj
    #This is to check answers with two questions. These should be tables, but there are many
    #errors in the dataset labeling.

    for i,entity in enumerate(entities):
        if entity is None:
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
                    print('Error: {} linked to {}'.format(entity,entities[other_i]))
                import pdb;pdb.set_trace()
            if len(links_to_headers)>0:
                for other_i in links_to_headers:
                    print('{} linked to {}'.format(entity,entities[other_i]))
                import pdb;pdb.set_trace()
            if len(links_to_questions)>2:
                for other_i in links_to_questions:
                    print('{} linked to {}'.format(entity,entities[other_i]))
                import pdb;pdb.set_trace()
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

                first_left_of = max(first[::2])<min(entity.box[::2])
                first_right_of = min(first[::2])>max(entity.box[::2])
                first_top_of = max(first[1::2])<min(entity.box[1::2])
                first_bottom_of = min(first[1::2])>max(entity.box[1::2])

                first_is_row_header = first_left_of or first_right_of
                first_is_col_header = first_top_of or first_bottom_of

                second_left_of = max(second[::2])<min(entity.box[::2])
                second_right_of = min(second[::2])>max(entity.box[::2])
                second_top_of = max(second[1::2])<min(entity.box[1::2])
                second_bottom_of = min(second[1::2])>max(entity.box[1::2])

                second_is_row_header = second_left_of or second_right_of
                second_is_col_header = second_top_of or second_bottom_of

                bad=False
                if first_is_row_header==first_is_col_header:
                    bad=True
                    print('{} has ambiguous header {}'.format(entity,entities[links_to_questions[0]]))
                if second_is_row_header==second_is_col_header:
                    bad=True
                    print('{} has ambiguous header {}'.format(entity,entities[links_to_questions[1]]))

                if first_is_row_header==second_is_row_header:
                    bad=True
                    print('{} has conflicting headers 0:{}, 1:{}'.format(
                            entity,
                            entities[links_to_questions[0]],
                            entities[links_to_questions[1]]))

                if bad:
                    import pdb;pdb.set_trace()
        elif entity.cls=='question':
            if len(links_to_headers)>1:
                for other_i in links_to_headers:
                    print('{} linked to {}'.format(entity,entities[other_i]))
                import pdb;pdb.set_trace()
        #elif entity.cls=='header':
        #    if len(links_to_headers)>0:
        #        for other_i in links_to_headers:
        #            print('{} linked to {}'.format(entity,entities[other_i]))
        #            #subheaders
        #        import pdb;pdb.set_trace()
    return entity_adj
