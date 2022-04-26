import json
import torch
from collections import defaultdict
import os
import utils.img_f as img_f
import numpy as np
import math
import random

from .form_qa import FormQA, collate, Entity, Line, Table
from .gen_daemon import GenDaemon


#put work images into paragraph form (single image)
def resizeAndJoinImgs(word_imgs,height,space_width,boundary_x):
    #word_imgs: list of (text,word_img)
    #height: the height to resive word images to
    #space_width: how much space between words
    #boundary_x: when to stop horizontally and wrap to new line

    full_text=''
    newline = round(height*0.1+0.9*random.random()*height)
    max_x=0
    max_y=0
    cur_x=0
    cur_y=0
    resized_words=[]

    text,img = word_imgs[0]
    width = max(1,round(img.shape[1]*height/img.shape[0]))
    img = img_f.resize(img,(height,width))
    resized_words.append((img,cur_x,cur_y))
    full_text += text
    cur_x+=width
    max_x=max(max_x,cur_x)
    max_y=max(max_y,cur_y+height)
    
    for text,img in word_imgs[1:]:
        width = max(1,round(img.shape[1]*height/img.shape[0]))
        img = img_f.resize(img,(height,width))
        if cur_x+width<boundary_x:
            full_text+=' '+text
        else:
            cur_x=0
            cur_y+=newline+height
            full_text+='\\'+text
        resized_words.append((img,cur_x,cur_y))
        cur_x+=space_width+width
        max_x=max(max_x,cur_x)
        max_y=max(max_y,cur_y+height)
    
    full_img = np.zeros([max_y,max_x],dtype=np.uint8)
    for img,x,y in resized_words:
        full_img[y:y+img.shape[0],x:x+img.shape[1]]=img

    return full_img,full_text



#This dataset creates synthetic form images on the fly
#The format is heavily based on the FUNSD dataset
class SynthFormDataset(FormQA):
    def __init__(self, dirPath, split, config):
        super(SynthFormDataset, self).__init__(dirPath,split,config)
        font_dir = config['font_dir'] #directory pointing to font dataset (has clean_fonts.csv)
        self.gen_daemon = GenDaemon(font_dir) #This makes the word images

        self.color=False

        self.image_size = config['image_size'] if 'image_size' in config else None #output size
        if type(self.image_size) is int:
            self.image_size = (self.image_size,self.image_size)
        
        #range of text heights
        self.min_text_height = config['min_text_height'] if 'min_text_height' in config else 8
        self.max_text_height = config['max_text_height'] if 'max_text_height' in config else 32

        #frequency of placing a table instead of label-value set
        self.table_prob = config['tables'] if 'tables' in config else 0.2

        #lots of other probabilities
        self.match_title_prob = 0.3
        self.match_label_prob = 0.5
        self.blank_value_prob = 0.1
        self.checkbox_prob = 0.005
        self.checkbox_blank_value_prob = 0.5
        self.new_col_chance = 0.3
        self.new_col_chance_checkbox = 0.6
        self.side_by_side_prob = 0.66
        self.min_qa_sep = 10
        self.max_qa_sep = 250
        self.block_pad_max = 500
        self.block_pad_min = 30

        self.max_table_cell_width=80
        self.max_table_colh_width=80
        self.max_table_rowh_width=200

        #Load results of gpt form generation
        with open(os.path.join(dirPath,'gpt2_form_generation.json')) as f:
            self.documents = json.load(f)

        self.warp_freq = 1.0
        if split=='train':
            self.augmentation = config['augmentation'] if 'augmentation' in config else None
        self.augment_shade = config.get('augment_shade',1)
	    

            
        
        self.images=[]
        for i in range(config['batch_size']*400): #we just randomly generate instances on the fly, but add this to give it instances to sample from (QADataset expects it)
            self.images.append({'id':'{}'.format(i), 'imagePath':None, 'annotationPath':0, 'rescaled':1.0, 'imageName':'0'})
        
        self.random_words = []
        self.stop_words = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]) #from https://www.ranks.nl/stopwords




    def parseAnn(self,annotations,s):
        #This defines the creation of the image with its GT entities, boxes etc.
        #Questions are generated by parent class with maskQuestions()



        image_h,image_w = self.image_size
        image = np.zeros([image_h,image_w],dtype=np.uint8)
        all_entities=[]
        entity_link=defaultdict(list)
        tables=[]
        
        #Main building loop
        debug=0
        while len(all_entities)==0 and debug<200:
            debug+=1
            success=True
            boxes = []
            prev_boxes=[(0,0,image_w,0)]
            furthest_right=0
            while len(prev_boxes)>0 and debug<200:
                debug+=1
                #pick starting point
                x1,y1,x2,y2 = prev_boxes.pop(random.randrange(len(prev_boxes)))
                if image_h-y2<60 and random.random()<0.5:
                    continue 
                if x1>=furthest_right:
                    x2 = image_w #can do to end of image

                if random.random()<self.table_prob:
                    success,box = self.addTable(x1,y2,x2,image,tables,all_entities,entity_link)
                else:
                    success,box = self.addForm(x1,y2,x2,image,all_entities,entity_link)
                
                if success:
                    prev_boxes.append(box) #add for the space under the just generated thing
                    if box[2]>furthest_right:
                        furthest_right = box[2]
                        prev_boxes.append((furthest_right,0,image_w,0)) #add non-box, just to the right




        

        
        #run through all entites to build bbs, assign bbid, and find ambiguity
        boxes = []
        text_line_counts = defaultdict(list)
        for ei,entity in enumerate(all_entities):
            for li,line in enumerate(entity.lines):
                text = self.punc_regex.sub('',line.text.lower())
                text_line_counts[text].append((ei,li))
                bbid = len(boxes)
                boxes.append(self.convertBB(s,line.box))
                line.bbid = bbid

        bbs = np.array(boxes)

        #assign ambiguity
        for line_ids in text_line_counts.values():
            if len(line_ids)>1:
                for ei,li in line_ids:
                    all_entities[ei].lines[li].ambiguous = True

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
                r_index = all_entities.index(r_header)
                for c,c_header in enumerate(table.col_headers):
                    c_index = all_entities.index(c_header)
                    v=table.cells[r][c]
                    if v is not None:
                        v_index = all_entities.index(v)
                        link_dict[r_index].append(v_index)
                        link_dict[c_index].append(v_index)
                        link_dict[v_index].append(r_index)
                        link_dict[v_index].append(c_index)
        link_dict = self.sortLinkDict(all_entities,link_dict)

        qa = self.makeQuestions(s,all_entities,entity_link,tables,all_entities,link_dict)

        return bbs, list(range(bbs.shape[0])), 255-image, None, qa

    

    def addTable(self,init_x,init_y,max_x,image,tables,entities,entity_link):
        #init_x, init_y: the top-left corner of the area availble
        #max_x: the furthes right it can draw (can always go to bottom of image)
        #image: the image were making
        #tables: list of tables to add this new table to
        #entities: " entities "
        #entity_link: " links "
        max_y = image.shape[0]
        
        #get starting position
        start_x = init_x+(self.block_pad_max if len(entities)>0 else self.block_pad_max//3)
        start_y = init_y+(self.block_pad_max if len(entities)>0 else self.block_pad_max//3)
        title_height = self.max_text_height
        label_height = self.max_text_height
        value_height = self.max_text_height
        if start_x>init_x+self.block_pad_min:
            start_x = random.randrange(init_x+2*self.block_pad_min,start_x)
        if start_y>init_y+self.block_pad_min:
            start_y = random.randrange(init_y+2*self.block_pad_min,start_y)

        if start_x>=image.shape[1]-16 or start_y>=image.shape[0]-10:
            #not enough room
            return False,None

        if random.random()<0.33:
            #This table gets a title!
            while len(self.random_words)<2:
                self.addRandomWords()
            num_words = random.randrange(1,min(len(self.random_words),6))
            title = ' '.join(self.random_words[-num_words:])
            self.random_words = self.random_words[:-num_words]
            title_words,title_font = self.gen_daemon.generate(title,ret_font=True) 
            if len(title_words)==0:
                #or maybe not
                title=None
        else:
            title = None

        
        
        #setup text height and spacing
        if title is not None:
            title_height = random.randrange((self.min_text_height+title_height)//2,1+max(self.min_text_height+2,title_height))
            em_approx = title_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
            min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/chara  cter-design-standards/whitespace
            max_space = 0.5*em_approx
            title_space_width = round(random.random()*(max_space-min_space) + min_space)
            title_newline_height = random.randrange(1,title_height) + title_height

        label_height = random.randrange(self.min_text_height,1+max(self.min_text_height+2,min(label_height,title_height)))
        em_approx = label_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
        min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/chara  cter-design-standards/whitespace
        max_space = 0.5*em_approx
        label_space_width = round(random.random()*(max_space-min_space) + min_space)
        label_newline_height = random.randrange(1,label_height) + label_height

        value_height = random.randrange(self.min_text_height,1+max(self.min_text_height+2,min(value_height,title_height)))
        em_approx = value_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
        min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/chara  cter-design-standards/whitespace
        max_space = 0.5*em_approx
        value_space_width = round(random.random()*(max_space-min_space) + min_space)
        value_newline_height = random.randrange(1,value_height) + value_height
        value_list_newline_height = value_newline_height + round(value_newline_height*random.random())
        
        if title is not None:
            title_word_widths = [max(1,round(img.shape[1]*(title_height/img.shape[0]))) for text,img in title_words]
        

        block_width = max_x-start_x


        #how wide will the title be?
        if title is not None:
            #let's lay it out and see
            if max(title_word_widths)>=block_width:
                return False,None
                #continue #not going to fit
            max_title_width = block_width
            max_title_x = start_x+max_title_width
            #layout the title to see how tall it is
            title_str=''
            title_str_lines=[]
            title_img_pos_lines=[]
            title_img_pos=[]
            cur_x=start_x
            cur_y=start_y
            rightmost_title_x=cur_x
            restart=False
            for title_w, (title_text,title_img) in zip(title_word_widths,title_words):
                if cur_x+title_w>max_title_x:
                    #newline
                    title_str_lines.append(title_str)
                    title_str=''
                    title_img_pos_lines.append(title_img_pos)
                    title_img_pos=[]
                    cur_x = start_x
                    cur_y += title_newline_height
                    if cur_y+title_height>=max_y:
                        #cannot fit
                        restart=True
                        break
                elif len(title_str)>0:
                    title_str+=' '#space
                title_x = cur_x
                title_y = cur_y
                
                cur_x+=title_w+title_space_width
                title_str += title_text
                title_img_pos.append((title_x,title_y,title_img,title_height,title_w))
                rightmost_title_x = max(rightmost_title_x,title_x+title_w)
            if restart:
                title = None
                end_title_y=start_y
                rightmost_title_x=start_x
                #return False,None,None
                #continue #retry, not room for title
            else:
                if len(title_img_pos)==0:
                    return False,None
                title_img_pos_lines.append(title_img_pos)
                title_str_lines.append(title_str)
                end_title_y=cur_y+title_newline_height+round(title_newline_height*random.random())
            
        else:
            end_title_y=start_y + random.randrange(70)
            rightmost_title_x=start_x

        if end_title_y+3*label_height>=max_y:
            return False,None
            #no room for fields




        num_rows=random.randrange(1,15)
        num_cols=random.randrange(1,10)

        if num_rows==1 and num_cols==1:
            #shouldn't have a table with one cell
            if random.random()<0.5:
                num_rows=random.randrange(2,15)
            else:
                num_cols=random.randrange(2,10)

        mean_height = random.randrange(self.min_text_height+1,self.max_text_height)

        table_entries = self.getTableValues(num_rows*num_cols)
        row_header_entries = self.getTableHeaders(num_rows)
        col_header_entries = self.getTableHeaders(num_cols)

        table_entries_1d = []
        font = None
        debug=0
        for text in table_entries:
            word_imgs,font = self.gen_daemon.generate(text,font=font,ret_font=True)
            while len(word_imgs)==0 and debug<2000:
                #for some reason, couldn't generate, try try again
                debug+=1
                text=self.getTableValues(1)[0]
                word_imgs,font = self.gen_daemon.generate(text,font=font,ret_font=True)
            img,label = resizeAndJoinImgs(word_imgs,value_height,value_space_width,self.max_table_cell_width)
            table_entries_1d.append([img,label])
        row_header_entries_1d = []
        font = None
        for text in row_header_entries:
            word_imgs,font = self.gen_daemon.generate(text,font=font,ret_font=True)
            while len(word_imgs)==0 and debug<2000:
                debug+=1
                text=self.getTableHeaders(1)[0]
                word_imgs,font = self.gen_daemon.generate(text,font=font,ret_font=True)
            img,label = resizeAndJoinImgs(word_imgs,label_height,label_space_width,self.max_table_rowh_width)
            row_header_entries_1d.append([img,label])
        col_header_entries_1d = []
        font = None
        for text in col_header_entries:
            word_imgs,font = self.gen_daemon.generate(text,font=font,ret_font=True)
            while len(word_imgs)==0 and debug<2000:
                debug+=1
                text=self.getTableHeaders(1)[0]
                word_imgs,font = self.gen_daemon.generate(text,font=font,ret_font=True)
            img,label = resizeAndJoinImgs(word_imgs,label_height,label_space_width,self.max_table_colh_width)
            col_header_entries_1d.append([img,label])

        row_headers = row_header_entries_1d
        col_headers = col_header_entries_1d
        table_entries = table_entries_1d
        table_entries_2d = []
        for r in range(num_rows):
            table_entries_2d.append(table_entries_1d[r*num_cols:(r+1)*num_cols])
        table_entries = table_entries_2d


        table_x = random.randrange(init_x,init_x+70)
        table_y = end_title_y 

        padding = random.randrange(0,30)

        max_height=0
        for c in range(num_cols):
            max_height = max(max_height,col_headers[c][0].shape[0])
        total_height = max_height+padding
        height_col_heading = max_height+padding
        
        if total_height+table_y >= self.image_size[0]:
            #NO TABLE
            return False,None
        height_row=[0]*num_rows
        for r in range(num_rows):
            max_height = row_headers[r][0].shape[0]
            for c in range(num_cols):
                max_height = max(max_height,table_entries[r][c][0].shape[0])
            height_row[r] = max_height+padding
            total_height+= max_height+padding

            if total_height+table_y >= self.image_size[0]:
                num_rows = r
                if num_rows==0:
                    return False,None
                total_height -= height_row[r]
                row_headers=row_headers[:num_rows]
                height_row=height_row[:num_rows]
                break

        max_width=0
        for r in range(num_rows):
            max_width = max(max_width,row_headers[r][0].shape[1])
        total_width = max_width+padding
        width_row_heading = max_width+padding
        
        if total_width+table_x >= max_x:
            #NO TABLE
            return False,None
        width_col=[0]*num_cols
        for c in range(num_cols):
            max_width = col_headers[c][0].shape[1]
            for r in range(num_rows):
                max_width = max(max_width,table_entries[r][c][0].shape[1])
            width_col[c] = max_width+padding
            total_width+= max_width+padding

            if total_width+table_x >= max_x:
                num_cols = c
                if num_cols==0:
                    return False,None
                total_width -= width_col[c]
                col_headers=col_headers[:num_cols]
                width_col=width_col[:num_cols]
                break
    

        #put row headers in image
        row_headers_e=[]
        cur_y = height_col_heading+table_y
        for r in range(num_rows):
            if width_row_heading-padding==row_headers[r][0].shape[1]:
                x=table_x
            else:
                x=table_x + random.randrange(0,width_row_heading-padding-row_headers[r][0].shape[1])
            if height_row[r]-padding==row_headers[r][0].shape[0]:
                y=cur_y
            else:
                y=cur_y + random.randrange(0,height_row[r]-padding-row_headers[r][0].shape[0])
            cur_y += height_row[r]

            diff = image.shape[0]-(y+row_headers[r][0].shape[0])
            if diff<0:
                row_headers[r][0] = row_headers[r][0][:diff]
            diff = image.shape[1]-(x+row_headers[r][0].shape[1])
            if diff<0:
                row_headers[r][0] = row_headers[r][0][:,:diff]
            image[y:y+row_headers[r][0].shape[0],x:x+row_headers[r][0].shape[1]] = row_headers[r][0]
            box = [x,y,x+row_headers[r][0].shape[1],y+row_headers[r][0].shape[0]]
            if row_headers[r][1] == '':
                string='-'
            else:
                string=row_headers[r][1]
            row_headers_e.append( Entity('answer',[Line(string,box)]) )

        #put col headers in image
        col_headers_e=[]
        cur_x = width_row_heading+table_x
        for c in range(num_cols):
            if height_col_heading-padding==col_headers[c][0].shape[0]:
                y=table_y
            else:
                y=table_y + random.randrange(0,height_col_heading-padding-col_headers[c][0].shape[0])
            if width_col[c]-padding==col_headers[c][0].shape[1]:
                x=cur_x
            else:
                x=cur_x + random.randrange(0,width_col[c]-padding-col_headers[c][0].shape[1])
            cur_x += width_col[c]
            
            diff = image.shape[0]-(y+col_headers[c][0].shape[0])
            if diff<0:
                col_headers[c][0] = col_headers[c][0][:diff]
            diff = image.shape[1]-(x+col_headers[c][0].shape[1])
            if diff<0:
                col_headers[c][0] = col_headers[c][0][:,:diff]
            image[y:y+col_headers[c][0].shape[0],x:x+col_headers[c][0].shape[1]] = col_headers[c][0]
            box = [x,y,x+col_headers[c][0].shape[1],y+col_headers[c][0].shape[0]]
            if col_headers[c][1] == '':
                string='-'
            else:
                string=col_headers[c][1]
            col_headers_e.append( Entity('answer',[Line(string,box)]) )

        table = Table(row_headers_e,col_headers_e)

        #put entries into image (and table)
        cur_x = width_row_heading+table_x
        for c in range(num_cols):
            cur_y = height_col_heading+table_y
            for r in range(num_rows):
                if random.random()>0.15 and len(table_entries[r][c][1])>0: #sometimes skip an entry
                    if width_col[c]-padding==table_entries[r][c][0].shape[1]:
                        x=cur_x
                    else:
                        x=cur_x + random.randrange(0,width_col[c]-padding-table_entries[r][c][0].shape[1])
                    if height_row[r]-padding==table_entries[r][c][0].shape[0]:
                        y=cur_y
                    else:
                        y=cur_y + random.randrange(0,height_row[r]-padding-table_entries[r][c][0].shape[0])
                    if y+table_entries[r][c][0].shape[0]>image.shape[0]:
                        diff = image.shape[0]-(y+table_entries[r][c][0].shape[0])
                    if x+table_entries[r][c][0].shape[1]>image.shape[1]:
                        diff = image.shape[1]-(x+table_entries[r][c][0].shape[1])
                        table_entries[r][c][0] = table_entries[r][c][0][:,:diff]
                    image[y:y+table_entries[r][c][0].shape[0],x:x+table_entries[r][c][0].shape[1]] = table_entries[r][c][0]
                    box = [x,y,x+table_entries[r][c][0].shape[1],y+table_entries[r][c][0].shape[0]]
                    table.cells[r][c]=Entity('answer',[Line(table_entries[r][c][1],box)])
                
                cur_y += height_row[r]
            cur_x += width_col[c]

        #add lines for headers
        line_thickness_h = random.randrange(1,max(2,min(10,padding)))
        #top
        img_f.line(image,
                (max(0,table_x+random.randrange(-5,5)),table_y+height_col_heading-random.randrange(0,1+padding)),
                (min(max_x-1,table_x+total_width+random.randrange(-5,5)),table_y+height_col_heading-random.randrange(0,1+padding)),
                random.randrange(0,100),
                line_thickness_h
                )
        #side
        img_f.line(image,
                (table_x+width_row_heading-random.randrange(0,padding+1),max(0,table_y+random.randrange(-5,5))),
                (table_x+width_row_heading-random.randrange(0,padding+1),min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                random.randrange(0,100),
                line_thickness_h
                )

        #outside of headers?
        if random.random()<0.5:
            line_thickness = random.randrange(1,max(2,min(10,padding)))
            #top
            img_f.line(image,
                    (max(0,table_x+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)),
                    (min(max_x-1,table_x+total_width+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)),
                    random.randrange(0,100),
                    line_thickness
                    )
            #side
            img_f.line(image,
                    (table_x-random.randrange(0,padding+1),max(0,table_y+random.randrange(-5,5))),
                    (table_x-random.randrange(0,padding+1),min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                    random.randrange(0,100),
                    line_thickness
                    )

        #value outline?
        if random.random()<0.5:
            line_thickness = random.randrange(1,max(2,min(10,padding)))
            #bot
            img_f.line(image,
                    (max(0,table_x+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+total_height),
                    (min(max_x-1,table_x+total_width+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+total_height),
                    random.randrange(0,100),
                    line_thickness
                    )
            #right
            img_f.line(image,
                    (table_x-random.randrange(0,padding+1)+total_width,max(0,table_y+random.randrange(-5,5))),
                    (table_x-random.randrange(0,padding+1)+total_width,min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                    random.randrange(0,100),
                    line_thickness
                    )

        #all inbetween lines?
        if random.random()<0.5:
            line_thickness = random.randrange(1,max(2,line_thickness_h))
            #horz
            cur_height = height_col_heading
            for r in range(num_rows-1):
                cur_height += height_row[r]
                img_f.line(image,
                        (max(0,table_x+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+cur_height),
                        (min(max_x-1,table_x+total_width+random.randrange(-5,5)),table_y-random.randrange(0,padding+1)+cur_height),
                        random.randrange(0,100),
                        line_thickness
                        )
            #right
            cur_width = width_row_heading
            for c in range(num_cols-1):
                cur_width += width_col[c]
                img_f.line(image,
                        (table_x-random.randrange(0,padding+1)+cur_width,max(0,table_y+random.randrange(-5,5))),
                        (table_x-random.randrange(0,padding+1)+cur_width,min(self.image_size[0]-1,table_y+total_height+random.randrange(-5,5))),
                        random.randrange(0,100),
                        line_thickness
                        )

        
        tables.append(table)

        first_index = len(entities)
        entities += tables[-1].allEntities()

        if title is not None:
            title_entity = self.makeAndDrawEntity(image,'header',title_str_lines,title_img_pos_lines)
            title_index= len(entities)
            entities.append(title_entity)
            entity_link[title_index] += range(first_index,first_index+len(table.row_headers)+len(table.col_headers))

        return True,(init_x, init_y, table_x+total_width+10, table_y+total_height+10)



    def addForm(self,init_x,init_y,max_x,image,entities,entity_link):
        debug=0
        while debug<100:
            debug+=1

            #get label-value set
            title,pairs = random.choice(self.documents)

            #fix bad parsing
            new_pairs = []
            for label,value in pairs:
                if not label.endswith('http'):
                    new_pairs.append((label,value))
            if len(new_pairs)>0:
                pairs=new_pairs
                break

        max_y = image.shape[0]
        label_matches_title = random.random()<self.match_title_prob and title is not None
        value_matches_label = random.random()<self.match_label_prob

        
        label_height = None

        #generate title images
        if title is not None:
            if random.random()<0.02 and title[-1]!=':':
                title+=':'
            title_words,title_font = self.gen_daemon.generate(title,ret_font=True) #(text,img)
        if label_matches_title:
            label_font = title_font
        else:
            label_font = None
        if value_matches_label:
            value_font = label_font
        else:
            value_font = None
    

        #get cue (and whether to use checkboxes)
        options=['colon','line','line+colon','dotted line','dotted line+colon','box','box+colon','to left','below','none']
        checkboxes = random.random()<self.checkbox_prob
        if checkboxes:
            #make check image
            cue = random.choice(['colon','box','box+colon','none','to left'])
            blank_value_prob = self.checkbox_blank_value_prob
            checkboxes = random.choice(['paren','bracket']) if 'box' not in cue else 'box'
            x_str = random.choice('xX')
            ws,value_font =  self.gen_daemon.generate(x_str,font=value_font,ret_font=True)
            x_str,x_im = ws[0]
            if checkboxes!='box':
                open_im,close_im = self.gen_daemon.getBrackets(value_font,paren=checkboxes=='paren')
                x_space=random.randrange(0,x_im.shape[1])
                blank_im = np.zeros((max(open_im.shape[0],x_im.shape[0],close_im.shape[0]),x_space+open_im.shape[1]+x_im.shape[1]+close_im.shape[1]),dtype=np.uint8)

                blank_im[0:open_im.shape[0],0:open_im.shape[1]] = open_im
                blank_im[0:close_im.shape[0],-close_im.shape[1]:] = close_im


        else:
            cue = random.choice(options)
            blank_value_prob = self.blank_value_prob
        if 'to left' in cue or 'below' in cue:
            if random.random()<0.5:
                cue+=' box'
            elif not checkboxes:
                cue+=' line'

        #generate label and value images 
        image_pairs = []
        for label,value in pairs:
            label_lower = label.lower()
            if (label_lower.endswith('http') or label_lower.endswith('https')) and len(pairs)>0:
                continue #bad parsing of gpt ouput
            label_words,label_font = self.gen_daemon.generate(label+(':' if 'colon' in cue else ''),font=label_font,ret_font=True)
            if random.random()<blank_value_prob and 'to left' not in cue and 'below' not in cue and (not checkboxes or checkboxes!='box'):
                value_words = []
                if checkboxes:
                    value_words=[[(None,blank_im)]]
                elif cue=='none' and not checkboxes:
                    cue = random.choice(options[:-1])

            elif checkboxes:
                if checkboxes=='box':
                    value_words = [[(x_str,x_im)]]
                else:
                    check_im = np.copy(blank_im)
                    x = open_im.shape[1]+random.randrange(0,x_space+1)
                    check_im[:x_im.shape[0],x:x+x_im.shape[1]] = x_im
                    value_words = [[(x_str,check_im)]]
            elif isinstance(value,str):
                value_words,value_font = self.gen_daemon.generate(value,font=value_font,ret_font=True)
                value_words = [value_words]
            else:
                #list answer
                list_values=[]
                for value_item in value:
                    value_words,value_font = self.gen_daemon.generate(value_item,font=value_font,ret_font=True)
                    list_values.append(value_words)
                value_words = list_values

            if 'to left' in cue or 'below' in cue:
                if len(value_words)>0:
                    #we switch the labels and values so that we can resuse the same layout processing
                    image_pairs.append((value_words[0],[label_words])) #switched! only allow one answer for ease of implementation
                    #  also, what would a list response look like for these?
                else:
                    cue = random.choice(options[:-3])
                    #and go back and change the previous:
                    image_pairs = [(l_w[0],[v_w]) for v_w,l_w in image_pairs]
                    image_pairs.append((label_words,value_words))
            else:
                image_pairs.append((label_words,value_words))
        
        start_x = init_x+(self.block_pad_max if len(entities)>0 else self.block_pad_max//3)
        start_x = min(start_x,max_x-60)
        if start_x<init_x:
            return False,None
        start_y = init_y+(self.block_pad_max if len(entities)>0 else self.block_pad_max//3)
        start_y = min(start_y,max_y-60)
        if start_y<init_y:
            return False,None
        title_height = self.max_text_height #retry will resample using the current height as a max
        label_height = self.max_text_height
        value_height = self.max_text_height
        num_pairs = len(pairs)

        for retry in range(5): #allow five attempts, shrinking things each time

            if start_x>init_x+self.block_pad_min:
                start_x = random.randrange(init_x+self.block_pad_min,start_x)
            if start_y>init_y+self.block_pad_min:
                start_y = random.randrange(init_y+self.block_pad_min,start_y)

            if start_x>=image.shape[1]-16 or start_y>=image.shape[0]-10:
                continue
            
            #setup text height and spacing
            if title is not None:
                title_height = random.randrange((self.min_text_height+title_height)//2,1+max(self.min_text_height+2,title_height))
            em_approx = title_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
            min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/chara  cter-design-standards/whitespace
            max_space = 0.5*em_approx
            title_space_width = round(random.random()*(max_space-min_space) + min_space)
            title_newline_height = random.randrange(1,title_height) + title_height

            if label_matches_title:
                label_height = title_height #same font, same size
            else:
                label_height = random.randrange(self.min_text_height,1+max(self.min_text_height+2,min(label_height,title_height)))
            em_approx = label_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
            min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/chara  cter-design-standards/whitespace
            max_space = 0.5*em_approx
            label_space_width = round(random.random()*(max_space-min_space) + min_space)
            label_newline_height = random.randrange(1,label_height) + label_height

            if value_matches_label:
                value_height = label_height #save font, same size
            else:
                value_height = random.randrange(self.min_text_height,1+max(self.min_text_height+2,min(value_height,title_height)))
            em_approx = value_height*1.6 #https://en.wikipedia.org/wiki/Em_(typography)
            min_space = 0.2*em_approx #https://docs.microsoft.com/en-us/typography/develop/chara  cter-design-standards/whitespace
            max_space = 0.5*em_approx
            value_space_width = round(random.random()*(max_space-min_space) + min_space)
            value_newline_height = random.randrange(1,value_height) + value_height

            if 'to left' in cue or 'below' in cue:
                #switch sizes, as we swapped the labels and values
                temp=(label_height,label_space_width,label_newline_height)
                label_height=value_height
                label_space_width=value_space_width
                label_newline_height=value_newline_height
                value_height,value_space_width,value_newline_height=temp

            value_list_newline_height = value_newline_height + round(value_newline_height*random.random())
            
            max_word_width = 0
            if title is not None:
                #get word widths
                title_word_widths = [max(1,round(img.shape[1]*(title_height/img.shape[0]))) for text,img in title_words]
                max_word_width = max(title_word_widths)
            
            new_image_pairs = []
            w_pairs = []
            max_value_word_width=max_label_word_width=0
            for label_words,value_words in image_pairs:
                if 'to left' in cue:
                    if len(value_words)>1:
                        continue
                if len(label_words)==0:
                    continue
                label_word_widths = [max(1,round(img.shape[1]*(label_height/img.shape[0]))) for text,img in label_words]
                max_label_word_width = max(max_label_word_width,*label_word_widths)
                value_word_widths = []
                for value_words_item in value_words:
                    value_word_widths_item = [max(1,round(img.shape[1]*(value_height/img.shape[0]))) for text,img in value_words_item]
                    max_word_width = max(max_word_width,*label_word_widths,*value_word_widths_item)
                    max_value_word_width = max([max_value_word_width,*value_word_widths_item])
                    value_word_widths.append(value_word_widths_item)

                new_image_pairs.append((label_words,value_words))
                w_pairs.append((label_word_widths,value_word_widths))
            image_pairs = new_image_pairs

            block_width = max_x-start_x
            if max_word_width>block_width:
                continue #retry, we need to shrink things


            #how tall and wide will the title be?
            if title is not None:
                if max(title_word_widths)>=block_width:
                    continue #not going to fit, try again with smaller size
                if int(block_width*0.66)<=max(title_word_widths):
                    max_title_width = block_width
                else:
                    max_title_width = random.randrange(max(title_word_widths),int(block_width*0.66))
                max_title_x = start_x+max_title_width
                #layout the title to see how tall it is
                title_str=''
                title_str_lines=[]
                title_img_pos_lines=[]
                title_img_pos=[]
                cur_x=start_x
                cur_y=start_y
                rightmost_title_x=cur_x
                restart=False
                for title_w, (title_text,title_img) in zip(title_word_widths,title_words):
                    if cur_x+title_w>max_title_x:
                        #newline
                        title_str_lines.append(title_str)
                        title_str=''
                        title_img_pos_lines.append(title_img_pos)
                        title_img_pos=[]
                        cur_x = start_x
                        cur_y += title_newline_height
                        if cur_y+title_height>=max_y:
                            #cannot fit
                            restart=True
                            break
                    elif len(title_str)>0:
                        title_str+=' '#space
                    title_x = cur_x
                    title_y = cur_y
                    
                    cur_x+=title_w+title_space_width
                    title_str += title_text
                    title_img_pos.append((title_x,title_y,title_img,title_height,title_w))
                    rightmost_title_x = max(rightmost_title_x,title_x+title_w)
                if restart or len(title_img_pos)==0:
                    continue #retry, not room for title
                title_img_pos_lines.append(title_img_pos)
                title_str_lines.append(title_str)
                end_title_y=cur_y+title_newline_height+round(title_newline_height*random.random())
            else:
                end_title_y=start_y
                rightmost_title_x=start_x

            if end_title_y+label_height>=max_y:
                continue #no room for fields


            can_do_side_by_side = 2+max_value_word_width+max_label_word_width+self.min_qa_sep < block_width and 'below' not in cue #room horizontally for labels and values in same line

            if ('to left' in cue or checkboxes) and not can_do_side_by_side:
                continue #to left requires side by side, try again
            side_by_side = can_do_side_by_side and (random.random()<self.side_by_side_prob or cue=='none' or 'to left' in cue or checkboxes) #labels and values will be on same line
            if side_by_side:
                aligned_cols = random.random()<0.5 or cue=='none'
                if 'to left' in cue:
                    algined_cols=True #labels will always have same x starting position. Values too
                if aligned_cols:
                    fixed_value_width = random.randrange(1+max_value_word_width,(block_width)-(max_label_word_width+1+self.min_qa_sep))
                    fixed_label_width = random.randrange(1+max_label_word_width,(block_width)-(fixed_value_width+self.min_qa_sep))
                    sep = (block_width)-(fixed_value_width+fixed_label_width)
                    max_qa_sep = self.max_qa_sep if start_y<image.shape[0]//2 else self.max_qa_sep//2
                    if sep>self.min_qa_sep:
                        sep = random.randrange(self.min_qa_sep,min(sep,self.max_qa_sep))
                    all_value_x = start_x+fixed_label_width+sep
                    max_label_x = start_x+fixed_label_width
                else:
                    if (block_width)-(max_label_word_width+max_value_word_width)<self.min_qa_sep:
                        continue #restart, too narrow
                    sep = random.randrange(self.min_qa_sep,min(self.max_qa_sep//2+1,1+(block_width)-(max_label_word_width+max_value_word_width)))
                    max_label_x = max_x-(max_value_word_width+sep+1)
                    all_value_x=None
            else:
                #the label will be above the value
                aligned_cols = False
                try:
                    sep=random.randrange(max(label_height,value_height)*3,min(max(label_height,value_height)*10,max_x-max(max_value_word_width,max_label_word_width)))//2
                except ValueError:
                    sep=max_x #can't do columns
                max_label_x=max_x-1
                all_value_x=None

            cur_x = start_x
            if title is not None:
                title_left = random.random()<0.5
                if title_left and random.random()<0.5:
                    #have title left of items
                    cur_x = start_x + random.randrange(0,min(rightmost_title_x+title_height*3,max_x-max_word_width-2))
            col_x = cur_x #where the column starts
            col_number = 0 #which column are we on
            cur_y = end_title_y
            rightmost_x_so_far = cur_x
            label_max_x = {}
            rightmost_value_x=defaultdict(int)
            pairs_to_draw = []
            num_pairs_to_draw_in_col =0 

            #This returns whether we have room horizontally to makea new column
            def roomForNewCol(col_x,rightmost_x_so_far):
                room_for_new_col = (aligned_cols and col_x+sep+2*(fixed_label_width+sep+fixed_value_width)<max_x) or (not aligned_cols and col_x+2*sep+rightmost_x_so_far+max_label_word_width+sep+max_value_word_width<max_x)
                return room_for_new_col

            #Some things for starting a new column
            def shiftCol(col_number,col_x,all_value_x,max_label_x,cur_y,rightmost_x_so_far):
                if aligned_cols:
                    col_x += fixed_label_width+2*sep+fixed_value_width
                    all_value_x += fixed_label_width+2*sep+fixed_value_width
                    max_label_x += fixed_label_width+2*sep+fixed_value_width
                    cur_y = end_title_y
                else:
                    col_x = rightmost_x_so_far+2*sep
                    cur_y = end_title_y
                num_pairs_to_draw_in_col=0
                return col_number+1,col_x, all_value_x, max_label_x,cur_y,num_pairs_to_draw_in_col

            for (label_words,value_words),(label_word_widths_list,value_word_widths_list) in zip(image_pairs,w_pairs):
                if 'to left' in cue and len(pairs_to_draw)>0:
                    #reset cur_y to correct position as "value" may have finished above "label"
                    #This is comparing the last line of the label lines
                    cur_y = max(cur_y,pairs_to_draw[-1][1][-1][-1][1]+label_height+label_newline_height)
                
                #Should we make a new column?
                if checkboxes:
                    change_new_col = random.random()<self.new_col_chance_checkbox*min(1,num_pairs_to_draw_in_col/3)
                else:
                    change_new_col = random.random()<self.new_col_chance*min(1,num_pairs_to_draw_in_col/3)
                if roomForNewCol(col_x,rightmost_x_so_far) and (cur_y+label_height>=max_y or change_new_col):
                    #we are making a new column
                    col_number,col_x, all_value_x, max_label_x,cur_y,num_pairs_to_draw_in_col=shiftCol(col_number,col_x,all_value_x,max_label_x,cur_y,rightmost_x_so_far)
                elif cur_y+label_height>=max_y:
                    break #cannot do pair, not enough room vertically. Stop putting in pairs

                cannot_do_pair = False
                restart=True
                catch_inf_loop=0
                while restart and catch_inf_loop<200: #for restarting as new column
                    catch_inf_loop+=1
                    cur_x = col_x
                    restart=False

                    #position label words
                    label_str='' #running words in line
                    label_str_lines=[] #holds all the line strings
                    label_img_pos=[] #holds image and it's position for each word in current line
                    label_img_pos_lines=[] #holds all the lines
                    label_max_x[col_number] = cur_x
                    for label_w,(label_text,label_img) in zip(label_word_widths_list,label_words):
                        if cur_x+label_w>max_label_x: #not enough room horizontally
                            if len(label_img_pos)==0:
                                #if we haven't added any words, making a newline won't help
                                cannot_do_pair=True
                                break
                            
                            #makde a newline
                            label_str_lines.append(label_str)
                            label_str=''
                            label_img_pos_lines.append(label_img_pos)
                            label_img_pos=[]
                            cur_x = col_x
                            cur_y += label_newline_height
                            if cur_y+label_height>=max_y:
                                #Not enough room vertically for new line
                                #do we have room for another column?
                                if roomForNewCol(col_x,rightmost_x_so_far):
                                    col_number,col_x, all_value_x, max_label_x,cur_y,num_pairs_to_draw_in_col=shiftCol(col_number,col_x,all_value_x,max_label_x,cur_y,rightmost_x_so_far)
                                    restart = True
                                    #restart on new column
                                    break
                                else:
                                    cannot_do_pair=True
                                    break
                        elif len(label_str)>0:
                            #normally adding word to the right
                            label_str+=' '#space
                        label_x = cur_x
                        label_y = cur_y
                        
                        cur_x+=label_w+label_space_width #adjust x position
                        label_str += label_text #add word text 
                        label_img_pos.append((label_x,label_y,label_img,label_height,label_w))
                        if label_x+label_w>max_x or label_y+label_height>max_y:
                            #we went off the image
                            cannot_do_pair=True
                            break

                        rightmost_x_so_far = max(rightmost_x_so_far,label_x+label_w)
                        label_max_x[col_number] = max(label_max_x[col_number],cur_x)

                    if cannot_do_pair:
                        break
                    
                    if len(label_img_pos)>0:
                        #add current line
                        label_str_lines.append(label_str)
                        label_img_pos_lines.append(label_img_pos)

                    #get starting position for value
                    if side_by_side:
                        if aligned_cols:
                            value_start_x = all_value_x
                        else:
                            value_start_x = label_max_x[col_number]+sep

                        lowest_y = max_y-(value_height+1)
                        if 'to left' in cue:
                            label_y = label_img_pos_lines[0][0][1] #first line instead of last
                            cur_y = label_y 
                        else:
                            #somewhere from level to below label y position
                            cur_y = random.randrange(min(label_y-round(0.15*min(label_height,value_height)),label_y+label_height-value_height)-4,min(max(label_y+value_height,label_y+label_height)+4,lowest_y))
                    else:
                        value_start_x = col_x + random.randrange(-4,label_height) #roughly beggining of column x
                        if 'below' in cue:
                            #has more signfificant verticle spacing
                            cur_y = label_y+round(label_height+5+label_height*random.random()*0.5)
                        else:
                            cur_y = label_y+round(label_newline_height+label_newline_height*random.random())


                    max_value_x = max_x
                    value_entities=[]
                    for value_word_widths_item,value_words_item in zip(value_word_widths_list,value_words):
                        if cur_y+value_height>=max_y:
                            #do we have room for another column?
                            if roomForNewCol(col_x,rightmost_x_so_far):
                                col_number,col_x, all_value_x, max_label_x,cur_y,num_pairs_to_draw_in_col=shiftCol(col_number,col_x,all_value_x,max_label_x,cur_y,rightmost_x_so_far)
                                restart = True
                                break
                            else:
                                cannot_do_pair=True
                                break
                        cur_x = value_start_x
                        value_str=''
                        value_str_lines_item=[]
                        value_img_pos_lines_item=[]
                        value_img_pos = []
                        for value_w,(value_text,value_img) in zip(value_word_widths_item,value_words_item):

                            
                            if cur_x+value_w>=max_value_x:
                                if cur_x == value_start_x or value_start_x+value_w>=max_value_x or len(value_img_pos)==0:
                                    cannot_do_pair=True
                                    break
                                #newline
                                value_str_lines_item.append(value_str)
                                value_str=''
                                value_img_pos_lines_item.append(value_img_pos)
                                value_img_pos=[]
                                cur_x = value_start_x
                                cur_y += value_newline_height

                                if cur_y+value_height>=max_y:
                                    #do we have room for another column?
                                    if roomForNewCol(col_x,rightmost_x_so_far):
                                        col_number,col_x, all_value_x, max_label_x,cur_y,num_pairs_to_draw_in_col=shiftCol(col_number,col_x,all_value_x,max_label_x,cur_y,rightmost_x_so_far)
                                        restart = True
                                        break
                                    else:
                                        cannot_do_pair=True
                                        #import pdb;pdb.set_trace()
                                        break
                            elif len(value_str)>0:
                                value_str+=' '#space
                            value_x = cur_x
                            value_y = cur_y
                            
                            cur_x+=value_w+value_space_width
                            if value_text is not None:
                                value_str += value_text
                            else:
                                value_str = None
                            value_img_pos.append((value_x,value_y,value_img,value_height,value_w))
                            if value_x+value_w>max_x or value_y+value_height>max_y:
                                cannot_do_pair=True
                                break

                            rightmost_x_so_far = max(rightmost_x_so_far,value_x+value_w)

                        if cannot_do_pair or restart:
                            break
                        if len(value_img_pos)>0:
                            value_str_lines_item.append(value_str)
                            value_img_pos_lines_item.append(value_img_pos)
                        if len(value_str_lines_item)>0:
                            value_entities.append((value_str_lines_item,value_img_pos_lines_item))
                        for value_img_pos in value_img_pos_lines_item:
                            rightmost_value_x[col_number]= max(value_img_pos[-1][0]+value_img_pos[-1][-1],rightmost_value_x[col_number])

                        cur_y += value_list_newline_height


                    if cannot_do_pair or restart:
                        break

                    #else add the info to be drawn
                    pairs_to_draw.append((label_str_lines,label_img_pos_lines,value_entities,col_number))
                    num_pairs_to_draw_in_col+=1

                    cur_y += round(label_newline_height*random.random()) + (value_list_newline_height if len(value_words)==0 else 0) +3

                        
                if cannot_do_pair:
                    break

                
            if len(pairs_to_draw)==0:
                #couldn't add this document
                continue #retry



            #place the header
            actual_block_width = max(rightmost_x_so_far,rightmost_title_x)-start_x
            title_width = rightmost_title_x-start_x
            wiggle = min(100,actual_block_width-title_width) #how much horizontal room we have to place the title in
            if title is not None:
                if title_left:
                    #left
                    if wiggle>0:
                        title_x_offset = random.randrange(wiggle)
                    else:
                        title_x_offset=0
                else:
                    #middle
                    title_x_offset = (actual_block_width//2)-(title_width//2) + (random.randrange(-wiggle//2,wiggle//2) if wiggle>0 else 0)
                rightmost_title_x+=title_x_offset
                #update title position
                for title_img_pos in title_img_pos_lines:
                    for i in range(len(title_img_pos)):
                        x,y,img,h,w = title_img_pos[i]
                        title_img_pos[i] = (x+title_x_offset,y,img,h,w)

            #draw the word images and add the entities + links
            lowest_y=0
            rightmost_x=0
            if title is not None:
                title_entity = self.makeAndDrawEntity(image,'header',title_str_lines,title_img_pos_lines)
                title_ei = len(entities)
                entities.append(title_entity)
                lowest_y = title_entity.box[-1]
                rightmost_x = title_entity.box[-2]

            for label_str_lines,label_img_pos_lines,value_entities,col_number in pairs_to_draw:
                if 'to left' in cue or 'below' in cue:
                    #switch back!
                    value_str_lines = label_str_lines
                    value_img_pos_lines = label_img_pos_lines
                    if len(value_entities)==0:
                        continue #something's wrong and we don't have a label anymore. Skip this one
                    label_str_lines,label_img_pos_lines = value_entities[0]
                    value_entities[0]=(value_str_lines,value_img_pos_lines)
                
                label_entity = self.makeAndDrawEntity(image,'question',label_str_lines,label_img_pos_lines)
                if isinstance(label_entity,tuple):
                    #it's blank
                    continue

                label_ei=len(entities)
                entities.append(label_entity)
                if title is not None:
                    entity_link[title_ei].append(label_ei)

                lowest_y = max(lowest_y,label_entity.box[-1])
                rightmost_x = max(rightmost_x,label_entity.box[-2])
                
                if not side_by_side:
                    rightmost_value_x[col_number] = max(rightmost_value_x[col_number],label_max_x[col_number])

                for value_str_lines,value_img_pos_lines in value_entities:
                    value_entity = self.makeAndDrawEntity(image,'answer',value_str_lines,value_img_pos_lines,rightmost_value_x[col_number],cue,side_by_side,checkboxes)
                    if isinstance(value_entity,tuple):
                        #it's blank (returns position information)
                        rightmost_x = max(rightmost_x,value_entity[0])
                        lowest_y = max(lowest_y,value_entity[1])
                    else:
                        value_ei=len(entities)
                        entities.append(value_entity)
                        entity_link[label_ei].append(value_ei)
                        lowest_y = max(lowest_y,value_entity.box[-1])
                        rightmost_x = max(rightmost_x,value_entity.box[-2])

                
                if len(value_entities)==0: #blank
                    entity_link[label_ei]=None
                    #draw empty line
                    line_thickness = random.randrange(1,5)
                    pad_w = random.randrange(1,5)
                    pad_h = random.randrange(1,5)
                    color = random.randrange(1,170)
                    dotting = random.randrange(1,5)
                    if side_by_side:
                        if aligned_cols:
                            x = all_value_x
                        else:
                            x = label_img_pos_lines[-1][-1][0]+label_img_pos_lines[-1][-1][-1]+sep
                        y = label_img_pos_lines[-1][-1][1]
                        if rightmost_value_x[col_number]==0:
                            rightmost_value_x[col_number] = x+6*value_height
                            rightmost_x_so_far = max(rightmost_x_so_far,rightmost_value_x[col_number])
                    else:
                        x = col_x+random.randrange(-4,label_height)
                        y = label_img_pos_lines[-1][-1][1] + label_newline_height + round(label_newline_height*random.random())
                        if rightmost_value_x[col_number]==0:
                            rightmost_value_x[col_number] = rightmost_x_so_far

                    img_h = label_img_pos_lines[-1][-1][-2]
                    if 'dotted line' in cue:
                        for x in range(max(0,x-pad_w),min(max_x,rightmost_value_x[col_number]+pad_w)):
                            if math.sin(x*math.pi/dotting)>0:
                                try:
                                    image[y+img_h+pad_h-line_thickness//2:1+y+img_h+pad_h+line_thickness//2,x]=color
                                except IndexError:
                                    pass

                    elif 'line' in cue:
                        img_f.line(image,(x-pad_w,y+img_h+pad_h),(rightmost_value_x[col_number]+pad_w,y+img_h+pad_h),color,line_thickness)
                    elif 'box' in cue:
                        img_f.rectangle(image,(x-pad_w,y-pad_h),(rightmost_value_x[col_number]+pad_w,y+img_h+pad_h),color,line_thickness)
                    lowest_y = max(lowest_y,y+img_h+pad_h)
                elif label_ei not in entity_link:
                    entity_link[label_ei]=None

            if rightmost_x<init_x:
                continue

            #properly finished
            return True,(init_x,init_y,rightmost_x,lowest_y)

        return False,None

    #Addes the word images to the document image, along with lines, boxes, etc.
    #Makes the Entity object and returns it
    def makeAndDrawEntity(self,image,cls,str_lines,img_pos_lines,max_line_x=None,cue=None,side_by_side=False,checkboxes=False):
        lines=[]
        if checkboxes:
            blank = checkboxes=='box' and random.random()<self.checkbox_blank_value_prob
        else:
            blank = cue is not None and ('to left' in cue or 'below' in cue) and random.random()<self.blank_value_prob
        if cue is not None and any(prompt in cue for prompt in ['box','line']):
            if random.random()<0.2 and 'to left' not in cue and not checkboxes:
                line_end_x = max_line_x
            elif (random.random()<0.8 or 'below' in cue) and not checkboxes:
                line_end_x = 0
                for img_pos_words in img_pos_lines:
                    line_max_x=img_pos_words[-1][0]+img_pos_words[-1][-1]
                    line_end_x=max(line_end_x,line_max_x)
            else:
                line_end_x=None
        else:
            line_end_x=None
        max_x=max_y = 0
        min_x=min_y = 9999999999999999999
        if cue is not None:
            pad_w = random.randrange(1,5)
            pad_h = random.randrange(1,5)
            color = random.randrange(100,255)
        lines_to_draw=[]
        heights = []
        for i,(text,img_pos_words) in enumerate(zip(str_lines,img_pos_lines)):
            line_max_x=img_pos_words[-1][0]+img_pos_words[-1][-1]
            line_max_y=img_pos_words[-1][1]+img_pos_words[-1][-2]
            line_min_x=img_pos_words[0][0]
            line_min_y=img_pos_words[0][1]

            heights.append(line_max_y-line_min_y)
            
            if not blank:
                for x,y,img,img_h,img_w in img_pos_words:
                    if x<0 or y<0 or x>=image.shape[1] or y>=image.shape[0]:
                        return (max_x,max_y)
                    img = img_f.resize(img,(img_h,img_w))
                    if x+img_w>image.shape[1]:
                        img = img[:,:image.shape[1]-(x+img_w)]
                        img_w = img.shape[1]
                    if y+img_h>image.shape[0]:
                        img = img[:image.shape[0]-(y+img_h),:]
                        img_h = img.shape[0]

                    image[y:y+img_h,x:x+img_w] = img

            if text is not None:
                lines.append(Line(text,(line_min_x,line_min_y,line_max_x,line_max_y)))


            if cue is not None:
                if line_end_x is not None:
                    line_max_x = line_end_x
                dotting = random.randrange(1,5)
                if 'line' in cue:
                    lines_to_draw.append((max(0,line_min_x-pad_w),min(max_line_x,line_max_x+pad_w),line_max_y+pad_h))
                elif 'box' in cue:
                    min_x = min(line_min_x-pad_w,min_x)
                    min_y = min(line_min_y-pad_h,min_y)
                    max_x = max(line_max_x+pad_w,max_x)
                    max_y = max(line_max_y+pad_h,max_y)

        if cue is not None:
            #print('line th range ({}):  {} , {}'.format(np.mean(heights),1,
            line_thickness = random.randrange(1,min(5,max(2,math.ceil(.25*np.mean(heights)))))

        if cue is not None and 'box' in cue:
            max_line_x = min(max_line_x,max_x)
            if checkboxes:
                #make it square
                h = max_y-min_y
                w = max_line_x-min_x
                dim = max(h,w)
                if h<dim:
                    diff = dim-h
                    top = random.randrange(0,diff+1)
                    bot = diff-top
                    min_y-=top
                    max_y+=bot
                elif w<dim:
                    diff = dim-w
                    left = random.randrange(0,1+diff//2)
                    right = diff-left
                    min_x-=left
                    max_line_x+=right
            img_f.rectangle(image,(min_x,min_y),(max_line_x,max_y),color,line_thickness)
        elif cue is not None and 'dotted line' in cue:
            if 'below' in cue or random.random()<0.1:
                lines_to_draw = lines_to_draw[-1:] #only do last line
            for x1,x2,y in lines_to_draw:
                for x in range(x1,x2):
                    if math.sin(x*math.pi/dotting)>0:
                        try:
                            image[y-line_thickness//2:1+y+line_thickness//2,x]=color
                        except IndexError:
                            pass

        elif cue is not None and 'line' in cue:
            if 'below' in cue or random.random()<0.1:
                lines_to_draw = lines_to_draw[-1:] #only do last line
            for x1,x2,y in lines_to_draw:
                img_f.line(image,(x1,y),(x2,y),color,line_thickness)
        return Entity(cls,lines) if not blank and len(lines)>0 else (max_x,max_y)
    
    #selects a formatting and gets random number (or word)
    def getTableValues(self,num):
        ret=[]
        for n in range(num):
            if random.random()<0.5:
                r = random.random()
                #number
                if r<0.1:
                    #percent
                    ret.append('{}%'.format(random.randrange(101)))
                elif r<0.2:
                    #percent
                    ret.append('{:.4}%'.format(random.random()*100))
                elif r<0.3:
                    ret.append('{}'.format(random.randrange(10000)))
                elif r<0.4:
                    ret.append('{:.4}'.format(random.random()*100))
                elif r<0.5:
                    ret.append('{}'.format(random.randrange(-1000,1000)))
                elif r<0.6:
                    ret.append('{:.3}'.format(random.random()))
                elif r<0.7:
                    ret.append('-{:.3}'.format(random.random()))
                elif r<0.8:
                    ret.append('${}'.format(random.randrange(10000)))
                elif r<0.9:
                    ret.append('${}'.format(random.randrange(1000)))
                else:
                    ret.append('{}'.format(random.randrange(101)))
            else:
                #words
                if len(self.random_words)==0:
                    self.addRandomWords()
                ret.append(self.random_words.pop())
        return ret


    #build table header from random words
    def getTableHeaders(self,num):
        ret=[]
        for n in range(num):
            #words
            if len(self.random_words)==0:
                self.addRandomWords()
            if len(self.random_words)>4 and random.random()<0.02:
                ret.append(' '.join(self.random_words[-4:]))
                self.random_words = self.random_words[:-4]
            elif len(self.random_words)>3 and random.random()<0.07:
                ret.append(' '.join(self.random_words[-3:]))
                self.random_words = self.random_words[:-3]
            elif len(self.random_words)>2 and random.random()<0.2:
                ret.append(' '.join(self.random_words[-2:]))
                self.random_words = self.random_words[:-2]
            else:
                ret.append(self.random_words.pop())
        return ret


    #get random non-stop words from Wikipedia
    def addRandomWords(self):
        prevent_inf=0
        while (len(self.random_words)==0 or prevent_inf==0) and prevent_inf<200:
            prevent_inf+=1
            words = self.gen_daemon.getTextSample()
            words = [w for w in words if w.lower() not in self.stop_words]
            random.shuffle(words)
            self.random_words+=words

        if prevent_inf==200:
            self.random_words+=['X']
