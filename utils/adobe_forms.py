import json
from utils import img_f
import numpy as np
import os,sys
from collections import defaultdict

INSIDE_THRESH=0.92

def parseGT(directory,name,scale):
    with open(os.path.join(directory,'jsons',name+'.json')) as f:
        data=json.load(f)


    all_bbs=[]
    x1s=[]
    x2s=[]
    y1s=[]
    y2s=[]
    for section,ldata in data.items():
        for i,bb in enumerate(ldata):
            x=round(bb['x'])
            y=round(bb['y'])
            w=round(bb['w'])
            h=round(bb['h'])
            t=bb['jsonClass']

            x2=x+w
            y2=y+h

            all_bbs.append((section,i))
            x1s.append(x)
            x2s.append(x2)
            y1s.append(y)
            y2s.append(y2)

    x1s=np.array(x1s)
    x2s=np.array(x2s)
    y1s=np.array(y1s)
    y2s=np.array(y2s)

    inter_rect_x1 = np.maximum(x1s[:,None],x1s[None,:])
    inter_rect_x2 = np.minimum(x2s[:,None],x2s[None,:])
    inter_rect_y1 = np.maximum(y1s[:,None],y1s[None,:])
    inter_rect_y2 = np.minimum(y2s[:,None],y2s[None,:])

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0,a_max=None) * np.clip(
            inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=None)
    areas = (x2s-x1s+1)*(y2s-y1s+1)

    inside_amounts = (inter_area/areas[None,:])
    insides = inside_amounts>INSIDE_THRESH


    #top to bottom
    #first choice groups, tables and lists
    #fields, choice fields
    #be sure any nested fields, etc are processed before their parent so they properly claim their childrem
    hier_level = {
            "ChoiceGroup":0,
            "Table":0,
            "ChoiceField":1,
            "Field":2,
            "List":0.5,
            "SectionTitle":4,
            "HeaderTitle":4,
            "ChoiceGroupTitle":4,
            "TableTitle":4,
            "TextBlock":5,
            "TextRun":6,
            "Widget":6
            }

    belongs_to = {}
    owns = defaultdict(list)
    bbs=[]

    done=set()
    for json_type in ["ChoiceGroup","Table","ChoiceField","Field","List","SectionTitle","HeaderTitle","ChoiceGroupTitle","TableTitle","TextBlock"]:#,"Widget","TextRun"]:
        if json_type in data:
            print(json_type)
            for i,bb in enumerate(data[json_type]):
                bb_i = all_bbs.index((json_type,i))
                x=round(bb['x'])
                y=round(bb['y'])
                w=round(bb['w'])
                h=round(bb['h'])
                t=bb['jsonClass']

                x2=x+w
                y2=y+h


                #hierarchy
                inside_me = np.nonzero(insides[bb_i])[0]
                already_owned=[]
                for i_bb_i in inside_me:
                    if bb_i!=i_bb_i:
                        if i_bb_i in belongs_to:
                            already_owned.append(i_bb_i)
                        else:
                            belongs_to[i_bb_i]=bb_i
                            owns[bb_i].append(i_bb_i)
                for i_bb_i in already_owned:
                    other_owner = belongs_to[i_bb_i]
                    if other_owner in belongs_to and belongs_to[other_owner] == bb_i:
                        #The other owner is a sub part of me, it should own it
                        pass
                    elif bb_i in belongs_to and belongs_to[bb_i] == other_owner:
                        #the other owner owns me, so I should own i_bb_i (it's lower in heir than me)
                        belongs_to[i_bb_i]=bb_i
                        owns[bb_i].append(i_bb_i)
                        owns[other_owner].remove(i_bb_i)
                    elif hier_level[json_type] > hier_level[all_bbs[i_bb_i][0]]:
                        #I can't claim this
                        pass
                    else:
                        print('this is wrong. i_bb_i claimed by two different fields')
                        print('current: {} {} l{}'.format(bb_i,all_bbs[bb_i],hier_level[json_type]))
                        print('I want to claim: {} {} l{}'.format(i_bb_i,all_bbs[i_bb_i],hier_level[all_bbs[i_bb_i][0]]))
                        print('But its claimed by: {} {} l{}'.format(other_owner,all_bbs[other_owner],hier_level[all_bbs[other_owner][0]]))
                        if bb_i in belongs_to:
                            print('I am owned by {} {}'.format(belongs_to[bb_i], all_bbs[belongs_to[bb_i]]))
                        print('me to want: {}, want to me: {}'.format(insides[bb_i,i_bb_i],insides[i_bb_i,bb_i]))
                        print('other to want: {}, want to other: {}'.format(insides[other_owner,i_bb_i],insides[i_bb_i,other_owner]))

                        a=[x for x in range(100)]
                        import pdb;pdb.set_trace()
                        a=[x for x in range(100)]
    
    print('Hierarchy!')
    def printRec(bb_i,depth):
        print(('\t'*depth)+'{} {} ({},{})'.format(all_bbs[bb_i][0],bb_i,x1s[bb_i],y1s[bb_i]))
        for o_i in owns[bb_i]:
            printRec(o_i,depth+1)
    for bb_i in range(len(all_bbs)):
        if bb_i not in belongs_to:
            printRec(bb_i,0)
    ##I think these are the only things needing grouped
    #A widget in a text block is fill-in-the-blank!!!!
    #groups=defaultdict(list)
    #if 'TextBlock' in data:
    #    for i,bb in enumerate(data['TextBlock']):
    #        bb_i = all_bbs.index(('TextBlock',i))
    #        for other_bb_i in owns[bb_i]:
    #            ssert(all_bbs[other_bb_i][0]=='TextRun') #could be widget
    #            groups[bb_i].append(other_bb_i)
    if True:
        #linking
        def linkTextBlock(bb_i,pairs):
            if len(owns[bb_i])==1:
                return owns[bb_i][0], owns[bb_i][0]
            #get read order, by y first
            #first get average height, so we know if things are on a new line
            avg_h=0
            with_y=[]
            for o_bb_i in owns[bb_i]:
                avg_h+=y2s[o_bb_i]-y1s[o_bb_i]
                with_y.append((o_bb_i,(y2s[o_bb_i]+y1s[o_bb_i])/2))
            avg_h/=len(owns[bb_i])
            with_y.sort(key=lambda a:a[1])
            lines=[]
            cur_line=[(with_y[0][0],x1s[with_y[0][0]])]
            cur_y=with_y[0][1]
            for o_bb_i,y in with_y[1:]:
                if y-cur_y>avg_h*0.6:
                    lines.append(cur_line)
                    cur_line=[]
                cur_line.append((o_bb_i,x1s[o_bb_i]))
                cur_y=y
            lines.append(cur_line)

            prev=None
            for line in lines:
                line.sort(key=lambda a:a[1])
                if prev is not None:
                    pairs.add((min(prev,line[0][0]),max(prev,line[0][0])))
                prev=line[0][0]
                for o_bb_i,x in line[1:]:
                    pairs.add((min(prev,o_bb_i),max(prev,o_bb_i)))
                    prev = o_bb_i
            return lines[0][0][0],prev   

        def linkField(bb_i,pairs,checkbox=False):
            #assert len(owns[bb_i])==2
            text_i=None
            widget_is=[]
            for o_bb_i in owns[bb_i]:
                if all_bbs[o_bb_i][0]=='TextBlock':
                    assert(text_i is None)
                    text_i = o_bb_i
                elif all_bbs[o_bb_i][0]=='Widget':
                    widget_is.append(o_bb_i)
            
            #assert(text_i is not None)
            if text_i is None:
                #it probably is a mislabel. See if it's just off by too much
                for o_bb_i in range(len(all_bbs)):
                    if o_bb_i not in belongs_to:
                        json_type = all_bbs[o_bb_i][0]
                        if json_type=='TextBlock' and inside_amounts[bb_i,o_bb_i]>0.6:
                            text_i = o_bb_i
                            print('Reclaimed TextBlock {} for Field {}'.format(text_i,bb_i))
                            break
            assert(text_i is not None)
            text_top, text_bot = linkTextBlock(text_i,pairs)
            if checkbox:
                assert(len(widget_is)<=2)

                if len(widget_is)==2:
                    #which widget is the check box? Probably the smaller one that is to the left
                    for widget_i in widget_is:
                        assert(len(owns[widget_i])==0)
                    area0 = (x2s[widget_is[0]]-x1s[widget_is[0]])*(y2s[widget_is[0]]-y1s[widget_is[0]])
                    area1 = (x2s[widget_is[1]]-x1s[widget_is[1]])*(y2s[widget_is[1]]-y1s[widget_is[1]])

                    pairs.add((min(widget_is[0],text_i),max(widget_is[0],text_i)))
                    pairs.add((min(widget_is[1],text_i),max(widget_is[1],text_i)))

                    if area0<area1:
                        assert(x1s[widget_is[0]]<x1s[widget_is[1]])
                        return widget_is[0]
                    elif area1<area0:
                        assert(x1s[widget_is[1]]<x1s[widget_is[0]])
                        return widget_is[1]
                    else:
                        assert False
                else:
                    pairs.add((min(widget_is[0],text_i),max(widget_is[0],text_i)))
                    return widget_is[0]
            else:
                assert(len(widget_is)==1)
                assert(len(owns[widget_is[0]])==0)
                pairs.add((min(text_bot,widget_is[0]),max(text_bot,widget_is[0])))
                return text_top

        def linkChoiceField(bb_i,pairs):
            assert len(owns[bb_i])==1 and all_bbs[owns[bb_i][0]][0]=='Field'
            return linkField(owns[bb_i][0],pairs,True)
        def linkChoiceGroup(bb_i,pairs):
            title_i = None
            to_link_pre=[]
            for other_bb_i in owns[bb_i]:
                if 'ChoiceGroupTitle' == all_bbs[other_bb_i][0]:
                    assert title_i is None
                    title_i = other_bb_i
                elif len(owns[other_bb_i])>0:
                    to_link_pre.append(other_bb_i)
                else:
                    print('unknown in choice group {} {}'.format(other_bb_i,all_bbs[other_bb_i]))
                    import pdb;pdb.set_trace() #not sure what this is

            to_link_to_ChoiceField=[]
            to_link_post=[]
            for other_bb_i in to_link_pre:
                if 'ChoiceField' == all_bbs[other_bb_i][0]:
                    to_link_bb_i = linkChoiceField(other_bb_i,pairs)
                    to_link_post.append(to_link_bb_i)
                elif 'ChoiceGroup' == all_bbs[other_bb_i][0]:
                    to_link_bb_is = linkChoiceGroup(other_bb_i,pairs)
                    #but these probably need linked to a ChoiceField...
                    #closest to top-and-left?
                    to_link_to_ChoiceField.append(to_link_bb_is)
                else:
                    print('Error, {} in a ChoiceGroup'.format( all_bbs[other_bb_i][0]))
                    assert False

            for bb_is in to_link_to_ChoiceField:
                xs=[]
                ys=[]
                for bb_i in bb_is:
                    xs.append(x1s[bb_i])
                    ys.append(y1s[bb_i])
                xs=np.array(xs)
                ys=np.array(ys)
                xg = xs.min()
                yg = ys.min()

                min_dist=9999999
                best_bb_i=None
                for bb_i in to_link_post:
                    x = x1s[bb_i]
                    y = y1s[bb_i]
                    if y<=yg or x<=xg:
                        dist = abs(y-yg)+abs(x-xg)
                        if dist<min_dist:
                            min_dist=dist
                            best_bb_i = bb_i

                for bb_i in bb_is:
                    pairs.add((min(bb_i,best_bb_i),max(bb_i,best_bb_i)))

            if title_i is not None:
                for to_link_bb_i in to_link_post:
                    pairs.add((min(title_i,to_link_bb_i),max(title_i,to_link_bb_i)))
                return [title_i]
            else:
                return to_link_post

        def linkList(bb_i,pairing):
            prev=owns[bb_i][0]
            assert(all_bbs[prev][0]=='TextBlock')
            for o_bb_i in owns[bb_i][1:]:
                pairs.add((min(prev,o_bb_i),max(prev,o_bb_i)))
                prev=o_bb_i
                assert(all_bbs[prev][0]=='TextBlock')


        pairs=set()
        #if 'ChoiceGroup' in data:
        #    for i,bb in enumerate(data['ChoiceGroup']):
        #        bb_i = all_bbs.index(('ChoiceGroup',i))
        for bb_i in range(len(all_bbs)):
            if bb_i not in belongs_to:
                json_type = all_bbs[bb_i][0]
                if json_type=='ChoiceGroup':
                    linkChoiceGroup(bb_i,pairs)
                elif json_type=='Field':
                    linkField(bb_i,pairs)
                elif json_type=='TextBlock':
                    linkTextBlock(bb_i,pairs)
                elif json_type=='Table':
                    linkTable(bb_i,pairs)
                elif json_type=='List':
                    linkList(bb_i,pairs)

            

            
        #for json_type in ["Widget","TextRun"]:
        #    if json_type in data:
        #        print(json_type)
        #        for i,bb in enumerate(data[json_type]):
        #            bb_i = all_bbs.index((json_type,i))
        #            x=round(bb['x'])
        #            y=round(bb['y'])
        #            w=round(bb['w'])
        #            h=round(bb['h'])
        #            t=bb['jsonClass']

        #            x2=x+w
        #            y2=y+h

        #            #linking and grouping

        #            #bounding box
        #            lXL=x
        #            rXL=x2
        #            tYL=y
        #            bYL=y2
        #            s=scale
        #            bb = np.empty(8+8+numClasses, dtype=np.float32)
        #            bb[0]=lXL*s
        #            bb[1]=tYL*s
        #            bb[2]=rXL*s
        #            bb[3]=tYL*s
        #            bb[4]=rXL*s
        #            bb[5]=bYL*s
        #            bb[6]=lXL*s
        #            bb[7]=bYL*s
        #            #we add these for conveince to crop BBs within window
        #            bb[8]=s*lXL
        #            bb[9]=s*(tYL+bYL)/2.0
        #            bb[10]=s*rXL
        #            bb[11]=s*(tYL+bYL)/2.0
        #            bb[12]=s*(lXL+rXL)/2.0
        #            bb[13]=s*tYL
        #            bb[14]=s*(rXL+lXL)/2.0
        #            bb[15]=s*bYL

        #            bb[16:]=0
        #            bb[classMap[label]]=1
        #            bbs.append(bb)


    if True:
        image = img_f.imread(os.path.join(directory,'imgs',name+'.png'))
        if len(image.shape)==2:
            image= np.stack([image,image,image],axis=2)
        print(list(data.keys()))

        for section,ldata in data.items():
            for bb in ldata:
                x=round(bb['x'])
                y=round(bb['y'])
                w=round(bb['w'])
                h=round(bb['h'])
                t=bb['jsonClass']

                x2=x+w
                y2=y+h #ChoiceGroup,Table,Field,ChoiceField,List,TextBlock,SectionTitle,HeaderTitle,ChoiceGroupTitle,TableTitle,Widget,TextRun

                if t=='Field':
                    img_f.rectangle(image,(x,y),(x2,y2),(255,0,0),2)
                elif t=='Widget':
                    image[y:y2,x+1:x2,1]=0
                elif t=='TextRun':
                    image[y+1:y2,x+1:x2,2]=0
                elif t=='SectionTitle':
                    img_f.rectangle(image,(x,y),(x2,y2),(255,0,255),2)
                elif t=='TextBlock':
                    img_f.rectangle(image,(x,y),(x2,y2),(0,255,255),2)
                elif t=='HeaderTitle':
                    img_f.rectangle(image,(x,y),(x2,y2),(0,0,255),2)
                elif t=='List':
                    img_f.rectangle(image,(x,y),(x2,y2),(0,255,0),2)
                elif t=='ChoiceGroup':
                    img_f.rectangle(image,(x,y),(x2,y2),(0,255,0),2)
                elif t=='ChoiceField':
                    img_f.rectangle(image,(x,y),(x2,y2),(255,255,0),2)
                elif t=='ChoiceGroupTitle':
                    img_f.rectangle(image,(x,y),(x2,y2),(255,150,150),2)
                elif t=='TableTitle':
                    img_f.rectangle(image,(x,y),(x2,y2),(150,255,150),2)
                elif t=='Table':
                    img_f.rectangle(image,(x,y),(x2,y2),(150,150,255),2)

                else:
                    print('UNKNOWN TYPE: {}'.format(t))
                    img_f.rectangle(image,(x,y),(x2,y2),(0,100,100),2)

        for bb_i1,bb_i2 in pairs:
            x1c = int(round((x1s[bb_i1]+x2s[bb_i1])/2))
            y1c = int(round((y1s[bb_i1]+y2s[bb_i1])/2))
            x2c = int(round((x1s[bb_i2]+x2s[bb_i2])/2))
            y2c = int(round((y1s[bb_i2]+y2s[bb_i2])/2))
            img_f.line(image,(x1c,y1c),(x2c,y2c),(0,200,0),2)

        img_f.imshow('image',image)
        img_f.show()

directory = sys.argv[1]
names = [f[:-5] for f in os.listdir(os.path.join(directory,'jsons')) if os.path.isfile(os.path.join(directory,'jsons',f))]
for name in names[1:]:
    print(name)
    parseGT(directory,name,1.0)


