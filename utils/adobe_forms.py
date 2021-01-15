import json
from utils import img_f
import numpy as np
import os,sys
from collections import defaultdict

INSIDE_THRESH=0.92
DISPLAY=False
LINKING=True

class TableError(Exception):
    pass


def parseGT(directory,name,scale):
    with open(os.path.join(directory,'jsons',name+'.json')) as f:
        data=json.load(f)


    image = img_f.imread(os.path.join(directory,'imgs',name+'.png'))
    all_bbs=[]
    x1s=[]
    x2s=[]
    y1s=[]
    y2s=[]
    #all_areas=[]
    for section,ldata in data.items():
        for i,bb in enumerate(ldata):
            x=round(bb['x'])
            y=round(bb['y'])
            #w=round(bb['w'])
            #h=round(bb['h'])
            t=bb['jsonClass']

            x2=round(bb['x']+bb['w'])-1
            y2=round(bb['y']+bb['h'])-1

            assert(bb['w']>0 and bb['h']>0)

            

            if x>=0 and x<image.shape[1] and y>=0 and y<image.shape[0] and x2>x and y2>y:
                all_bbs.append((section,i))
                x1s.append(x)
                x2s.append(x2)
                y1s.append(y)
                y2s.append(y2)
                #all_areas.append(bb['h']*bb['w'])

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
    #insides[area<


    #top to bottom
    #first choice groups, tables and lists
    #fields, choice fields
    #be sure any nested fields, etc are processed before their parent so they properly claim their childrem
    hier_level = {
            "ChoiceGroup":0,
            "Table":0,
            "ChoiceField":1,
            "Field":2,
            "List":0,
            "SectionTitle":4,
            "HeaderTitle":4,
            "ChoiceGroupTitle":4,
            "TableTitle":4,
            "TextBlock":5,
            "TextRun":6,
            "Widget":6
            }
    recursive = ['ChoiceGroup','TextBlock']
    illegal=[
            ('Field','ChoiceField'),
            ('TextBlock','ChoiceGroupTitle'),
            ('TextBlock','HeaderTitle'),
            ('TextBlock','SectionTitle'),
            ('TextBlock','TableTitle'),
            ('TextBlock','Field'),
            ('ChoiceGroupTitle','ChoiceGroup')
            ]

    belongs_to = {}
    owns = defaultdict(list)
    bbs=[]
    bb_i_groups=[]

    done=set()
    bottom_up=True

    #build
    if True:
        conflict_fields=[]
        #ordered_types = ["ChoiceGroup","Table","List","ChoiceField","Field","SectionTitle","HeaderTitle","ChoiceGroupTitle","TableTitle","TextBlock"] #,"Widget","TextRun"]
        ordered_types = [(k,v) for k,v in hier_level.items() if v<6]
        ordered_types.sort(key=lambda a:a[1], reverse=bottom_up)
        grouped_types=[]
        group=[]
        level=ordered_types[0][1]
        for typ,lvl in ordered_types:
            if lvl==level:
                group.append(typ)
            else:
                grouped_types.append(group)
                group=[typ]
                level=lvl
        grouped_types.append(group)
        ordered_bbs=[]
        for json_types in grouped_types:
            #print(json_types)
            for json_type in json_types:
                if json_type in data:
                    for i,bb in enumerate(data[json_type]):
                        try:
                            bb_i = all_bbs.index((json_type,i))
                            ordered_bbs.append((areas[bb_i],bb_i))
                        except ValueError:
                            pass

        ordered_bbs.sort(key=lambda a:a[0], reverse=not bottom_up)
        for area,bb_i in ordered_bbs:
            json_type = all_bbs[bb_i][0]

            #hierarchy
            inside_me = np.nonzero(insides[bb_i])[0]
            already_owned=[]
            for i_bb_i in inside_me:
                if bb_i!=i_bb_i:
                    if i_bb_i in belongs_to:
                        already_owned.append(i_bb_i)
                    elif (
                            #not bottom_up or 
                            #hier_level[json_type]<=hier_level[all_bbs[i_bb_i][0]] or 
                            ##(json_type==all_bbs[i_bb_i][0] and json_type in recursive) or 
                            #(json_type=='TextBlock' and inside_amounts[bb_i,i_bb_i]>0.999999999 and 'Title' not in all_bbs[i_bb_i][0])
                            area>=areas[i_bb_i] and
                            (json_type,all_bbs[i_bb_i][0]) not in illegal
                            ):
                        if json_type=='List' and 'Text' not in all_bbs[i_bb_i][0]:
                            print('skipping image, List present')
                            return 
                        if (not (json_type=='TextBlock' and all_bbs[i_bb_i][0]=='ChoiceField')) or area>areas[i_bb_i]:
                            if json_type=='TextBlock' and all_bbs[i_bb_i][0]=='ChoiceField':
                                print('Hey! {} TextBlock is claiming {} ChoiceField. inside: {}'.format(bb_i,i_bb_i,inside_amounts[bb_i,i_bb_i]))
                            belongs_to[i_bb_i]=bb_i
                            owns[bb_i].append(i_bb_i)
            #if json_type=='ChoiceGroup':
            #    print(bb_i)
            #    print('inside me {}'.format(inside_me))
            #    print('already owned {}'.format(already_owned))
            #    print('difference {}'.format(set(inside_me).difference(set(already_owned))))
            #    for i_bb_i in already_owned:
            #        if 'ChoiceField'==all_bbs[i_bb_i][0]:
            #            print('ChoiceField {} belongs to {} {}'.format(i_bb_i,belongs_to[i_bb_i],all_bbs[belongs_to[i_bb_i]][0]))
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
                #elif hier_level[json_type] > hier_level[all_bbs[i_bb_i][0]]:
                #    #I can't claim this
                #    pass
                #elif bottom_up and hier_level[json_type] < hier_level[all_bbs[other_owner][0]]:
                #    #since it's lower, it keeps it
                #    pass
                #elif inside_amounts[bb_i,i_bb_i]!=inside_amounts[other_owner,i_bb_i]:
                #    if inside_amounts[bb_i,i_bb_i]>inside_amounts[other_owner,i_bb_i]:
                #        belongs_to[i_bb_i]=bb_i
                #        owns[bb_i].append(i_bb_i)
                #        owns[other_owner].remove(i_bb_i)
                #    else:
                #        pass
                #elif ('Field' in all_bbs[other_owner][0] or 'Group' in all_bbs[other_owner][0]) and ('Field' in all_bbs[bb_i][0] or 'Group' in all_bbs[bb_i][0]) and 'TextBlock' in all_bbs[i_bb_i][0]:
                #    #give it to the one without a textbox
                #    other_has_other_text = any('TextBlock'==all_bbs[obi][0] for obi in owns[other_owner] if obi!=i_bb_i)
                #    i_have_text = any('TextBlock'==all_bbs[obi][0] for obi in owns[bb_i])

                #    assert(other_has_other_text!=i_have_text)
                #    if i_have_text:
                #        pass
                #    elif other_has_other_text:
                #        belongs_to[i_bb_i]=bb_i
                #        owns[bb_i].append(i_bb_i)
                #        owns[other_owner].remove(i_bb_i)
                #elif 'TextRun'==all_bbs[i_bb_i][0] and 'Text' not in all_bbs[other_owner][0] and 'Text' not in all_bbs[bb_i][0]:
                #    #we'll just punt and let this be  picked up by the correct text block
                #    del belongs_to[i_bb_i]
                #    owns[other_owner].remove(i_bb_i)
                #elif inside_amounts[bb_i,i_bb_i]==inside_amounts[other_owner,i_bb_i]:# and inside_amounts[bb_i,other_owner]<=inside_amounts[other_owner,bb_i]:
                #    print('this is wrong. i_bb_i claimed by two different fields')
                #    print('current: {} {} l{}'.format(bb_i,all_bbs[bb_i],hier_level[json_type]))
                #    print('I want to claim: {} {} l{}'.format(i_bb_i,all_bbs[i_bb_i],hier_level[all_bbs[i_bb_i][0]]))
                #    print('But its claimed by: {} {} l{}'.format(other_owner,all_bbs[other_owner],hier_level[all_bbs[other_owner][0]]))
                #    if bb_i in belongs_to:
                #        print('I am owned by {} {}'.format(belongs_to[bb_i], all_bbs[belongs_to[bb_i]]))
                #    print('me to want: {}, want to me: {}'.format(insides[bb_i,i_bb_i],insides[i_bb_i,bb_i]))
                #    print('other to want: {}, want to other: {}'.format(insides[other_owner,i_bb_i],insides[i_bb_i,other_owner]))

        
        def checkFix(bb_i):
            for o_i in list(owns[bb_i]):
                checkFix(o_i)
            #print('check {} {} -> {}'.format(bb_i,all_bbs[bb_i],owns[bb_i]))
            if all_bbs[bb_i][0]=='ChoiceGroupTitle' and (bb_i not in belongs_to or all_bbs[belongs_to[bb_i]][0]!='ChoiceGroup'):
                #import pdb;pdb.set_trace()
                assert(len(owns[bb_i])==1 and all_bbs[owns[bb_i][0]][0]=='TextBlock')

                if bb_i in belongs_to:
                    owns[belongs_to[bb_i]].remove(bb_i)
                    owns[belongs_to[bb_i]].append(owns[bb_i][0])
                    belongs_to[owns[bb_i][0]]=belongs_to[bb_i]
                    belongs_to[bb_i]=None
                    del owns[bb_i]
                else:
                    belongs_to[bb_i]=None
                    del belongs_to[owns[bb_i][0]]
                    del owns[bb_i]
                print('Stray {} ChoiceGroupTitle converted to TextBlock'.format(bb_i))


            if all_bbs[bb_i][0]=='TextRun' and (bb_i not in belongs_to or all_bbs[belongs_to[bb_i]][0]!='TextBlock'):
                possiblities=[]
                for o_bb_i in range(len(all_bbs)):
                    if all_bbs[o_bb_i][0]=='TextBlock':
                        possiblities.append((o_bb_i,inside_amounts[o_bb_i,bb_i],inside_amounts[bb_i,o_bb_i]))
                possiblities.sort(key=lambda a:a[1]+0.1*a[2],reverse=True)
                
                best_i = possiblities[0][0]
                if bb_i in belongs_to:
                    owns[belongs_to[bb_i]].remove(bb_i)
                belongs_to[bb_i]=best_i
                owns[best_i].append(bb_i)
                print('TextRun {} (re)assigned to TextBlock {}'.format(bb_i,best_i))
            
            
            if all_bbs[bb_i][0]=='ChoiceField':
                if bb_i not in belongs_to or all_bbs[belongs_to[bb_i]][0]!='ChoiceGroup':
                    possiblities=[]
                    for o_bb_i in range(len(all_bbs)):
                        if all_bbs[o_bb_i][0]=='ChoiceGroup' and inside_amounts[o_bb_i,bb_i]>0.05:
                            possiblities.append((o_bb_i,inside_amounts[o_bb_i,bb_i]+inside_amounts[bb_i,o_bb_i]))
                            #owns[o_bb_i].append(bb_i)
                            #belongs_to[bb_i]=o_bb_i
                            #break
                    possiblities.sort(key=lambda a:a[1],reverse=True)
                    if len(possiblities)>0:
                        
                        g_i = possiblities[0][0]
                        i_own_g=False
                        if g_i in belongs_to:
                            g_belongs=belongs_to[g_i]
                            while g_belongs in belongs_to:
                                g_belongs = belongs_to[g_belongs]
                                if g_belongs==bb_i:
                                    i_own_g=True
                        
                        if i_own_g:
                            if bb_i in belongs_to:
                                my_owner = belongs_to[bb_i]
                                owns[my_owner].remove(bb_i)
                                owns[g_i].append(bb_i)
                                belongs_to[bb_i]=g_i
                                
                                owns[belongs_to[g_i]].remove(g_i)
                                owns[my_owner].append(g_i)
                                belongs_to[g_i]=my_owner
                            else:
                                owns[g_i].append(bb_i)
                                belongs_to[bb_i]=g_i
                                owns[belongs_to[g_i]].remove(g_i)
                                del belongs_to[g_i]


                        else:
                            owns[g_i].append(bb_i)
                            if bb_i in belongs_to:
                                owns[belongs_to[bb_i]].remove(bb_i)
                            belongs_to[bb_i]=g_i
                            print('ChoiceField {} (re)assigned to ChoiceGroup {}'.format(bb_i,g_i))
                        

            if all_bbs[bb_i][0]=='Field':
                has_text=False
                has_widget=False
                for o_i in owns[bb_i]:
                    if all_bbs[o_i][0]=='TextBlock':
                        has_text=True
                        for oo_i in owns[o_i]:
                            if all_bbs[oo_i][0]=='Widget':
                                has_widget=True
                    if all_bbs[o_i][0]=='Widget':
                        has_widget=True
                if not has_text or not has_widget:
                    me_belongs=[]
                    xx=bb_i
                    while xx in belongs_to:
                        xx=belongs_to[xx]
                        if xx in me_belongs:
                            #loop!
                            assert(False) #loop
                        me_belongs.append(xx)
                    for o_bb_i in range(len(all_bbs)):
                        if o_bb_i not in belongs_to or belongs_to[o_bb_i] in me_belongs:
                            json_type = all_bbs[o_bb_i][0]
                            if not has_text and json_type=='TextBlock' and inside_amounts[bb_i,o_bb_i]>0.6:
                                #text_i = o_bb_i
                                if o_bb_i in belongs_to:
                                    owns[belongs_to[o_bb_i]].remove(o_bb_i)
                                owns[bb_i].append(o_bb_i)
                                belongs_to[o_bb_i]=bb_i
                                print('Fix Reclaimed TextBlock {} for Field {}'.format(o_bb_i,bb_i))
                                has_text=True
                                if has_widget:
                                    break
                            if not has_widget and json_type=='Widget' and inside_amounts[bb_i,o_bb_i]>0.6:
                                #text_i = o_bb_i
                                if o_bb_i in belongs_to:
                                    owns[belongs_to[o_bb_i]].remove(o_bb_i)
                                owns[bb_i].append(o_bb_i)
                                belongs_to[o_bb_i]=bb_i
                                print('Fix Reclaimed Widget {} for Field {}'.format(o_bb_i,bb_i))
                                has_widget=True
                                if has_text:
                                    break
                if not has_text:
                    #perhaps it was claimed by something else
                    possiblities=[]
                    for o_bb_i in range(len(all_bbs)):
                        if insides[bb_i,o_bb_i] and all_bbs[o_bb_i][0]=='TextBlock':
                            possiblities.append(o_bb_i)
                    if len(possiblities)>0:
                        if len(possiblities)==1:
                            text_bb_i = possiblities[0]
                        else:
                            not_field=[]
                            others=[]
                            for poss in possiblities:
                                dist = abs(x1s[bb_i]-x1s[poss])+abs(y1s[bb_i]-y1s[poss])
                                bel = belongs_to[poss]
                                if 'Field' not in all_bbs[bel][0]:
                                    not_field.append((dist,poss))
                                else:
                                    others.append((dist,poss))
                            if len(not_field)>0:
                                not_field.sort(key=lambda a:a[0])
                                text_bb_i = not_field[0][1]
                            else:
                                others.sort(key=lambda a:a[0])
                                text_bb_i = others[0][1]

                        
                        owns[bb_i].append(text_bb_i)
                        owns[belongs_to[text_bb_i]].remove(text_bb_i)
                        belongs_to[text_bb_i]=bb_i
                        print('Fix stole TextBlock {} for Field {}'.format(text_bb_i,bb_i))
                    else:
                        print('Field {} has no text'.format(bb_i))

                if not has_widget:
                    #perhaps it was claimed by something else
                    possiblities=[]
                    for o_bb_i in range(len(all_bbs)):
                        if insides[bb_i,o_bb_i] and all_bbs[o_bb_i][0]=='Widget':
                            possiblities.append(o_bb_i)
                    if len(possiblities)==1:
                        widget_bb_i = possiblities[0]
                    else:
                        assert len(possiblities)>1
                        not_field=[]
                        others=[]
                        for poss in possiblities:
                            dist = abs(x1s[bb_i]-x1s[poss])+abs(y1s[bb_i]-y1s[poss])
                            bel = belongs_to[poss]
                            if 'Field' not in all_bbs[bel][0]:
                                not_field.append((dist,poss))
                            else:
                                others.append((dist,poss))
                        if len(not_field)>0:
                            not_field.sort(key=lambda a:a[0])
                            widget_bb_i = not_field[0][1]
                        else:
                            others.sort(key=lambda a:a[0])
                            widget_bb_i = others[0][1]

                    
                    owns[bb_i].append(widget_bb_i)
                    owns[belongs_to[widget_bb_i]].remove(widget_bb_i)
                    belongs_to[widget_bb_i]=bb_i
                    print('Fix stole Widget {} for Field {}'.format(widget_bb_i,bb_i))



        for bb_i in range(len(all_bbs)):
            if bb_i not in belongs_to:
                checkFix(bb_i)

        accounted=[False]*len(all_bbs)
        print('Hierarchy!')
        def printRec(bb_i,depth):
            accounted[bb_i]=True
            print(('\t'*depth)+'{} {} ({},{})'.format(all_bbs[bb_i][0],bb_i,x1s[bb_i],y1s[bb_i]))
            for o_i in owns[bb_i]:
                printRec(o_i,depth+1)
        for bb_i in range(len(all_bbs)):
            if bb_i not in belongs_to:
                printRec(bb_i,0)
            elif belongs_to[bb_i] is None:
                accounted[bb_i]=True

        assert all(accounted) #[(e,r) for e,r in enumerate(accounted)]
        ##I think these are the only things needing grouped
        #A widget in a text block is fill-in-the-blank!!!!
        #groups=defaultdict(list)
        #if 'TextBlock' in data:
        #    for i,bb in enumerate(data['TextBlock']):
        #        bb_i = all_bbs.index(('TextBlock',i))
        #        for other_bb_i in owns[bb_i]:
        #            ssert(all_bbs[other_bb_i][0]=='TextRun') #could be widget
        #            groups[bb_i].append(other_bb_i)

        if LINKING:
            #linking
            def putInReadOrder(bb_is):
                elements=[]
                avg_h=0
                for o_bb_i in bb_is:
                    elements.append((o_bb_i,(x1s[o_bb_i]+x2s[o_bb_i])/2,(y1s[o_bb_i]+y2s[o_bb_i])/2))
                    avg_h += y2s[o_bb_i]-y1s[o_bb_i]
                avg_h/=len(elements)
                elements.sort(key=lambda a:a[2])

                
                lines=[]
                cur_line=[elements[0][0:2]]
                cur_y=elements[0][2]
                for o_bb_i,x,y in elements[1:]:
                    if y-cur_y>avg_h*0.6:
                        lines.append(cur_line)
                        cur_line=[]
                    cur_line.append((o_bb_i,x))
                    cur_y = y
                lines.append(cur_line)

                in_order = []
                for line in lines:
                    line.sort(key=lambda a:a[1])
                    in_order+=[a[0] for a in line]
                return in_order
                
            def addPair(pairs,a,b):
                pairs.add((min(a,b),max(a,b)))
            def linkTitle(bb_i,pairs):
                assert('Title' in all_bbs[bb_i][0])
                assert(len(owns[bb_i])==1)
                assert(all_bbs[owns[bb_i][0]][0]=='TextBlock')
                return linkTextBlock(owns[bb_i][0],pairs)
            def linkTextBlock(bb_i,pairs):
                if len(owns[bb_i])==1:
                    assert(all_bbs[owns[bb_i][0]][0] == 'TextRun')
                    return owns[bb_i][0], owns[bb_i][0]
                #get read order, by y first
                #first get average height, so we know if things are on a new line
                avg_h=0
                with_y=[]
                new_owns=[]
                group=[]
                for o_bb_i in owns[bb_i]:
                    if all_bbs[o_bb_i][0]=='TextBlock':
                        new_owns+=owns[o_bb_i]
                        for o_i in owns[o_bb_i]:
                            if all_bbs[o_i][0]=='TextRun':
                                group.append(o_i)
                            elif all_bbs[o_i][0]=='Widget':
                                if len(group)>0:
                                    bb_i_groups.append(group)
                                    group=[]
                            else:
                                assert False
                    elif all_bbs[o_bb_i][0]=='ChoiceGroup':
                        assert all(all_bbs[cf][0]=='ChoiceField' for cf in owns[o_bb_i])
                        for cf in owns[o_bb_i]:
                            new_owns.append(linkChoiceField(cf,pairs))
                        if len(group)>0:
                            bb_i_groups.append(group)
                            group=[]
                    else:
                        if all_bbs[o_bb_i][0]=='TextRun':
                            group.append(o_bb_i)
                        elif all_bbs[o_bb_i][0]=='Widget':
                            if len(group)>0:
                                bb_i_groups.append(group)
                                group=[]
                        elif all_bbs[o_bb_i][0]=='List' and len(owns[o_bb_i])==0:
                            pass
                        else:
                            assert False
                        new_owns.append(o_bb_i)
                if len(group)>0:
                    bb_i_groups.append(group)
                    group=[]

                ordered_owns = putInReadOrder(new_owns)
                prev=None
                for o_i in ordered_owns:
                    if prev is not None:
                        addPair(pairs,prev,o_i)
                    prev=o_i
                return ordered_owns[0],prev
                    

            def linkField(bb_i,pairs,checkbox=False):
                #assert len(owns[bb_i])==2
                assert('Field' in all_bbs[bb_i][0])
                text_i=None
                text_is=[]
                widget_is=[]
                for o_bb_i in owns[bb_i]:
                    if all_bbs[o_bb_i][0]=='TextBlock':
                        #assert(text_i is None)
                        if text_i is None:
                            text_i = o_bb_i
                        else:
                            #assert(not checkbox)
                            text_is+=[text_i,o_bb_i]
                    elif all_bbs[o_bb_i][0]=='Widget':
                        widget_is.append(o_bb_i)
                
                        
                if checkbox and len(text_is)==0:
                    assert(text_i is not None)
                    text_top, text_bot = linkTextBlock(text_i,pairs)
                    #assert(len(widget_is)<=2)

                    if len(widget_is)==2:
                        #which widget is the check box? Probably the smaller one that is to the left
                        for widget_i in widget_is:
                            assert(len(owns[widget_i])==0)
                        #area0 = (x2s[widget_is[0]]-x1s[widget_is[0]])*(y2s[widget_is[0]]-y1s[widget_is[0]])
                        #area1 = (x2s[widget_is[1]]-x1s[widget_is[1]])*(y2s[widget_is[1]]-y1s[widget_is[1]])

                        addPair(pairs,widget_is[0],text_bot)
                        addPair(pairs,widget_is[1],text_bot)

                        #if area0<area1:
                        #    assert(x1s[widget_is[0]]<x1s[widget_is[1]])
                        #    #return widget_is[0]
                        #elif area1<area0:
                        #    assert(x1s[widget_is[1]]<x1s[widget_is[0]])
                        #    #return widget_is[1]
                        #else:
                        #    assert False
                    elif len(widget_is)>2:
                        ordered_is = putInReadOrder(widget_is+[text_i])
                        part1=[]
                        part2=[]
                        hit_text=False
                        for o_i in ordered_is:
                            if o_i == text_i:
                                hit_text=True
                            elif not hit_text:
                                part1.append(o_i)
                            else:
                                part2.append(o_i)
                        if len(part1)>0:
                            prev = part1[0]
                            for o_i in part1[1:]:
                                addPair(pairs,prev,o_i)
                                prev=o_i
                            addPair(pairs,prev,text_top)
                        if len(part2)>0:
                            prev = part2[0]
                            for o_i in part2[1:]:
                                addPair(pairs,prev,o_i)
                                prev=o_i
                            addPair(pairs,part2[0],text_bot)
                    elif len(widget_is)==1:
                        addPair(pairs,widget_is[0],text_bot)
                        #return widget_is[0]
                    return text_top
                elif len(text_is)==0:
                    if text_i is not None:
                        text_top, text_bot = linkTextBlock(text_i,pairs)
                        #assert(len(widget_is)==1)
                        if len(widget_is)==1:
                            assert(len(owns[widget_is[0]])==0)
                            addPair(pairs,text_bot,widget_is[0])
                        elif len(widget_is)>1:
                            widget_is = putInReadOrder(widget_is)
                            prev=text_bot
                            for w_i in widget_is:
                                assert(len(owns[w_i])==0)
                                addPair(pairs,prev,w_i)
                                prev=w_i
                        return text_top
                    else:
                        assert(len(widget_is)==1)
                        return widget_is[0]
                else:
                    #exception, just do everything read order
                    ordered_is = putInReadOrder(text_is+widget_is)

                    prev=None
                    toptop=None
                    for o_i in ordered_is:
                        if all_bbs[o_i][0]=='TextBlock':
                            top,bot = linkTextBlock(o_i,pairs)
                        else:
                            top=bot=o_i
                        if prev is not None:
                            addPair(pairs,prev,top)
                        else:
                            assert(toptop is None)
                            toptop=top
                        prev=bot
                    return toptop


            def linkChoiceField(bb_i,pairs):
                if len(owns[bb_i])==1 and all_bbs[owns[bb_i][0]][0]=='Field':
                    return linkField(owns[bb_i][0],pairs,True)
                else:
                    return linkField(bb_i,pairs,True)
            def linkChoiceGroup(bb_i,pairs):
                title_i = None
                title_is = None #multi line title, mislabeled
                to_link_pre=[]
                ordered_is = putInReadOrder(owns[bb_i])
                prev=None
                for other_bb_i in ordered_is:
                    if 'ChoiceGroupTitle' == all_bbs[other_bb_i][0]:
                        #assert title_i is None
                        if title_i is None:
                            title_i = other_bb_i
                            title_is = [other_bb_i]
                        elif prev=='ChoiceGroupTitle':
                            title_is.append(other_bb_i)
                        else:
                            #I'm just going to assume it's actually Text
                            assert(len(owns[other_bb_i])==1 and all_bbs[owns[other_bb_i][0]][0]=='TextBlock')
                            to_link_pre.append(owns[other_bb_i][0])
                            #change ownerships
                            belongs_to[other_bb_i]=None
                            belongs_to[owns[other_bb_i][0]]=bb_i
                            #don't need to chagne owns... I think
                    elif len(owns[other_bb_i])>0:
                        to_link_pre.append(other_bb_i)
                    elif all_bbs[other_bb_i][0]=='Widget':
                        pass #error, ignore it
                    else:
                        print('unknown in choice group {} {}'.format(other_bb_i,all_bbs[other_bb_i]))
                        import pdb;pdb.set_trace() #not sure what this is
                    prev=all_bbs[other_bb_i][0]

                to_link_to_ChoiceField=[]
                to_link_post=[]
                to_link_pre = putInReadOrder(to_link_pre)
                texts=[]
                prev=None
                for other_bb_i in to_link_pre:
                    if 'ChoiceField' == all_bbs[other_bb_i][0]:
                        to_link_bb_i = linkChoiceField(other_bb_i,pairs)
                        to_link_post.append(to_link_bb_i)
                    elif 'ChoiceGroup' == all_bbs[other_bb_i][0]:
                        to_link_bb_is = linkChoiceGroup(other_bb_i,pairs)
                        if type(to_link_bb_is) is list:
                            #but these probably need linked to a ChoiceField...
                            #closest to top-and-left?
                            to_link_to_ChoiceField.append(to_link_bb_is)
                        else:
                             to_link_post.append(to_link_bb_is)
                    elif 'TextBlock' == all_bbs[other_bb_i][0]:
                        #This is generally just a comment or instructions. We won't link it to anything
                        texts.append(linkTextBlock(other_bb_i,pairs))
                    elif 'List' == all_bbs[other_bb_i][0]:
                        #This is generally just a comment or instructions. We won't link it to anything
                        linkList(other_bb_i,pairs)
                    elif 'Field' ==all_bbs[other_bb_i][0]:
                        to_link_bb_i = linkField(other_bb_i,pairs)
                        if prev=='ChoiceField':
                            to_link_to_ChoiceField.append([to_link_bb_i])
                        else:
                            to_link_post.append(to_link_bb_i)
                    elif 'Table' ==all_bbs[other_bb_i][0]:
                        to_link_bb_is = linkTable(other_bb_i,pairs)
                        if prev=='ChoiceField':
                            to_link_to_ChoiceField.append(to_link_bb_is)
                        else:
                            to_link_post.append(to_link_bb_i)
                    else:
                        print('Error, {} in a ChoiceGroup'.format( all_bbs[other_bb_i][0]))
                        assert False
                    prev = all_bbs[other_bb_i][0]

                if len(texts)==len(to_link_to_ChoiceField):
                    for top,bot in texts:
                        to_link_post.append(top)
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
                    y_diffs = [(p_i,abs(y1s[p_i]-yg),abs(x1s[p_i]-xg)) for p_i in to_link_post]
                    y_diffs.sort(key=lambda a:a[1]+a[2]*0.5)

                    first_parent = belongs_to[y_diffs[0][0]]
                    first_y_diff=None
                    for p_i,y,x in y_diffs[1:]:
                        if belongs_to[p_i]!=first_parent:
                            first_y_diff = y-y_diffs[0][1]
                            #first_y_diff = x-y_diffs[0][2]
                            one_x_diff = x
                            second_i=p_i
                            break


                    if first_y_diff is not None and first_y_diff<20:
                        if y_diffs[0][2]<one_x_diff and y_diffs[0][2]<20:
                            for p_i in bb_is:
                                addPair(pairs,p_i,y_diffs[0][0])
                        elif y_diffs[0][2]>one_x_diff and one_x_diff<20:
                            for p_i in bb_is:
                                addPair(pairs,p_i,second_i)
                        else:

                            #there are two even fields. just punt
                            for p_i in bb_is:
                                to_link_post+=bb_is
                    else:
                        for p_i in bb_is:
                            addPair(pairs,p_i,y_diffs[0][0])

                if title_i is not None:
                    if len(title_is)==1:
                        top,bot = linkTitle(title_i,pairs)
                    else:
                        t,b = linkTitle(title_is[0],pairs)
                        top=t
                        for title_i in title_is[1:]:
                            t,bot = linkTitle(title_i,pairs)
                            addPair(pairs,b,t)
                            b=bot
                    for to_link_bb_i in to_link_post:
                        addPair(pairs,bot,to_link_bb_i)
                    return top
                else:
                    return to_link_post

            def linkList(bb_i,pairing):
                ordered_is = putInReadOrder(owns[bb_i])
                prev=None
                toptop=None
                tops_and_bottoms=[]
                all_text=True
                for o_bb_i in ordered_is:
                    if all_bbs[o_bb_i][0]=='TextBlock':
                        top,bot = linkTextBlock(o_bb_i,pairs)
                        top = [top]
                        bot = [bot]
                    else:
                        all_text=False
                        top = linkAll(o_bb_i,pairs)
                        bot=top
                    #addPair(pairs,prev,top)
                    #prev = bot
                    tops_and_bottoms.append((top,bot))

                for top,bot in tops_and_bottoms:
                    if prev is not None:
                        for p in prev:
                            for t in top:
                                addPair(pairs,p,t)
                    else:
                        assert(toptop is None)
                        toptop=top

                    if all_text:
                        prev=bot
                    else:
                        prev=top
                return toptop

            def linkTable(bb_i,pairs):
                #ugh
                #does it have headers for rows, cols, or both?

                #First, put everything in read order
                elements=[]
                avg_h=0
                for o_bb_i in owns[bb_i]:
                    elements.append((o_bb_i,x1s[o_bb_i],y1s[o_bb_i]))
                    avg_h += y2s[o_bb_i]-y1s[o_bb_i]
                avg_h/=len(elements)
                elements.sort(key=lambda a:a[2])

                
                lines=[]
                cur_line=[elements[0][0:2]]
                cur_y=elements[0][2]
                for o_bb_i,x,y in elements[1:]:
                    if y-cur_y>avg_h*0.6:
                        lines.append(cur_line)
                        cur_line=[]
                    cur_line.append((o_bb_i,x))
                    cur_y = y
                lines.append(cur_line)


                prev=None
                counts=[]
                for line in lines:
                    line.sort(key=lambda a:a[1])
                    counts.append(len(line))
                if len(lines[0])==1 and all_bbs[lines[0][0][0]][0]=='TextBlock':
                    has_title=True
                    title_i = lines[0][0][0]
                    lines=lines[1:]
                    counts=counts[1:]
                else:
                    has_title = False

                min_counts=min(counts)
                max_counts=max(counts)
                has_row_header = all([all_bbs[line[0][0]][0]=='TextBlock' for line in lines[1:]])
                if min_counts==max_counts:
                    #perfect grid
                    has_col_header = all([all_bbs[i][0]=='TextBlock' for i,x in lines[0][1:]])
                    if not has_title:
                        has_title =  has_row_header and has_col_header
                        title_i = lines[0][0][0]
                else:
                    if not (len(lines[0])==min_counts and all(len(line)==max_counts for line in lines[1:])):
                        raise TableError
                    #assert(len(lines[0])==min_counts and all(len(line)==max_counts for line in lines[1:])) #normal table
                    has_col_header = all([all_bbs[i][0]=='TextBlock' for i,x in lines[0]])
                    if not has_col_header:
                        raise TableError
                    #assert(has_col_header)
                    has_title=False
                
                if has_title:
                    if all_bbs[title_i][0]=='TextBlock':
                        title_top,title_bot = linkTextBlock(title_i,pairs)
                    else:
                        title_top,title_bot = linkTitle(title_i,pairs)
                        #title_owns = owns[lines[0][0][0]]
                        #assert(len(title_owns)==1 and all_bbs[title_owns[0]][0]=='TextBlock')
                        #title_top,title_bot = linkTextBlock(title_owns[0],pairs)
                if has_col_header:
                    start_row=1
                else:
                    start_row=0
                if has_row_header:
                    start_col=1
                else:
                    start_col=0
                if has_col_header:
                    col_hs=[]
                    for h_i,x in lines[0][start_col:]:
                        if all_bbs[h_i][0] != 'TextBlock':
                            raise TableError
                        h_top,h_bot = linkTextBlock(h_i,pairs)
                        col_hs.append(h_bot)
                        if has_title:
                            addPair(pairs,title_bot,h_top)
                if has_row_header:
                    row_hs=[]
                    for line in lines[start_row:]:
                        h_i = line[0][0]
                        if all_bbs[h_i][0] != 'TextBlock':
                            raise TableError
                        h_top,h_bot = linkTextBlock(h_i,pairs)
                        row_hs.append(h_bot)
                        if has_title:
                            addPair(pairs,title_bot,h_top)


                #top_cell=None
                cells=[]
                for col in range(start_col,max_counts):
                    for row in range(start_row,len(lines)):
                        cell_is = linkAll(lines[row][col][0],pairs)
                        for cell_i in cell_is:
                            if has_col_header:
                                if len(col_hs)<=col-start_col:
                                    raise TableError
                                addPair(pairs,col_hs[col-start_col],cell_i)
                            if has_row_header:
                                if len(row_hs)<=row-start_row:
                                    raise TableError
                                addPair(pairs,row_hs[row-start_row],cell_i)
                        #if top_cell is None:
                        #    top_cell = cell_i
                        cells+=cell_is
                if has_title:
                    return [title_top]
                elif has_col_header:
                    return col_hs
                elif has_row_header:
                    return row_hs
                else:
                    return cells#[top_cell]

            def linkAll(bb_i,pairs):
                json_type = all_bbs[bb_i][0]
                if json_type=='ChoiceGroup':
                    bb_is=linkChoiceGroup(bb_i,pairs)
                    if type(bb_is) is not list:
                        bb_is = [bb_is]
                    return bb_is
                    #if type(bb_is) is list:
                    #    bb_is.sort(key=lambda i:y1s[i])
                    #    return bb_is[0]
                    #else:
                    #    return bb_is
                elif json_type=='Field':
                    return [linkField(bb_i,pairs)]
                elif json_type=='TextBlock':
                    return [linkTextBlock(bb_i,pairs)[0]]
                elif json_type=='Table':
                    return linkTable(bb_i,pairs)
                elif json_type=='List':
                    return linkList(bb_i,pairs)
                elif json_type=='Widget':
                    return [bb_i]
                else:
                    print('linkAll doesnt have {}'.format(json_type))
                    assert False


            pairs=set()
            #if 'ChoiceGroup' in data:
            #    for i,bb in enumerate(data['ChoiceGroup']):
            #        bb_i = all_bbs.index(('ChoiceGroup',i))
            for bb_i in range(len(all_bbs)):
                if bb_i not in belongs_to:
                    try:
                        linkAll(bb_i,pairs)
                    except TableError:
                        print('odd table, skipping')
                        return

                

                
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
        else:
            pairs=set()
    else:
        pairs=set()



    #display
    if DISPLAY:
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

                if x>=0 and x2<image.shape[1] and y>=0 and y2<image.shape[0]:

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

            assert(len(owns[bb_i1])==0)
            assert(len(owns[bb_i2])==0)

        #for group in bb_i_groups:


        img_f.imshow('image',image)
        img_f.show()
    else:

        for bb_i1,bb_i2 in pairs:
            assert(len(owns[bb_i1])==0)
            assert(len(owns[bb_i2])==0)
    return True

directory = sys.argv[1]
names = [f[:-5] for f in os.listdir(os.path.join(directory,'jsons')) if os.path.isfile(os.path.join(directory,'jsons',f))]

if len(sys.argv)==3:
    if sys.argv[2][0]=='D':
        DISPLAY=True
        LINKING=False
    elif sys.argv[2][0]=='S':
        DISPLAY=True

start=442
i=start
good=0
for name in names[start:]: #53? 54 are identicle but annotated differently
    print('{}  <<< {}'.format(name,i))
    ret=parseGT(directory,name,1.0)
    if ret is True:
        good+=1
    i+=1

print('good {}/{}'.format(good,i))
