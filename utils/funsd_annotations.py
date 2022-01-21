import numpy as np
from collections import defaultdict


def combineLine(classMap,line,bbs,trans,lineTrans,s,label):
    numClasses=len(classMap)
    bb = np.empty(8+8+numClasses, dtype=np.float32)
    lXL = min([w[0] for w in line])
    rXL = max([w[2] for w in line])
    tYL = min([w[1] for w in line])
    bYL = max([w[3] for w in line])
    bb[0]=lXL*s
    bb[1]=tYL*s
    bb[2]=rXL*s
    bb[3]=tYL*s
    bb[4]=rXL*s
    bb[5]=bYL*s
    bb[6]=lXL*s
    bb[7]=bYL*s
    #we add these for conveince to crop BBs within window
    bb[8]=s*lXL
    bb[9]=s*(tYL+bYL)/2.0
    bb[10]=s*rXL
    bb[11]=s*(tYL+bYL)/2.0
    bb[12]=s*(lXL+rXL)/2.0
    bb[13]=s*tYL
    bb[14]=s*(rXL+lXL)/2.0
    bb[15]=s*bYL
    
    bb[16:]=0
    if numClasses>0:
        bb[classMap[label]]=1
    #if boxinfo['label']=='header':
    #    bb[16]=1
    #elif boxinfo['label']=='question':
    #    bb[17]=1
    #elif boxinfo['label']=='answer':
    #    bb[18]=1
    #elif boxinfo['label']=='other':
    #    bb[19]=1
    bbs.append(bb)
    trans.append(' '.join(lineTrans))
    #nex = j<len(boxes)-1
    #numNeighbors.append(len(boxinfo['linking'])+(1 if prev else 0)+(1 if nex else 0))
    #prev=True



def createLines(annotations,classMap,scale):
    numClasses=len(classMap)
    boxes = annotations['form']
    origIdToIndexes={}
    annotations['linking']=defaultdict(list)
    groups=[]
    bbs=[]
    trans=[]
    line=[]
    lineTrans=[]

    numBBs = len(boxes)
    #new line
    line=[]
    lineTrans=[]
    for j,boxinfo in enumerate(boxes):
        prev=False
        line=[]
        lineTrans=[]
        startIdx=len(bbs)
        for word in boxinfo['words']:
            lX,tY,rX,bY = word['box']
            if len(line)==0:
                line.append(word['box']+[(lX+rX)/2,(tY+bY)/2])
                lineTrans.append(word['text'])
            else:
                difX = lX-line[-1][2]
                difY = (tY+bY)/2 - line[-1][5]
                pW = line[-1][2]-line[-1][0]
                pH = line[-1][3]-line[-1][1]
                if difX<-pW*0.25 or difY>pH*0.75:
                    combineLine(classMap,line,bbs,trans,lineTrans,scale,boxinfo['label'])
                    line=[]
                    lineTrans=[]
                line.append(word['box']+[(lX+rX)/2,(tY+bY)/2])
                lineTrans.append(word['text'])
        combineLine(classMap,line,bbs,trans,lineTrans,scale,boxinfo['label'])
        endIdx=len(bbs)
        groups.append(list(range(startIdx,endIdx)))
        for idx in range(startIdx,endIdx-1):
            annotations['linking'][idx].append(idx+1) #we link them in read order. The group supervises dense connections. Read order is how the NAF dataset is labeled.
        origIdToIndexes[j]=(startIdx,endIdx-1)
    
    annotations['linking_groups']=[]
    for j,boxinfo in enumerate(boxes):
        for linkIds in boxinfo['linking']:
            linkId = linkIds[0] if linkIds[1]==j else linkIds[1]
            annotations['linking_groups'].append((linkId,j))
            j_first_x = np.mean(bbs[origIdToIndexes[j][0]][0:8:2])
            j_first_y = np.mean(bbs[origIdToIndexes[j][0]][1:8:2])
            link_first_x = np.mean(bbs[origIdToIndexes[linkId][0]][0:8:2])
            link_first_y = np.mean(bbs[origIdToIndexes[linkId][0]][1:8:2])
            j_last_x = np.mean(bbs[origIdToIndexes[j][1]][0:8:2])
            j_last_y = np.mean(bbs[origIdToIndexes[j][1]][1:8:2])
            link_last_x = np.mean(bbs[origIdToIndexes[linkId][1]][0:8:2])
            link_last_y = np.mean(bbs[origIdToIndexes[linkId][1]][1:8:2])

            above = link_last_y<=j_first_y+2
            below = link_first_y>=j_last_y-2
            left = link_last_x<=j_first_x+2
            right = link_first_x>=j_last_x-2
            if above or left:
                annotations['linking'][origIdToIndexes[j][0]].append(origIdToIndexes[linkId][1])
            elif below or right:
                annotations['linking'][origIdToIndexes[j][1]].append(origIdToIndexes[linkId][0])
            else:
                print("!!!!!!!!")
                print("Print odd para align, unhandeled case.")
                print("trans:{}, ({},{}), trans:{}, ({},{})   , trans:{}, ({},{}), trans:{}, ({},{})".format(trans[origIdToIndexes[j][0]],j_first_x,j_first_y,trans[origIdToIndexes[j][1]],j_last_x,j_last_y,trans[origIdToIndexes[linkId][0]],link_first_x,link_first_y,trans[origIdToIndexes[linkId][1]],link_last_x,link_last_y))
                import pdb;pdb.set_trace()
                #annotations['linking'][origIdToIndexes[j][1]].append(origIdToIndexes[linkId][0])
    #import pdb;pdb.set_trace()
    numNeighbors = [len(annotations['linking'][index]) for index in range(len(bbs))]
    bbs = np.stack(bbs,axis=0)
    bbs = bbs[None,...] #add batch dim

    return bbs, numNeighbors, trans, groups



#These are manual corrections I'm making to some training set images
rulebook={
        '0011838621': [ ('add link',29,32),
                         ('add link',29,7),
                         ('change class',60,'answer'),
                         ('change class',61,'answer'),
                         ('change text',61,'M. D. Davis'),
                         ],

        '0000999294': [
            ('change class',36,'header'),
            ('change class',41,'question'),

            ('add link',41,51),
            ('add link',41,61),
            ('add link',41,62),
            ('add link',41,65),
            ('add link',41,66),
            ('add link',41,71),
            ('add link',41,70),
            ],
        '0001476912': [
            ('change class',11,'header'),
            ('change class',12,'question'),
            ('change class',48,'question'),
            ('remove link',11,21),
            ('remove link',11,22),
            ('add link',21,48),
            ('add link',22,12),
            ('add link',20,46),
            ('change text',48,'Los Angeles'),
            ],
        '0001485288':[
            ('change class',20,'question'),
            ],
        '0011973451': [
                         ('add link',13,4)
                         ],
        '0060029036': [
            ('change class',42,'header'),
            ('add link',42,43),
            ('add link',42,44),
            ('add link',42,45),
            ('change class',35,'header'),
            ('add link',35,36),
            ('add link',35,11),
            ('add link',35,12),
            ],
        '0060094595': [
            ('change class',24,'header'),
            ('change class',27,'header'),
            ('change class',50,'header'),
            ('change text',50,'Burn Test'),

            ('add link',50,51),
            ('add link',50,52),
            ('add link',50,53),
            ('add link',50,53),
            ('add link',50,54),
            ('add link',50,55),
            ('add link',50,56),

             ('add link',27,0),
             ('add link',27,1),
             ('add link',27,2),
             ('add link',27,3),
             ('add link',27,36),

            ],
        '01073843': [
                #85, 86,87,88,89,90
                ('change class',94,'header'),
                ('change text',94,'COMPOUND (ug plate)'),
                ('change class',24,'question'),
                ('change class',25,'question'),
                ('change class',26,'question'),
                ('add link',24,30),
                ('add link',86,30),
                ('add link',24,61),
                ('add link',88,61),
                ('add link',24,63),
                ('add link',90,63),
                ('add link',25,29),#1.10 
                ('add link',85,29),#1.10 
                ('add link',25,31),#1.12
                ('add link',86,31),#1.12
                ('add link',25,33),#1.07
                ('add link',87,33),#1.07
                ('add link',25,59),#.82
                ('add link',88,59),#.82
                ('add link',25,62),#.78
                ('add link',89,62),#.78
                ('add link',25,64),#.71
                ('add link',90,64),#.71
                ('add link',26,32),
                ('add link',86,32),
                ('add link',26,60),
                ('add link',88,60),
                ('add link',26,65),
                ('add link',90,65),

                ('change class',97,'header'),
                ('change class',35,'question'),
                ('change class',36,'question'),
                ('add link',35,39),
                ('add link',35,43),
                ('add link',35,47),
                ('add link',35,51),
                ('add link',35,55),
                ('add link',35,57),
                ('add link',36,40),
                ('add link',36,44),
                ('add link',36,48),
                ('add link',36,52),
                ('add link',36,56),
                ('add link',36,58),


                ('add link',85,39),
                ('add link',86,43),
                ('add link',87,47),
                ('add link',88,51),
                ('add link',89,55),
                ('add link',90,57),
                ('add link',85,40),
                ('add link',86,44),
                ('add link',87,48),
                ('add link',88,52),
                ('add link',89,56),
                ('add link',90,58),

                ('change class',13,'header'),
                ('change class',37,'question'),
                ('change class',38,'question'),
                ('add link',37,41),
                ('add link',37,45),
                ('add link',37,49),
                ('add link',37,53),
                ('add link',38,42),
                ('add link',38,46),
                ('add link',38,50),
                ('add link',38,54),

                ('add link',85,41),
                ('add link',86,45),
                ('add link',87,49),
                ('add link',88,53),
                ('add link',85,42),
                ('add link',86,46),
                ('add link',87,50),
                ('add link',88,54),

                ('remove link',95,30),
                ('remove link',95,31),
                ('remove link',95,32),
                ('remove link',95,33),
                ('remove link',95,59),
                ('remove link',95,60),
                ('remove link',95,61),
                ('remove link',95,62),
                ('remove link',95,63),
                ('remove link',95,64),
                ('remove link',95,65),

                ],
        }
#'change text': [id,text]

for doc in rulebook:
    new_rules = defaultdict(list)
    for typ,info0,info1 in rulebook[doc]:
        if typ=='add link':
            new_rules[info0].append((typ,info1))
            new_rules[info1].append((typ,info0))
        elif typ=='remove link':
            new_rules[info0].append((typ,info1))
            new_rules[info1].append((typ,info0))
        elif typ=='change text' or typ=='change class':
            new_rules[info0].append((typ,info1))
    rulebook[doc] = new_rules


def fixFUNSD(annotations):
    for name,rules in rulebook.items():
        if name in annotations['XX_imageName']:
            #apply rules
            #links_to_add,links_to_remove = rules
            for entity in annotations['form']:
                if entity['id'] in rules:
                    for rule in rules[entity['id']]:
                        typ,info = rule
                        if typ=='remove link':
                            for i,link in enumerate(entity['linking']):
                                if info in link:
                                    del entity['linking'][i]
                                    break
                        elif typ=='add link':
                            entity['linking'].append([info,entity['id']])
                        elif typ=='change text':
                            if isinstance(info,str):
                                entity['text']=info
                                for i,word in enumerate(info.split(' ')):
                                    entity['words'][i]['text']=word
                            else:
                                assert False
                        elif typ=='change class':
                            entity['label'] = info
                

