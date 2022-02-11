from collections import defaultdict
import numpy as np
import torch
import math

def calcCorners(x,y,r,h,w):
    tlX = -w*math.cos(r) -h*math.sin(r) +x
    tlY = -h*math.cos(r) +w*math.sin(r) +y
    trX =  w*math.cos(r) -h*math.sin(r) +x
    trY = -h*math.cos(r) -w*math.sin(r) +y
    brX =  w*math.cos(r) +h*math.sin(r) +x
    brY =  h*math.cos(r) -w*math.sin(r) +y
    blX = -w*math.cos(r) +h*math.sin(r) +x
    blY =  h*math.cos(r) +w*math.sin(r) +y
    return [[tlX,tlY],[trX,trY],[brX,brY],[blX,blY]]

def calcCornersTorch(x,y,r,h,w):
    tlX = -w*torch.cos(r) -h*torch.sin(r) +x
    tlY = -h*torch.cos(r) +w*torch.sin(r) +y
    trX =  w*torch.cos(r) -h*torch.sin(r) +x
    trY = -h*torch.cos(r) -w*torch.sin(r) +y
    brX =  w*torch.cos(r) +h*torch.sin(r) +x
    brY =  h*torch.cos(r) -w*torch.sin(r) +y
    blX = -w*torch.cos(r) +h*torch.sin(r) +x
    blY =  h*torch.cos(r) +w*torch.sin(r) +y
    return tlX,tlY,trX,trY,brX,brY,blX,blY


def avg_y(bb):
    points = bb['poly_points']
    return (points[0][1]+points[1][1]+points[2][1]+points[3][1])/4.0
def avg_x(bb):
    points = bb['poly_points']
    return (points[0][0]+points[1][0]+points[2][0]+points[3][0])/4.0
def left_x(bb):
    points = bb['poly_points']
    return (points[0][0]+points[3][0])/2.0
def right_x(bb):
    points = bb['poly_points']
    return (points[1][0]+points[2][0])/2.0


def convertBBs(bbs,rotate,numClasses):
    if bbs.shape[1]==0:
        return None
    new_bbs = np.empty((1,bbs.shape[1], 5+8+numClasses), dtype=np.float32) #5 params, 8 points (used in loss), n classes
    
    tlX_ = bbs[:,:,0]
    tlY_ = bbs[:,:,1]
    trX_ = bbs[:,:,2]
    trY_ = bbs[:,:,3]
    brX_ = bbs[:,:,4]
    brY_ = bbs[:,:,5]
    blX_ = bbs[:,:,6]
    blY_ = bbs[:,:,7]

    if not rotate:
        tlX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        tlY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        trX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        trY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        brX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        brY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
        blX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        blY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
    else:
        tlX =  tlX_
        tlY =  tlY_
        trX =  trX_
        trY =  trY_
        brX =  brX_
        brY =  brY_
        blX =  blX_
        blY =  blY_

    lX = (tlX+blX)/2.0
    lY = (tlY+blY)/2.0
    rX = (trX+brX)/2.0
    rY = (trY+brY)/2.0
    d=np.sqrt((lX-rX)**2 + (lY-rY)**2)

    if (d==0).any():
        print('ERROR: zero length bb {}'.format(bbs[0][d[0]==0]))
        d[d==0]=1

    hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpose horz run
    hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transpose horz run
    h = (hl+hr)/2.0

    #tX = lX + h*-(rY-lY)/d
    #tY = lY + h*(rX-lX)/d
    #bX = lX - h*-(rY-lY)/d
    #bY = lY - h*(rX-lX)/d

    #etX =tX + rX-lX
    #etY =tY + rY-lY
    #ebX =bX + rX-lX
    #ebY =bY + rY-lY

    cX = (lX+rX)/2.0
    cY = (lY+rY)/2.0
    rot = np.arctan2(-(rY-lY),rX-lX)
    height = np.abs(h)    #this is half height
    width = d/2.0 #and half width


    height[ np.logical_or(np.isnan(height),height==0) ] =1
    width[ np.logical_or(np.isnan(width),width==0) ] =1
    #topX = (tlX+trX)/2.0
    #topY = (tlY+trY)/2.0
    #botX = (blX+brX)/2.0
    #botY = (blY+brY)/2.0
    topX = cX-np.sin(rot)*height
    botX = cX+np.sin(rot)*height
    topY = cY-np.cos(rot)*height
    botY = cY+np.cos(rot)*height
    leftX = lX
    leftY = lY
    rightX = rX
    rightY = rY


    new_bbs[:,:,0]=cX
    new_bbs[:,:,1]=cY
    new_bbs[:,:,2]=rot
    new_bbs[:,:,3]=height
    new_bbs[:,:,4]=width
    new_bbs[:,:,5]=leftX
    new_bbs[:,:,6]=leftY
    new_bbs[:,:,7]=rightX
    new_bbs[:,:,8]=rightY
    new_bbs[:,:,9]=topX
    new_bbs[:,:,10]=topY
    new_bbs[:,:,11]=botX
    new_bbs[:,:,12]=botY
    #print("{} {}, {} {}".format(new_bbs.shape,new_bbs[:,:,13:].shape,bbs.shape,bbs[:,:,-numClasses].shape))
    if numClasses>0:
        new_bbs[:,:,13:]=bbs[:,:,-numClasses:]

    assert(not np.isnan(new_bbs).any())


    return torch.from_numpy(new_bbs)


#This annotation corrects assumptions made during GTing, modifies the annotations for the current parameterization, and slightly changes the format
#if this is None, it gets processed for QA-JSON stuff
def fixAnnotations(this,annotations):
    if this is None:
        def isSkipField(this,bb):
            return (   #((bb['isBlank']=='blank' or bb['isBlank']==3) and 'P' not in bb['type']) or
                        bb['type']=='graphic' or
                        bb['type'] == 'fieldRegion'
                    )
    else:
        def isSkipField(this,bb):
            return (    (this.no_blanks and (bb['isBlank']=='blank' or bb['isBlank']==3)) or
                        (this.no_print_fields and (bb['isBlank']=='print' or bb['isBlank']==2)) or
                        (this.no_graphics and bb['type']=='graphic') or
                        bb['type'] == 'fieldRow' or
                        bb['type'] == 'fieldCol' or
                        bb['type'] == 'fieldRegion'
                    )
    def fixIsBlank(bb):
        if 'isBlank' in bb:
            if bb['type'] == 'fieldCircle':
                bb['isBlank'] = 'print'
            else:
                if type(bb['isBlank']) is int:
                    isBlankMap=['print', 'handwriting', 'print', 'blank', 'signature','ERROR5?','ERROR6?','ERROR7?','ERROR8?','ERROR9?']
                    if bb['isBlank']>len(isBlankMap):
                        print('bad isBlank: {} for {}'.format(bb['isBlank'], annotations['imageFilename']))
                        bb['isBlank'] = 'unknown'
                    else:
                        bb['isBlank'] = isBlankMap[bb['isBlank']]
        else:
            bb['isBlank'] = 'print'



    #restructure
    annotations['byId']={}
    for iii,bb in enumerate(annotations['textBBs']):
        fixIsBlank(bb)
        annotations['byId'][bb['id']]=bb
    for bb in annotations['fieldBBs']:
        fixIsBlank(bb)
        annotations['byId'][bb['id']]=bb
    if 'samePairs' in annotations:
        if this is None or not this.only_opposite_pairs:
            annotations['pairs']+=annotations['samePairs']
        del annotations['samePairs']

    #if this is None:
    #    #find bbs linked to blanks, as the blanks will be removed, but we want to know these are "questions" rather than "other" text
    #    isQuestion = []
    #    for id1,id2 in annotations['pairs']:
    #        b1 = annotations['byId'][id1]['isBlank']
    #        b2 = annotations['byId'][id2]['isBlank']
    #        if b1=='blank' or b1==3:
    #            isQuestion.append(id2)
    #        elif b2=='blank' or b2==3:
    #            isQuestion.append(id1)
    #    annotations['isBlankQuestion'] = isQuestion


    numPairsWithoutBB=0
    for id1,id2 in annotations['pairs']:
        if id1 not in annotations['byId'] or id2 not in annotations['byId']:
            numPairsWithoutBB+=1

    toAdd=[]
    idsToRemove=set()

    #enumerations inside a row they are paired to should be removed
    #enumerations paired with the left row of a chained row need to be paired with the right
    pairsToRemove=[]
    pairsToAdd=[]
    for bb in annotations['textBBs']:
        if bb['type']=='textNumber':
            for pair in annotations['pairs']:
                if bb['id'] in pair:
                    if pair[0]==bb['id']:
                        otherId=pair[1]
                    else:
                        otherId=pair[0]
                    otherBB=annotations['byId'][otherId]
                    if otherBB['type']=='fieldRow':
                        if avg_x(bb)>left_x(otherBB) and avg_x(bb)<right_x(otherBB):
                            idsToRemove.add(bb['id'])
                        #else TODO chained row case



    #remove fields we're skipping
    #reconnect para chains we broke by removing them
    #print('removing fields')
    idsToFix=[]
    circleIds=[]
    for bb in annotations['fieldBBs']:
        id=bb['id']
        #print('skip:{}, type:{}'.format(isSkipField(this,bb),bb['type']))
        if isSkipField(this,bb):
            #print('remove {}'.format(id))
            idsToRemove.add(id)
            if bb['type']=='fieldP':
                idsToFix.append(id)
        elif bb['type']=='fieldCircle':
            circleIds.append(id)
            if this is None or this.swapCircle:
                annotations['byId'][id]['type']='textCircle'

    del annotations['fieldBBs']
    del annotations['textBBs']

    
    parasLinkedTo=defaultdict(list)
    pairsToRemove=[]
    for i,pair in enumerate(annotations['pairs']):
        assert(len(pair)==2)
        if pair[0] not in annotations['byId'] or pair[1] not in annotations['byId']:
            pairsToRemove.append(i)
        elif pair[0] in idsToFix and annotations['byId'][pair[1]]['type'][-1]=='P':
            parasLinkedTo[pair[0]].append(pair[1])
            pairsToRemove.append(i)
        elif pair[1] in idsToFix and annotations['byId'][pair[0]]['type'][-1]=='P':
            parasLinkedTo[pair[1]].append(pair[0])
            pairsToRemove.append(i)
        elif pair[0] in idsToRemove or pair[1] in idsToRemove:
            pairsToRemove.append(i)
        elif (this is not None and this.only_opposite_pairs and 
                ( (annotations['byId'][pair[0]]['type'][:4]=='text' and 
                   annotations['byId'][pair[1]]['type'][:4]=='text') or
                  (annotations['byId'][pair[0]]['type'][:4]=='field' and 
                    annotations['byId'][pair[1]]['type'][:4]=='field') )):
            pairsToRemove.append(i)

    pairsToRemove.sort(reverse=True)
    last=None
    for i in pairsToRemove:
        if i==last:#in case of duplicated
            continue
        #print('del pair: {}'.format(annotations['pairs'][i]))
        del annotations['pairs'][i]
        last=i
    for _,ids in parasLinkedTo.items():
        if len(ids)==2:
            if ids[0] not in idsToRemove and ids[1] not in idsToRemove:
                #print('adding: {}'.format([ids[0],ids[1]]))
                #annotations['pairs'].append([ids[0],ids[1]])
                toAdd.append([ids[0],ids[1]])
        #else I don't know what's going on


    for id in idsToRemove:
        #print('deleted: {}'.format(annotations['byId'][id]))
        del annotations['byId'][id]


    #skipped link between col and enumeration when enumeration is between col header and col
    for pair in annotations['pairs']:
        notNum=num=None
        if pair[0] in annotations['byId'] and annotations['byId'][pair[0]]['type']=='textNumber':
            num=annotations['byId'][pair[0]]
            notNum=annotations['byId'][pair[1]]
        elif pair[1] in annotations['byId'] and annotations['byId'][pair[1]]['type']=='textNumber':
            num=annotations['byId'][pair[1]]
            notNum=annotations['byId'][pair[0]]

        if notNum is not None and notNum['type']!='textNumber':
            for pair2 in annotations['pairs']:
                if notNum['id'] in pair2:
                    if notNum['id'] == pair2[0]:
                        otherId=pair2[1]
                    else:
                        otherId=pair2[0]
                    if annotations['byId'][otherId]['type']=='fieldCol' and avg_y(annotations['byId'][otherId])>avg_y(annotations['byId'][num['id']]):
                        toAdd.append([num['id'],otherId])

    for pair in annotations['pairs']:
        assert(len(pair)==2)
    #heirarchy labels.
    #for pair in annotations['samePairs']:
    #    text=textMinor=None
    #    if annotations['byId'][pair[0]]['type']=='text':
    #        text=pair[0]
    #        if annotations['byId'][pair[1]]['type']=='textMinor':
    #            textMinor=pair[1]
    #    elif annotations['byId'][pair[1]]['type']=='text':
    #        text=pair[1]
    #        if annotations['byId'][pair[0]]['type']=='textMinor':
    #            textMinor=pair[0]
    #    else:#catch case of minor-minor-field
    #        if annotations['byId'][pair[1]]['type']=='textMinor' and annotations['byId'][pair[0]]['type']=='textMinor':
    #            a=pair[0]
    #            b=pair[1]
    #            for pair2 in annotations['pairs']:
    #                if a in pair2:
    #                    if pair2[0]==a:
    #                        otherId=pair2[1]
    #                    else:
    #                        otherId=pair2[0]
    #                    toAdd.append([b,otherId])
    #                if b in pair2:
    #                    if pair2[0]==b:
    #                        otherId=pair2[1]
    #                    else:
    #                        otherId=pair2[0]
    #                    toAdd.append([a,otherId])

    #    
    #    if text is not None and textMinor is not None:
    #        for pair2 in annotations['pairs']:
    #            if textMinor in pair2:
    #                if pair2[0]==textMinor:
    #                    otherId=pair2[1]
    #                else:
    #                    otherId=pair2[0]
    #                toAdd.append([text,otherId])
    #        for pair2 in annotations['samePairs']:
    #            if textMinor in pair2:
    #                if pair2[0]==textMinor:
    #                    otherId=pair2[1]
    #                else:
    #                    otherId=pair2[0]
    #                if annotations['byId'][otherId]['type']=='textMinor':
    #                    toAddSame.append([text,otherId])

    for pair in toAdd:
        assert(len(pair)==2)
        if pair not in annotations['pairs'] and [pair[1],pair[0]] not in annotations['pairs']:
             annotations['pairs'].append(pair)
    #annotations['pairs']+=toAdd

    #handle groups of things that are intended to be circled or crossed out
    #first identify groups
    circleGroups={}
    circleGroupId=0
    #also find text-field pairings
    paired = set()
    for pair in annotations['pairs']:
        if pair[0] in circleIds and pair[1] in circleIds:
            group0=None
            group1=None
            for id,group in circleGroups.items():
                if pair[0] in group:
                    group0=id
                if pair[1] in group:
                    group1=id
            if group0 is not None:
                if group1 is None:
                    circleGroups[group0].append(pair[1])
                elif group0!=group1:
                    circleGroups[group0] += circleGroups[group1]
                    del circleGroups[group1]
            elif group1 is not None:
                circleGroups[group1].append(pair[0])
            else:
                circleGroups[circleGroupId] = pair.copy()
                circleGroupId+=1

        if pair[0] in annotations['byId'] and pair[1] in annotations['byId']:
            cls0 = annotations['byId'][pair[0]]['type'][:4]=='text'
            cls1 = annotations['byId'][pair[1]]['type'][:4]=='text'
            if cls0!=cls1:
                paired.add(pair[0])
                paired.add(pair[1])

    for pair in annotations['pairs']:
        assert(len(pair)==2)

    #what pairs to each group?
    groupPairedTo=defaultdict(list)
    for pair in annotations['pairs']:
        if pair[0] in circleIds and pair[1] not in circleIds:
            for id,group in circleGroups.items():
                if pair[0] in group:
                    groupPairedTo[id].append(pair[1])

        if pair[1] in circleIds and pair[0] not in circleIds:
            for id,group in circleGroups.items():
                if pair[1] in group:
                    groupPairedTo[id].append(pair[0])


    for pair in annotations['pairs']:
        assert(len(pair)==2)
    #add pairs
    toAdd=[]
    if this is None or not this.only_opposite_pairs:
        for gid,group in  circleGroups.items():
            for id in group:
                for id2 in group:
                    if id!=id2:
                        toAdd.append([id,id2])
                for id2 in groupPairedTo[gid]:
                    toAdd.append([id,id2])
    for pair in toAdd:
        assert(len(pair)==2)
        if pair not in annotations['pairs'] and [pair[1],pair[0]] not in annotations['pairs']:
             annotations['pairs'].append(pair)

    #mark each bb that is chained to a cross-class pairing
    while True:
        size = len(paired)
        for pair in annotations['pairs']:
            if pair[0] in paired:
                paired.add(pair[1])
            elif pair[1] in paired:
                paired.add(pair[0])
        if len(paired)<=size:
            break #at the end of every chain
    for id in paired:
        if id in annotations['byId']:
            annotations['byId'][id]['paired']=True

    for pair in annotations['pairs']:
        assert(len(pair)==2)


    return numPairsWithoutBB

def getBBWithPoints(useBBs,s,useBlankClass=False,usePairedClass=False, useAllClass=[]):

    numClasses=2
    if useBlankClass:
        numClasses+=1
    if usePairedClass:
        numClasses+=1
    bbs = np.empty((1,len(useBBs), 8+8+numClasses), dtype=np.float32) #2x4 corners, 2x4 cross-points, 2 classes
    for j,bb in enumerate(useBBs):
        tlX = bb['poly_points'][0][0]
        tlY = bb['poly_points'][0][1]
        trX = bb['poly_points'][1][0]
        trY = bb['poly_points'][1][1]
        brX = bb['poly_points'][2][0]
        brY = bb['poly_points'][2][1]
        blX = bb['poly_points'][3][0]
        blY = bb['poly_points'][3][1]

        ###DEBUG
        #h = math.sqrt(((tlX+trX)/2 - (blX+brX)/2)**2 + ((tlY+trY)/2 - (blY+brY)/2)**2)
        #w = math.sqrt(((tlX+blX)/2 - (brX+trX)/2)**2 + ((tlY+blY)/2 - (trY+brY)/2)**2)
        #assert(h/w<5 or min(h,w)<1)
        ###DEBUG
            

        bbs[:,j,0]=tlX*s
        bbs[:,j,1]=tlY*s
        bbs[:,j,2]=trX*s
        bbs[:,j,3]=trY*s
        bbs[:,j,4]=brX*s
        bbs[:,j,5]=brY*s
        bbs[:,j,6]=blX*s
        bbs[:,j,7]=blY*s
        #we add these for conveince to crop BBs within window
        bbs[:,j,8]=s*(tlX+blX)/2.0
        bbs[:,j,9]=s*(tlY+blY)/2.0
        bbs[:,j,10]=s*(trX+brX)/2.0
        bbs[:,j,11]=s*(trY+brY)/2.0
        bbs[:,j,12]=s*(tlX+trX)/2.0
        bbs[:,j,13]=s*(tlY+trY)/2.0
        bbs[:,j,14]=s*(brX+blX)/2.0
        bbs[:,j,15]=s*(brY+blY)/2.0

        #classes
        if bb['type']=='detectorPrediction':
            bbs[:,j,16]=bb['textPred']
            bbs[:,j,17]=bb['fieldPred']
        else:
            field = bb['type'][:4]!='text'
            text=not field
            bbs[:,j,16]=1 if text else 0
            bbs[:,j,17]=1 if field else 0
        index = 18
        if len(useAllClass)>0:
            for clas in useAllClass:
                bbs[:,j,index]=bb['type']==clas
                index+=1
        if useBlankClass:
            if bb['type']=='detectorPrediction':
                bbs[:,j,index]=bb['blankPred']
            else:
                blank = (bb['isBlank']=='blank' or bb['isBlank']==3) if 'isBlank' in bb else False
                bbs[:,j,index]=1 if blank else 0
            index+=1
        if usePairedClass:
            assert(bb['type']!='detectorPrediction')
            paired = bb['paired'] if 'paired' in bb else False
            bbs[:,j,index]=1 if paired else 0
            index+=1
    return bbs

def getStartEndGT(useBBs,s,useBlankClass=False):


    if useBlankClass:
        numClasses=3
    else:
        numClasses=2
    start_gt = np.empty((1,len(useBBs), 4+numClasses), dtype=np.float32) #x,y,r,h, x classes
    end_gt = np.empty((1,len(useBBs), 4+numClasses), dtype=np.float32) #x,y,r,h, x classes
    j=0
    for bb in useBBs:
        tlX = bb['poly_points'][0][0]
        tlY = bb['poly_points'][0][1]
        trX = bb['poly_points'][1][0]
        trY = bb['poly_points'][1][1]
        brX = bb['poly_points'][2][0]
        brY = bb['poly_points'][2][1]
        blX = bb['poly_points'][3][0]
        blY = bb['poly_points'][3][1]

        field = bb['type'][:4]!='text'
        if useBlankClass and (bb['isBlank']=='blank' or bb['isBlank']==3):
            field=False
            text=False
            blank=True
        else:
            text=not field
            blank=False
            
        lX = (tlX+blX)/2.0
        lY = (tlY+blY)/2.0
        rX = (trX+brX)/2.0
        rY = (trY+brY)/2.0
        d=math.sqrt((lX-rX)**2 + (lY-rY)**2)

        hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpose horz run
        hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transpose horz run
        h = (hl+hr)/2.0

        tX = lX + h*-(rY-lY)/d
        tY = lY + h*(rX-lX)/d
        bX = lX - h*-(rY-lY)/d
        bY = lY - h*(rX-lX)/d
        start_gt[:,j,0] = tX*s
        start_gt[:,j,1] = tY*s
        start_gt[:,j,2] = bX*s
        start_gt[:,j,3] = bY*s

        etX =tX + rX-lX
        etY =tY + rY-lY
        ebX =bX + rX-lX
        ebY =bY + rY-lY
        end_gt[:,j,0] = etX*s
        end_gt[:,j,1] = etY*s
        end_gt[:,j,2] = ebX*s
        end_gt[:,j,3] = ebY*s

        #classes
        start_gt[:,j,4]=1 if text else 0
        start_gt[:,j,5]=1 if field else 0
        if useBlankClass:
            start_gt[:,j,6]=1 if blank else 0
        end_gt[:,j,4]=1 if text else 0
        end_gt[:,j,5]=1 if field else 0
        if useBlankClass:
            end_gt[:,j,6]=1 if blank else 0
        j+=1
    return start_gt, end_gt

def getBBInfo(bb,rotate,useBlankClass=False):

    tlX_ = bb['poly_points'][0][0]
    tlY_ = bb['poly_points'][0][1]
    trX_ = bb['poly_points'][1][0]
    trY_ = bb['poly_points'][1][1]
    brX_ = bb['poly_points'][2][0]
    brY_ = bb['poly_points'][2][1]
    blX_ = bb['poly_points'][3][0]
    blY_ = bb['poly_points'][3][1]

    if not rotate:
        tlX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        tlY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        trX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        trY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        brX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        brY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
        blX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        blY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
    else:
        tlX =  tlX_
        tlY =  tlY_
        trX =  trX_
        trY =  trY_
        brX =  brX_
        brY =  brY_
        blX =  blX_
        blY =  blY_


    if bb['type']=='detectorPrediction':
        text=bb['textPred']
        field=bb['fieldPred']
        blank = bb['blankPred'] if 'blankPred' in bb else None
        nn = bb['nnPred'] if 'nnPred' in bb else None
    else:
        field = bb['type'][:4]!='text'
        if useBlankClass:
            blank = bb['isBlank']=='blank' or bb['isBlank']==3
        else:
            blank=None
        text=not field
        nn=None
        
    lX = (tlX+blX)/2.0
    lY = (tlY+blY)/2.0
    rX = (trX+brX)/2.0
    rY = (trY+brY)/2.0
    d=math.sqrt((lX-rX)**2 + (lY-rY)**2)

    #orthX = -(rY-lY)
    #orthY = (rX-lX)
    #origLX = blX-tlX
    #origLY = blY-tlY
    #origRX = brX-trX
    #origRY = brY-trY
    #hl = (orthX*origLX + orthY*origLY)/d
    #hr = (orthX*origRX + orthY*origRY)/d
    hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpose horz run
    hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transpose horz run
    h = (np.abs(hl)+np.abs(hr))/2.0
    #h=0

    cX = (lX+rX)/2.0
    cY = (lY+rY)/2.0
    rot = np.arctan2(-(rY-lY),rX-lX)
    height = h*2 #use full height
    width = d

    return cX,cY,height,width,rot,text,field,blank,nn


def getResponseBBIdList_(this,queryId,annotations):
    responseBBList=[]
    for pair in annotations['pairs']: #done already +annotations['samePairs']:
        if queryId in pair:
            if pair[0]==queryId:
                otherId=pair[1]
            else:
                otherId=pair[0]
            if otherId in annotations['byId'] and (not this.onlyFormStuff or ('paired' in bb and bb['paired'])):
                #responseBBList.append(annotations['byId'][otherId])
                responseBBList.append(otherId)
    return responseBBList

def computeRotation(bb):
    tlX = bb['poly_points'][0][0]
    tlY = bb['poly_points'][0][1]
    trX = bb['poly_points'][1][0]
    trY = bb['poly_points'][1][1]
    brX = bb['poly_points'][2][0]
    brY = bb['poly_points'][2][1]
    blX = bb['poly_points'][3][0]
    blY = bb['poly_points'][3][1]

    lX = (tlX+blX)/2
    lY = (tlY+blY)/2
    rX = (trX+brX)/2
    rY = (trY+brY)/2

    return math.atan2(rY-lY,rX-lX)

def computeRotationDiff(bb1,bb2):
    r1 = computeRotation(bb1)
    r2 = computeRotation(bb2)

    diff = r1-r2
    if diff>np.pi:
        diff-=2*np.pi
    elif diff<-np.pi:
        diff+=2*np.pi
    return abs(diff)
def getCenterPoint(bb):
    tlX = bb['poly_points'][0][0]
    tlY = bb['poly_points'][0][1]
    trX = bb['poly_points'][1][0]
    trY = bb['poly_points'][1][1]
    brX = bb['poly_points'][2][0]
    brY = bb['poly_points'][2][1]
    blX = bb['poly_points'][3][0]
    blY = bb['poly_points'][3][1]
    return (tlX+trX+blX+brX)/4, (tlY+trY+blY+brY)/4
def connectionNotParallel(bb1,bb2):
    r1 = computeRotation(bb1)
    r2 = computeRotation(bb2)
    cx1,cy1 = getCenterPoint(bb1)
    cx2,cy2 = getCenterPoint(bb2)

    a = math.atan2(cy2-cy1,cx2-cx1)

    r = (r1+r2)/2
    if r<0:
        r+=np.pi
    if a<0:
        a+=np.pi

    return abs(r-a) > (15/180 * np.pi)
def horizontalOverlap(bb1,bb2):
    tlX1 = bb1['poly_points'][0][0]
    tlY1 = bb1['poly_points'][0][1]
    trX1 = bb1['poly_points'][1][0]
    trY1 = bb1['poly_points'][1][1]
    brX1 = bb1['poly_points'][2][0]
    brY1 = bb1['poly_points'][2][1]
    blX1 = bb1['poly_points'][3][0]
    blY1 = bb1['poly_points'][3][1]
    tlX2 = bb2['poly_points'][0][0]
    tlY2 = bb2['poly_points'][0][1]
    trX2 = bb2['poly_points'][1][0]
    trY2 = bb2['poly_points'][1][1]
    brX2 = bb2['poly_points'][2][0]
    brY2 = bb2['poly_points'][2][1]
    blX2 = bb2['poly_points'][3][0]
    blY2 = bb2['poly_points'][3][1]
    r1 = computeRotation(bb1)
    r2 = computeRotation(bb2)
    r = (r1+r2)/2
    r = r/np.pi * 180
    if (45>r and r>-45) or r>135 or r<-135: #horizontal
        lX1 = (tlX1+blX1)/2
        lX2 = (tlX2+blX2)/2
        rX1 = (trX1+brX1)/2
        rX2 = (trX2+brX2)/2
    
        aX1=min(lX1,rX1)
        aX2=min(lX2,rX2)
        bX1=max(lX1,rX1)
        bX2=max(lX2,rX2)
        overlap = min(bX1,bX2)-max(aX1,aX2)
        return max(0,overlap/(bX1-aX1),overlap/(bX2-aX2))
    else:
        lY1 = (tlY1+blY1)/2
        lY2 = (tlY2+blY2)/2
        rY1 = (trY1+brY1)/2
        rY2 = (trY2+brY2)/2

        aY1=min(lY1,rY1)
        aY2=min(lY2,rY2)
        bY1=max(lY1,rY1)
        bY2=max(lY2,rY2)
        overlap = min(bY1,bY2)-max(aY1,aY2)
        #print(overlap)
        return max(0,overlap/(bY1-aY1),overlap/(bY2-aY2))
def areFar(bb1,bb2):
    tlX1 = bb1['poly_points'][0][0]
    tlY1 = bb1['poly_points'][0][1]
    trX1 = bb1['poly_points'][1][0]
    trY1 = bb1['poly_points'][1][1]
    brX1 = bb1['poly_points'][2][0]
    brY1 = bb1['poly_points'][2][1]
    blX1 = bb1['poly_points'][3][0]
    blY1 = bb1['poly_points'][3][1]
    tlX2 = bb2['poly_points'][0][0]
    tlY2 = bb2['poly_points'][0][1]
    trX2 = bb2['poly_points'][1][0]
    trY2 = bb2['poly_points'][1][1]
    brX2 = bb2['poly_points'][2][0]
    brY2 = bb2['poly_points'][2][1]
    blX2 = bb2['poly_points'][3][0]
    blY2 = bb2['poly_points'][3][1]

    lX1 = (tlX1+blX1)/2
    rX1 = (trX1+brX1)/2
    lY1 = (tlY1+blY1)/2
    rY1 = (trY1+brY1)/2
    lX2 = (tlX2+blX2)/2
    rX2 = (trX2+brX2)/2
    lY2 = (tlY2+blY2)/2
    rY2 = (trY2+brY2)/2

    h1 = ((blX1-tlX1)+(brX1-blX1))/2
    h2 = ((blX2-tlX2)+(brX2-blX2))/2

    thresh = 1.2*max(h1,h2)
    

    dist = math.sqrt(min( (lX1-rX2)**2 + (lY1-rY2)**2, (lX2-rX1)**2 + (lY2-rY1)**2 ))
    return dist>thresh

def formGroups(annotations,group_only_same=False):
    #printTypes(annotations)
    groups={}
    groupMap={}
    curGroupId=0
    rot_diff = 40/180.0 * np.pi
    rightThresh = -160/180.0 * np.pi
    leftThresh = -20/180.0 * np.pi

    #These are for collecting info used to split bad groups later
    relative_rel_angles=defaultdict(dict)
    hasMinorNeighbor = defaultdict(lambda: False)
    textNeighbors = defaultdict(list)

    for pair in annotations['pairs']:
        if annotations['byId'][pair[0]]['type']=='text' and annotations['byId'][pair[1]]['type']=='text':
            textNeighbors[pair[0]].append(pair[1])
            textNeighbors[pair[1]].append(pair[0])
        if annotations['byId'][pair[0]]['type']=='textMinor' and annotations['byId'][pair[1]]['type']=='field':
            hasMinorNeighbor[pair[1]]=True
        elif annotations['byId'][pair[1]]['type']=='textMinor' and annotations['byId'][pair[0]]['type']=='field':
            hasMinorNeighbor[pair[0]]=True
        #this is the rule that determines which bbs get grouped
        if group_only_same:
            should_be_grouped = (
                    (annotations['byId'][pair[0]]['type'] == annotations['byId'][pair[1]]['type']) 
                and (
                    computeRotationDiff(annotations['byId'][pair[0]],annotations['byId'][pair[1]]) < rot_diff and
                    (annotations['byId'][pair[0]]['type']!='textMinor' or horizontalOverlap(annotations['byId'][pair[0]],annotations['byId'][pair[1]]) > 0.2 )
              ) and (
                    #horz_overlapped
                    horizontalOverlap(annotations['byId'][pair[0]],annotations['byId'][pair[1]]) > 0
              ) )
        else:
            #if you have the same label, if your joined para, if you are a minor text or circle that's in para
            should_be_grouped = ( (
                (annotations['byId'][pair[0]]['type'] == annotations['byId'][pair[1]]['type']) or
                ('P' in annotations['byId'][pair[0]]['type'] and 'P' in annotations['byId'][pair[1]]['type']) or
                ( ('fieldP' in annotations['byId'][pair[0]]['type'] or 'fieldP' in annotations['byId'][pair[1]]['type']) and ('textMinor' in annotations['byId'][pair[0]]['type'] or 'textMinor' in annotations['byId'][pair[1]]['type']) )
                or ( ('P' in annotations['byId'][pair[0]]['type'] or 'P' in annotations['byId'][pair[1]]['type']) and ('Circle' in annotations['byId'][pair[0]]['type'] or 'Circle' in annotations['byId'][pair[1]]['type']) )
                ) and (
                    computeRotationDiff(annotations['byId'][pair[0]],annotations['byId'][pair[1]]) < rot_diff and
                    (annotations['byId'][pair[0]]['type']!='textMinor' or horizontalOverlap(annotations['byId'][pair[0]],annotations['byId'][pair[1]]) > 0.2 )
                    #(annotations['byId'][pair[0]]['type']!='textMinor' or connectionNotParallel(annotations['byId'][pair[0]],annotations['byId'][pair[1]]) )
                ) )
        if should_be_grouped:
                
            #print('adding grouping between: {} and {}'.format(annotations['byId'][pair[0]]['type'],annotations['byId'][pair[1]]['type']))

            #add to appropriate group or form a new one
            if pair[0] not in groupMap and pair[1] not in groupMap:
                groups[curGroupId] = list(pair)
                groupMap[pair[0]] = curGroupId
                groupMap[pair[1]] = curGroupId
                curGroupId+=1
            elif pair[1] not in groupMap:
                groupId = groupMap[pair[0]]
                groups[groupId].append(pair[1])
                groupMap[pair[1]]=groupId
            elif pair[0] not in groupMap:
                groupId = groupMap[pair[1]]
                groups[groupId].append(pair[0])
                groupMap[pair[0]]=groupId
            elif groupMap[pair[1]] != groupMap[pair[0]]:
                goneGroupId = groupMap[pair[1]]
                goneGroup = groups[goneGroupId]
                del groups[goneGroupId]
                groupId = groupMap[pair[0]]
                groups[groupId] += goneGroup
                for bbId in goneGroup:
                    groupMap[bbId]=groupId

            #store angle of relationship for later processing
            rot0 = computeRotation(annotations['byId'][pair[0]])
            rot1 = computeRotation(annotations['byId'][pair[1]])

            cx0,cy0 = getCenterPoint(annotations['byId'][pair[0]])
            cx1,cy1 = getCenterPoint(annotations['byId'][pair[1]])

            a0_to_1 = math.atan2(cy1-cy0,cx1-cx0)
            a1_to_0 = math.atan2(cy0-cy1,cx0-cx1)
            rela0 = a0_to_1 - rot0
            rela1 = a1_to_0 - rot1
            if rela0>np.pi:
                rela0-=2*np.pi
            elif rela0<-np.pi:
                rela0+=2*np.pi
            if rela1>np.pi:
                rela1-=2*np.pi
            elif rela1<-np.pi:
                rela1+=2*np.pi
            relative_rel_angles[pair[0]][pair[1]]= rela0
            relative_rel_angles[pair[1]][pair[0]]= rela1

    #Now, we're going to examine each group to see if it needs split
    removeGroupIds=[]
    allNewGroups=[]
    for groupId,group in groups.items():
        splitCandidates=[] #These will be instances of side-by-side paragraphs, particularly the last line in a paragraph
        splitAllTextDownCandidates=[]#These are instances of a title with sub-texts below it
        allHaveMinorNeighbor=True
        for bbId in group:
            bbType = annotations['byId'][bbId]['type']
            downTexts=[] if bbType=='text' or bbType=='textMinor'  else None
            #toprint='{}[{}]: '.format(bbId,annotations['byId'][bbId]['type'])
            upPairs=[]
            angles=[]
            for otherId,angle in relative_rel_angles[bbId].items():
                #toprint+='[{}] {}, '.format(annotations['byId'][otherId]['type'],angle)
                #is the neighbor above?
                if leftThresh>angle and angle>rightThresh: # and areFar(annotations['byId'][bbId],annotations['byId'][otherId]):
                    upPairs.append(otherId)
                    angles.append(angle)

                #if downTexts is not None and annotations['byId'][otherId]['type']=='text':
                #    print('{} - {}: {}'.format(bbId,otherId,angle))
                if downTexts is not None and annotations['byId'][otherId]['type']==bbType and angle<np.pi and angle>0:
                    downTexts.append(otherId)
            if len(upPairs)>1:
                
                splitCandidates.append((bbId,upPairs))
                #print('split cand: {} {}'.format(bbId,list(zip(upPairs,angles))))
            if downTexts is not None and ( (len(downTexts)>2 and bbType=='text') or (len(downTexts)>1 and bbType=='textMinor')):
                splitAllTextDownCandidates.append((bbId,downTexts))
                #print('Down split {}[{}]: {}'.format(bbId,bbType,downTexts))
                    
            #print(toprint)
            if not hasMinorNeighbor[bbId]:
                allHaveMinorNeighbor=False
        if allHaveMinorNeighbor: #This is not group fields that are together, but have individual minor labels
            assert(len(splitCandidates)==0)
            removeGroupIds.append(groupId)
            allNewGroups+=[[bbId] for bbId in group]

        recreate=False
        newPairs = list(annotations['pairs'])
        if len(splitCandidates)>0:
            recreate=True
            #print('split cand {}'.format(splitCandidates))
            
            for bbId,upPairs in splitCandidates:
                cx,cy = getCenterPoint(annotations['byId'][bbId])
                furthestDist=0
                cutThis=None
                for otherId in upPairs:
                    cxo,cyo = getCenterPoint(annotations['byId'][otherId])
                    dist = ((cx-cxo)**2) + ((cy-cyo)**2)
                    if dist>furthestDist:
                        furthestDist=dist
                        cutThis=otherId
                if [bbId,cutThis] in newPairs:
                    newPairs.remove([bbId,cutThis])
                elif [cutThis,bbId] in newPairs:
                    newPairs.remove([cutThis,bbId])

        for headId, subIds in splitAllTextDownCandidates:
            recreate=True
            for subId in subIds:
                if [headId,subId] in newPairs:
                    newPairs.remove([headId,subId])
                elif [subId,headId] in newPairs:
                    newPairs.remove([subId,headId])

        if recreate:
            #recreate groups
            newGroups={}
            newGroupMap={}
            for bbId in group:
                partOf=[]
                #for otherId,newGroupId in newGroupMap.items():
                #    if (bbId,otherId) in newPairs or (otherId,bbId) in newPairs:
                #        partOf.append(newGroupId)
                for newGroupId,newGroup in newGroups.items():
                    for otherId in newGroup:
                        if [bbId,otherId] in newPairs or [otherId,bbId] in newPairs:
                            partOf.append(newGroupId)
                            break
                if len(partOf)>1:
                    #merge
                    finalGroupId = partOf[0]
                    for goneGroupId in partOf[1:]:
                        newGroups[finalGroupId]+=newGroups[goneGroupId]
                        #for otherId in 0-
                        del newGroups[goneGroupId]
                if len(partOf)>0:
                    newGroups[partOf[0]].append(bbId)
                else:
                    newGroups[curGroupId]=[bbId]
                    curGroupId+=1

            allNewGroups+=[group for gid,group in newGroups.items()]
            removeGroupIds.append(groupId)




    groups = [group for gid,group in groups.items() if gid not in removeGroupIds]
    groups += allNewGroups

    #groups of single elements
    for bbId in annotations['byId']:
        if bbId not in groupMap:
            groups.append([bbId])
    
    #for group in groups:
    #    toprint=''
    #    for bbId in group:
    #        toprint+=annotations['byId'][bbId]['type']+', '
    #    toprint+=':'
    #    print(toprint)

    return groups

def printTypes(annotations):
    toprint=''
    n=6
    for i,(bbId,bb) in enumerate(annotations['byId'].items()):
        if len(bb['type'])>7:
            toprint+=bb['type']+'\t'
        else:
            toprint+=bb['type']+'\t\t'
        if i%n==n-1:
            print(toprint)
            toprint=''
    print(toprint)
