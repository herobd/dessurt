import json
from utils import img_f
import numpy as np
import os,sys

def parseGT(directory,name):
    with open(os.path.join(directory,'jsons',name+'.json')) as f:
        data=json.load(f)

    image = img_f.imread(os.path.join(directory,'imgs',name+'.png'))
    if len(image.shape)==2:
        image= np.stack([image,image,image],axis=2)

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

    inter_rect_x1 = np.max(x1s[:,None],x1s[None,:])
    inter_rect_x2 = np.min(x2s[:,None],x2s[None,:])
    inter_rect_y1 = np.max(y1s[:,None],y1s[None,:])
    inter_rect_y2 = np.min(y2s[:,None],y2s[None,:])

    inter_area = np.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * np.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )
    areas = (x2-x1+1)*(y2-y1_1)

    insides = (inter_area/areas[None,:])>0.9


    #top to bottom
    #first choice groups, tables and lists
    #fields, choice fields
    #be sure any nested fields, etc are processed before their parent so they properly claim their childrem

    if 'ChoiceGroup' in data:
        for i,bb in enumerate(data['ChoiceGroup']):
            bb_i = all_bbs.index(('ChoiceGroup',i))
            x=round(bb['x'])
            y=round(bb['y'])
            w=round(bb['w'])
            h=round(bb['h'])
            t=bb['jsonClass']

            x2=x+w
            y2=y+h


            np.nonzero(inside[bb_i])

    print(list(data.keys()))

    for section,ldata in data.items():
        for bb in ldata:
            x=round(bb['x'])
            y=round(bb['y'])
            w=round(bb['w'])
            h=round(bb['h'])
            t=bb['jsonClass']

            x2=x+w
            y2=y+h

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

    img_f.imshow('image',image)
    img_f.show()


parseGT(sys.argv[1],sys.argv[2])


