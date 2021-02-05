from utils.yolo_tools import newGetTargIndexForPreds_textLines
from utils.bb_merging import TextLine
import torch
import utils.img_f as img_f
import numpy as np
from evaluators.draw_graph import plotRect
from utils.forms_annotations import calcCorners
from utils.yolo_tools import classPolyIOU
#def plotRect(img,color,xyrhw,lineWidth=1):
#    tl,tr,br,bl=getCorners(xyrhw)
#
#    img_f.line(img,tl,tr,color,lineWidth)
#    img_f.line(img,tr,br,color,lineWidth)
#    img_f.line(img,br,bl,color,lineWidth)
#    img_f.line(img,bl,tl,color,lineWidth)

gt = [
        [50,15,0,15,50,1,0], #0
        [50,65,0,15,50,1,0], #1
        [50,115,0,15,50,1,0],#2
        [50,165,0,15,50,1,0],#3
        [50,215,0,15,50,1,0],#4
        [50,265,0,15,50,1,0],#5
        ]
gt = torch.FloatTensor(gt)

pred = [
        [0.9,54,16,0,14,49,1,0], #0 0
        [0.9,29,67,0,14,29,1,0], #1 1
        [0.9,82,63,0,14,20,1,0], #  2

        [0.9,54,116,0,14,49,1,0],#2 3
        [0.9,54,116,0,17,49,1,0],#  4

        [0.9,15,166,0,15,16,1,0],#3 5
        [0.9,46,166,0,15,16,1,0],#  6
        [0.9,85,166,0,15,16,1,0],#  7
        [0.9,29,167,0,18,29,1,0],#  8
        [0.9,82,163,0,18,20,1,0],#  9

        [0.9,50,216,0,14,49,1,0],#4 10
        [0.9,29,217,0,18,29,1,0],#  11
        [0.9,82,213,0,18,20,1,0],#  12

        [0.9,15,266,0,15,16,1,0],#5 13
        [0.9,46,266,0,15,16,1,0],#  14
        [0.9,85,266,0,15,16,1,0],#  15
        [0.9,54,266,0,17,49,1,0],#  16
        ]
#pred = torch.FloatTensor(pred)
pred_corners=[]
for p in pred:
    tl,tr,br,bl=calcCorners(*p[1:6])#getCorners(p[1:])
    pred_corners.append([0.9,tl[0],tl[1],br[0],br[1],0,1,0])
pred_corners = torch.FloatTensor(pred_corners)
pred_lines=[]
for p in pred_corners:
    #print(p)
    pred_lines.append(TextLine(p))



targs = newGetTargIndexForPreds_textLines(gt,pred_lines,0.5,2,True,True)
print(targs)

canvas = np.zeros([400,200,3],dtype=np.uint)
for g in gt:
    plotRect(canvas,(0,255,0),g)

print('============')
for i,p in enumerate(pred_lines):
    if targs[i]==-1:
        color = (255,0,0)
    else:
        color = (0,0,255)
    #plotRect(canvas,color,p[1:])
    #print(p)
    pts = p.polyPoints()
    pts = pts.reshape((-1,1,2))
    img_f.polylines(canvas,pts.astype(np.int),'transparent',color)

#1
merged = TextLine(pred_corners[1])
merged.merge(TextLine(pred_corners[2]))
pts = merged.polyPoints()
pts = pts.reshape((-1,1,2))
img_f.polylines(canvas,pts.astype(np.int),'transparent',(240,0,240))
iou = classPolyIOU(gt[1:2],[merged],0)[0]
print(iou)

#3
merged = TextLine(pred_corners[5])
merged.merge(TextLine(pred_corners[6]))
merged.merge(TextLine(pred_corners[7]))
pts = merged.polyPoints()
pts = pts.reshape((-1,1,2))
img_f.polylines(canvas,pts.astype(np.int),'transparent',(240,0,240))
iou = classPolyIOU(gt[3:4],[merged],0)[0]
print(iou)

#5
merged = TextLine(pred_corners[13])
merged.merge(TextLine(pred_corners[14]))
merged.merge(TextLine(pred_corners[15]))
pts = merged.polyPoints()
pts = pts.reshape((-1,1,2))
img_f.polylines(canvas,pts.astype(np.int),'transparent',(240,0,240))
iou = classPolyIOU(gt[5:6],[merged],0)[0]
print(iou)

img_f.imshow('im',canvas)
img_f.show()
