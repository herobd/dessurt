from utils.yolo_tools import newGetTargIndexForPreds_textLines
from utils.bb_merging import TextLine
import torch
import utils.img_f as img_f
from utils.forms_annotations import calcCorners
def plotRect(img,color,xyrhw,lineWidth=1):
    tl,tr,br,bl=getCorners(xyrhw)

    img_f.line(img,tl,tr,color,lineWidth)
    img_f.line(img,tr,br,color,lineWidth)
    img_f.line(img,br,bl,color,lineWidth)
    img_f.line(img,bl,tl,color,lineWidth)

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
        [0.9,54,16,0,14,49,1,0], #0
        [0.9,29,67,0,14,29,1,0], #1
        [0.9,82,63,0,14,20,1,0],

        [0.9,54,116,0,14,49,1,0], #2
        [0.9,54,116,0,17,49,1,0],

        [0.9,15,166,0,15,16,1,0], #3
        [0.9,46,166,0,15,16,1,0],
        [0.9,75,166,0,15,16,1,0],
        [0.9,29,67,0,18,29,1,0],
        [0.9,82,63,0,18,20,1,0],

        [0.9,54,16,0,14,49,1,0], #4
        [0.9,29,67,0,18,29,1,0],
        [0.9,82,63,0,18,20,1,0],

        [0.9,15,166,0,15,16,1,0], #5
        [0.9,46,166,0,15,16,1,0],
        [0.9,75,166,0,15,16,1,0],
        [0.9,54,116,0,17,49,1,0],
        ]
pred = torch.FloatTensor(pred)
pred_lines=[]
for p in pred:
    pred_lines.append(TextLine(p))



targs = newGetTargIndexForPreds_textLines(gt,pred_lines,0.5,2,True,True)
print(targs)

canvas = np.zeros(200,400,3,dtype=np.uint)
for g in gt:
    plotRect(canvas,(0,256,0),g)

for i,p in enumerate(pred):
    if targs[i]==-1:
        color = (256,0,0)
    else:
        color = (0,0,256)
    plotRect(canvas,color,p)

img_f.imshow('im',canvas)
img_f.show()
