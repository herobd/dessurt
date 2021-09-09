import torch
from utils.util import calcXYWH
from model.qa_imdoc_gpt3 import affineTransform


char_prob = torch.FloatTensor([
    [1,1,0],
    [0.8,1,0.2],
    [0.6,1,0.5],
    [0.4,1,0.7],
    [0.2,1,0.9],
    [0,1,0.9],
    ])

points=[]
tlX,tlY = (50,25)
trX,trY = (60,3)
brX,brY = (80,30)
blX,blY = (70,50)
points.append([tlX,tlY,trX,trY,brX,brY,blX,blY])
tlX,tlY = (1,1)
trX,trY = (3,1)
brX,brY = (3,3)
blX,blY = (1,4)
points.append([tlX,tlY,trX,trY,brX,brY,blX,blY])
tlX,tlY = (3,1)
trX,trY = (9,7)
brX,brY = (8,9)
blX,blY = (1,4)
points.append([tlX,tlY,trX,trY,brX,brY,blX,blY])
tlX,tlY = (3,1)
trX,trY = (90,70)
brX,brY = (80,90)
blX,blY = (1,4)
points.append([tlX,tlY,trX,trY,brX,brY,blX,blY])
tlX,tlY = (60,3)
trX,trY = (80,30)
brX,brY = (70,50)
blX,blY = (50,25)
points.append([tlX,tlY,trX,trY,brX,brY,blX,blY])

for  tlX,tlY,trX,trY,brX,brY,blX,blY in points:
    canvas = torch.FloatTensor(3,100,100).fill_(-1)
    lX,lY,rX,rY,width,height,rot = calcXYWH(tlX,tlY,trX,trY,brX,brY,blX,blY)
    print(rot)

    left_x = min(tlX,trX,blX,brX)
    top_y = min(tlY,trY,blY,brY)
    h=max(tlY,trY,blY,brY) - min(tlY,trY,blY,brY)
    w=max(tlX,trX,blX,brX) - min(tlX,trX,blX,brX)
    patch_size = (1,3,h,w)
    print(width/char_prob.size(0),width,height)
    scalew = w/width
    scaleh = h/height
    im_patch = affineTransform(
          char_prob.permute(1,0)[None,:,None],#.expand(1,-1,5,-1),#make sequance an image,
          patch_size, #canvas to draw in
          scalew,#(width/char_prob.size(0)), #strech or shrink char prob to fit
          scaleh,#(height), #expand height of char prob to fill vert space
          rot)
    #print(im_patch.size())
    #print(im_patch)

    mask = im_patch[0].sum(dim=0)!=0
    canvas[:,top_y:top_y+h,left_x:left_x+w][:,mask] = im_patch[0][:,mask]
    #canvas[:,top_y:top_y+h,left_x:left_x+w] = im_patch[0]

    from utils import img_f
    import numpy as np
    canvas=(canvas.permute(1,2,0).numpy()+1)*(255/2)
    img_f.line(canvas,(tlX,tlY),(trX,trY),(255,0,0))
    img_f.line(canvas,(brX,brY),(trX,trY),(255,0,0))
    img_f.line(canvas,(brX,brY),(blX,blY),(255,0,0))
    img_f.line(canvas,(tlX,tlY),(blX,blY),(255,0,0))
    img_f.imshow('x',canvas.astype(np.uint8))
    img_f.show()
