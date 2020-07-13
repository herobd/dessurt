from model.oversegment_loss import build_oversegmented_targets_multiscale
import torch
import math

def calcPoints(x,y,r,h,w):
    rx = x + math.cos(r)*w
    ry = y - math.sin(r)*w
    lx = x - math.cos(r)*w
    ly = y + math.sin(r)*w

    tx = x - math.sin(r)*h
    ty = y - math.cos(r)*h
    bx = x + math.sin(r)*h
    by = y + math.cos(r)*h

    return lx,ly,rx,ry,tx,ty,bx,by

num_classes=1
H=1500
W=1500
scale = [ (16,16), (32,32), (64,64) ]
grid_sizesH=[H//s[0] for s in scale]
grid_sizesW=[W//s[0] for s in scale]
pred_boxes = [torch.zeros(1,1,1,1,1)]*3
pred_cls = [torch.zeros(1,1,1,1,1)]*3
pred_conf = [torch.zeros(1,1,1,1)]*3

#varying sizes
yb=100
t=0
targs=[]
for h in range(8,40,4):
    w = h*2
    r = 0#-math.pi*(7/10)
    y = yb
    for x in range(100,1300,150):
        lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
        targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])
        t+=1
        y+=1
        r += math.pi/10
        if r>math.pi:
            r-=math.pi*2
    yb+=1.5*(h+w)
target_sizes= [t]
target= torch.FloatTensor(1,t,13+1)
for t,targ in enumerate(targs):
    target[0,t,:13]=torch.FloatTensor(targ)

build_oversegmented_targets_multiscale(pred_boxes, pred_conf, pred_cls, target, target_sizes, num_classes, grid_sizesH, grid_sizesW,scale=scale, assign_mode='split', close_anchor_rule='unmask')
