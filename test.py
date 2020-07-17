#from model.optimize import optimizeRelationships
#import torch
#relPred=torch.tensor([.5,.4,.6,.2,.3,.1,.1])
#relNodes=[(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)]
#gtNodeN=[2,1,1,1,2]
#res=optimizeRelationships(relPred,relNodes,gtNodeN)
#import pdb;pdb.set_trace()
#print(res)
#

#import numpy as np
#import matplotlib.pyplot as plt
#
#iterations=100000
#
#warmup_steps=10000
#def lrf(step_num):
#    return min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)
#
#cycle_size=5000
#decay_rate=0.99995
#min_lr_mul=0.0001
#low_lr_mul=0.3
#def decayCycle (step_num):
#                cycle_num = step_num//cycle_size
#                decay = decay_rate**step_num
#                if cycle_num%2==0: #even, rising
#                    return decay*((1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + min_lr_mul
#                else: #odd
#                    return -decay*(1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1) + 1-(1-min_lr_mul)*(1-decay)
#
#iters_in_trailoff = iterations-(2*cycle_size)
#def oneCycle (step_num):
#                cycle_num = step_num//cycle_size
#                if step_num<cycle_size: #rising
#                    return ((1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + low_lr_mul
#                elif step_num<cycle_size*2: #falling
#                    return (1-(1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
#                else: #trail off
#                    t_step_num = step_num-(2*cycle_size)
#                    return low_lr_mul*(iters_in_trailoff-t_step_num)/iters_in_trailoff + min_lr_mul*t_step_num/iters_in_trailoff
#
#x = [i for i in range(0,iterations)]
##y = [lrf(i) for i in x]
##y = [decayCycle(i) for i in x]
#y = [oneCycle(i) for i in x]
#y=np.array(y)
#
#print('max val: {}'.format(y.max()))
#print('mean val: {}'.format(y.mean()))
#plt.plot(x,y,'.')
#plt.show()
from model.oversegment_loss import build_oversegmented_targets_multiscale
import torch
import math,random

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
#scale = [ (16,16), (32,32), (64,64) ]
scale = [ (32,32), (64,64) ]
grid_sizesH=[H//s[0] for s in scale]
grid_sizesW=[W//s[0] for s in scale]
pred_boxes = [torch.zeros(1,1,1,1,1)]*3
pred_cls = [torch.zeros(1,1,1,1,1)]*3
pred_conf = [torch.zeros(1,1,1,1)]*3

targs=[]
#varying sizes
yb=100
t=0
for h in range(8,15,4):
    w = h*2
    r = 0#-math.pi*(7/10)
    y = yb
    for x in range(100,300,300):
        lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
        targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])
        t+=1
        y+=1
        #r += math.pi/10
        #if r>math.pi:
        #    r-=math.pi*2
        #r = (random.random()*math.pi/2)-math.pi/4
    yb+=0.5*(h+w)

#Totally random
#t=50
#for i in range(t):
#    w = random.random()*1000+8
#    h = random.random()*50+8
#    x = random.random()*(H-200) +100
#    y = random.random()*(W-200) +100
#    #r = (random.random()*math.pi*2)-math.pi
#    sig = math.pi/24
#    if random.random()>0.2:
#        if random.random()>0.5:
#            r = random.gauss(0,sig)
#        else:
#            r = random.gauss(math.pi,sig)
#    else:
#        if random.random()>0.5:
#            r = random.gauss(math.pi/2,sig)
#        else:
#            r = random.gauss(-math.pi/2,sig)
#    if r>math.pi:
#        r-=math.pi*2
#    elif r<-math.pi:
#        r+=math.pi*2
#    lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
#    targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])

target_sizes= [t]
target= torch.FloatTensor(1,t,13+1)
for t,targ in enumerate(targs):
    target[0,t,:13]=torch.FloatTensor(targ)

build_oversegmented_targets_multiscale(pred_boxes, pred_conf, pred_cls, target, target_sizes, num_classes, grid_sizesH, grid_sizesW,scale=scale, assign_mode='split', close_anchor_rule='unmask')
