#from model.optimize import optimizeRelationships
#import torch
#relPred=torch.tensor([.5,.4,.6,.2,.3,.1,.1])
#relNodes=[(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)]
#gtNodeN=[2,1,1,1,2]
#res=optimizeRelationships(relPred,relNodes,gtNodeN)
#import pdb;pdb.set_trace()
#print(res)
#

import numpy as np
import matplotlib.pyplot as plt

iterations=100000

warmup_steps=10000
def lrf(step_num):
    return min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)

cycle_size=5000
decay_rate=0.99995
min_lr_mul=0.0001
low_lr_mul=0.3
def decayCycle (step_num):
                cycle_num = step_num//cycle_size
                decay = decay_rate**step_num
                if cycle_num%2==0: #even, rising
                    return decay*((1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + min_lr_mul
                else: #odd
                    return -decay*(1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1) + 1-(1-min_lr_mul)*(1-decay)

iters_in_trailoff = iterations-(2*cycle_size)
def oneCycle (step_num):
                cycle_num = step_num//cycle_size
                if step_num<cycle_size: #rising
                    return ((1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + low_lr_mul
                elif step_num<cycle_size*2: #falling
                    return (1-(1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
                else: #trail off
                    t_step_num = step_num-(2*cycle_size)
                    return low_lr_mul*(iters_in_trailoff-t_step_num)/iters_in_trailoff + min_lr_mul*t_step_num/iters_in_trailoff

x = [i for i in range(0,iterations)]
#y = [lrf(i) for i in x]
#y = [decayCycle(i) for i in x]
y = [oneCycle(i) for i in x]
y=np.array(y)

print('max val: {}'.format(y.max()))
print('mean val: {}'.format(y.mean()))
plt.plot(x,y,'.')
plt.show()
