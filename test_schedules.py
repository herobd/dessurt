

import numpy as np
import matplotlib.pyplot as plt

iterations=70000

warmup_steps=1000
steps=[1000,2000]
down_steps= 5000
swa_start=60000
warmup_cap=1
swa_lr_mul=0.1

lr_lambda = lambda step_num: min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)

lr_lambda2 = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)

def lrf(step_num):
    return min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)

steps=[0]+steps
def riseLR(step_num):
                if step_num<swa_start-down_steps:
                    for i,step in enumerate(steps[1:]):
                        if step_num<step:
                            return warmup_cap*(step_num-steps[i])/(step-steps[i])
                    return 1.0
                elif step_num<swa_start:
                    return 1 - (1-swa_lr_mul)*(down_steps-(swa_start-step_num))/down_steps
                else:
                    return swa_lr_mul

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
y = [lr_lambda(i) for i in x]
y=np.array(y)

print('max val: {}'.format(y.max()))
print('mean val: {}'.format(y.mean()))
plt.plot(x,y,'r.')

x2 = [i for i in range(0,iterations)]
y2 = [riseLR(i) for i in x]
y2=np.array(y2)

print('max val: {}'.format(y.max()))
print('mean val: {}'.format(y.mean()))
plt.plot(x2,y2,'b.')
plt.show()
