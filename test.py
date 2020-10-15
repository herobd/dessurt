from model.oversegment_loss import build_oversegmented_targets_multiscale
from model.overseg_box_detector import build_box_predictions
from utils.bb_merging import TextLine
import torch
import torch.nn.functional as F
import utils.img_f as img_f
import numpy as np
import math,random
from utils.forms_annotations import calcCorners
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

def drawPoly(img,pts,color,thck=1):
    for i in range(-1,len(pts)-1):
        img_f.line(img,(int(pts[i][0]),int(pts[i][1])),(int(pts[i+1][0]),int(pts[i+1][1])),color,thck)

num_classes=2
numBBTypes=2
numAnchors=4
numBBParams=6
H=300
W=700
#scale = [ (16,16), (32,32), (64,64) ]
scale = [ (32,32), (64,64) ]
grid_sizesH=[H//s[0] for s in scale]
grid_sizesW=[W//s[0] for s in scale]
pred_boxes = [torch.zeros(1,1,1,1,1)]*3
pred_cls = [torch.zeros(1,1,1,1,1)]*3
pred_conf = [torch.zeros(1,1,1,1)]*3

targs=[]
#varying sizes
#yb=100
#t=0
#for h in range(40,41,4):
#    w = h*3
#    r = math.pi/10#-math.pi*(7/10)
#    y = yb
#    for x in range(150,300,300):
#        lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
#        targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])
#        t+=1
#        y+=1
#        r += math.pi/10
#        if r>math.pi:
#            r-=math.pi*2
#        #r = (random.random()*math.pi/2)-math.pi/4
#    yb+=1.1*(h+w)

boxes = [
        [450,110,-math.pi*3/10,40,120],
        #[150,110,-math.pi*7/10,40,120],
        #[150,110,-math.pi*1/10,40,120],
        [299,105,math.pi/20,40,40],
        [345,80,-math.pi/40,40,80],
        #[140,100,0,40,120]
        ]
t=len(boxes)
for x,y,r,h,w in boxes:
    lx,ly,rx,ry,tx,ty,bx,by = calcPoints(x,y,r,h,w)
    targs.append([x,y,r,h,w,lx,ly,rx,ry,tx,ty,bx,by])

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

nGT, masks, conf_masks, t_Ls, t_Ts, t_Rs, t_Bs, t_rs, tconf_scales, tcls_scales, pred_covered, gt_covered, recall, precision, pred_covered_noclass, gt_covered_noclass, recall_noclass, precision_noclass = build_oversegmented_targets_multiscale(pred_boxes, pred_conf, pred_cls, target, target_sizes, num_classes, grid_sizesH, grid_sizesW,scale=scale, assign_mode='split', close_anchor_rule='unmask')


# Handle target variables
t_Ls = [t.type(torch.FloatTensor) for t in t_Ls]
t_Ts = [t.type(torch.FloatTensor) for t in t_Ts]
t_Rs = [t.type(torch.FloatTensor) for t in t_Rs]
t_Bs = [t.type(torch.FloatTensor) for t in t_Bs]
t_rs = [t.type(torch.FloatTensor) for t in t_rs]
tconf_scales = [t.type(torch.FloatTensor) for t in tconf_scales]
tcls_scales = [t.type(torch.FloatTensor) for t in tcls_scales]

ys = []
for level in range(len(t_Ls)):
    level_y = torch.cat([ torch.stack([tconf_scales[level],t_Ls[level],t_Ts[level],t_Rs[level], t_Bs[level],t_rs[level]],dim=2), tcls_scales[level].permute(0,1,4,2,3)], dim=2)
    ys.append(level_y.view(level_y.size(0),level_y.size(1)*level_y.size(2),level_y.size(3),level_y.size(4)))

    display_y = level_y[0].permute(2,3,0,1).contiguous().view(-1,level_y.size(2))

    #print('level_y: {}'.format(display_y[display_y[:,0]>0.5]))
gt_boxes = build_box_predictions(ys,scale,ys[0].device,numAnchors,numBBParams,numBBTypes)
assert((gt_boxes[0,gt_boxes[0,:,0]>0.5,2] < gt_boxes[0,gt_boxes[0,:,0]>0.5,4]).all())
#print('gt: {}'.format(gt_boxes[0,gt_boxes[0,:,0]>0.5]))

img =np.zeros([H,W,3]).astype(np.uint8)
#draw pattern on img
x=W//2
y=H//2
length=1
cur_l=1
colors=[(100,0,0),(100,100,0),(0,100,0),(0,100,100),(0,0,100),(100,0,100)]
color_idx=0
dir_x=1
dir_y=0
while x<W and y<H and x>=0 and y>=0:
    img[y,x]=colors[color_idx]
    x+=dir_x
    y+=dir_y
    cur_l-=1
    if cur_l<=0:
        length+=1
        cur_l=length
        color_idx=(color_idx+1)%len(colors)
        old_dir_x=dir_x
        if dir_x==1 or dir_x==-1:
            dir_x=0
        elif dir_y==1:
            dir_x=-1
        else:
            dir_x=1
        if dir_y==1 or dir_y==-1:
            dir_y=0
        elif old_dir_x==1:
            dir_y=1
        else:
            dir_y=-1
##
for t in targs:
    corners = calcCorners(*t[0:5])
    #img_f.polylines(img,np.array([[t[5],t[10]],[t[7],t[10]],[t[7],t[12]],[t[5],t[12]]],np.int32).reshape((-1,1,2)),True,(0,255,0),3)
    #drawPoly(img,[[t[5],t[10]],[t[7],t[10]],[t[7],t[12]],[t[5],t[12]]],(0,255,0),3)
    #print([[t[5],t[10]],[t[7],t[10]],[t[7],t[12]],[t[5],t[12]]])
    drawPoly(img,corners,(0,255,0),3)


all_textlines=[]
for bb in gt_boxes[0]:
    if bb[0]>0.5:
        #all_textlines.append(TextLine(torch.FloatTensor([1,t[5],t[10],t[7],t[12],0,0,1])))
        #img_f.rectangle(img,(t[5],t[10]),(t[7],t[12]),(0,0,255),1)
        all_textlines.append(TextLine(torch.FloatTensor(bb)))
        #img_f.rectangle(img,bb[1:3].numpy(),bb[3:5].numpy(),(0,0,255))
        #img_f.line(img,(round(bb[1].item()),round(bb[2].item())),(round(bb[3].item()),round(bb[4].item())),(0,0,255))
        img_f.rectangle(img,(round(bb[1].item()),round(bb[2].item())),(round(bb[3].item()),round(bb[4].item())),(0,0,255),2)
        print(bb)

first_textline=all_textlines[0]

for tl in all_textlines[1:]:
    first_textline.merge(tl)


#img_f.polylines(img,np.array(first_textline.polyPoints(),np.int32).reshape((-1,1,2)),True,(255,0,0),1)
drawPoly(img,first_textline.polyPoints(),(255,0,0),1)
#for p in first_textline.polyPoints():
#    img[int(p[1]),int(p[0]),1]=255
first_textline.point_pairs=[[[65.21669158964258, 77.14328562594461], [65.2075060002131, 88.9007364792096]], [[96.61047620053407, 77.16781223631895], [96.60129061110459, 88.92526308958392]], [[161.22792336099965, 82.31366184896892], [155.33209246401657, 95.68772699007339]], [[186.27192437994506, 152.58398816607735], [183.98673040521123, 153.02354357220406]], [[199.7422775571269, 222.30047877453023], [197.34052273584177, 222.762454575079]], [[213.2126307343089, 292.0169693829836], [210.69431506647246, 292.5013655779543]], [[226.68298391149077, 361.73345999143646], [224.04810739710297, 362.24027658082923]], [[239.99088060733126, 413.6787780543838], [233.70542765067663, 415.42193623116145]], [[255.53034298604817, 408.48283047016116], [264.21479200516984, 428.36429622971156]], [[277.52670079412866, 371.97737703983876], [277.74250086602757, 371.9806364947108]], [[278.6009160599562, 301.4923288382379], [278.7975092330128, 301.49529819117015]], [[279.6751313257834, 231.00728063665156], [279.85251759999784, 231.00995988764043]], [[280.7493465916107, 160.5222324350616], [280.907525966983, 160.52462158410708]], [[281.82356185743816, 90.03718423346436], [281.9625343339682, 90.03928328057009]], [[291.50563029203437, 70.74403781843006], [295.4305020743513, 68.37452961671698]], [[300.32287750858063, 122.38241149886198], [301.3423533786961, 122.28138233745949]], [[306.92629331507294, 189.92752758575762], [308.12450537354306, 189.80878582540936]], [[313.5297091215654, 257.4726436726537], [314.90665736839, 257.3361893133597]], [[337.74602111910644, 285.4644266971028], [329.45439757303404, 273.9892127142956]], [[343.3936065527864, 217.0338411772087], [339.59540711908863, 228.67654192012208]], [[342.9042376788019, 148.30557038514735], [340.0847759930731, 160.77010303607662]], [[358.49675900157996, 102.71485213390011], [351.6114951865873, 89.53335538193892]], [[385.98489066294115, 118.69956388229889], [390.6884824727121, 116.87907852936667]], [[410.3432746137601, 181.72970621944467], [415.11104889476275, 179.88437958488805]], [[434.70165856457896, 244.7598485565902], [439.5336153168131, 242.8896806404092]], [[459.0600425153977, 307.78999089373576], [463.9561817388636, 305.8949816959307]], [[483.41842646621654, 370.8201332308812], [488.3787481609141, 368.90028275145175]], [[512.2609223573583, 427.7332089491025], [516.5165075555101, 420.53209197908416]], [[539.1375756785009, 405.76779938847176], [539.0879505355716, 405.76761890761554]], [[539.3538831287099, 343.31323971241363], [539.3089103486665, 343.3130850471789]], [[539.5951419810486, 272.4678841253044], [539.5552359282938, 272.46774688496953]], [[539.8364008333875, 201.62252853813698], [539.8015615079213, 201.62240872273105]], [[551.2335494706666, 140.23650018087835], [536.415657170144, 131.02473897730306]], [[563.2278977724335, 171.39249030728365], [570.0688070489025, 81.56104191857611]], [[557.0684979220293, 178.14626666763797], [600.7529441250301, 178.13191050174646]], [[592.0128549833953, 187.5673463816864], [565.8062368134797, 308.61824762601765]], [[536.6163059826291, 328.1036989955593], [620.5324783785172, 291.18265522548313]], [[611.4411662517434, 304.98493561196165], [565.9163628470873, 430.12004170594787]], [[599.9968313436336, 413.76284576772014], [605.0573677196238, 413.7512960653403]], [[600.0437477560993, 434.31941113003995], [605.1042841320896, 434.3078614276601]]]
for t,b in first_textline.pairPoints():
    color = [100+random.randint(0,154),100+random.randint(0,154),100+random.randint(0,154)]
    img[int(t[1]-1):int(t[1]+1),int(t[0]-1):int(t[0]+1),:]=color
    img[int(b[1]-1):int(b[1]+1),int(b[0]-1):int(b[0]+1),:]=color
for p in first_textline._top_points:
    img[int(p[1]-2):int(p[1]+2),int(p[0]-2):int(p[0]+2),:]=[0,1,1]
    img[int(p[1]-1):int(p[1]+1),int(p[0]-1):int(p[0]+1),:]=[1,0,1]
for p in first_textline._bot_points:
    img[int(p[1]-2):int(p[1]+2),int(p[0]-2):int(p[0]+2),:]=[1,0,1]
    img[int(p[1]-1):int(p[1]+1),int(p[0]-1):int(p[0]+1),:]=[1,1,0]
#print('final poly: {}'.format(first_textline.polyPoints()))

img_f.imshow('x',img)
img_f.show()

t_img = torch.from_numpy(img).permute(2,0,1)
grid1 = first_textline.getGrid(100,t_img.device) 
#grid2 = first_textline.getGrid2(32) 
print(grid1)
print('{}\t{}'.format(grid1[0,0],grid1[0,-1]))
print('{}\t{}'.format(grid1[-1,0],grid1[-1,-1]))

grid1[...,1] = 2*grid1[...,1]/t_img.size(1) -1
grid1[...,0] = 2*grid1[...,0]/t_img.size(2) -1

#grid2[...,1]/=t_img.size(1)
#grid2[...,0]/=t_img.size(2)

#print('----')
#print(grid2)

#sampled=F.grid_sample(t_img[None,...].expand(2,-1,-1,-1).float(),torch.stack((grid1,grid2),dim=0))
sampled=F.grid_sample(t_img[None,...].float(),grid1[None,...])
sampled=sampled.permute(0,2,3,1).numpy().astype(np.uint8)
img_f.imshow('grid1',sampled[0])
#img_f.show()
#img_f.imshow('grid2',sampled[1])

img_f.show()


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
