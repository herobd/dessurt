from utils import img_f
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def sigmoid_BCE_mask_loss(y_input, y_target):
    loss= F.binary_cross_entropy_with_logits(y_input, y_target,reduction='none')
    if y_target.sum()==0:
        return loss.mean()
    positive_loss = (loss*y_target).sum()/y_target.sum()
    anti_target = 1-y_target
    negative_loss = (loss*anti_target).sum()/anti_target.sum()
    return (positive_loss+ negative_loss)/2
def focalLoss(x, target,alpha=1,gamma=2):
    eps = np.finfo(float).eps
    p_t = torch.where(target == 1, x, 1-x)
    fl = - 1 * (1 - p_t) ** gamma * torch.log(p_t + eps)
    fl = torch.where(target == 1, fl * alpha, fl * (1 - alpha))
    return fl.mean()

def IoULoss(pred,target):
    if len(pred.size())==4:
        assert pred.size(1)==1
        pred = pred[:,0]
    eps=0.001
    intersection = (pred*target).sum(dim=2).sum(dim=1)
    comb = (pred+target).sum(dim=2).sum(dim=1)
    iou = (intersection+eps)/(comb-intersection+eps)
    return -iou.mean()

gt = img_f.imread('/home/brian/Downloads/gt.png')[:,:,0]
print(gt.shape)
bad1 = img_f.imread('/home/brian/Downloads/bad1.png')[:,:,0]
good1 = img_f.imread('/home/brian/Downloads/good1.png')[:,:,0]
good2 = img_f.imread('/home/brian/Downloads/good2.png')[:,:,0]

gt = torch.from_numpy(gt).float()/255
bad1 = torch.from_numpy(bad1).float()/255
good1 = torch.from_numpy(good1).float()/255
good2 = torch.from_numpy(good2).float()/255

gt = gt[None,...]
bad1 = bad1[None,...]
good1 = good1[None,...]
good2 = good2[None,...]

print('IoU')
print('bad1 {}'.format(IoULoss(bad1,gt)))
print('good1 {}'.format(IoULoss(good1,gt)))
print('good2 {}'.format(IoULoss(good2,gt)))
print('1-good2 {}'.format(IoULoss(1-good2,gt)))


print('focal')
print('bad1 {}'.format(focalLoss(bad1,gt)))
print('good1 {}'.format(focalLoss(good1,gt)))
print('good2 {}'.format(focalLoss(good2,gt)))
print('1-good2 {}'.format(focalLoss(1-good2,gt)))

print('BCE')
print('bad1 {}'.format(sigmoid_BCE_mask_loss(bad1,gt)))
print('good1 {}'.format(sigmoid_BCE_mask_loss(good1,gt)))
print('good2 {}'.format(sigmoid_BCE_mask_loss(good2,gt)))
print('1-good2 {}'.format(sigmoid_BCE_mask_loss(1-good2,gt)))
#from skimage import data, future, io
#from utils import img_f
#import numpy as np
#import sys
#
#img = img_f.imread(sys.argv[1])
##mask = future.manual_polygon_segmentation(img)
#mask = future.manual_lasso_segmentation(img)
#if mask.sum()==0:
#    mask = np.zeros_like(mask)
#print(mask.flags)
#print(mask.shape)
#print(img.min(),img.max())
#print(mask.min(),mask.max())
#img_f.imshow('x',img*mask)
#img_f.show()
