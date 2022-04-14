import torch
import numpy as np
import torch.nn.functional as F
import utils
from model.unlikelihood_loss import unlikelihoodLoss
#import torch.nn as nn

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def sigmoid_BCE_loss(y_input, y_target):
    return F.binary_cross_entropy_with_logits(y_input, y_target)
def sigmoid_BCE_mask_loss(y_input, y_target):
    loss= F.binary_cross_entropy_with_logits(y_input, y_target,reduction='none')
    if y_target.sum()==0:
        return loss.mean()
    positive_loss = (loss*y_target).sum()/y_target.sum()
    anti_target = 1-y_target
    negative_loss = (loss*anti_target).sum()/anti_target.sum()
    return (positive_loss+ negative_loss)/2
def padded_seq_cross_entropy(x,target):
    batchsize = x.size(0)
    lenn = x.size(1)
    x_o = x
    x = x.contiguous().view(-1,x.size(2))
    target_flat = target.contiguous().view(-1)

    ce = F.cross_entropy(x,target_flat,reduction='none')

    #zero padding
    ce[target_flat==0]=0
    
    #average each by their respective len
    ce = ce.view(batchsize,lenn)
    #import pdb;pdb.set_trace()
    ce = ce.sum(dim=1)/(target!=0).sum(dim=1)
    
    #then average the batch
    return ce.sum()/batchsize

def MSE(y_input, y_target):
    return F.mse_loss(y_input, y_target.float())



def detect_alignment_loss(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop)
def detect_alignment_loss_points(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop,points=True)

#def lf_point_loss(prediction,target):
#    return point_loss(prediction,target)
#def lf_line_loss(prediction,target):
#    return special_loss(prediction,target)
#def lf_xyrs_loss(prediction,target):
#    return xyrs_loss(prediction,target)
#def lf_end_loss(end_pred,path_xyxy,end_point):
    #    return end_pred_loss(end_pred,path_xyxy,end_point)

def CTCLoss(input,target,input_len,target_len):
    ret = F.ctc_loss(input,target,input_len,target_len)
    return torch.where(torch.isinf(ret), torch.zeros_like(ret), ret)

#class LabelSmoothing(nn.Module):
#    "Implement label smoothing."
#    "From the Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html"
#    def __init__(self, size, padding_idx, smoothing=0.0):
#        super(LabelSmoothing, self).__init__()
#        self.criterion = nn.KLDivLoss(size_average=False)
#        self.padding_idx = padding_idx
#        self.confidence = 1.0 - smoothing
#        self.smoothing = smoothing
#        self.size = size
#        self.true_dist = None
#
#    def forward(self, x, target):
#        
#        assert x.size(1) == self.size
#        true_dist = x.data.clone()
#        true_dist.fill_(self.smoothing / (self.size - 2))
#        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#        true_dist[:, self.padding_idx] = 0
#        mask = torch.nonzero(target.data == self.padding_idx)
#        if mask.dim() > 0:
#            true_dist.index_fill_(0, mask.squeeze(), 0.0)
#        self.true_dist = true_dist
#        return self.criterion(x, true_dist)

def label_smoothing(x, target, padding_idx=0, smoothing=0.0): #huggingface padds with 0
    if len(x.size())==3:
        batchsize = x.size(0)
        lenn = x.size(1)
        x = x.contiguous().view(-1,x.size(2))
        target = target.contiguous().view(-1)
        orig_sizes = 'x:{}, target:{}'.format(x.size(),target.size())
    else:
        batchsize=1
    
    size = x.size(1)
    confidence = 1.0 - smoothing
    true_dist = x.data.clone()
    true_dist.fill_(smoothing / (size - 2))
    try:
        true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
    except RuntimeError as e:
        print('scatter FAILED')
        print('target.data.size = {}'.format(target.data.size()))
        print('true_dist.size = {}'.format(true_dist.size()))
        print(orig_sizes)
        raise e
    true_dist[:, padding_idx] = 0
    mask = torch.nonzero(target.data == padding_idx)
    if mask.dim() > 0:
        true_dist.index_fill_(0, mask.squeeze(), 0.0)
        #x.index_fill_(0, mask.squeeze(), 0.0)
        mask10 = torch.ones_like(x)
        mask10.index_fill_(0, mask.squeeze(), 0.0)
        x=x*mask10
    return F.kl_div(x.view(batchsize,lenn,size), true_dist.view(batchsize,lenn,size),reduction='batchmean')


def IoULoss(pred,target):
    if len(pred.size())==4:
        assert pred.size(1)==1
        pred = pred[:,0]
    eps=0.001
    intersection = (pred*target).sum(dim=2).sum(dim=1)
    comb = (pred+target).sum(dim=2).sum(dim=1)
    iou = (intersection+eps)/(comb-intersection+eps)
    return -iou.mean()

#from https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py
#MIT license
def focalLoss(x, target,alpha=1,gamma=2):
    #fix binary cases
    if len(x.size())==3:
        x = torch.stack((x,1-x),dim=1)
        target = torch.stack((target,1-target),dim=1)
    elif x.size(1)==1:
        x = torch.cat((x,1-x),dim=1)
        target = torch.cat((target,1-target),dim=1)
    eps = np.finfo(float).eps
    p_t = torch.where(target == 1, x, 1-x)
    fl = - 1 * (1 - p_t) ** gamma * torch.log(p_t + eps)
    fl = torch.where(target == 1, fl * alpha, fl * (1 - alpha))
    if torch.isnan(fl).any():
        import pdb;pdb.set_trace()
    return fl.mean()
