# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
# https://github.com/idiap/fullgrad-saliency
# Heavily modified by Brian Davis


"""  
    Implement a simpler FullGrad-like saliency algorithm.
    Instead of exactly computing bias-gradients, we only
    extract gradients w.r.t. biases, which are simply
    gradients of intermediate spatial features *before* ReLU.
    The rest of the algorithm including post-processing
    and the aggregation is the same.
    Note: this algorithm is only provided for convenience and
    performance may not be match that of FullGrad. 
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

import numpy as np
from utils import img_f

from collections import defaultdict
import json

def getCorners(xyrhw):
    xc=xyrhw[0].item()
    yc=xyrhw[1].item()
    rot=xyrhw[2].item()
    h=xyrhw[3].item()
    w=xyrhw[4].item()
    h = min(30000,h)
    w = min(30000,w)
    #tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(w*math.sin(rot)+h*math.cos(rot) + yc) )
    #tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
    #br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(w*math.sin(rot)-h*math.cos(rot) + yc) )
    #bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(-w*math.sin(rot)-h*math.cos(rot) + yc) )
    #return tr,tl,br,bl
    tl,tr,br,bl= calcCorners(xc,yc,rot,h,w)
    return [int(x) for x in tl],[int(x) for x in tr],[int(x) for x in br],[int(x) for x in bl]

def getBounds(bbs):
    xs=[]
    ys=[]
    for bb in bbs:
        x,y = zip(*getCorners(bb))
        xs+=x
        ys+=y
    return min(xs), max(xs), min(ys), max(ys)
def _postProcessGrad(input, eps=1e-6):
    # Absolute value
    input = abs(input)

    # Rescale operations to ensure gradients lie between 0 and 1
    flatin = input.view((input.size(0),-1))
    temp, _ = flatin.min(1, keepdim=True)
    input = input - temp.unsqueeze(1).unsqueeze(1).detach()

    flatin = input.view((input.size(0),-1))
    temp, _ = flatin.max(1, keepdim=True)
    input = input / (temp.unsqueeze(1).unsqueeze(1).detach() + eps)
    return input
def _postProcessFlatGrad(input, eps=1e-6):
    #Here, we're dealing with the graph, the batch dim is actually the node or edge dim (batch size of 1)
    # Absolute value
    input = abs(input)

    # Rescale operations to ensure gradients lie between 0 and 1
    input = input - input.min().detach()

    input = input / (input.max().detach() + eps)
    return input


            

class InputGradModel():
    """
    Compute simple input Grad saliency map for x
    """

    def __init__(self, model):
        self.model = model

        self.easyocr_chars = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


    def _getGradients(self, image,ocr_res,questions):
        """
        Compute intermediate gradients for an image
        """
        image_gradients = []
        ocr_gradients = []
        with torch.enable_grad():
            image = image.requires_grad_()
            self.model.all_grad=True

            pred_a, target_a, string_a, pred_mask,im_tokens,ocr_tokens = self.model(image,ocr_res,questions,RUN=True,get_tokens=True)

            max_pred_a, pred_index = pred_a.max(dim=-1)

            a_output_scalar = -1. * F.binary_cross_entropy_with_logits(max_pred_a,torch.ones_like(max_pred_a),reduction='mean')
            a_1st_output_scalar = -1. * F.binary_cross_entropy_with_logits(max_pred_a[...,0],torch.ones_like(max_pred_a[...,0]),reduction='mean')
            mask_output_scalars = -1. * F.binary_cross_entropy_with_logits(pred_mask,torch.ones_like(pred_mask),reduction='mean')


            #res = self.model_ext.getFeatureGrads(image, output_scalar)+(node_bb_info,allEdgeIndexes[mf:])
            self.model.zero_grad()

            for output_i,output_scalar in enumerate([a_output_scalar,a_1st_output_scalar,mask_output_scalars]):
                #self.do_only='pix'
                if len(ocr_res[0])>0:
                    ocr_gradients.append( torch.autograd.grad(
                        outputs = output_scalar, 
                        inputs = ocr_tokens, 
                        retain_graph=True,
                        create_graph=False)[0].mean(dim=-1).cpu().detach() )
                else:
                    ocr_gradients.append(None)
                image_gradients.append( torch.autograd.grad(
                    outputs = output_scalar, 
                    inputs = im_tokens, 
                    retain_graph=True if output_i<2 else False,
                    create_graph=False)[0].mean(dim=-1).cpu().detach() )

                
        #import pdb;pdb.set_trace()                     
        #input_gradients = torch.cat(input_gradients,dim=0
        return zip(ocr_gradients,image_gradients),string_a,pred_mask


    def saliency(self, image,ocr_res,questions,path_prefix=None):
        #Simple FullGrad saliency

        #image = image[:,:,400:-400,100:-100]
        
        self.model.eval()
        #answer_grad, char0_grad, out_mask_grad = self._getGradients(image)
        res_grad,p_answer, p_mask = self._getGradients(image,ocr_res,questions)
        print(p_answer)

        
        im_size = image.size()
        #im_size[2] //= 2
        #im_size[2] //= 2
        assert(im_size[0]==1)

        image=image[0].cpu()
        draw_image = np.uint8((1-image[0:1,:,:].numpy()) * 127).transpose(1,2,0)
        if draw_image.shape[2]==1:
            draw_image = np.repeat(draw_image,3,2)


        #get location of ocr characters
        ocr_res=ocr_res[0]#remove batch dim
        all_char_pred = []
        for i,(bb,(string,char_prob),score) in enumerate(ocr_res):
            char_pred = char_prob.argmax(dim=1)
            char_loc = char_pred!=0
            new_char_pred = char_pred[char_loc]
            all_char_pred.append(new_char_pred.cpu())
        if len(all_char_pred)>0:
            all_char_pred = torch.cat(all_char_pred,dim=0)

        ocr_chars=''
        for pred in all_char_pred:
            ocr_chars+=self.easyocr_chars[pred-1]




        for name,(ocr_grad,image_grad) in zip(['answer','first_char','image'],res_grad):

            # Input-gradient * image
            # = image_grad * image[0]
            if image_grad.size(1)>0:
                H,W = self.model.patches_resolution
                image_grad = image_grad.view(H,W)
                saliency_image = image_grad.cpu().numpy()

                print('min={}, max={}'.format(saliency_image.min(),saliency_image.max()))

                saliency_image = saliency_image - saliency_image.min()
                saliency_image = saliency_image / saliency_image.max()
                saliency_image = saliency_image.clip(0,1)

                #test = np.uint8(saliency_image * 255)
                #img_f.imshow('',test)
                #img_f.show()

                saliency_image = img_f.resize(saliency_image,draw_image.shape[0:2],order=0)



                draw_image_this = np.copy(draw_image)
                draw_image_this[:,:,1]=draw_image_this[:,:,1].astype(float)*saliency_image
                draw_image_this[:,:,2]=draw_image_this[:,:,2].astype(float)*(1-saliency_image)
                saliency_image_e = np.uint8(saliency_image * 255)
                draw_image_this[:,:,0]=saliency_image_e


            #filename = path_prefix+'_{}_image.png'.format(name)
            #img_f.imwrite(filename, image)
            if ocr_grad is not None:
                saliency_ocr = ocr_grad[0].cpu().numpy()
                saliency_ocr = saliency_ocr - saliency_ocr.min()
                saliency_ocr = saliency_ocr / saliency_ocr.max()
                saliency_ocr = saliency_ocr.clip(0,1)

                assert len(ocr_chars) == saliency_ocr.shape[0]
                CSI = "\x1B["
                #Blue 0-0.2 Cyan 0.2-0.4  Green 0.4-0.6  Yellow 0.6-0.8  Red 0.8-1
                char_sal=''
                for char,score in zip(ocr_chars,saliency_ocr):
                    if score<0.2:
                        color = 34
                    elif score<0.4:
                        color = 36
                    elif score<0.6:
                        color=32
                    elif score<0.8:
                        color = 33
                    else:
                        color=31

                    if score<0.9:
                        w=10
                    else:
                        w=40

                    char_sal += CSI+"{};{}m".format(color,w) + char + CSI + "0m"

                print(char_sal)
            if image_grad.size(1)>0:
                img_f.imshow('',draw_image_this)
                img_f.show()

        return p_answer,p_mask
