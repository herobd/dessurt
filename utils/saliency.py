
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
# https://github.com/idiap/fullgrad-saliency


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

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullGradExtractor:
    #Extract tensors needed for FullGrad using hooks
    
    def __init__(self, model):
        self.model = model

        self.biases = []
        self.feature_grads = []
        self.grad_handles = []

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                
                # Register feature-gradient hooks for each layer
                handle_g = m.register_backward_hook(self._extract_layer_grads)
                self.grad_handles.append(handle_g)

                # Collect model biases
                b = self._extract_layer_bias(m)
                if (b is not None): self.biases.append(b)


    def _extract_layer_bias(self, module):
        # extract bias of each layer

        # for batchnorm, the overall "bias" is different 
        # from batchnorm bias parameter. 
        # Let m -> running mean, s -> running std
        # Let w -> BN weights, b -> BN bias
        # Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b) 
        # Thus (-m*w/s + b) is the effective bias of batchnorm

        if isinstance(module, nn.BatchNorm2d):
            b = - (module.running_mean * module.weight 
                    / torch.sqrt(module.running_var + module.eps)) + module.bias
            return b.data
        elif module.bias is None:
            return None
        else:
            return module.bias.data

    def getBiases(self):
        # dummy function to get biases
        return self.biases

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        # from each layer

        if not module.bias is None:
            self.feature_grads.append(out_grad[0])

    def getFeatureGrads(self, x, output_scalar):
        
        # Empty feature grads list 
        self.feature_grads = []

        self.model.zero_grad()
        # Gradients w.r.t. input
        input_gradients = torch.autograd.grad(outputs = output_scalar, inputs = x)[0]

        return input_gradients, self.feature_grads

class SimpleFullGradMod():
    """
    Compute simple FullGrad saliency map for my graph model
    """

    def __init__(self, model):
        self.model = model
        self.model_ext = FullGradExtractor(model)

    def _getGradients(self, image):
        """
        Compute intermediate gradients for an image
        """
        with torch.enable_grad():
            self.model.eval()
            image = image.requires_grad_()
            allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final = self.model(image)
            edge_preds = allEdgeOuts[-1][:,-1,0]


            # Select the output unit corresponding to the target class
            # -1 compensates for negation in nll_loss function
            #output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')
            output_scalar = -1. * F.binary_cross_entropy_with_logits(edge_preds,torch.ones_like(edge_preds),reduction='sum')

            return self.model_ext.getFeatureGrads(image, output_scalar)

    def _postProcess(self, input, eps=1e-6):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.min(1, keepdim=True)
        input = input - temp.unsqueeze(1).unsqueeze(1)

        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.max(1, keepdim=True)
        input = input / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return input

    def saliency(self, image):
        #Simple FullGrad saliency
        
        self.model.eval()
        input_grad, intermed_grad = self._getGradients(image)
        
        im_size = image.size()
        assert(im_size[0]==1)
        image = image.expand(input_grad.size(0),-1,-1,-1) #expand to number of edges

        # Input-gradient * image
        grd = input_grad * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        # Aggregate Intermediate-gradients
        for i in range(len(intermed_grad)):

            # Select only Conv layers 
            if len(intermed_grad[i].size()) == len(im_size):
                temp = self._postProcess(intermed_grad[i])
                gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True) 
                cam += gradient.sum(1, keepdim=True)

        return cam, [None]*cam.size(0)


def save_saliency_map(image, saliency_map, filename):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension
    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    image = np.uint8(image * 255).transpose(1,2,0)
    image[:,:,1]*=saliency_map
    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    image[:,:,0]=saliency_map

    img_f.imwrite(filename, image)

