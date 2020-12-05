
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

import numpy as np
from utils import img_f
from utils.gpu import get_gpu_memory_map

from evaluators.draw_graph import getCorners
from utils.bb_merging import TextLine
from collections import defaultdict

def getBounds(bbs):
    if type(bbs[0]) is TextLine:
        bounds=[]
        for bb in bbs:
            bounds.append(bb.boundingRect())
        bounds = np.array(bounds)
        return bounds[:,0].min(), bounds[:,2].max(), bounds[:,1].min(), bounds[:,3].max()
    else:
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
    input = input - temp.unsqueeze(1).unsqueeze(1)

    flatin = input.view((input.size(0),-1))
    temp, _ = flatin.max(1, keepdim=True)
    input = input / (temp.unsqueeze(1).unsqueeze(1) + eps)
    return input
def _postProcessFlatGrad(input, eps=1e-6):
    #Here, we're dealing with the graph, the batch dim is actually the node or edge dim (batch size of 1)
    # Absolute value
    input = abs(input)

    # Rescale operations to ensure gradients lie between 0 and 1
    input = input - input.min()

    input = input / (input.max() + eps)
    return input

class FullGradExtractor:
    #Extract tensors needed for FullGrad using hooks
    
    def __init__(self, model):
        self.model = model

        self.biases = []
        self.backbone_feature_grads = []
        self.graph_node_feature_grads = defaultdict(list)
        self.graph_edge_feature_grads = defaultdict(list)
        self.grad_handles = []

        # Iterate through layers
        for name,m in self.model.named_modules():
            import pdb;pdb.set_trace()
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.GroupNorm):
                if 'graphnets' in name:
                    giter = int(name[10]) #assuming no more than 9 iters
                    if 'node' in name:
                        store_here = self.graph_node_feature_grads[giter]
                    elif 'edge' in name:
                        store_here = self.graph_edge_feature_grads[giter]
                elif 'detector' in name:
                    store_here = self.backbone_feature_grads
                # Register feature-gradient hooks for each layer
                handle_g = m.register_backward_hook(lambda m,i,o: self._extract_layer_grads(m,i,o,store_here))
                self.grad_handles.append(handle_g)

                # Collect model biases
                b = self._extract_layer_bias(m)
                if (b is not None): self.biases.append(b)

    def _clear_feature_grads(self):
        self.backbone_feature_grads = []
        self.graph_node_feature_grads = defaultdict(list)
        self.graph_edge_feature_grads = defaultdict(list)


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

    def _extract_layer_grads(self, module, in_grad, out_grad,store_here):
        # function to collect the gradient outputs
        # from each layer

        if not module.bias is None:
            store_here.append(out_grad[0])

    def getFeatureGrads(self, x, output_scalars):
        
        # Empty feature grads list 
        self.feature_grads = []

        # Gradients w.r.t. input
        input_gradients = []
        backbone_features_grads=[]
        graph_node_feature_grads = defaultdict(list)
        graph_edge_feature_grads = defaultdict(list)
        self.model.zero_grad()

        #import gc
        #prevTensors=None
        #import torch.autograd.profiler as profiler
        #with profiler.profile(profile_memory=True,record_shapes=True) as prof:

        for output_i,output_scalar in enumerate(output_scalars):
            input_gradients.append( torch.autograd.grad(
                outputs = output_scalar, 
                inputs = x, 
                retain_graph=True if output_i<len(output_scalars)-1 else False,
                create_graph=False)[0].cpu().detach() )
            #torch.cuda.empty_cache()
            #print(get_gpu_memory_map())
            backbone_features_grads.append([_postProcessGrad(fg).sum(1, keepdim=True).cpu() for fg in self.backbone_feature_grads])# if len(fg.size())==4 and fg.size(0)==1]) #only keep gradients from the CNN backbone. I dont want to go and refit all the RoIs
            for giter,grads in self.graph_node_feature_grads:
                graph_node_feature_grads[giter].append([_postProcessFlatGrad(fg).sum(1, keepdim=True).cpu() for fg in grads])
            for giter,grads in self.graph_edge_feature_grads:
                graph_edge_feature_grads[giter].append([_postProcessFlatGrad(fg).sum(1, keepdim=True).cpu() for fg in grads])
            self._clear_feature_grads()

            #curTensors=[]
            #for obj in gc.get_objects():
            #    try:
            #        if (torch.is_tensor(obj) and obj.is_cuda) or (hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda):
            #            curTensors.append((type(obj), obj.size()))
            #    except:
            #        pass
            #if prevTensors is not None:
            #    print('prev size:{}, cur size:{}'.format(len(prevTensors),len(curTensors)))
            #    used=set()
            #    if len(prevTensors)<len(curTensors):
            #        for ct in curTensors:
            #            found=-1
            #            for i,pt in enumerate(prevTensors):
            #                if ct[1]==pt[1]:
            #                    found=i
            #                    break
            #            if found==-1:
            #                print('new tensor {}'.format(ct))
            #            else:
            #                del prevTensors[i]

            #prevTensors=curTensors
            #stats=torch.cuda.memory_stats(0)
            #print('Total alloc: {}'.format(stats['allocated_bytes.all.current']))

        #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
             
        input_gradients = torch.cat(input_gradients,dim=0)
        backbone_features_grads = [torch.cat(fgs,dim=0) for fgs in zip(*backbone_features_grads)]
        for giter,grads in graph_node_feature_grads:
            graph_node_feature_grads[giter] = [torch.stack(fgs,dim=0) for fgs in zip(*grads)] #stack here to create batch dim
        for giter,grads in graph_edge_feature_grads:
            graph_edge_feature_grads[giter] = [torch.stack(fgs,dim=0) for fgs in zip(*grads)] #stack here to create batch dim

        return input_gradients, backbone_features_grads, graph_node_feature_grads, graph_edge_feature_grads

            

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
        #is_train = self.model.training
        #if not is_train:
        #    self.model.train()
        with torch.enable_grad():
            image = image.requires_grad_()
            self.model.all_grad=True
            try:
                allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final = self.model(image)
                edge_preds = allEdgeOuts[-1][:,-1,0]
                output_scalar = -1. * F.binary_cross_entropy_with_logits(edge_preds,torch.ones_like(edge_preds),reduction='none')

                #extra stuff here to extract which edge goes to which map
                assert(output_scalar.size(0) == len(allEdgeIndexes[-1]))
                node_bb_info=[]
                for giter in range(len(allOutputBoxes)):
                    iter_info=[]
                    for g1id,g2id in allEdgeIndexes[giter]:
                        info1 = getBounds([allOutputBoxes[giter][bbid] for bbid in allGroups[giter][g1id]])
                        info2 = getBounds([allOutputBoxes[giter][bbid] for bbid in allGroups[giter][g2id]])
                        iter_info.append((info1,info2))
                    node_bb_info.append(iter_info)

                res = self.model_ext.getFeatureGrads(image, output_scalar)+(node_bb_info,)
            except RuntimeError as e:
                print(e)
                print('Skipping saliency for this image')
                return None, None, None
        #if not is_train:
        #    self.model.eval()
        return res


    def saliency(self, image):
        #Simple FullGrad saliency

        #image = image[:,:,400:-400,100:-100]
        
        self.model.eval()
        input_grad, backbone_grad, graph_node_grad, graph_edge_grad, node_bb_info= self._getGradients(image)
        if input_grad is None:
            return None, None
        
        im_size = image.size()
        #im_size[2] //= 2
        #im_size[2] //= 2
        assert(im_size[0]==1)
        image = image.expand(input_grad.size(0),-1,-1,-1) #expand to number of edges

        # Input-gradient * image
        grd = input_grad.cuda() * image
        gradient = _postProcessGrad(grd).sum(1, keepdim=True)
        grd=None
        cam = gradient

        cam_graph = some_tensor

        # Aggregate Intermediate-gradients
        for i in range(len(backbone_grad)):

            # Select only Conv layers 
            if len(backbone_grad[i].size()) == len(im_size):
                #temp = self._postProcess(backbone_grad[i].cpu())
                #temp = temp.sum(1, keepdim=True)
                gradient = F.interpolate(backbone_grad[i].cuda(), size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True) 
                cam += gradient#.sum(1, keepdim=True)

        return cam, node_bb_info


def save_saliency_map(image, saliency_map, edges_info, edge_id, filename):
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
    if image.shape[2]==1:
        image = np.repeat(image,3,2)
    image[:,:,1]=image[:,:,1].astype(float)*saliency_map
    image[:,:,2]=image[:,:,2].astype(float)*(1-saliency_map)
    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    image[:,:,0]=saliency_map[:,:,0]
    
    edge_info = edges_info[-1][edge_id]
    x1,x2,y1,y2=edge_info[0]
    img_f.polylines(image,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))
    x1,x2,y1,y2=edge_info[1]
    img_f.polylines(image,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))

    img_f.imwrite(filename, image)

