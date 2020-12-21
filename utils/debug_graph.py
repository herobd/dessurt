
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
import json

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

def _extract_layer_grads(self, module, in_grad, out_grad,store_here,name):
    # function to collect the gradient outputs
    # from each layer

    if (store_here is None or type(store_here) is bool):
        print('error store {}'.format(store_here))
        print(name)
    if not module.bias is None and store_here is not None:# and (self.do_only is None or self.do_only==is_type):
        #store_here.append(out_grad[0])
        store_here.append(_postProcessGrad(out_grad[0]).sum(1, keepdim=True).cpu())

class FullGradExtractor:
    #Extract tensors needed for FullGrad using hooks
    
    def __init__(self, model):
        self.model = model

        self.biases = []
        self.backbone_feature_grads = []
        self.graph_node_feature_grads = defaultdict(dict)
        self.graph_edge_feature_grads = defaultdict(dict)
        self.grad_handles = []
        self.do_only=False

        # Iterate through layers
        for name,m in self.model.named_modules():
            
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.GroupNorm):
                #if 'graphnets' in name:
                #    giter = int(name[10]) #assuming no more than 9 iters
                #    if 'node' in name:
                #        is_type='node'
                #        store_here = self.graph_node_feature_grads[giter]
                #    elif 'edge' in name:
                #        is_type='edge'
                #        store_here = self.graph_edge_feature_grads[giter]
                #    else:
                #        store_here =None
                #        is_type=None
                #elif 'detector' in name:
                #    is_type='pix'
                #    store_here = self.backbone_feature_grads
                #else:
                #    is_type=None
                #    store_here = None

                #if store_here is not None:
                #    # Register feature-gradient hooks for each layer
                #    handle_g = m.register_backward_hook(lambda m,i,o: self._extract_layer_grads(m,i,o,store_here,str(name)))
                #    self.grad_handles.append(handle_g)
                def save_plz(typ,name,this):
                    def actually_save(module,in_grad,out_grad):
                        in_grad=None
                        if not module.bias is None:
                            if typ=='mhAtt':
                                #print('{} {}'.format(name, out_grad[0].size()))
                                if out_grad[0].size(1)==this.connected.shape[0]:
                                    for ni in range(out_grad[0].size(1)):
                                        if not this.connected[ni,this.cur_edge[0]]:
                                            if (out_grad[0][0,ni]!=0).any():
                                                print('bad node grad for [{}/{}] {}'.format(ni,out_grad[0].size(1),name))
                                else: #edges
                                    #print(len(this.edge_indexes))
                                    for ei in range(out_grad[0].size(1)):
                                        a_node = this.edge_indexes[ei if ei<len(this.edge_indexes) else ei-len(this.edge_indexes)][0] 
                                        #print('{}/{}  {}/{}'.format(ei,out_grad[0].size(1),a_node,this.connected.shape[0]))
                                        if not this.connected[a_node,this.cur_edge[0]]:
                                            if (out_grad[0][0,ei]!=0).any():
                                                print('bad edge grad for [{}/{}] {}'.format(ei,out_grad[0].size(1),name))
                            if typ=='node':
                                if (out_grad[0].size(0)==this.connected.shape[0]):
                                    for ni in range(out_grad[0].size(0)):
                                        if not this.connected[ni,this.cur_edge[0]]:

                                            #print('{} and {} not connected'.format(ni,this.cur_edge))
                                            #print(out_grad[0][ni])
                                            if (out_grad[0][ni]!=0).any():
                                                print('bad node grad for [{}/{}] {}'.format(ni,out_grad[0].size(0),name))
                                            #print('fine')
                                #else:
                                #    print('{} : {} != {}'.format(name,out_grad[0].size(),this.connected.shape[0]))
                            if typ=='edge':
                                for ei in range(out_grad[0].size(0)):
                                    a_node = this.edge_indexes[ei if ei<len(this.edge_indexes) else ei-len(this.edge_indexes)][0] 
                                    #print('{}/{}  {}/{}'.format(ei,out_grad[0].size(1),a_node,this.connected.shape[0]))
                                    if not this.connected[a_node,this.cur_edge[0]]:
                                        if (out_grad[0][ei]!=0).any():
                                            print('bad edge grad for [{}/{}] {}'.format(ei,out_grad[0].size(0),name))

                                    
                            out_grad=None
                        #stats=torch.cuda.memory_stats(0)
                        #print('total alloc: {}\t{}'.format(stats['allocated_bytes.all.current'],name))
                    return actually_save
                if 'graphnets' in name:
                    giter = int(name[10])
                    node_grads = None#self.graph_node_feature_grads[giter]
                    edge_grads = None#self.graph_edge_feature_grads[giter]
                    if giter==1:
                        if 'mhAtt' in name:
                            handle_g = m.register_backward_hook(save_plz('mhAtt',name,self))
                        elif 'node' in name:
                            handle_g = m.register_backward_hook(save_plz('node',name,self))
                        elif 'edge' in name:
                            handle_g = m.register_backward_hook(save_plz('edge',name,self))
                else:
                    handle_g = None
                if handle_g is not None:
                    self.grad_handles.append(handle_g)


                # Collect model biases
                b = self._extract_layer_bias(m)
                if (b is not None): self.biases.append(b)
        #assert(len(self.graph_node_feature_grads)==3)
        #assert(len(self.graph_edge_feature_grads)==3)



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


    def getFeatureGrads(self, x, output_scalars, connected, edge_indexes):
        self.connected=connected
        self.edge_indexes=edge_indexes[1]
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
            self.cur_edge = edge_indexes[-1][output_i]
            assert(self.cur_edge[0]<connected.shape[0])
            assert(self.cur_edge[1]<connected.shape[0])
            #self.do_only='pix'
            input_gradients.append( torch.autograd.grad(
                outputs = output_scalar, 
                inputs = x, 
                retain_graph=True if output_i<len(output_scalars)-1 else False,
                create_graph=False)[0].cpu().detach() )
            #torch.cuda.empty_cache()
            #print(get_gpu_memory_map())

            

class GraphChecker():
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
            #try:
            if True:
                allOutputBoxes, offsetPredictions, allEdgeOuts, allEdgeIndexes, allNodeOuts, allGroups, rel_prop_scores,merge_prop_scores, final = self.model(image)
                edge_preds = allEdgeOuts[-1][:,-1,0]
                output_scalar = -1. * F.binary_cross_entropy_with_logits(edge_preds,torch.ones_like(edge_preds),reduction='none')

                #extra stuff here to extract which edge goes to which map
                assert(output_scalar.size(0) == len(allEdgeIndexes[-1]))
                node_bb_info=[]
                mf = 1 if self.model.merge_first else 0
                for giter in range(mf,len(allOutputBoxes)):
                    iter_info=[]
                    #for g1id,g2id in allEdgeIndexes[giter]:
                    for g1id in range(len(allGroups[giter])):
                        info1 = getBounds([allOutputBoxes[giter][bbid] for bbid in allGroups[giter][g1id]])
                        #info2 = getBounds([allOutputBoxes[giter][bbid] for bbid in allGroups[giter][g2id]])
                        #iter_info.append((info1,info2))
                        iter_info.append(info1)
                    node_bb_info.append(iter_info)

                edge_indexes = allEdgeIndexes[mf:]
                giter=1
                total_steps=6
                num_nodes=len(node_bb_info[giter])
                adj = np.zeros((num_nodes,num_nodes))
                for n1,n2 in edge_indexes[giter]:
                    adj[n1,n2] = 1
                    adj[n2,n1] = 1
                connected=adj.copy()
                for i in range(total_steps-1):
                    adj = np.matmul(adj,adj)
                    connected+=adj
                connected = connected>0

                self.model_ext.getFeatureGrads(image, output_scalar, connected, edge_indexes)
            #except RuntimeError as e:
            #    print(e)
            #    print('Skipping saliency for this image')


    def check(self, image):
        #Simple FullGrad saliency

        #image = image[:,:,400:-400,100:-100]
        
        self.model.eval()
        self._getGradients(image)
