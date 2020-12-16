
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
                if 'graphnets' in name:
                    giter = int(name[10])
                    if giter==0:
                        if 'node' in name:
                            handle_g = m.register_backward_hook(self._extract_layer_node0_grads)
                        elif 'edge' in name:
                            handle_g = m.register_backward_hook(self._extract_layer_edge0_grads)
                    elif giter==1:
                        if 'node' in name:
                            handle_g = m.register_backward_hook(self._extract_layer_node1_grads)
                        elif 'edge' in name:
                            handle_g = m.register_backward_hook(self._extract_layer_edge1_grads)
                    elif giter==2:
                        if 'node' in name:
                            handle_g = m.register_backward_hook(self._extract_layer_node2_grads)
                        elif 'edge' in name:
                            handle_g = m.register_backward_hook(self._extract_layer_edge2_grads)
                elif 'detector' in name:
                    handle_g = m.register_backward_hook(self._extract_layer_backbone_grads)
                else:
                    handle_g = None
                if handle_g is not None:
                    self.grad_handles.append(handle_g)

                # Collect model biases
                b = self._extract_layer_bias(m)
                if (b is not None): self.biases.append(b)

    def _clear_feature_grads(self):
        self.backbone_feature_grads.clear()# = []
        for giter in self.graph_node_feature_grads:
            self.graph_node_feature_grads[giter].clear()
            self.graph_edge_feature_grads[giter].clear()


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

    def _extract_layer_backbone_grads(self, module, in_grad, out_grad):
        if not module.bias is None:
            self.backbone_feature_grads.append(_postProcessGrad(out_grad[0]).sum(1, keepdim=True).cpu())
    def _extract_layer_edge0_grads(self, module, in_grad, out_grad):
        if not module.bias is None:
            self.graph_edge_feature_grads[0].append(_postProcessFlatGrad(out_grad[0]).sum(1, keepdim=True).cpu())
    def _extract_layer_edge1_grads(self, module, in_grad, out_grad):
        if not module.bias is None:
            self.graph_edge_feature_grads[1].append(_postProcessFlatGrad(out_grad[0]).sum(1, keepdim=True).cpu())
    def _extract_layer_edge2_grads(self, module, in_grad, out_grad):
        if not module.bias is None:
            self.graph_edge_feature_grads[2].append(_postProcessFlatGrad(out_grad[0]).sum(1, keepdim=True).cpu())
    def _extract_layer_node0_grads(self, module, in_grad, out_grad):
        if not module.bias is None:
            self.graph_node_feature_grads[0].append(_postProcessFlatGrad(out_grad[0]).sum(1, keepdim=True).cpu())
    def _extract_layer_node1_grads(self, module, in_grad, out_grad):
        if not module.bias is None:
            self.graph_node_feature_grads[1].append(_postProcessFlatGrad(out_grad[0]).sum(1, keepdim=True).cpu())
    def _extract_layer_node2_grads(self, module, in_grad, out_grad):
        if not module.bias is None:
            self.graph_node_feature_grads[2].append(_postProcessFlatGrad(out_grad[0]).sum(1, keepdim=True).cpu())
            #print('self.graph_node_feature_grads[2] len {}'.format(len(self.graph_node_feature_grads[2])))
    def _extract_layer_grads(self, module, in_grad, out_grad,store_here,name):
        # function to collect the gradient outputs
        # from each layer

        if (store_here is None or type(store_here) is bool):
            print('error store {}'.format(store_here))
            print(name)
        if not module.bias is None and store_here is not None:# and (self.do_only is None or self.do_only==is_type):
            #store_here.append(out_grad[0])
            if torch.is_nan(out_grad[0]).any():
                print('nan in {}'.format(name))
            store_here.append(_postProcessGrad(out_grad[0]).sum(1, keepdim=True).cpu())
            if torch.is_nan(store_here[-1]).any():
                print('nan out {}'.format(name))

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
            #self.do_only='pix'
            input_gradients.append( torch.autograd.grad(
                outputs = output_scalar, 
                inputs = x, 
                retain_graph=True if output_i<len(output_scalars)-1 else False,
                create_graph=False)[0].cpu().detach() )
            #torch.cuda.empty_cache()
            #print(get_gpu_memory_map())
            backbone_features_grads.append(list(self.backbone_feature_grads))# if len(fg.size())==4 and fg.size(0)==1]) #only keep gradients from the CNN backbone. I dont want to go and refit all the RoIs
            #self._clear_feature_grads()

            #self.do_only='node'
            #torch.autograd.grad(
            #    outputs = output_scalar, 
            #    inputs = x, 
            #    retain_graph=True if output_i<len(output_scalars)-1 else False,
            #    create_graph=False)

            for giter,grads in self.graph_node_feature_grads.items():
                graph_node_feature_grads[giter].append(list(grads))
            #print('graph_node_feature_grads[2] len {}'.format(len(graph_node_feature_grads[2])))
            #self._clear_feature_grads()

            #self.do_only='edge'
            #torch.autograd.grad(
            #    outputs = output_scalar, 
            #    inputs = x, 
            #    retain_graph=True if output_i<len(output_scalars)-1 else False,
            #    create_graph=False)
            for giter,grads in self.graph_edge_feature_grads.items():
                graph_edge_feature_grads[giter].append(list(grads))
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
            #print('total alloc: {}'.format(stats['allocated_bytes.all.current']))

        #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
             
        input_gradients = torch.cat(input_gradients,dim=0)
        backbone_features_grads = [torch.cat(fgs,dim=0) for fgs in zip(*backbone_features_grads)]
        for giter,grads in graph_node_feature_grads.items():
            graph_node_feature_grads[giter] = [torch.stack(fgs,dim=0) for fgs in zip(*grads)] #stack here to create batch dim
        for giter,grads in graph_edge_feature_grads.items():
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

                res = self.model_ext.getFeatureGrads(image, output_scalar)+(node_bb_info,allEdgeIndexes[mf:])
            #except RuntimeError as e:
            #    print(e)
            #    print('Skipping saliency for this image')
            #    return None, None, None, None, None, None
        #if not is_train:
        #    self.model.eval()
        return res


    def saliency(self, image,draw_image,path_prefix):
        #Simple FullGrad saliency

        #image = image[:,:,400:-400,100:-100]
        
        self.model.eval()
        input_grad, backbone_grad, graph_node_grad, graph_edge_grad, node_bb_info, edge_indexes= self._getGradients(image)
        if input_grad is None:
            return

        
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

        #cam_graph = torch.FloatTensor(*im_size).zero_()

        # Aggregate Intermediate-gradients
        for i in range(len(backbone_grad)):

            # Select only Conv layers 
            if len(backbone_grad[i].size()) == len(im_size):
                #temp = self._postProcess(backbone_grad[i].cpu())
                #temp = temp.sum(1, keepdim=True)
                gradient = F.interpolate(backbone_grad[i].cuda(), size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True) 
                cam += gradient#.sum(1, keepdim=True)
        
        num_giters = len(node_bb_info)
        cam_edges = [None]*num_giters
        cam_nodes = [None]*num_giters
        for giter in range(num_giters):
            num_edge=len(edge_indexes[giter])
            cam_e = torch.FloatTensor(input_grad.size(0),num_edge,1).zero_()
            for e_grad in graph_edge_grad[giter]:
                if e_grad.size(1)>num_edge:
                    e_grad = (e_grad[:,:num_edge]+e_grad[:,num_edge:])/2 #collapse directions
                cam_e += e_grad
            cam_edges[giter] = cam_e

            num_node=len(node_bb_info[giter])
            cam_n = torch.FloatTensor(input_grad.size(0),num_node,1).zero_()
            for n_grad in graph_node_grad[giter]:
                if cam_n.size() == n_grad.size():
                    cam_n += n_grad
            cam_nodes[giter] = cam_n
        #return cam, cam_graph, node_bb_info, edge_indexes

        #images: pixel,pixel with each giter individualy, pixel with all giters together

        draw_image = draw_image.data.cpu().numpy()
        saliency_map = cam.data.cpu().numpy()

        saliency_map = saliency_map - saliency_map.min()
        saliency_map = saliency_map / saliency_map.max()
        saliency_map = saliency_map.clip(0,1)

        saliency_graph_edges=[cam_e.data.cpu().numpy() for cam_e in cam_edges]
        for giter in range(num_giters):
            saliency_graph_edges[giter] = saliency_graph_edges[giter] - saliency_graph_edges[giter].min()
            saliency_graph_edges[giter] = saliency_graph_edges[giter] / saliency_graph_edges[giter].max()
            saliency_graph_edges[giter] = saliency_graph_edges[giter].clip(0,1)

        saliency_graph_nodes=[cam_n.data.cpu().numpy() for cam_n in cam_nodes]
        for giter in range(num_giters):
            saliency_graph_nodes[giter] = saliency_graph_nodes[giter] - saliency_graph_nodes[giter].min()
            saliency_graph_nodes[giter] = saliency_graph_nodes[giter] / saliency_graph_nodes[giter].max()
            saliency_graph_nodes[giter] = saliency_graph_nodes[giter].clip(0,1)

        draw_image = np.uint8(draw_image * 255).transpose(1,2,0)
        if draw_image.shape[2]==1:
            draw_image = np.repeat(draw_image,3,2)

        for e in range(input_grad.size(0)):
            image = np.copy(draw_image)
            image[:,:,1]=image[:,:,1].astype(float)*saliency_map[e]
            image[:,:,2]=image[:,:,2].astype(float)*(1-saliency_map[e])
            saliency_map_e = np.uint8(saliency_map[e] * 255).transpose(1, 2, 0)
            image[:,:,0]=saliency_map_e[:,:,0]

            pixel_image=np.copy(image)
            
            n1,n2 = edge_indexes[-1][e]
            x1,x2,y1,y2=node_bb_info[-1][n1]
            img_f.polylines(image,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))
            x1,x2,y1,y2=node_bb_info[-1][n2]
            img_f.polylines(image,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))

            filename = path_prefix+'_{}_pixels.png'.format(e)
            img_f.imwrite(filename, image)

            image_all = np.copy(pixel_image)
            start_line_width = 2*num_giters
            colors=np.array([[0,1,1],[1,0,1],[1,0,0],[1,1,0],[0,1,0]])
            for giter,(saliency_e,saliency_n) in enumerate(zip(saliency_graph_edges,saliency_graph_nodes)):
                image = np.copy(draw_image)
                image[:,:,0:2]=0

                to_draw = defaultdict(list)
                for e_iter,(n1,n2) in enumerate(edge_indexes[giter]):
                    x1,x2,y1,y2=node_bb_info[giter][n1]
                    xc1 = (x1+x2)/2
                    yc1 = (y1+y2)/2
                    x1,x2,y1,y2=node_bb_info[giter][n2]
                    xc2 = (x1+x2)/2
                    yc2 = (y1+y2)/2
                    
                    shade = int(255*saliency_graph_edges[giter][e][e_iter])

                    to_draw[shade].append(((int(xc1),int(yc1)),(int(xc2),int(yc2))))
                
                width = start_line_width-2*giter
                shades = list(to_draw.keys())
                shades.sort()
                for shade in shades:
                    color = colors[giter]*shade
                    for p1,p2 in to_draw[shade]:
                        img_f.line(image,p1,p2,(0,shade,shade),2)

                        img_f.line(image_all,p1,p2,color,width)



                for n in range(len(node_bb_info[giter])):
                    x1,x2,y1,y2=node_bb_info[giter][n]
                    image[int(y1):int(y2+1),int(x1):int(x2+1),1:]=255*saliency_n[e][n]
                    h = (1+y2-y1)//2
                    seg_vert = h/num_giters
                    if seg_vert<1:
                        yc = (y1+y2)/2
                        y1 = yc-num_giters
                        y2 = yc+num_giters
                        seg_vert=1
                    w = (1+x2-x1)//2
                    seg_horz = w/num_giters
                    if seg_horz<1:
                        xc = (x1+x2)/2
                        x1 = xc-num_giters
                        x2 = xc+num_giters
                        seg_horz=1
                    color = colors[giter]*int(255*saliency_n[e][n])
                    image_all[int(y1+giter*seg_vert):int(1+y2-giter*seg_vert),int(x1+giter*seg_horz):1+int(x2-giter*seg_horz)]=color


                x1,x2,y1,y2=node_bb_info[-1][n1]
                img_f.polylines(image,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))
                x1,x2,y1,y2=node_bb_info[-1][n2]
                img_f.polylines(image,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))
                filename = path_prefix+'_{}_graph_g{}.png'.format(e,giter)
                img_f.imwrite(filename, image)

            x1,x2,y1,y2=node_bb_info[-1][n1]
            img_f.polylines(image_all,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))
            x1,x2,y1,y2=node_bb_info[-1][n2]
            img_f.polylines(image_all,np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]),False,(0,255,0))
            filename = path_prefix+'_{}_graph_all.png'.format(e)
            img_f.imwrite(filename, image_all)


