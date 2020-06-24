import torch
from torch import nn
from base import BaseModel
import math
import json
import numpy as np
from .net_builder import make_layers


MAX_H_PRED=2
MAX_W_PRED=1.1



class OverSegBoxDetector(nn.Module): #BaseModel
    def __init__(self, config): # predCount, base_0, base_1):
        super(OverSegBoxDetector, self).__init__()
        self.config = config
        self.rotation = True
        self.numBBTypes = config['number_of_box_types']
        self.numBBParams = 6 #conf,L-off,T-off,R-off,B-off,rot
        self.predNumNeighbors=False
        self.anchors=None
        self.numAnchors=2

        self.predPixelCount = config['number_of_pixel_types'] if 'number_of_pixel_types' in config else 0


        in_ch = 3 if 'color' not in config or config['color'] else 1
        norm = config['norm_type'] if "norm_type" in config else None
        if norm is None:
            print('Warning: OverSegBoxDetector has no normalization!')
        dilation = config['dilation'] if 'dilation' in config else 1
        dropout = config['dropout'] if 'dropout' in config else None
        #self.cnn, self.scale = vgg.vgg11_custOut(self.predLineCount*5+self.predPointCount*3,batch_norm=batch_norm, weight_norm=weight_norm)
        self.numOutBB = (self.numBBTypes+self.numBBParams)*self.numAnchors

        if 'down_layers_cfg' in config:
            #Don't define in channels
            layers_cfg = config['down_layers_cfg']
        else:
            layers_cfg=[[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]]

        self.net_down_modules = nn.ModuleList()
        self.net_out_modules = nn.ModuleList()
        last_ch_out = in_ch
        for i,layers_cfg_set in enumerate(layers_cfg):
            #assert(layers_cfg_set[0] == last_ch_out)
            modules, down_last_channels = make_layers([last_ch_out]+layers_cfg_set, dilation,norm,dropout=dropout)
            self.net_down_modules.append(nn.Sequential(*modules))

            #TODO exception if creating Unet to not classify all resolutions
            if i==len(layers_cfg)-1:
                self.net_out_modules.append(nn.Conv2d(down_last_channels, self.numOutBB, kernel_size=1))
            else: 
                #if it isn't the last layer, include an extra regression layer
                self.net_out_modules.append(nn.Sequential(
                    nn.Conv2d(down_last_channels, down_last_channels, kernel_size=1),
                    nn.ReLU(True),
                    nn.Conv2d(down_last_channels, self.numOutBB, kernel_size=1)))
            last_ch_out = down_last_channels
        self.final_features=None 
        self.last_channels=down_last_channels
        #self.net_down = nn.Sequential(*self.net_down_modules)
        self.scale=[]
        scaleX=1
        scaleY=1
        for layers_cfg_set in layers_cfg:
            for a in layers_cfg_set:
                if a=='M' or (type(a) is str and a[0]=='D'):
                    scaleX*=2
                    scaleY*=2
                elif type(a) is str and a[0]=='U':
                    scaleX/=2
                    scaleY/=2
                elif type(a) is str and a[0:4]=='long': #long pool
                    scaleX*=3
                    scaleY*=2
            self.scale.append((scaleX,scaleY))


        if self.predPixelCount>0:
            raise NotImplementedError('need to copy format for down layers')

        #self.base_0 = config['base_0']
        #self.base_1 = config['base_1']
        if 'DEBUG' in config:
            self.setDEBUG()

    def forward(self, img):
        #import pdb; pdb.set_trace()
        if self.predPixelCount>0:
            raise NotImplementedError('need to ...')
            levels=[img]
            for module in self.net_down_modules:
                levels.append(module(levels[-1]))
            y=levels[-1]
        else:
            ys=[]
            x=img
            for down_l,out_l in zip(self.net_down_modules,self.net_out_modules):
                x=down_l(x)
                ys.append(out_l(x))


        #priors_0 = Variable(torch.arange(0,y.size(2)).type_as(img.data), requires_grad=False)[None,:,None]
        offsetPredictions_scales=[]
        bbPredictions_scales=[]
        for level,y in enumerate(ys):
            priors_0 = torch.arange(0,y.size(2)).type_as(img.data)[None,:,None]
            priors_0 = (priors_0 + 0.5) * self.scale[level][1] #self.base_0
            priors_0 = priors_0.expand(y.size(0), priors_0.size(1), y.size(3))
            priors_0 = priors_0[:,None,:,:].to(img.device)

            #priors_1 = Variable(torch.arange(0,y.size(3)).type_as(img.data), requires_grad=False)[None,None,:]
            priors_1 = torch.arange(0,y.size(3)).type_as(img.data)[None,None,:]
            priors_1 = (priors_1 + 0.5) * self.scale[level][0] #elf.base_1
            priors_1 = priors_1.expand(y.size(0), y.size(2), priors_1.size(2))
            priors_1 = priors_1[:,None,:,:].to(img.device)

            pred_offsets=[] #we seperate anchor predictions here. And compute actual bounding boxes
            pred_boxes=[]
            for i in range(self.numAnchors):

                offset = i*(self.numBBParams+self.numBBTypes)

                stackedPred = [
                    torch.sigmoid(y[:,0+offset:1+offset,:,:]),                              #0. confidence
                    torch.tanh(y[:,1+offset:2+offset,:,:])*self.scale[level][0]*MAX_W_PRED + priors_1, #1. x1
                    torch.tanh(y[:,2+offset:3+offset,:,:])*self.scale[level][1]*MAX_H_PRED + priors_0, #2. y1
                    torch.tanh(y[:,3+offset:4+offset,:,:])*self.scale[level][0]*MAX_W_PRED + priors_1, #3. x2
                    torch.tanh(y[:,4+offset:5+offset,:,:])*self.scale[level][1]*MAX_H_PRED + priors_0, #4. y2
                    torch.sin(y[:,5+offset:6+offset,:,:]*np.pi)*np.pi,        #5. rotation (radians)
                ]


                for j in range(self.numBBTypes):
                    stackedPred.append(torch.sigmoid(y[:,6+j+offset:7+j+offset,:,:]))         #x. class prediction
                    #stackedOffsets.append(y[:,6+j+offset:7+j+offset,:,:])         #x. class prediction
                pred_boxes.append(torch.cat(stackedPred, dim=1))
                #pred_offsets.append(torch.cat(stackedOffsets, dim=1))
                pred_offsets.append(y[:,offset:offset+self.numBBParams+self.numBBTypes,:,:])
            offsetPredictions = torch.stack(pred_offsets, dim=1)
            offsetPredictions = offsetPredictions.permute(0,1,3,4,2).contiguous()
            offsetPredictions_scales.append(offsetPredictions)

            bbPredictions = torch.stack(pred_boxes, dim=1)
            bbPredictions = bbPredictions.transpose(2,4).contiguous()#from [batch, anchors, channel, rows, cols] to [batch, anchros, cols, rows, channels]
            bbPredictions = bbPredictions.view(bbPredictions.size(0),bbPredictions.size(1),-1,bbPredictions.size(4))#flatten to [batch, anchors, instances, channel]
            #avg_conf_per_anchor = bbPredictions[:,:,:,0].mean(dim=0).mean(dim=1)
            bbPredictions = bbPredictions.view(bbPredictions.size(0),-1,bbPredictions.size(3)) #[batch, instances+anchors, channel]
            bbPredictions_scales.append(bbPredictions)

        bbPredictions = torch.cat(bbPredictions_scales,dim=1) #stack the scales
        bbPredictions_scales=None

        pixelPreds=None
        if self.predPixelCount>0:
            startLevel = len(self.net_up_modules)-len(levels) -1
            y2=levels[-2]
            p=startLevel-1
            for module in self.net_up_modules[:-1]:
                #print('uping {} , {}'.format(y2.size(), levels[p].size()))
                y2 = module(y2,levels[p])
                p-=1
            pixelPreds = self.net_up_modules[-1](y2)
            



        return bbPredictions, offsetPredictions_scales, None,None,None, pixelPreds #, avg_conf_per_anchor

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters: {}'.format(params))
        print(self)

    def setDEBUG(self):
        #self.debug=[None]*5
        #for i in range(0,1):
        #    def save_layer(module,input,output):
        #        self.debug[i]=output.cpu()
        #    self.net_down_modules[i].register_forward_hook(save_layer)

        def save_layer0(module,input,output):
            self.debug0=output.cpu()
        self.net_down_modules[0].register_forward_hook(save_layer0)
        def save_layer1(module,input,output):
            self.debug1=output.cpu()
        self.net_down_modules[1].register_forward_hook(save_layer1)
        def save_layer2(module,input,output):
            self.debug2=output.cpu()
        self.net_down_modules[2].register_forward_hook(save_layer2)
        def save_layer3(module,input,output):
            self.debug3=output.cpu()
        self.net_down_modules[3].register_forward_hook(save_layer3)
        def save_layer4(module,input,output):
            self.debug4=output.cpu()
        self.net_down_modules[4].register_forward_hook(save_layer4)
