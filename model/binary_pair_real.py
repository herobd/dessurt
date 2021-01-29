import torch
from torch import nn
#from base import BaseModel
import torch.nn.functional as F
#from torch.nn.utils.weight_norm import weight_norm
import math
import json
from .net_builder import make_layers
from model.simpleNN import SimpleNN

#This assumes the classification of edges was done by the pairing_graph modules featurizer

class BinaryPairReal(nn.Module):
    def __init__(self, config): # predCount, base_0, base_1):
        super(BinaryPairReal, self).__init__()
        self.undirected = (not config['directed']) if 'directed' in config else True
        numBBOut = config['bb_out'] if 'bb_out' in config else (config['node_out'] if 'node_out' in config else 0)
        numRelOut = config['rel_out'] if 'rel_out' in config else (config['edge_out'] if 'edge_out' in config else 1)

        in_ch=config['in_channels']

        norm = config['norm'] if 'norm' in config else 'group_norm'
        dropout = config['dropout'] if 'dropout' in config else True

        self.use_node_for_rel_pred = config['use_node_for_rel_pred'] if 'use_node_for_rel_pred' in config else False
        self.in_ch=in_ch
        if self.use_node_for_rel_pred:
            rel_in_ch = in_ch*3
        else:
            rel_in_ch = in_ch

        layer_desc = config['layers'] if 'layers' in config else ['FC256','FC256','FC256']
        if 'FCnR' in layer_desc[-1]: #no ReLU
            assert 'Norm' in layer_desc[0]
            act= ['GroupNorm','ReLU']
        else:
            assert 'Norm' not in layer_desc[0]
            act =[]
        layer_desc = [rel_in_ch]+layer_desc#+['FCnR{}'.format(numRelOut)]
        layers, last_ch_relC = make_layers(layer_desc,norm=norm,dropout=dropout)
        self.layersRel = nn.Sequential(*layers)
        final, fin_ch_iout = make_layers([last_ch_relC,*act,'FCnR{}'.format(numRelOut)],norm=norm,dropout=dropout)
        self.finalRel = nn.Sequential(*final)

        if numBBOut>0:
            layer_desc = config['layers_bb'] if 'layers_bb' in config else ['FC256','FC256','FC256']
            layer_desc = [in_ch]+layer_desc#+['FCnR{}'.format(numBBOut)]
            layers, last_ch_bbC = make_layers(layer_desc,norm=norm,dropout=dropout)
            self.layersBB = nn.Sequential(*layers)
            final, fin_ch_out = make_layers([last_ch_bbC,*act,'FCnR{}'.format(numBBOut)],norm=norm,dropout=dropout)
            self.finalBB = nn.Sequential(*final)

        #This is written to by the PairingGraph object (which holds this one)
        self.numShapeFeats = config['num_shape_feats'] if 'num_shape_feats' in config else 16

        

        if 'shape_layers' in config:
            if type(config['shape_layers']) is list:
                layer_desc = config['shape_layers']
                layer_desc = [self.numShapeFeats]+layer_desc+['FCnR{}'.format(numRelOut)]
                layers, last_ch_relC = make_layers(layer_desc,norm=norm,dropout=dropout)
                self.shape_layers = nn.Sequential(*layers)
                self.frozen_shape_layers=False
            else:
                checkpoint = torch.load(config['shape_layers'])
                shape_config = checkpoint['config']['model']
                if 'state_dict' in checkpoint:
                    self.shape_layers =  eval(checkpoint['config']['arch'])(shape_config)
                    self.shape_layers.load_state_dict(checkpoint['state_dict'])
                else:
                    self.shape_layers = checkpoint['model']
                for param in self.shape_layers.parameters():
                    param.requires_grad=False
                self.frozen_shape_layers=True
            if 'weight_split' in config:
                if type(config['weight_split']) is float:
                    init = config['weight_split']
                else:
                    init = 0.5
                self.split_weighting = nn.Parameter(torch.tensor(init, requires_grad=True))
            else:
                self.split_weighting = None
        else:
            self.shape_layers=None



    def forward(self, features, adjacencyMatrix=None, numBBs=None):
        if adjacencyMatrix is None and numBBs is None:
            node_features, edge_indexes, edge_features, u_features = features #graph input
        else:
            edge_features = features[numBBs:]
            if numBBs>0:
                node_features = features[:numBBs]
            else:
                node_features = None
        if self.use_node_for_rel_pred:
            #set up both directions of relationships, appending node features to edge features
            #node_feats_for_edges = edge_features.new_empty(edge_features.size(0),self.in_ch*2)
            #for i,(n1,n2) in enumerate(edge_indexes):
            #    node_feats_for_edges[i,0:self.in_ch] = node_features[n1]
            #    node_feats_for_edges[i,self.in_ch:] = node_features[n2]
            #    node_feats_for_edges[i+len(edge_indexes),0:self.in_ch] = node_features[n2]
            #    node_feats_for_edges[i+len(edge_indexes),self.in_ch:] = node_features[n1]
            #edge_features = torch.cat([edge_features.repeat(2,1),node_feats_for_edges],dim=1)
            edge_features = torch.cat([edge_features,node_features[edge_indexes[0]],node_features[edge_indexes[1]]],dim=1)
        featsRel = self.layersRel(edge_features)
        res = self.finalRel(featsRel)

        if self.shape_layers is not None:
            if self.frozen_shape_layers:
                self.shape_layers.eval()
            res2 = self.shape_layers(edge_features[:,-self.numShapeFeats:])
            if self.split_weighting is None:
                res = (res+res2)/2
            else:
                weight = self.split_weighting.clamp(0,1)
                res = weight*res + (1-weight)*res2
        if node_features is not None:
            featsB = self.layersBB(node_features)
            resB = self.finalBB(featsB)
        else:
            featsB=None
            resB=None
        #import pdb;pdb.set_trace()
        if self.undirected:
            res = (res[:res.size(0)//2] + res[res.size(0)//2:])/2
            featsRel = (featsRel[:featsRel.size(0)//2] + featsRel[featsRel.size(0)//2:])/2
        if adjacencyMatrix is None and numBBs is None:
            return resB[:,None,:],res[:,None,:], featsB,featsRel, u_features
        else:
            return resB, res


