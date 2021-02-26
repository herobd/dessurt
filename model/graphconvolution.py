#from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
import math, copy

import torch

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from .simpleNN import SimpleNN
from .net_builder import getGroupSize
from .attention import MultiHeadedAttention


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() #does this do parameter init?

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj[0], support)
        #normalize based on how many things are summed
        output /= adj[1][:,None]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvWithAct(nn.Module):
        def __init__(self,in_ch,out_ch,norm='split_norm',dropout=0.4):
            super(GraphConvWithAct, self).__init__()
            self.graph_conv=GraphConvolution(in_ch,out_ch)
            self.split_normBB=None
            act_layers = []
            if norm=='batch_norm':
                act_layers.append(nn.BatchNorm1d(out_ch)) #essentially all the nodes compose a batch. There aren't enough some times
            elif norm=='group_norm':
                act_layers.append(nn.GroupNorm(4,out_ch))
            elif norm=='split_norm':
                #act_layers.append(nn.InstanceNorm1d(out_ch))
                self.split_normBB = nn.GroupNorm(4,out_ch)
                self.split_normRel = nn.GroupNorm(4,out_ch)
            if type(dropout) is float:
                act_layers.append(nn.Dropout(p=dropout,inplace=True))
            elif dropout:
                act_layers.append(nn.Dropout(p=0.3,inplace=True))
            act_layers.append(nn.ReLU(inplace=True))
            self.act_layers = nn.Sequential(*act_layers)

        def forward(self,node_features,adjacencyMatrix,numBBs):
            if self.split_normBB is not None:
                bb = self.split_normBB(node_features[:numBBs])
                rel = self.split_normRel(node_features[numBBs:])
                node_featuresX = torch.cat((bb,rel),dim=0)
            else:
                node_featuresX = node_features
            node_featuresX = self.act_layers(node_featuresX)
            node_featuresX = self.graph_conv(node_featuresX,adjacencyMatrix)
            return node_featuresX

class GraphResConv(nn.Module):
    """
    Two graph conv residual layer
    """

    def __init__(self, num_features,norm='group_norm',dropout=0.1, depth=2):
        super(GraphResConv, self).__init__()
        #self.side1=GraphConvWithAct(num_features,num_features,norm,dropout)
        #self.side2=GraphConvWithAct(num_features,num_features,norm,dropout)
        self.sideLayers=nn.ModuleList()
        for i in range(depth):
            sideLayers.append(GraphConvWithAct(num_features,num_features,norm,dropout))
            

    def forward(self,node_features,adjacencyMatrix,numBBs):
        #side = self.side1(node_features,adjacencyMatrix,numBBs)
        #side = self.side2(side,adjacencyMatrix,numBBs)
        side=node_features
        for layer in self.sideLayers:
            side = layer(side,adjacencyMatrix,numBBs)
        return node_features+side


class GraphSelfAttention(nn.Module):
    """
    Graph convolution using self attention across neighbors
    """

    def __init__(self, in_features, heads=8):
        super(GraphSelfAttention, self).__init__()
        self.mhAtt = MultiHeadedAttention(heads,in_features)

    def forward(self, input, adj):
        #construct mask s.t. 1s where edges exist
        #locs = torch.LongTensor(adj).t()
        #ones = torch.ones(len(adj))
        #mask = torch.sparse.FloatTensor(locs,ones,torch.Size([input.size(0),input.size(0)]))
        mask=adj
        input_ = input[None,...] # add batch dim
        return self.mhAtt(input_,input_,input_,mask)

    def __repr__(self):
        return self.__class__.__name__ +'(heads:{})'.format(self.mhAtt.h)

class GraphTransformerBlock(nn.Module):

    def __init__(self, features, num_heads, num_ffnn_layers=2, ffnn_features=None,split=False):
        super(GraphTransformerBlock, self).__init__()
        if ffnn_features is None:
            ffnn_features=features

        config_ffnn = {
                'feat_size': features,
                'num_layers': num_ffnn_layers-1,
                'hidden_size': ffnn_features,
                'out_size': features,
                #'reverse': True,
                'norm': None,
                'dropout': 0.1
                }
        self.ffnn = SimpleNN(config_ffnn)
        self.att = GraphSelfAttention(features,num_heads)
        #self.norm1 = nn.GroupNorm(getGroupSize(features),features)
        #self.norm2 = nn.GroupNorm(getGroupSize(features),features)
        self.norm1 = nn.GroupNorm(1,features)
        self.norm2 = nn.GroupNorm(1,features)

    def forward(self,input,adj=None,numBBs=None):
        if adj is None:
            input,adjMine,numBBs = input
        else:
            adjMine=adj
        #import pdb;pdb.set_trace()
        side1=self.att(input,adjMine)[0]
        side1+=input
        #TODO allow splitting into rel and box sides
        side1 = self.norm1(side1)
        side2=self.ffnn(side1)
        #return self.norm2(side2),adj,numBBs
        if adj is None:
            return self.norm2(side2+side1),adj,numBBs
        else:
            return self.norm2(side2+side1)
