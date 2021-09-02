import os
import numpy as np
import torch
import utils.img_f as img_f
import math
from model.loss import *
from data_sets.test_random_message import display



def RandomMessagesDataset_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    def __to_tensor(instance,gpu):
        features, adjaceny, gt, num = instance

        if gpu is not None:
            features = features.float().to(gpu)
            adjaceny = adjaceny.to(gpu)
            gt = gt.float().to(gpu)
        else:
            features = features.float()
            gt = gt.float()
        return features, adjaceny, gt, num

    
    features, adj, gt, num = __to_tensor(instance,gpu)
    if True : #type(model) is MetaGraphNet:
        output,_ = model((features,adj._indices(),None,None))
        output=output[:num]
        gt=gt[:,None,:].expand(num,output.size(1),gt.size(1))
    else:
        output,_ = model(features,(adj,None),num)
    if lossFunc is not None:
        loss = lossFunc(output,gt)
        loss = loss.item()
    else:
        loss=0

    #print(loss)
    display(instance,torch.sigmoid(output[:,-1].cpu()))

    return (
            {'loss':loss},
             loss
            )


