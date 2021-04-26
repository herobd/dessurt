from skimage import color, io
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.img_f as img_f
from utils import util
from utils.util import plotRect
from model.alignment_loss import alignment_loss
import math
from model.loss import *
from collections import defaultdict
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from model.optimize import optimizeRelationships, optimizeRelationshipsSoft
import json
from utils.forms_annotations import fixAnnotations, getBBInfo
from evaluators.draw_graph import draw_graph

def NobrainGraphPair_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    return NobrainQA_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc, toEval)

def NobrainQA_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    losses,run_log,out = trainer.run(instance,get=['strings'])
    aa=''
    for a in out['strings']:
        aa+=a+'\n'
    print('')
    print(aa)
    #losses.update(run_log)
    return (
             #dict(losses),
             run_log,
             None
            )


