from utils import img_f
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from data_sets.gen_daemon import GenDaemon
a=GenDaemon('../data/fonts')
a.generateLabelValuePairs()
