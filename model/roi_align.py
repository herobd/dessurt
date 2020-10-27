import torch
from torch import nn
from torchvision.ops import roi_align


class ROIAlign(nn.Module):
    def __init__(self, output_H, output_W, spatial_scale, sampling_ratio=-1):
        super(ROIAlign, self).__init__()
        self.output_size = (output_H,output_W)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
