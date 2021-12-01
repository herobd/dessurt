from base import BaseModel
import torch
from torch import nn
from utils.util import getGroupSize2
from model.swin_transformer import to_2tuple



class ResLayer(nn.Module):
    def __init__(self, indim):
        super().__init__()
        dim = indim//2
        self.conv = nn.Sequential(
                nn.Conv2d(indim, dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(getGroupSize2(dim),dim),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                nn.Conv2d(dim, indim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(getGroupSize2(indim),indim),
                nn.Dropout2d(p=0.0625,inplace=True),
                )
        self.act = nn.ReLU(True)

    def forward(self,x):
        side = self.conv(x)
        return self.act(x+side)

class ResConvPatchEmbed(nn.Module):
    r""" Image to Patch Embedding, but with more convolutions
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=8, in_chans=1, embed_dim=256):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1, stride=1)

        self.norm = nn.GroupNorm(getGroupSize2(embed_dim),embed_dim)

        self.cnn = nn.Sequential(
                nn.Conv2d(in_chans, 64, kernel_size=5, stride=1, padding=2),
                nn.GroupNorm(getGroupSize2(64),64),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #downsample 32->16
                nn.GroupNorm(getGroupSize2(128),128),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                ResLayer(128),
                ResLayer(128),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), #downsample 16->8
                nn.GroupNorm(getGroupSize2(256),256),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                ResLayer(256),
                ResLayer(256),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), #downsample 8->4
                nn.GroupNorm(getGroupSize2(512),512),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                ResLayer(512),
                ResLayer(512),
            )



    def forward(self, x):
        B, C, H, W = x.shape
        x = self.cnn(x)
        x = self.proj(x).flatten(2)# B C h*w #.transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x.transpose(1,2)#B h*w C

class HWRWithEmb(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        n_class = config['num_class']
        self.embedder = ResConvPatchEmbed(32,embed_dim=1,in_chans=config.get('in_dim',1))
        #self.pool = nn.AvgPool2d((4,1))
        self.pool = nn.Sequential(
                nn.Conv2d(512,512,(4,3),1,(0,1)),
                nn.GroupNorm(getGroupSize2(512),512),
                #nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True)
                )
        self.conv1d = nn.Sequential(
                #nn.Conv1d(512,512,3,1,1),
                nn.Conv1d(512,128,3,1,1),
                nn.GroupNorm(getGroupSize2(128),128),
                #nn.Dropout1d(p=0.0625,inplace=True),
                nn.ReLU(True),
                nn.Conv1d(128,512,3,1,1),
                nn.GroupNorm(getGroupSize2(512),512),
                #nn.Dropout1d(p=0.0625,inplace=True),
                #nn.ReLU(True),
                )
        self.pred = nn.Sequential(
                nn.ReLU(True),
                #nn.Conv1d(512,n_class,1,1,0),
                nn.ConvTranspose1d(512,n_class,2,2,0),
                nn.LogSoftmax(dim=1)
                )

    def forward(self, x,ocr_res=None):
        x = self.embedder.cnn(x) #skip the extra things used in Transformer
        b,dim,h,w = x.size()
        assert h==4
        x = self.pool(x).view(b,dim,w)
        x = self.conv1d(x) + x
        x = self.pred(x)
        return x.permute(2,0,1) #batch,dim,witdh to width,batch,dim as recurrent output

