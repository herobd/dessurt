
class ResLayer(nn.Module):
    def __init__(self, dim):
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2),
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),

class ConvPatchEmbed(nn.Module):
    r""" Image to Patch Embedding, but with more convolutions
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=8, in_chans=1, embed_dim=256, norm_layer=None,cnn_model_small=True,lighter=False):
        super().__init__()
        #From cnn_lstm_skip_forSwin.py
        ks = [7, 3, 3, 3, 3, 3, 3]
        ps = [3, 1, 1, 1, 1, 1, 1]
        ss = [2, 1, 1, 1, 1, 1, 1]
        if lighter:
            nm = [64, 128, 128, 256, 256, 256, 256]
        else:
            nm = [64, 128, 128, 256, 512, 512, 512]
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(nm[-1], embed_dim, kernel_size=1, stride=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None



        cnn = nn.Sequential()
        norm = 'group'
        nc = in_chans

        def ResLayer(i, norm=None):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if norm is not None and 'group' in norm:
                cnn.add_module('groupnorm{0}'.format(i), nn.GroupNorm(getGroupSize(nOut),nOut))
            elif norm is not None and 'layer' in norm:
                cnn.add_module('layernorm{0}'.format(i), nn.LayerNorm(nOut))
            elif norm:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('dropout{0}'.format(i), nn.Dropout2d(p=0.1,inplace=True))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0,norm) #32>16 or 64>32
        if not cnn_model_small:
            cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16
        convRelu(1,norm)
        convRelu(2, norm)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8
        convRelu(3,norm)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2,2))  # 256x4
        convRelu(4, norm)
        convRelu(5,norm)                                           # 512x4
        #cnn.add_module('pooling{0}'.format(3),
        #               nn.MaxPool2d((2, 1), (2, 1)))  # 512x2x4
        convRelu(6, norm)                                     #512x1x1 even 32x32 to 4x4, that's a 8 reduction


        self.cnn = nn.Sequential(
                nn.Conv2d(nIn, 64, kernel_size=5, stride=1, padding=2),
                nn.GroupNorm(getGroupSize(64),64),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #downsample 32->16
                ResLayer(128),
                ResLayer(128),
                nn.GroupNorm(getGroupSize(128),128),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), #downsample 16->8
                ResLayer(256),
                ResLayer(256),
                nn.GroupNorm(getGroupSize(256),256),
                nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), #downsample 8->4
                ResLayer(512),
                ResLayer(512),
            )



    def forward(self, x):
        B, C, H, W = x.shape
        x = self.cnn(x)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
