from base import BaseModel
import torch
from torch import nn
from utils.util import getGroupSize2
from model.qa_imdoc_perceiver import QAImDocPerceiver
from model.cnn_hwr import ResConvPatchEmbed

class PseudoModel:
    def __init__(self,ocr_out_dim,patch_scale_x,patch_scale_y):
        self.ocr_out_dim = ocr_out_dim
        self.patches_resolution = [32,0]
        self.patch_scale_x = patch_scale_x
        self.patch_scale_y = patch_scale_y
        self.one_hot_conf = 0.9
        self.zero_hot_conf = (1-self.one_hot_conf)/(self.ocr_out_dim-1)
        self.ocr_append_image = True
class HWRWithEmbAndOCR(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        n_class = config['num_class']
        self.embedder = ResConvPatchEmbed(32,embed_dim=1,in_chans=2+97)
        self.pseudo_model = PseudoModel(97,1/self.embedder.patch_size[1],1/self.embedder.patch_size[0])
        self.pool = nn.Sequential(
                nn.Conv2d(512,512,(4,3),1,(0,1)),
                nn.GroupNorm(getGroupSize2(512),512),
                #nn.Dropout2d(p=0.0625,inplace=True),
                nn.ReLU(True)
                )
        self.conv1d = nn.Sequential(
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

    def forward(self, x, ocr_results):
        device = x.device
        self.pseudo_model.patches_resolution[1]=x.size(3)
        grid_ocr = QAImDocPerceiver.appendOCRToVisual(self.pseudo_model,ocr_results,device)
        x = torch.cat((x,grid_ocr),dim=1)
        x = self.embedder.cnn(x) #skip the extra things used in Transformer
        b,dim,h,w = x.size()
        assert h==4
        x = self.pool(x).view(b,dim,w)
        x = self.conv1d(x) + x
        x = self.pred(x)
        return x.permute(2,0,1) #batch,dim,witdh to width,batch,dim as recurrent output
