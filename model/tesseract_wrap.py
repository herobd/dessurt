from base import BaseModel
from PIL import Image
import numpy as np
from torch import nn
from torch import functional as F
import pytesseract

#from .pretrained_gen import Print


class TesseractWrap(nn.Module):

    def __init__(self, config):
        super(TesseractWrap, self).__init__()
        self.height=config['height']
        psm = config['psm'] if 'psm' in config else 7
        self.custom_oem_psm_config = r'--psm {}'.format(psm) #or 13

    def forward(self, input, style=None):
        if input.size(2) != self.height:
            new_width = round(input.size(3)*self.height/input.size(2))
            input = F.interpolate(image,(self.height,new_width),mode='bilinear')
        input = (1-input)*127.5
        input = input.permute(0,2,3,1).cpu().numpy().astype(np.uint8) #move color channel to back
        if input.shape[-1] == 1:
            input = input[:,:,:,0]
        toRet=[]
        for b in range(input.shape[0]):
            image = Image.fromarray(input[b])
            text = pytesseract.image_to_string(image,config=self.custom_oem_psm_config)
            toRet.append(text[:-2]) #not sure what they are, but there are a couple extrac chars
        return toRet

    #def eval(self):
    #    pass

    def summary(self):
        print('Tesseract OCR wrapper')
