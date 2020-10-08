from base import BaseModel
from PIL import Image
from torch import nn
from torch import functional as F
try:
    import pytesseract
except:
    pass

#from .pretrained_gen import Print


class TesseractWrap(nn.Module):

    def __init__(self, height=32,psm=7):
        #super(TesseractWrap, self).__init__(None)
        self.height=height
        self.custom_oem_psm_config = r'--psm {}'.format(psm) #or 13

    def forward(self, input, style=None):
        if input.size(2) != self.height:
            new_width = round(input.size(3)*self.height/input.size(2))
            input = F.interpolate(image,(self.height,new_width),mode='bilinear')
        input = input.permute(0,2,3,1).cpu().numpy() #move color channel to back
        toRet=[]
        for b in input.size(0):
            image = Image.fromarray(input[b])
            text = pytesseract.image_to_string(image,config=self.custom_oem_psm_config)
            toRet.append(text)
        return toRet

    #def eval():
    #    pass
