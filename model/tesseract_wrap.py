from base import BaseModel
from PIL import Image
import numpy as np
from torch import nn
from torch import functional as F
import threading
try:
    import pytesseract
except:
    pass

#from .pretrained_gen import Print


class TesseractWrap(nn.Module):

    def __init__(self, config):
        super(TesseractWrap, self).__init__()
        self.height=config['height']
        psm = config['psm'] if 'psm' in config else 7
        self.custom_oem_psm_config = r'--psm {}'.format(psm) #or 13
        self.threads = config['threads'] if 'threads' in config else 0

        def read(task):
            i,image = task
            image = Image.fromarray(image)
            try:
                text = pytesseract.image_to_string(image,config=self.custom_oem_psm_config)
            except:
                #Assume local installation of tesseract
                self.custom_oem_psm_config += r' --tessdata-dir "$HOME/local/share/tessdata"'
                text = pytesseract.image_to_string(image,config=self.custom_oem_psm_config)
            return i,text[:-2] #not sure what they are, but there are a couple extrac chars
        self.read = read

    def forward(self, input, style=None):
        if input.size(2) != self.height:
            new_width = round(input.size(3)*self.height/input.size(2))
            input = F.interpolate(image,(self.height,new_width),mode='bilinear')
        input = (1-input)*127.5
        input = input.permute(0,2,3,1).cpu().numpy().astype(np.uint8) #move color channel to back
        if input.shape[-1] == 1:
            input = input[:,:,:,0]
        if self.threads <= 1:
            toRet=[]
            for b in range(input.shape[0]):
                toRet.append(self.read.__call__((b,input[b]))[1])
                #image = Image.fromarray(input[b])
                #try:
                #    text = pytesseract.image_to_string(image,config=self.custom_oem_psm_config)
                #except pytesseract.pytesseract.TesseractError:
                #    #Assume local installation of tesseract
                #    self.custom_oem_psm_config += r' --tessdata-dir "$HOME/local/share/tessdata"'
                #    text = pytesseract.image_to_string(image,config=self.custom_oem_psm_config)
                #toRet.append(text[:-2]) #not sure what they are, but there are a couple extrac chars
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                res = excecutor.map(self.read,zip(range(input.shape[0],input))
            toRet=[None]*input.shape[0]
            for i,text in res:
                toRet[i]=text
        return toRet

    #def eval(self):
    #    pass

    def summary(self):
        print('Tesseract OCR wrapper')
