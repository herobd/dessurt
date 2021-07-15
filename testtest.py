from utils import img_f
import numpy as np
PRINT_ATT=False
ATT_VIS=[]
ATT_VIS_TEXT=[]
ATT_TEXT=[]
NUMS=[]



def attDisplay(img,ocr,q,a,pred):
    a=a[:-1]
    img = (1-img[0].cpu().numpy())/2
    swin = zip(ATT_VIS,ATT_VIS_TEXT,ATT_TEXT[:len(ATT_VIS)],NUMS[:len(ATT_VIS)])

    full = zip(ATT_TEXT[len(ATT_VIS):],NUMS[len(ATT_VIS):])

    size_bar = 20

    for layer,(im,im_text,text,nums) in enumerate(swin):
        att = img_f.resize(im.detach().numpy(),img.shape)
        img_color = np.stack((att,1-att,img),axis=2)
        img_f.imwrite('att/swin{}.png'.format(layer),(img_color*255))

        num_ocr,num_q,num_a = nums

        assert(num_ocr+num_q+num_a==im_text.size(0))

        text_att = text.detach().numpy()
        text_att[text_att>1] = 1
        img_f.imwrite('att/text_att{}.png'.format(layer),(text_att*255).astype(np.uint8))
        text = text.sum(dim=0)
        ocr_att = text[:num_ocr]
        q_att = text[num_ocr:num_ocr+num_q]
        a_att = text[num_ocr+num_q:num_ocr+num_q+num_a]
        
        char_per_att = ocr_att.size(0)//len(ocr)

        
        print('\nLayer {}'.format(layer))
        #print('OCR ATTENTION')
        #for i,c in enumerate(ocr):
        #    string='[{}]:'.format(c)
        #    for ai in range(char_per_att):
        #        string+='#'*int(size_bar*ocr_att[char_per_att*i + ai])
        #        string+='\t\t'+str(ocr_att[char_per_att*i + ai])
        #        string+='\n    '
        #    string = string[:-1]
        #    print(string)
        #print('\nQ ATTENTION')
        #for i,c in enumerate(q):
        #    string='[{}]:'.format(c)
        #    for ai in range(char_per_att):
        #        string+='#'*int(size_bar*q_att[char_per_att*i + ai])
        #        string+='\t\t'+str(q_att[char_per_att*i + ai])
        #        string+='\n    '
        #    string = string[:-1]
        #    print(string)
        att_aa = text_att[-num_a:,-num_a:]
        aa = ' '
        #print('\nA ATTENTION')
        for i,c in enumerate(a):
            string='[{}]:'.format(c)
            
            for ai in range(char_per_att):
                string+='#'*int(size_bar*a_att[char_per_att*i + ai])
                string+='\t\t'+str(a_att[char_per_att*i + ai])
                string+='\n    '
                aa+=c
            string = string[:-1]
            #print(string)
        aa+='\n'
        for i,c in enumerate(a):
            for ai in range(char_per_att):
                I = char_per_att*i + ai
                aa+=c
                for j in range(len(a)):
                    for aj in range(char_per_att):
                        J = char_per_att*j + aj
                        v = att_aa[I,J]
                        if v==0:
                            aa+=' '
                        elif v< 0.1:
                            aa+='.'
                        elif v< 1:
                            aa+=str(v)[2]
                        else:
                            aa+='#'
                aa+='\n'
        print('A to A')
        print(aa)
        #import pdb;pdb.set_trace()
        assert (im_text[-num_a:]==0).all
        
