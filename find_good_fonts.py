import pytesseract
from synthetic_text_gen import SyntheticWord
from utils import img_f
import editdistance
import numpy as np

for start in [0,1000,2000,3000,4000,5000,6000]:

    THRESHOLD=3
    font_dir = '../data/fonts'
    gen = SyntheticWord(font_dir)
    fonts = gen.getTestFontImages(start)
    height=24

    custom_oem_psm_config = r'--psm 7'

    all_fonts=[]
    for index,(findex,fontfile,images) in enumerate(fonts):
        print('Eval {}/{}'.format(index,len(fonts)),end='\r')
        total_dist=0
        for text,image in images:
            image = 255*(1-image)
            if image.shape[0] != height:
                new_width = round(image.shape[1]*height/image.shape[0])
                image = img_f.resize(image,(height,new_width))
            image = image.astype(np.uint8)
            image = np.pad(image,4,constant_values=255)
            #if total_dist==0:
            #    img_f.imshow('x',image)
            #    img_f.show()
            pred_text = pytesseract.image_to_string(image,config=custom_oem_psm_config)
            dist = editdistance.eval(text,pred_text)
            #print('{} = {} : {}'.format(dist,text,pred_text))
            total_dist+=dist

        all_fonts.append((total_dist,findex,fontfile))
        if total_dist<THRESHOLD:
            print('GOOD: {},{}'.format(findex,fontfile))


    with open('scored_fonts_{}.csv'.format(start),'w') as out:
        for d,i,f in all_fonts:
            out.write('{},{},{}\n'.format(d,i,f))
    print('wrote {}'.format(start))

print('DONE')
