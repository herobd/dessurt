from data_sets.gen_daemon import GenDaemon
import numpy as np
from utils import img_f

H=1300
W=1500
word='Dessurt'
word_height=16

gen = GenDaemon('../data/fonts',clear_fonts=True)
used = set()

page = np.zeros((2000,1500),dtype=np.uint8)
x=2
y=2
max_x=0
count=0
while True:
    ows = gen.generate(word,used=used)
    w,img=ows[0]

    scale = 16/img.shape[0]
    img = img_f.resize(img,[0],fx=scale,fy=scale)

    if x+img.shape[1]<W:

        page[y:y+img.shape[0],x:x+img.shape[1]]=img
        count+=1

    y+=word_height+5
    max_x = max(max_x,x+img.shape[1])
    
    if y+word_height>=H:
        y=2
        x=max_x+7
        max_x=x

        if x+7>=W:
            break

    if count==586:
        break

page = 255-page

print('Count {}'.format(count))
img_f.imwrite('easy_font_image.png',page)



