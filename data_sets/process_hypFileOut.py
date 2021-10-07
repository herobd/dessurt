from unidecode import unidecode
import img_f
import numpy as np

def resolveOverlap(lines,bbs):
    bbs = np.array(bbs)
    x1s = bbs[:,0]
    y1s = bbs[:,1]
    x2s = bbs[:,2]
    y2s = bbs[:,3]

    assert (x1s<x2s).all() and (y1s<y2s).all()
    
    area = (x2s-x1s)*(y2s-y1s)

    x1s_a = x1s[None,:].repeat(len(bbs),axis=0)
    x1s_b = x1s[:,None].repeat(len(bbs),axis=1)
    y1s_a = y1s[None,:].repeat(len(bbs),axis=0)
    y1s_b = y1s[:,None].repeat(len(bbs),axis=1)
    x2s_a = x2s[None,:].repeat(len(bbs),axis=0)
    x2s_b = x2s[:,None].repeat(len(bbs),axis=1)
    y2s_a = y2s[None,:].repeat(len(bbs),axis=0)
    y2s_b = y2s[:,None].repeat(len(bbs),axis=1)
    area_a = area[None,:].repeat(len(bbs),axis=0)
    area_b = area[:,None].repeat(len(bbs),axis=1)

    inter_rect_x1 = np.maximum(x1s_a,x1s_b)
    inter_rect_x2 = np.minimum(x2s_a,x2s_b)
    inter_rect_y1 = np.maximum(y1s_a,y1s_b)
    inter_rect_y2 = np.minimum(y2s_a,y2s_b)
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * np.clip(
            inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)

    iou = inter_area/(area_a+area_b-inter_area)
    #import pdb;pdb.set_trace()
    iou[np.tri(len(bbs)).astype(np.bool)]=0 #no duplicate and self intersection
    overlapped = iou>0.6
    rs,cs = np.nonzero(overlapped)

    for r,c in zip(rs,cs):
        print('Overlap')
        print(lines[r])
        print(lines[c])

remove='⌨┃'

last_image=None
#check='005449024_00310.jpg-A-00200'

with open('all_states_random_4k_KEYOUTS/hypFileOut.all_states_random_4k_Mx8M.BBOX') as f:
    textlines = f.readlines()

for textline in textlines:
    #print('before {}'.format(textline))
    for c in remove:
        textline = textline.replace(c,'')
    #print('replace {}'.format(textline))
    textline = unidecode(textline)
    textline.replace('  ',' ')
    #print('after {}'.format(textline))
    words = textline.split(' ')
    header=words[0]
    w_bbs = []#words[1::2]
    w_words = []#words[2::2]
    i=1
    while i < len(words)-1:
        w_bbs.append(words[i])
        w_words.append(words[i+1])
        i+=2
        while i < len(words)-1 and not (len(words[i])>5 and words[i][0]=='[' and words[i][-1]==']'):
            w_words[-1]+=' '+words[i]
            i+=1

    #import pdb;pdb.set_trace()
    #if header==check:
    #    import pdb;pdb.set_trace()

    image = header.split('-')[0]

    if image!=last_image:
        print(image)
        if last_image is not None:
            resolveOverlap(lines,bbs)
            img_f.imshow('x',draw_img)
            img_f.show()
        draw_img = img_f.imread('1940_all-states_random_4k/'+image)
        draw_img = np.stack([draw_img]*3,axis=2)
        lines=[]
        bbs=[]
    last_image = image
    
    full_line=None
    for bb,word in zip(w_bbs,w_words):
        if len(word)>0:
            p1,p2 = bb[1:-1].split(':')
            x1,y1 = [int(a) for a in p1.split(',')]
            x2,y2 = [int(a) for a in p2.split(',')]
            #assert x1<x2
            #assert y1<y2
            if full_line is None:
                top_left_p = (x1,y1)
                #determine orientation
                if x1<x2 and y1<y2:
                    orientation = 'normal'
                    bot_left_p = (x1,y2)
                elif x1<x2 and y1>y2:
                    orientation = 'counterclockwise'
                    bot_left_p = (x2,y1)
                elif x1>x2 and y1>y2:
                    orientation = 'upsidedown'
                    bot_left_p = (x1,y2)
                elif x1>x2 and y1<y2:
                    orientation = 'clockwise'
                    bot_left_p = (x2,y1)
                else:
                    assert False #0 size
            else:
                if orientation=='normal':
                    assert x1<x2 and y1<y2
                    assert y1==top_left_p[1]
                    overlap = x1<=last_x2
                elif orientation=='counterclockwise':
                    assert x1<x2 and y1>y2
                    assert x1==top_left_p[0]
                    overlap = y1>=last_y2
                elif orientation=='upsidedown':
                    assert x1>x2 and y1>y2
                    assert y1==top_left_p[1]
                    overlap = x1>=last_x2
                elif orientation=='clockwise':
                    assert x1>x2 and y1<y2
                    assert x1==top_left_p[0]
                    overlap = y1<=last_y2
            first_char_is_punc = any(word[0]==punc for punc in ',.:;?!')
            if full_line is None:
                full_line=word
            elif overlap and first_char_is_punc:
                full_line+=word
            else:
                full_line+=' '+word
            
            #print('   >> ({},{},{},{}) {}'.format(x1,y1,x2,y2,word))
            last_x2=x2
            last_y2=y2
    if full_line is None:
        continue
    bot_right_p = (x2,y2)
    if orientation=='normal':
        top_right_p = (x2,y1)
        bb = [*top_left_p,*bot_right_p]
        color = (0,0,255)
    elif orientation=='counterclockwise':
        top_right_p = (x1,y2)
        bb = [*top_right_p,*bot_left_p]
        color = (0,255,0)
    elif orientation=='upsidedown':
        top_right_p = (x2,y1)
        bb = [*bot_right_p,*top_left_p]
        color = (255,255,0)
    elif orientation=='clockwise':
        top_right_p = (x1,y2)
        bb = [*bot_left_p,*top_right_p]
        color = (0,255,255)

    lines.append((full_line,top_left_p,top_right_p,bot_right_p,bot_left_p))
    bbs.append(bb)
    print('{} : {}'.format(image,full_line))


    



    img_f.line(draw_img,top_left_p,bot_left_p,(255,0,0),2)
    img_f.line(draw_img,top_left_p,top_right_p,color,2)
    img_f.line(draw_img,bot_right_p,top_right_p,color,2)
    img_f.line(draw_img,bot_right_p,bot_left_p,color,2)
