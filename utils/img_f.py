import skimage

#These are all based on the OpenCV functions, to make the conversion to scikit image easier (also should make future changes easier as well)

def line(img,p1,p2,color,thickness=1):
    rr,cc = skimage.draw.line(p1[1],p1[0],p2[1],p2[0])
    img[rr,cc]=color
    if thicnkness>1:
        rr,cc = skimage.draw.line(p1[1]+1,p1[0]+1,p2[1]+1,p2[0]+1)
        img[rr,cc]=color
        rr,cc = skimage.draw.line(p1[1],p1[0]+1,p2[1],p2[0]+1)
        img[rr,cc]=color
        rr,cc = skimage.draw.line(p1[1]+1,p1[0],p2[1]+1,p2[0])
        img[rr,cc]=color
    if thickness>2:
        rr,cc = skimage.draw.line(p1[1]-1,p1[0]-1,p2[1]-1,p2[0]-1)
        img[rr,cc]=color
        rr,cc = skimage.draw.line(p1[1],p1[0]-1,p2[1],p2[0]-1)
        img[rr,cc]=color
        rr,cc = skimage.draw.line(p1[1]-1,p1[0],p2[1]-1,p2[0])
        img[rr,cc]=color
        rr,cc = skimage.draw.line(p1[1]+1,p1[0]-1,p2[1]+1,p2[0]-1)
        img[rr,cc]=color
        rr,cc = skimage.draw.line(p1[1]-1,p1[0]+1,p2[1]-1,p2[0]+1)
        img[rr,cc]=color
        assert(thickness<4)


def imread(path,color=True):
    return skimage.io.imread(path,not color)

def imwrite(path,img):
    return skimage.io.imsave(path,img)

def imshow(name,img):
    return skimage.io.imshow(img)

def show(): #replaces cv2.waitKey()
    return skimage.io.imshow(img)

def resize(img,dim,fx=None,fy=None): #remove ",interpolation = cv2.INTER_CUBIC"
    hasColor = len(img.shape)==3
    if dim[0]==0:
        downsize = fx<1 and fy<1
        
        return skimage.transform.rescale(img,(fy,fx),3,multichannel=hasColor,anti_aliasing=downsize)
    else:
        downsize = dim[0]<img.shape[0] and dim[1]<img.shape[1]
        return skimage.transform.resize(img,dim,3,multichannel=hasColor,anti_aliasing=downsize)

def otsuThreshold(img):
    t = skimage.filters.threshold_otsu(img)
    return (img>t)*255,t

def rgb2hsv(img):
    return skimage.color.rgb2hsv(img)
def hsv2rgb(img):
    return skimage.color.hsv2rgb(img)
def rgb2gray(img):
    return skimage.color.rgb2gray(img)
def gray2rgb(img):
    return skimage.color.gray2rgb(img)
