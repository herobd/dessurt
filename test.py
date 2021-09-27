from skimage import data, future, io
from utils import img_f
import numpy as np
import sys

img = img_f.imread(sys.argv[1])
#mask = future.manual_polygon_segmentation(img)
mask = future.manual_lasso_segmentation(img)
if mask.sum()==0:
    mask = np.zeros_like(mask)
print(mask.flags)
print(mask.shape)
print(img.min(),img.max())
print(mask.min(),mask.max())
img_f.imshow('x',img*mask)
img_f.show()
