import utils.img_f as img_f
import numpy as np

def tensmeyer_brightness(img, foreground=0, background=0):
    if img.shape[2]==3:
        gray = img_f.rgb2gray(img)
    else:
        gray = img
    try:
        ret,th = img_f.otsuThreshold(gray)
    except ValueError:
        th=img/2


    th = (th.astype(np.float32) / 255)#[...,None]

    if img.shape[2]==3:
        th = th[...,None]


    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background

    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_tensmeyer_brightness(img, sigma=20, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    if kwargs.get("better",False):
        foreground = (random_state.beta(1.2,2)-0.1)*256/0.9
        background = ((-random_state.beta(1.2,2))+0.1)*256/0.9
    else:
        foreground = random_state.normal(0,sigma)
        background = random_state.normal(0,sigma)
    #print('fore {}, back {}'.format(foreground,background))

    img = tensmeyer_brightness(img, foreground, background)

    return img


def increase_brightness(img, brightness=0, contrast=1):
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_random_brightness(img, b_range=[-50,51], **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    brightness = random_state.randint(b_range[0], b_range[1])

    img = increase_brightness(img, brightness)

    return input_data

def apply_random_color_rotation(img, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    shift = random_state.randint(0,255)

    hsv = img_f.rgb2hsv(img)
    hsv[...,0] = hsv[...,0] + shift
    img = img_f.hsv2rgb(hsv)

    return img
