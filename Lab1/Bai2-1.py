
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw 
from PIL import ImageFont
import os
import matplotlib.pyplot as plt
import numpy as np

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst



if __name__ == "__main__":
    imagePath = r"F:\P\HK7\XLA\Lab-01\baboon.png"
    im = Image.open(imagePath)
    im_flip = ImageOps.flip(im)
    im_mirror = ImageOps.mirror(im)
    plt.imshow(get_concat_h(im_flip, im_mirror))
    plt.show()
    