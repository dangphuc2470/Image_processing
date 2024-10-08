
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

if __name__ == "__main__":
    imagePath = r"F:\P\HK7\XLA\Lab-01\baboon.png"
    im = cv2.imread(imagePath)
    im_converted = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_flip = cv2.flip(im_converted, 0)
    im_mirror = cv2.flip(im_converted, 1)

    plt.imshow(im_flip)
    plt.show()
    plt.imshow(im_mirror)
    plt.show()
