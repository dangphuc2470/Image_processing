
from PIL import Image
from PIL import ImageOps
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
    imagePath = r"F:\P\HK7\XLA\Lab-01\lenna.jpg"
    image = Image.open(imagePath)
    blue_array = np.array(image)
    blue_array[:,:,0] = 0
    blue_array[:,:,1] = 0
    plt.imshow(blue_array)
    plt.show()
    