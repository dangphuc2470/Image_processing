import cv2
import matplotlib.pyplot as plt

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

if __name__ == "__main__":
    imagePath = r"F:\P\HK7\XLA\Lab-01\baboon.png"
    image = cv2.imread(imagePath)
    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    smaller_image = cv2.resize(image_converted, (0, 0), fx=2, fy=2)
    print(smaller_image.shape)
    plt.imshow(smaller_image)
    plt.show()