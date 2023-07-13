import torch
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = mpimg.imread('./CK+48/anger/S010_004_00000017.png')
    resized_image = cv2.resize(image, (224, 224))
    plt.imshow(resized_image, cmap='gray')
    plt.show()