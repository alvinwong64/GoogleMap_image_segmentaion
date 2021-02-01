from PIL import Image
from PIL import ImageOps
import cv2
import numpy as np
im = Image.open(r'./data/masks/0.png').convert('RGB')
im= ImageOps.grayscale(im)
# im = np.array(im)
# im2 = Image.open(r'./data/imgs/1.jpg')
# im= cv2.imread('./data/masks/1.png')
# img_nd = np.array(im)im
# print(len(set(im.flatten())))
img_nd = np.array(im)

img_nd[[0,15,48,75]]=0

# img_trans = img_nd.transpose((2, 0, 1))

print(type(im))

# print(im[:][0:3][0])
