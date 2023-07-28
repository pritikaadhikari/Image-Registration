#Author: Om Sinkar

import cv2
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
import torch
import torchvision.transforms as T
from PIL import Image
#from pathlib import Path
from matplotlib import pyplot as plt


# %matplotlib inline

orig_img = cv2.imread('/input/before.jpg')
orig_img = np.asarray(orig_img)
orig_img = Image.fromarray(orig_img)

perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
x=0
while(True):
  perspective_imgs = perspective_transformer(orig_img)
  plt.imshow(np.squeeze(perspective_imgs))
  plt.show()
  y=str(x)
  # print(x)
  cv2.imwrite('',np.squeeze(perspective_imgs))
  x=x+1
  if(x>=200):
    break

