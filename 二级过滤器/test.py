import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import cv2

# 读取目录下所有的jpg图片
img =cv2.imread('./data/0.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()