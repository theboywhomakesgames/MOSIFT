import cv2
from cv2 import imread
import numpy as np
from pandas import array
import scipy.ndimage.filters as fi
from skimage.feature import peak_local_max
from scipy.signal import argrelextrema
import math

frame_idx = 0
last_frame = 0
original_resolution = 128
resolution_step = 4
s = 1
sample_count = 5
delta = 1.6

max_size = 20

# Prepare
frame = cv2.imread("./Flowers_before_difference_of_gaussians.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.resize(frame, (512, 512))

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(frame ,None)