import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d 
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import skimage.measure as measure

#%% 1D convolution
n = 10
m = 3

x = np.random.rand(n)
w = np.random.rand(m)

y = np.convolve(x,w, mode='valid')
print(y)

#%% 2D convolution

size = 5
n2 = (10,10)
m2 = (size, size)

x2 = np.random.rand(*n2)
ws = [[np.random.rand(*m2) for _ in range(5)] for _ in range(3)]
y2 = convolve2d(x2, ws[0][0], mode='valid')

#%% 2D Image convolution (RGB --> 3D)

img = cv2.imread('./gestures/20180304175203_0.png', 0)

#b,g,r = img[:,:,0],img[:,:,1],img[:,:,2]

number = 1
for i,c in enumerate([img]):
    for j,w in enumerate(ws[i]):
        
        y_c = convolve2d(c,w, mode='valid')
        y_c = measure.block_reduce(y_c, (3,3), np.max)
        y_c = convolve2d(y_c,w, mode='valid')
        y_c = measure.block_reduce(y_c, (3,3), np.max)
        y_c = convolve2d(y_c,w, mode='valid')
        y_c = measure.block_reduce(y_c, (2,2), np.max)
        y_c = convolve2d(y_c,w, mode='valid')
        y_c = measure.block_reduce(y_c, (2,2), np.max)

        plt.subplot(3,5,number)
        plt.imshow(y_c, norm=Normalize(vmin=0, vmax=600))
        number += 1

#%% 

















