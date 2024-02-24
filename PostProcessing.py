import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import io
import PySimpleGUI as sg
from PIL import Image

# Diffusive Background Substraction
# Anisotropic Diffusion Background Subtraction as implemented in this paper: https://iopscience.iop.org/article/10.1088/1361-6501/aafca9
# diffusion not in default opencv package. Must use pip install opencv-python opencv-contrib-python

# alpha:	The amount of time to step forward by on each iteration (normally, it's between 0 and 1).

# K: sensitivity to the edges
# It is observed that for small values of the threshold parameter (K  =  5), the particle images are not diffused sufficiently.
# Conversely, a large value of K (K  =  50 in the example), causes diffusion of the sharp reflection along with the particle images.
# It is observed that an intermediate value of K (K  =  10) yields better results than those for K  =  5 and K  =  50, 
# by diffusing the particles sufficiently and by retaining the sharp reflection in the background image (middle column in figure 3).

# niters: number of iterations
# The plots for K  =  10 show that the particles are removed sufficiently after about 300 iterations; 
# further increasing the number of iterations does not produce any improvement in the background image.
# If K is lower, more niters needed, plateau at 500 niters
# If K is higher, less niters are needed but still reach optimum performance in the 200-300 range.
def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

src = cv.imread('TestImages/microbeads.png')
src = rescaleFrame(src)

dest = cv.ximgproc.anisotropicDiffusion(src, alpha=0.1, K=10, niters=300)
res = cv.subtract(src, dest)
cv.imshow('source', src)
cv.imshow('result', res)

cv.waitKey(0)

# Segmentation