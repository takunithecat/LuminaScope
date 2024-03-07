import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import io
import PySimpleGUI as sg
from PIL import Image
import scipy
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import imutils

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

src = cv.imread('Microbeads\Set3-F0-RGB.png')
src = rescaleFrame(src)

#Background Subtraction
dest = cv.ximgproc.anisotropicDiffusion(src, alpha=0.1, K=10, niters=300)
src = cv.subtract(src, dest)

shifted = cv.pyrMeanShiftFiltering(src, 21, 51)

# Segmentation

# gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
# thresh = cv.threshold(gray, 0, 255,
# 	cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

# # compute the exact Euclidean distance from every binary
# # pixel to the nearest zero pixel, then find peaks in this
# # distance map
# D = ndimage.distance_transform_edt(thresh)
# localMax = peak_local_max(D, min_distance=20,
# 	labels=thresh)
# peaks_mask = np.zeros_like(D, dtype=bool)
# peaks_mask[localMax] = True

# markers = ndimage.label(peaks_mask, structure=np.ones((3, 3)))[0]
# labels = watershed(-D, markers, mask=thresh)

# for label in np.unique(labels):
# 	# if the label is zero, we are examining the 'background'
# 	# so simply ignore it
# 	if label == 0:
# 		continue
# 	# otherwise, allocate memory for the label region and draw
# 	# it on the mask
# 	mask = np.zeros(gray.shape, dtype="uint8")
# 	mask[labels == label] = 255
# 	# detect contours in the mask and grab the largest one
# 	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
# 		cv.CHAIN_APPROX_SIMPLE)
# 	cnts = imutils.grab_contours(cnts)
# 	c = max(cnts, key=cv.contourArea)
# 	# draw a circle enclosing the object
# 	((x, y), r) = cv.minEnclosingCircle(c)
# 	cv.circle(src, (int(x), int(y)), int(r), (0, 255, 0), 2)

	
cv.imshow("Output", src)
cv.waitKey(0)