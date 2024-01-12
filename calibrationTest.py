import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import os

# calibration python script
# -open image of coloured boxes
# -isolate every box
# - calculate the phasor transformation of every box
# -and plot the mean  G and mean S coordinates of every box in a phasor plot

# assign directory
directory = 'Images'

# define rescaling function
def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('Images/image0.jpg')

rescaled = rescaleFrame(img, 0.25)

# cropping step was manual to determine location
cropped = rescaled[130:700, 90:500]

median = cv.medianBlur(cropped, 3)
gauss = cv.GaussianBlur(cropped, (3,3), 0)

# Show image
# syntax: Picture name, variable
cv.imshow('Median', median)
cv.imshow('Gaussian', gauss)

cv.waitKey(0)