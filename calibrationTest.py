import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import os

# calibration python script
# - open image of coloured boxes
# - isolate every box
# - calculate the phasor transformation of every box
# - and plot the mean  G and mean S coordinates of every box in a phasor plot

# assign directory
directory = 'TestImages'

# define rescaling function
def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# define phasor plot function
def phasor(Gsmt):
    fft=np.fft.fft(Gsmt, axis=2)
   
    G=fft[:,:,1].real/fft[:,:,0].real
    G=np.nan_to_num(G, nan=0.0)
    # plt.figure()
    # plt.imshow(G)
    # # plt.axis('off')
    # plt.title('G')
    # plt.colorbar()
   
    S=fft[:,:,1].imag/fft[:,:,0].real
    S=np.nan_to_num(S, nan=0.0)
    # plt.figure()
    # plt.imshow(S)
    # # plt.axis('off')
    # plt.title('S')
    # plt.colorbar()
   
    Ph=np.arctan2(S[:,:], G[:,:])+np.pi
    Ph=np.nan_to_num(Ph, nan=0.0)
    # plt.figure()
    # plt.imshow(Ph)
    # # plt.axis('off')
    # plt.title('Phase')
    # plt.colorbar()
   
    Mod=np.sqrt(G**2+S**2)
    Mod=np.nan_to_num(Mod, nan=0.0)
    # plt.figure()
    # plt.imshow(Mod)
    # # plt.axis('off')
    # plt.title('Mod')
    # plt.colorbar()
   
   
    return G, S, Ph, Mod

img = cv.imread('TestImages/image0.jpg')

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