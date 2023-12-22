import numpy as np
import cv2 as cv
import tifffile as tiff

# This is an sample program for performing hyperspectral analysis using the LuminaScope system.
# This software is distributed under the Apache 2.0 License.
# For this analysis we need the three main libraries above.
# They are numpy, opencv-python, and tifffile in that order.

# Opening tiff file from file location
# File location is variable and depends on folders
img = cv.imread('autumn.tif')

# Show image
# syntax: Picture name, variable
cv.imshow('Autumn', img)

# Reading videos
# capture = cv.VideoCapture('Videos/dog.mp4')
# while True:
#   isTrue, frame = capture.read()
#   cv.imshow('Video', frame)
#   if cv.waitkey(20) & 0xFF==ord('d')
#       break
# capture.relase()
# cv.destroyAllWindows()

# Rescaling images
# Works on images, videos, and live video
# def rescaleFrame(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)

#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Changing live video resolution
# def changeRes(width,height):
#     capture.set(3, width)
#     capture.set(4,height)

# Isolate blue, green. red image
# Resulting grayscale images show concentration/intensity of r,b,g values
# b,g,r = cv.split(img)

# cv.imshow('Blue', b)
# cv.imshow('Green', g)
# cv.imshow('Red', r)

# Merging images from different channels
# merge = cv.merge([b,g,r])
# cv.imshow('Merged', merge)

# Reconstructing images to show color images that show concentration of r,g,b
# blank = np.zeros(img.shape[:2], dtype='uint8')
# blue = cv.merge([b,blank,blank])
# green = cv.merge([blank,g,blank])
# red = cv.merge([blank,blank,r])

# cv.imshow('Blue', blue)
# cv.imshow('Green', green)
# cv.imshow('Red', red)

# Calculate histogram of color? levels


# Crop image by indexing pixel ranges
# cropped = img[50:200, 200:400]
# cv.imshow('Cropped', cropped)

# Smooth image can be done by many techniques
# Average blur, higher number is more blurring
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

# Gaussian blur
# More natural compared to averaging, less blurred, due to weights on averaging
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)

# Median blur

# Threshhold image

# Calculate phasor with fast fourier transform

# Hyperspectral analysis

cv.waitKey(0)