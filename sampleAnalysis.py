import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import os

# assign directory
directory = 'Images'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # read in image
        img = cv.imread(f)

        # crop images

        # smooth images
        img= cv.medianBlur(img, 3)

        # isolate r,g,b images
        b,g,r = cv.split(img)

        # threshhold images
        b_thresh = cv.adaptiveThreshold(b, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
        g_thresh = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
        r_thresh = cv.adaptiveThreshold(r, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)

        # calculate histogram of levels
        # blue_hist = cv.calcHist([b], [0], None, [256], [0, 256])
        # red_hist = cv.calcHist([b], [0], None, [256], [0, 256])
        # green_hist = cv.calcHist([b], [0], None, [256], [0, 256])

        plt.figure()
        plt.title(filename + ' Multichannel Histogram')
        plt.xlabel('Bins')
        plt.ylabel('Num Pixels')
        colors = ('b', 'g', 'r')

        for i, col in enumerate(colors):
            hist = cv.calcHist([img], [i], None, [256], [0,256])
            plt.plot(hist, color = col)
            plt.xlim([0, 256])
        
        plt.show()

    cv.waitKey(0)