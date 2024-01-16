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

# initialize class for image isolation and organized phasor plot generation
# TODO: implement setters and getters
class Image():
    def __init__(self, img, low=None, high=None):
        self.img = img
        self.G = None
        self.S = None
        self.Mod = None
        self.Ph = None
        self.low = low
        self.high = high

    # phasor function calculate values
    def phasor(self):
        fft=np.fft.fft(self.img, axis=2)
    
        G=fft[:,:,1].real/fft[:,:,0].real
        G=np.nan_to_num(G, nan=0.0)
    
        S=fft[:,:,1].imag/fft[:,:,0].real
        S=np.nan_to_num(S, nan=0.0)
    
        Ph=np.arctan2(S[:,:], G[:,:])+np.pi
        Ph=np.nan_to_num(Ph, nan=0.0)
    
        Mod=np.sqrt(G**2+S**2)
        Mod=np.nan_to_num(Mod, nan=0.0)
    
        self.G = G
        self.S = S
        self.Ph = Ph
        self.Mod = Mod
    
    # plots each value separately
    def plot_phasors(self):
        plt.figure()
        plt.imshow(self.Mod)
        # plt.axis('off')
        plt.title('Mod')
        plt.colorbar()

        plt.figure()
        plt.imshow(self.Ph)
        # plt.axis('off')
        plt.title('Phase')
        plt.colorbar()

        plt.figure()
        plt.imshow(self.S)
        # plt.axis('off')
        plt.title('S')
        plt.colorbar()

        plt.figure()
        plt.imshow(self.G)
        # plt.axis('off')
        plt.title('G')
        plt.colorbar()

        plt.show()
    
    # define rescaling function
    def rescaleFrame(self, scale=0.5):
        width = int(self.img.shape[1] * scale)
        height = int(self.img.shape[0] * scale)

        dimensions = (width, height)
        return cv.resize(self.img, dimensions, interpolation=cv.INTER_AREA)
    
    def setmask(self):
        # Using inRange method, to create a mask
        self.mask = cv.inRange(self.img, self.low, self.high)

    def isolate(self):
        self.isolate = cv.bitwise_and(self.img, self.img, mask=self.mask)

def main():
    # assign directory
    directory = 'Images'

    # init image into class
    img = cv.imread('Images/image0.jpg')
    img = Image(img)

    # rescale image down so it's not covering the whole screen
    rescaled = img.rescaleFrame(0.25)

    # cropping step was manual to determine location
    cropped = rescaled[130:700, 90:500]

    # blurring step
    median = cv.medianBlur(cropped, 3)
    gauss = cv.GaussianBlur(cropped, (3,3), 0)

    # Convert BGR to HSV
    img_hsv = cv.cvtColor(median, cv.COLOR_BGR2HSV)
    
    # Remember that in HSV space, Hue is color from 0..180. Red 320-360, and 0 - 30.
    # We keep Saturation and Value within a wide range but note not to go too low or we start getting black/gray
    # set HSV lows and highs for each color
    red650 = Image(img_hsv, low=np.array([0,0,0]), high=np.array([10,255,255]))
    red625 = Image(img_hsv, low=, high=)
    red610 = Image(img_hsv, low=, high=)
    yellow600 = Image(img_hsv, low=, high=)
    yellow580 = Image(img_hsv, low=, high=)
    yellow575 = Image(img_hsv, low=, high=)
    green550 = Image(img_hsv, low=, high=)
    green540 = Image(img_hsv, low=, high=)
    green525 = Image(img_hsv, low=, high=)
    green500 = Image(img_hsv, low=, high=)
    blue475 = Image(img_hsv, low=, high=)
    blue450 = Image(img_hsv, low=, high=)
    blue440 = Image(img_hsv, low=, high=)
    blue430 = Image(img_hsv, low=, high=)
    blue420 = Image(img_hsv, low=, high=)
    violet410 = Image(img_hsv, low=, high=)
    violet400 = Image(img_hsv, low=, high=)

    # masks
    red650.setmask()
    red625.setmask()
    red610.setmask()
    yellow600.setmask()
    yellow580.setmask()
    yellow575.setmask()
    green550.setmask()
    green540.setmask()
    green525.setmask()
    green500.setmask()
    blue475.setmask()
    blue450.setmask()
    blue440.setmask()
    blue430.setmask()
    blue420.setmask()
    violet410.setmask()
    violet400.setmask()

    cv.imshow('isolate', forest)
    # Show image
    # syntax: Picture name, variable
    cv.imshow('Median', median)
    # cv.imshow('Gaussian', gauss)

    cv.waitKey(0)

if __name__ == '__main__':
    main()