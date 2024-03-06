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
        self.isolate = None
        self.mask = None
        self.G = None
        self.S = None
        self.Mod = None
        self.Ph = None
        self.low = low
        self.high = high

    # phasor function calculate values
    def calculate_phasors(self, img=None):
        if img is None:
            img = self.img

        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

        fft=np.fft.fft(img, axis=2)
    
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

    def plot_polar(self, Gval=None, Sval=None):
        if Gval is None:
            Gval = self.G.flatten()
        if Sval is None:
            Sval = self.S.flatten() * 180 / np.pi

        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(Sval, Gval, cmap='hsv', alpha=0.75)
        plt.title('G vs S')
        plt.show()
        

    # define rescaling function
    def rescale_frame(self, scale=0.5):
        width = int(self.img.shape[1] * scale)
        height = int(self.img.shape[0] * scale)

        dimensions = (width, height)
        return cv.resize(self.img, dimensions, interpolation=cv.INTER_AREA)
    
    def set_mask(self):
        # Using inRange method, to create a mask
        self.mask = cv.inRange(self.img, self.low, self.high)

    def set_isolate(self):
        self.isolate = cv.bitwise_and(self.img, self.img, mask=self.mask)

def main():
    # assign directory
    directory = 'Images'

    # init image into class
    img = cv.imread('TestImages/image0.jpg')
    img = Image(img)

    # rescale image down so it's not covering the whole screen
    rescaled = img.rescale_frame(0.25)

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
    red650 = Image(img_hsv, low=np.array([170,0,0]), high=np.array([180,255,255]))
    red625 = Image(img_hsv, low=np.array([0,0,0]), high=np.array([5,255,255]))
    red610 = Image(img_hsv, low=np.array([10,0,0]), high=np.array([15,255,255]))
    yellow600 = Image(img_hsv, low=np.array([15,0,0]), high=np.array([25,255,255]))
    yellow580 = Image(img_hsv, low=np.array([25,0,190]), high=np.array([29,255,255]))
    yellow575 = Image(img_hsv, low=np.array([29,0,0]), high=np.array([45,255,255]))
    green550 = Image(img_hsv, low=np.array([45,0,0]), high=np.array([55,255,255]))
    green540 = Image(img_hsv, low=np.array([55,0,0]), high=np.array([70,90,255]))
    green525 = Image(img_hsv, low=np.array([60,88,0]), high=np.array([70,255,255]))
    green500 = Image(img_hsv, low=np.array([70,0,0]), high=np.array([100,255,255]))
    blue475 = Image(img_hsv, low=np.array([100,0,0]), high=np.array([105,255,255]))
    blue450 = Image(img_hsv, low=np.array([105,0,0]), high=np.array([111,150,150]))
    blue440 = Image(img_hsv, low=np.array([112,0,0]), high=np.array([112,200,200]))
    blue430 = Image(img_hsv, low=np.array([113,0,0]), high=np.array([115,150,150]))
    blue420 = Image(img_hsv, low=np.array([115,0,0]), high=np.array([120,150,150]))
    violet410 = Image(img_hsv, low=np.array([120,0,0]), high=np.array([130,150,150]))
    violet400 = Image(img_hsv, low=np.array([130,0,0]), high=np.array([160,150,150]))

    # masks
    red650.set_mask()
    red625.set_mask()
    red610.set_mask()
    yellow600.set_mask()
    yellow580.set_mask()
    yellow575.set_mask()
    green550.set_mask()
    green540.set_mask()
    green525.set_mask()
    green500.set_mask()
    blue475.set_mask()
    blue450.set_mask()
    blue440.set_mask()
    blue430.set_mask()
    blue420.set_mask()
    violet410.set_mask()
    violet400.set_mask()

    # set isolates
    red650.set_isolate()
    red625.set_isolate()
    red610.set_isolate()
    yellow600.set_isolate()
    yellow580.set_isolate()
    yellow575.set_isolate()
    green550.set_isolate()
    green540.set_isolate()
    green525.set_isolate()
    green500.set_isolate()
    blue475.set_isolate()
    blue450.set_isolate()
    blue440.set_isolate()
    blue430.set_isolate()
    blue420.set_isolate()
    violet410.set_isolate()
    violet400.set_isolate()

    # Show image
    # cv.imshow('red625isolate', violet400.isolate)
    # syntax: Picture name, variable
    # cv.imshow('Median', median)
    # cv.imshow('Gaussian', gauss)

    # Calculate phasors and plot
    red650.calculate_phasors(red650.isolate)
    # red650.plot_phasors()
    red625.calculate_phasors(red625.isolate)
    # red625.plot_phasors()
    red610.calculate_phasors(red610.isolate)
    # red610.plot_phasors()
    yellow600.calculate_phasors(yellow600.isolate)
    # yellow600.plot_phasors()
    yellow580.calculate_phasors(yellow580.isolate)
    # yellow580.plot_phasors()
    yellow575.calculate_phasors(yellow575.isolate)
    # yellow575.plot_phasors()
    green550.calculate_phasors(green550.isolate)
    # green550.plot_phasors()
    green540.calculate_phasors(green540.isolate)
    # green540.plot_phasors()
    green525.calculate_phasors(green525.isolate)
    # green525.plot_phasors()
    green500.calculate_phasors(green500.isolate)
    # green500.plot_phasors()
    blue450.calculate_phasors(blue450.isolate)
    # blue450.plot_phasors()
    blue440.calculate_phasors(blue440.isolate)
    # blue440.plot_phasors()
    blue430.calculate_phasors(blue430.isolate)
    # blue430.plot_phasors()
    blue420.calculate_phasors(blue420.isolate)
    # blue420.plot_phasors()
    violet410.calculate_phasors(violet410.isolate)
    # violet410.plot_phasors()
    violet400.calculate_phasors(violet400.isolate)
    violet400.plot_polar()

    cv.waitKey(0)

if __name__ == '__main__':
    main()