import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import io
import PySimpleGUI as sg
from PIL import Image

# Image class from object based processing
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
    
    # define rescaling function
    def rescale_frame(self, scale=0.5):
        width = int(self.img.shape[1] * scale)
        height = int(self.img.shape[0] * scale)

        dimensions = (width, height)
        return cv.resize(self.img, dimensions, interpolation=cv.INTER_AREA)
    
    # separate rescaling function that fixes the maximum size to 700x700
    def rescale_fixed(self):
        width = int(self.img.shape[1])
        height = int(self.img.shape[0])

        larger_dimension = max(width, height)

        rescaled_width = int(width * (700 / larger_dimension))
        rescaled_height = int(height * (700 / larger_dimension))

        dimensions = (rescaled_width, rescaled_height)
        return cv.resize(self.img, dimensions, interpolation=cv.INTER_AREA)

    # Setters
    def set_mask(self):
        # Using inRange method, to create a mask
        self.mask = cv.inRange(self.img, self.low, self.high)

    def set_isolate(self):
        self.isolate = cv.bitwise_and(self.img, self.img, mask=self.mask)

    # Getters
    def get_mask(self):
        return self.mask
    
    def get_img(self):
        return self.img
    
    def get_isolate(self):
        return self.isolate
    
    def get_g(self):
        return self.G
    
    def get_s(self):
        return self.S
    
    def get_mod(self):
        return self.Mod
    
    def get_ph(self):
        return self.Ph
    
    def get_low(self):
        return self.low
    
    def get_high(self):
        return self.high

# GUI wrapper for viewing phasors

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("TIFF (*.tif)", "*.tif"),
              ("All files (*.*)", "*.*")]

def main():
    lowSliders = [
        [   
            # Hue Slider
            sg.Radio("Hue", "Radio", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,179),
                90,
                1,
                orientation='horizontal',
                size=(60,15),
                key="-LO HUE SLIDER-"      
            )
        ],
        [
            # Low Sat slider
            sg.Radio("Sat", "Radio", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                90,
                1,
                orientation='horizontal',
                size=(60,15),
                key="-LO SAT SLIDER-" 
            )
        ],
        [
            # Low Value Slider
            sg.Radio("Val", "Radio", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                90,
                1,
                orientation='horizontal',
                size=(60,15),
                key="-LO VAL SLIDER-" 
            )
        ]
    ]

    highSliders = [
        [
            # High Hue slider
            sg.Radio("Hue", "Radio", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,179),
                179,
                1,
                orientation='horizontal',
                size=(60,15),
                key="-HI HUE SLIDER-"      
            )
        ],
        [
            # High Sat slider
            sg.Radio("Sat", "Radio", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                255,
                1,
                orientation='horizontal',
                size=(60,15),
                key="-HI SAT SLIDER-" 
            )
        ],
        [
            # High Value Slider
            sg.Radio("Val", "Radio", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                255,
                1,
                orientation='horizontal',
                size=(60,15),
                key="-HI VAL SLIDER-" 
            )
        ]
    ]

    blurSlider = [
        [
            # Blur Slider
            sg.Radio("Blur", "Radio", size=(10, 1), key="-BLUR-"),
            sg.Slider(
                (3,7),
                3,
                2,
                orientation='horizontal',
                size=(20,15),
                key="-BLUR SLIDER-"      
            )
        ]
    ]

    file_list_column = [
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), enable_events=True, key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image")
        ]
    ]

    layout = [
        [
            # Image viewer
            sg.Image(key="-IMAGE-")
        ],
        [
            sg.Column(file_list_column)
        ],
        [
            sg.Column(blurSlider)
        ],
        [   sg.Frame("Lows", lowSliders),
            sg.VSeperator(),
            sg.Frame("Highs", highSliders)
        ]
    ]

    window = sg.Window("Phasor Viewer", layout, location=(800, 400))

    filename = ''
    HSV_flag = False

    while True:
        if filename != '':
            frame = cv.imread(filename=filename)

        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == "Load Image":
            filename = values["-FILE-"]

        if values["-BLUR-"]:
            try:
                if HSV_flag is True:
                    frame = cv.medianBlur(frame, int(values["-BLUR SLIDER-"]))
                else:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    frame = cv.medianBlur(frame, int(values["-BLUR SLIDER-"]))
                    HSV_flag = True
            except UnboundLocalError:
                print('ERROR: Frame not defined prior to blur manipulation')

        if values["-HSV-"]:
            try:
                if HSV_flag is True:
                    image_frame = Image(frame, 
                                    low=np.array([values["-LO HUE SLIDER-"], values["-LO SAT SLIDER-"], values["-LO VAL SLIDER-"]]),
                                    high=np.array([values["-HI HUE SLIDER-"], values["-HI SAT SLIDER-"], values["-HI VAL SLIDER-"]]))
                    image_frame.set_mask()
                    image_frame.set_isolate()
                    frame = image_frame.get_isolate()
                else:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    image_frame = Image(frame, 
                                    low=np.array([values["-LO HUE SLIDER-"], values["-LO SAT SLIDER-"], values["-LO VAL SLIDER-"]]),
                                    high=np.array([values["-HI HUE SLIDER-"], values["-HI SAT SLIDER-"], values["-HI VAL SLIDER-"]]))
                    image_frame.set_mask()
                    image_frame.set_isolate()
                    frame = image_frame.get_isolate()
                    HSV_flag = True
                
            except UnboundLocalError:
                print('ERROR: Frame not defined prior to HSV manipulation')

        if filename != '':
            try:
                show_frame = Image(frame)
                show_frame = show_frame.rescale_fixed()
                imencode = cv.imencode(".png", show_frame)[1]
                imgbytes = np.array(imencode).tobytes()
                window["-IMAGE-"].update(data=imgbytes)
            except UnboundLocalError:
                print("ERROR: Something wrong while viewing")

    window.close()

if __name__ == '__main__':
    main()