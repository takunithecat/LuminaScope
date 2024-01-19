import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
from calibrationTest import *
import os
import PySimpleGUI as sg

# GUI wrapper for viewing phasors

lowSliders = [
    [   
        # Hue Slider
        sg.Slider(
            (0,179),
            90,
            1,
            orientation='horizontal',
            size=(60,15),
            key="-LO HUE SLIDER"      
        )
    ],
    [
        # Low Sat slider
        sg.Slider(
            (0,255),
            90,
            1,
            orientation='horizontal',
            size=(60,15),
            key="-LO SAT SLIDER" 
        )
    ],
    [
        # Low Value Slider
        sg.Slider(
            (0,255),
            90,
            1,
            orientation='horizontal',
            size=(60,15),
            key="-LO LUM SLIDER" 
        )
    ]
]

highSliders = [
    [
        # High Hue slider
        sg.Slider(
            (0,179),
            90,
            1,
            orientation='horizontal',
            size=(60,15),
            key="-LO HUE SLIDER"      
        )
    ],
    [
        # High Sat slider
        sg.Slider(
            (0,255),
            90,
            1,
            orientation='horizontal',
            size=(60,15),
            key="-LO SAT SLIDER" 
        )
    ],
    [
        # High Value Slider
        sg.Slider(
            (0,255),
            90,
            1,
            orientation='horizontal',
            size=(60,15),
            key="-LO LUM SLIDER" 
        )
    ]
]

layout = [
    [
        # Image viewer
    ],
    [   sg.Column(lowSliders),
        sg.VSeperator(),
        sg.Column(highSliders)
    ]
]

if __name__ == '__main__':
    main()