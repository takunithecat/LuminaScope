import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
from calibrationTest import *
import os
import PySimpleGUI as sg

# GUI wrapper for viewing phasors

# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

if __name__ == '__main__':
    main()