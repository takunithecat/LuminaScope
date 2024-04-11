# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:09:03 2024

@author: fpalomba
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 as cv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def main(Gsmt, axis):
    fft=np.fft.fft(Gsmt, axis=axis)
    
    G=fft[1].real/fft[0].real
    G=np.nan_to_num(G, nan=0.0)
    
    S=-fft[1].imag/fft[0].real
    S=np.nan_to_num(S, nan=0.0)

    Ph=np.arctan2(S, G)+np.pi
    Ph=np.nan_to_num(Ph, nan=0.0)
    
    Mod=np.sqrt(G**2+S**2)
    Mod=np.nan_to_num(Mod, nan=0.0)
    
    I=fft[0].real
    return G, S, Ph, Mod, I



image=cv.imread('C:/Users/fpalomba/Downloads/Test_RGB_BME180.jpg')
image=cv.cvtColor(image, cv.COLOR_BGR2RGB) 

gray=np.sum(image, axis=2)

plt.figure()
plt.imshow(gray)
plt.colorbar()

Mgray=gray*(gray>40)

plt.figure()
plt.imshow(Mgray)

distance = ndi.distance_transform_edt(Mgray)
coords = peak_local_max(distance, min_distance=8)
print(len(coords))
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=Mgray)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, vmin=1, vmax=2)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

for i in range(np.max(labels)):
    mask=labels==i
    