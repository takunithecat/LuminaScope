import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 as cv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import seaborn as sns
import pandas as pd
import os

class ObjectPhasor:
    def __init__(self, G, S, img):
        # G and S are both arrays
        self.img = img
        self.G = G
        self.S = S

    def get_g(self):
        return self.G
    
    def get_s(self):
        return self.S
    
def phasors(img, axis):
    fft=np.fft.fft(img, axis=axis)
    
    G=fft[:,:,1].real/fft[:,:,0].real
    G=np.nan_to_num(G, nan=0.0)

    S=fft[:,:,1].imag/fft[:,:,0].real
    S=np.nan_to_num(S, nan=0.0)

    Ph=np.arctan2(S[:,:], G[:,:])+np.pi
    Ph=np.nan_to_num(Ph, nan=0.0)

    Mod=np.sqrt(G**2+S**2)
    Mod=np.nan_to_num(Mod, nan=0.0)
    
    I=fft[0].real
    return G, S, Ph, Mod, I

def background_subtraction(src):
    # Background Subtraction
    # Little too strong for image at low exposure
    dest = cv.ximgproc.anisotropicDiffusion(src, alpha=0.1, K=10, niters=300)
    src = cv.subtract(src, dest)

    shifted = cv.pyrMeanShiftFiltering(src, 21, 51)
    return shifted 

def watershed_object(image):
    # perform watershed on image and return individual objects in a list

    list_masks = []

    gray=np.sum(image, axis=2)
    Mgray=gray*(gray>40)

    distance = ndi.distance_transform_edt(Mgray)
    coords = peak_local_max(distance, min_distance=8)
    print(len(coords))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=Mgray)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(Mgray.shape, dtype='uint8')
        mask[labels == label] = 255

        masked = cv.bitwise_and(image, image, mask=mask)

        list_masks.append(masked)
    
    return list_masks

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# kmeans loop
# For image in folder, watershed and extract images
# For watershed image, calculate phasor transform, calculate average color? and save G S coordinate
# For each set of G S coordinate, format into array and plot as 3D
# Perform Kmeans clustering on set of G S coordinate with make dummies and tag
# plot kmeans in 3d

def kmeans_points():
    directory = 'Microbeads'

    # init list of object phasors
    phasor_list = []

    # Iterate files in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        if os.path.isfile(f):
            image=cv.imread(filename=f)
            image=cv.cvtColor(image, cv.COLOR_BGR2RGB) 
            image = cv.medianBlur(image, 5)

            objects = watershed_object(image)

            for obj in objects:
                # G and S are 2D Arrays of shape (image dim, 3)
                G, S, Ph, Mod, I = phasors(obj, axis=2)
                temp = ObjectPhasor(G, S, obj)
                phasor_list.append(temp)
            
    # Flatten all the X, Y, Z and make them into array shape (dim, 3)
    X = []
    Y = []
    Z = []

    for i in range(len(phasor_list)):
        x = phasor_list[i].get_g().flatten()
        y = phasor_list[i].get_s().flatten()
        shape = x.shape
        z = np.full(shape=shape, fill_value=i)

        X.extend(x)
        Y.extend(y)
        Z.extend(z)

    df = pd.DataFrame({"X" : X,
                       "Y" : Y,
                       "Z" : Z})
    
    df = df[(df['X'] != 0) & (df['Y'] != 0)] 

    dataset = df.to_numpy()
    xy_sample = df[["X", "Y"]]

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(xy_sample)
    labels = kmeans.predict(xy_sample)

    # Plot the data points and their cluster assignments
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=labels, cmap='viridis')

    # Set light blue background 
    ax.xaxis.set_pane_color((0.8, 0.8, 1.0, 1.0)) 
    ax.yaxis.set_pane_color((0.8, 0.8, 1.0, 1.0)) 
    ax.zaxis.set_pane_color((0.8, 0.8, 1.0, 1.0))
    ax.set_title("K-means Clustering on Phasors")
    ax.set_xlabel("G Value")
    ax.set_ylabel("S Value")
    ax.set_zlabel("Object Number")
    plt.show()

def kmeans_colors(img, n):
    image = img
    number_of_colors = n

    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    plt.title('Colors Detection', fontsize=20)
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    plt.show()
    return hex_colors, rgb_colors

def color_computing(rgb_colors):
    DIFF = []
    # init list of object phasors
    phasor_list = []

    directory = 'Microbeads'

    # Iterate files in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        if os.path.isfile(f):
            image=cv.imread(filename=f)
            image=cv.cvtColor(image, cv.COLOR_BGR2RGB) 
            image = cv.medianBlur(image, 5)

            objects = watershed_object(image)

            for obj in objects:
                # G and S are 2D Arrays of shape (image dim, 3)
                G, S, Ph, Mod, I = phasors(obj, axis=2)
                temp = ObjectPhasor(G, S, obj)
                phasor_list.append(temp)
    
    for phasor_object in phasor_list:
        DIFF_COLOR = []
        for color in range(len(rgb_colors)):
            # there is took much black space in the image that this apprach is broken
            diff = np.abs(phasor_object.img - rgb_colors[color])
            DIFF_COLOR.append(diff.mean())
        DIFF.append(DIFF_COLOR)

    return np.array(DIFF)

# classification loop
def main():
    # read in our best quality image to get the color codes
    image = cv.imread('Microbeads/10x RGB Fluorescence Set 4 4-5-24.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    hex_colors, rgb_colors = kmeans_colors(img=image, n=3)

    for i in range(len(rgb_colors)):
        rgb_colors[i] = rgb_colors[i].astype(int)

    results = color_computing(rgb_colors=rgb_colors)

    cols = ['Particle Number'] + hex_colors
    sorted_results = pd.DataFrame(columns=cols)
    k=0

    for r in results:
        d = {'Particle Number': [int(k)]}
        for c in range(len(hex_colors)):
            d[hex_colors[c]] = r[c]*100/r.sum()

        d = pd.DataFrame.from_dict(d)
        d = d.drop_duplicates(hex_colors)
        print(d)
        sorted_results = pd.concat([sorted_results, d], ignore_index=True)
        k=k+1

    sorted_results['Particle Number'] = sorted_results['Particle Number'].astype(int)
    
    sorted_results.head()

if __name__ == '__main__':
    main()

