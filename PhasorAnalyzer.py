import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 as cv
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
import os
import seaborn as sns

class ObjectPhasor:
    def __init__(self, G, S):
        # G and S are both arrays
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


# Main loop
# For image in folder, watershed and extract images
# For watershed image, calculate phasor transform, calculate average color? and save G S coordinate
# For each set of G S coordinate, format into array and plot as 3D
# Perform Kmeans clustering on set of G S coordinate with make dummies and tag
# plot kmeans in 3d

def test():
    phasor_list = []

    image=cv.imread('Microbeads/10x RGB Fluorescence Set 1 4-5-24.png')
    objects = watershed_object(image)

    for object in objects:
                # G and S are 2D Arrays of shape (image dim, 3)
                G, S, Ph, Mod, I = phasors(object, axis=2)
                temp = ObjectPhasor(G, S)
                phasor_list.append(temp)
    
    for i in range(len(phasor_list)):
        plt.figure()
        # plt.scatter(x = Gval, y = Sval)
        print(phasor_list[i].get_g().flatten())
        sns.histplot(x=phasor_list[i].get_g().flatten(), y=phasor_list[i].get_s().flatten())
        plt.title('G vs S')
        plt.show()

def main():
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
            image = cv.medianBlur(image, 3)

            objects = watershed_object(image)

            for object in objects:
                # G and S are 2D Arrays of shape (image dim, 3)
                G, S, Ph, Mod, I = phasors(object, axis=2)
                temp = ObjectPhasor(G, S)
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

    dataset = np.column_stack((X,Y,Z))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(dataset)

    labels = kmeans.predict(dataset)

    # Plot the data points and their cluster assignments
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=labels, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
            marker='x', color='red', s=100 , linewidths=3)
    # Set light blue background 
    ax.xaxis.set_pane_color((0.8, 0.8, 1.0, 1.0)) 
    ax.yaxis.set_pane_color((0.8, 0.8, 1.0, 1.0)) 
    ax.zaxis.set_pane_color((0.8, 0.8, 1.0, 1.0))
    ax.set_title("K-means Clustering on Swiss Roll Dataset")
    ax.set_xlabel("G Value")
    ax.set_ylabel("S Value")
    ax.set_zlabel("Object Number")
    plt.show()
        
if __name__ == '__main__':
    main()

