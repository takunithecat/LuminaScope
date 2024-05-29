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
import statistics
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score 
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import scipy
from scipy import stats

class ObjectPhasor:
    def __init__(self, G, S, Ph, img):
        # G and S are both arrays
        self.img = img
        self.G = G
        self.S = S
        self.Ph = Ph

    def get_g(self):
        return self.G
    
    def get_s(self):
        return self.S
    
    def get_ph(self):
        return self.Ph
    
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

def isolate(img, low, high):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img, low, high)
    isolate = cv.bitwise_and(img, img, mask=mask)
    mask = background_subtraction(isolate)
    img = cv.cvtColor(isolate, cv.COLOR_HSV2BGR)
    return img

def watershed_object(image):
    # perform watershed on image and return individual objects in a list

    list_masks = []

    gray=np.sum(image, axis=2)
    Mgray=gray*(gray>40)

    distance = ndi.distance_transform_edt(Mgray)
    coords = peak_local_max(distance, min_distance=16)
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

        # cv.imshow(f'Object #{label}', masked)
        # # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
        # cv.waitKey(500)
        # cv.destroyAllWindows()

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

def kmeans_points(dir):
    directory = dir

    # init list of object phasors
    phasor_list = []

    # Iterate files in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        if os.path.isfile(f):
            image=cv.imread(filename=f)
            image = isolate(image, low=np.array([0,0,70]), high=np.array([179,255,255]))
            image=cv.cvtColor(image, cv.COLOR_BGR2RGB) 
            image = cv.medianBlur(image, 5)

            objects = watershed_object(image)

            for obj in objects:
                # G and S are 2D Arrays of shape (image dim, 3)
                G, S, Ph, Mod, I = phasors(obj, axis=2)
                temp = ObjectPhasor(G, S, Ph, obj)
                phasor_list.append(temp)
    
    data = []
    for i in range(len(phasor_list)):
        temp = [phasor_list[i].get_g(), phasor_list[i].get_s(), phasor_list[i].get_ph(), i, phasor_list[i]]
        data.append(temp)
        
    export_df = pd.DataFrame(data, columns = ['G', 'S', 'Ph', 'ObjNum', 'ObjRef'])

    # Instead of having all the phasors as a list to plot, we combine to have just one point - the medians of G and S, which helps limit outliers
    export_df['G'] = export_df.apply(lambda x: statistics.median([i for i in x['G'].flatten() if i != 0]), axis=1)
    export_df['S_IQR'] = export_df.apply(lambda x: stats.iqr([i for i in x['S'].flatten() if i != 0]), axis=1)
    export_df['S_MAD'] = export_df.apply(lambda x: stats.median_abs_deviation([i for i in x['S'].flatten() if i != 0]), axis=1)
    export_df['S'] = export_df.apply(lambda x: statistics.median([i for i in x['S'].flatten() if i != 0]), axis=1)
    export_df['Ph'] = export_df.apply(lambda x: statistics.median([i for i in x['Ph'].flatten() if i != 0]), axis=1)

    # Flatten all the X, Y, Z and make them into array shape (dim, 3)
    X = export_df['G'].to_list()
    Y = export_df['S'].to_list()
    Z = export_df['ObjNum'].to_list()

    df = pd.DataFrame({"X" : X,
                       "Y" : Y,
                       "Z" : Z})

    dataset = df.to_numpy()
    xy_sample = df[["X", "Y"]]

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(xy_sample)
    labels = kmeans.predict(xy_sample)
    
    df['Labels'] = labels
    df = (df.groupby('Z')['Labels'].value_counts()
         .rename('counts').reset_index()
         .drop_duplicates('Z'))

    export_df = export_df.join(df, how='inner',lsuffix='ObjNum', rsuffix='Z')
    export_df = export_df[['G', 'S', 'S_IQR', 'S_MAD', 'Ph', 'ObjNum', 'Labels', 'ObjRef']]

    # Plot the data points and their cluster assignments

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=labels, cmap='viridis')

    # # Set light blue background 
    # ax.xaxis.set_pane_color((0.8, 0.8, 1.0, 1.0)) 
    # ax.yaxis.set_pane_color((0.8, 0.8, 1.0, 1.0)) 
    # ax.zaxis.set_pane_color((0.8, 0.8, 1.0, 1.0))
    # ax.set_title("K-means Clustering on Phasors")
    # ax.set_xlabel("G Value")
    # ax.set_ylabel("S Value")
    # ax.set_zlabel("Object Number")
    # fig.show()
    mb = export_df[(export_df['Labels'] == 0)]['S'].to_numpy()
    mcf = export_df[(export_df['Labels'] == 1)]['S'].to_numpy()

    fig, axs = plt.subplots(1, 2)
    axs[0].boxplot(mb)
    axs[0].set_title('MB231 Phasor Distribution')
    axs[0].set_ylabel("S Value")
    axs[0].set_xticklabels([''],
                    rotation=45, fontsize=8)

    axs[1].boxplot(mcf)
    axs[1].set_title('MCF10a Phasor Distribution')
    axs[1].set_xticklabels([''],
                    rotation=45, fontsize=8)

    fig, ax2 = plt.subplots()
    ax2.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='viridis')
    # Set light blue background 
    ax2.set_title("K-means Clustering on Phasors")
    ax2.set_xlabel("G Value")
    ax2.set_ylabel("S Value")

    xticks = ['MB231', 'MCF10a']
    fig, ax = plt.subplots()
    ax.boxplot([mb, mcf])
    ax.set_title('Distribution of Phasor S Values')
    ax.set_xticklabels(xticks,
                    rotation=45, fontsize=8)
    return export_df

def user_grouping(df, dir, mode=0):
    # input a dataframe to automatically determine hard limits based on S
    # no input dataframe means the user needs to hard code a value

    if mode == 0:
        directory = dir

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
                    temp = ObjectPhasor(G, S, Ph, obj)
                    phasor_list.append(temp)

        data = []

        for i in range(len(phasor_list)):
            temp = [phasor_list[i].get_g(), phasor_list[i].get_s(), phasor_list[i].get_ph(), i, phasor_list[i].get_img()]
            data.append(temp)

        export_df = pd.DataFrame(data, columns = ['G', 'S', 'Ph', 'ObjNum', 'ObjRef']) 

        # Instead of having all the phasors as a list to plot, we combine to have just one point - the medians of G and S, which helps limit outliers
        export_df['G'] = export_df.apply(lambda x: statistics.median([i for i in x['G'].flatten() if i != 0]), axis=1)
        export_df['S_IQR'] = export_df.apply(lambda x: stats.iqr([i for i in x['S'].flatten() if i != 0]), axis=1)
        export_df['S_MAD'] = export_df.apply(lambda x: stats.median_abs_deviation([i for i in x['S'].flatten() if i != 0]), axis=1)
        export_df['S'] = export_df.apply(lambda x: statistics.median([i for i in x['S'].flatten() if i != 0]), axis=1)
        export_df['Ph'] = export_df.apply(lambda x: statistics.median([i for i in x['Ph'].flatten() if i != 0]), axis=1)

        export_df = export_df[['G', 'S', 'S_IQR', 'S_MAD', 'Ph', 'ObjNum', 'Labels', 'ObjRef']]

        # Pick one: threshold on S or G or phase
        # Thresholding on phase is probably the best though
        export_df['Labels'] = export_df.apply(lambda x: 1 if x['S'] >= -0.5 else 0, axis=1)
        # export_df['Labels'] = export_df.apply(lambda x: 1 if x['S'] >= -0.5 else 0, axis=1)
        
        return export_df
    
    else:
        export_df = df[['G', 'S', 'S_IQR', 'S_MAD', 'Ph', 'ObjNum', 'ObjRef']]

        medians = df.groupby('Labels')['S'].median()
        stddev = df.groupby('Labels')['S'].std()

        maxside = max(medians[0], medians[1])

        if maxside == medians[0]:
            top = medians[0] - stddev[0]
            bottom = medians[1] + stddev[1]
        else:
            top = medians[0] + stddev[0]
            bottom = medians[1] - stddev[1]
        
        limit = statistics.mean([top, bottom])

        # Pick one: threshold on S or G or phase, have to change above medians to match whatever picked
        # Thresholding on phase is probably the best though
        # export_df['Labels'] = export_df.apply(lambda x: 1 if x['Ph'] >= limit else 0, axis=1)
        export_df['Labels'] = export_df.apply(lambda x: 1 if x['S'] >= limit else 0, axis=1)

        return export_df

def manual_grouping(dir1, dir2):
    # dir1 and dir2 should both be strings
    directory1 = dir1
    directory2 = dir2

    # init list of object phasors
    phasor_list = []

    # Iterate files in directory
    for filename in os.listdir(directory1):
        f = os.path.join(directory1, filename)

        # checking if it is a file
        if os.path.isfile(f):
            image=cv.imread(filename=f)
            image = isolate(image, low=np.array([0,0,70]), high=np.array([179,255,255]))
            image=cv.cvtColor(image, cv.COLOR_BGR2RGB) 
            image = cv.medianBlur(image, 5)

            objects = watershed_object(image)

            for obj in objects:
                # G and S are 2D Arrays of shape (image dim, 3)
                G, S, Ph, Mod, I = phasors(obj, axis=2)
                temp = ObjectPhasor(G, S, Ph, obj)
                phasor_list.append(temp)
    
    data = []

    for i in range(len(phasor_list)):
        temp = [phasor_list[i].get_g(), phasor_list[i].get_s(), phasor_list[i].get_ph(), i, phasor_list[i], directory1]
        data.append(temp)
        
    export_df1 = pd.DataFrame(data, columns = ['G', 'S', 'Ph', 'ObjNum', 'ObjRef', 'Labels'])

    # second pass for second dir
    # init list of object phasors
    phasor_list = []

    # Iterate files in directory
    for filename in os.listdir(directory2):
        f = os.path.join(directory2, filename)

        # checking if it is a file
        if os.path.isfile(f):
            image=cv.imread(filename=f)
            image = isolate(image, low=np.array([0,0,70]), high=np.array([179,255,255]))
            image=cv.cvtColor(image, cv.COLOR_BGR2RGB) 
            image = cv.medianBlur(image, 5)

            objects = watershed_object(image)

            for obj in objects:
                # G and S are 2D Arrays of shape (image dim, 3)
                G, S, Ph, Mod, I = phasors(obj, axis=2)
                temp = ObjectPhasor(G, S, Ph, obj)
                phasor_list.append(temp)
    
    data = []
    
    for i in range(len(phasor_list)):
        temp = [phasor_list[i].get_g(), phasor_list[i].get_s(), phasor_list[i].get_ph(), i, phasor_list[i], directory2]
        data.append(temp)
        
    export_df2 = pd.DataFrame(data, columns = ['G', 'S', 'Ph', 'ObjNum', 'ObjRef', 'Labels'])

    export_df = pd.concat([export_df1, export_df2], ignore_index=True)

    # Instead of having all the phasors as a list to plot, we combine to have just one point - the medians of G and S, which helps limit outliers
    export_df['G'] = export_df.apply(lambda x: statistics.median([i for i in x['G'].flatten() if i != 0]), axis=1)
    export_df['S_IQR'] = export_df.apply(lambda x: stats.iqr([i for i in x['S'].flatten() if i != 0]), axis=1)
    export_df['S_MAD'] = export_df.apply(lambda x: stats.median_abs_deviation([i for i in x['S'].flatten() if i != 0]), axis=1)
    export_df['S'] = export_df.apply(lambda x: statistics.median([i for i in x['S'].flatten() if i != 0]), axis=1)
    export_df['Ph'] = export_df.apply(lambda x: statistics.median([i for i in x['Ph'].flatten() if i != 0]), axis=1)

    export_df = export_df[['G', 'S', 'S_IQR', 'S_MAD', 'Ph', 'ObjNum', 'Labels', 'ObjRef']]

    mb = export_df[(export_df['Labels'] == directory1)]['S'].to_numpy()
    mcf = export_df[(export_df['Labels'] == directory2)]['S'].to_numpy()
    labels = export_df['Labels'].tolist()

    for n in range (len(labels)):
        if labels[n] == directory1:
            labels[n] = 0
        else:
            labels[n] = 1

    # fig, axs = plt.subplots(1, 2)
    # axs[0].boxplot(mb)
    # axs[0].set_title(f'{directory1} Phasor Distribution')
    # axs[0].set_ylabel("S Value")
    # axs[0].set_xticklabels([''],
    #                 rotation=45, fontsize=8)

    # axs[1].boxplot(mcf)
    # axs[1].set_title(f'{directory2} Phasor Distribution')
    # axs[1].set_xticklabels([''],
    #                 rotation=45, fontsize=8)

    # fig, ax2 = plt.subplots()
    # ax2.scatter(export_df['G'], export_df['S'], c=labels, cmap='viridis')
    # # Set light blue background 
    # ax2.set_title("K-means Clustering on Phasors")
    # ax2.set_xlabel("G Value")
    # ax2.set_ylabel("S Value")

    # xticks = [f'{directory1}', f'{directory2}']
    # fig, ax = plt.subplots()
    # ax.boxplot([mb, mcf])
    # ax.set_title('Distribution of Phasor S Values')
    # ax.set_xticklabels(xticks,
    #                 rotation=45, fontsize=8)
    
    # plt.show()

    return export_df

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

def classify_phasors(df):
    # classify into color based on G and S arrays

    X = df.drop(['Labels', 'ObjNum', 'ObjRef'], axis=1)
    y = df['Labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    rf = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    feature_scores = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_scores)

    # y_pred_prob = rf.predict_proba(X_test)[:, 1] 
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    
    # # Compute the ROC AUC score 
    # roc_auc = roc_auc_score(y_test, y_pred_prob) 
        
    # # Plot the ROC curve 
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    # ax.set_title('ROC Curve')
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.legend(loc="lower right")
    # plt.show()

    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    # # roc curve for tpr = fpr  
    # plt.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    # plt.xlabel('False Positive Rate') 
    # plt.ylabel('True Positive Rate') 
    # plt.title('ROC Curve') 
    # plt.legend(loc="lower right") 
    # plt.show()

    y_pred = rf.predict(X_test)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # View the classification report for test data and predictions
    print(classification_report(y_test, y_pred))

    return rf

def test_accuracy(kmeans_df, user_df):
    kmeans_dict = kmeans_df['Labels'].to_dict()
    user_dict = user_df['Labels'].to_dict()
    accuracy = []
    
    user_labels = []
    for key in kmeans_dict:
        user_labels.append(user_dict[key])
        if kmeans_dict[key] == user_dict[key]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    
    score = sum(accuracy) / len(kmeans_dict) * 100

    # in the case that the 0 and 1 labels are switched
    if score < 50:
        score = 100 - score

    print(score)
    # Create the confusion matrix
    # cm2 = confusion_matrix(np.array(user_labels), np.array(kmeans_df['Labels']))

    # ConfusionMatrixDisplay(confusion_matrix=cm2).plot()
    return score

def test_normality(df):
    t_stats = []
    p_values = []
    variance = []

    groups = df.groupby("Labels")

    for name, group in groups:
        print(name)
        Svals = group['S'].to_numpy()

        if len(Svals) < 50:
            tstat, pval = stats.shapiro(Svals)
            t_stats.append(tstat)
            p_values.append(pval)

            var = np.var(Svals)
            variance.append(var)

        else:
            tstat, pval = stats.kstest(Svals, stats.norm.cdf)
            t_stats.append(tstat)
            p_values.append(pval)

            var = np.var(Svals)
            variance.append(var)

    return t_stats, p_values, variance

def test_homogeneity(df, mode=0):
    # mode 0 is normal data
    # mode 1 is non normal data
    # there should only be 2 tags in df
    if mode == 0:
        # assume normality, continuous, randomly sampled, similar variance
        # two tailed
        # two sample
        # unpaired
        groupnames = df.groupby("Labels")  
        groupnames = list(groupnames.groups.keys())

        df_A = df[df['Labels'] == groupnames[0]]
        df_B = df[df['Labels'] == groupnames[1]]

        df_A = df_A.sample(n = 30, random_state=42)
        df_B = df_B.sample(n = 30, random_state=42)

        a = df_A['S'].to_numpy()
        b = df_B['S'].to_numpy()

        stat, p , deg = stats.ttest_ind(a, b)

    else:
        # one independent and dependent variable
        # independence of observations
        # If distributions different shape, differences in distribution
        # IF distributions same shape, differences in median
        groupnames = df.groupby("Labels")  
        groupnames = list(groupnames.groups.keys())

        df_A = df[df['Labels'] == groupnames[0]]
        df_B = df[df['Labels'] == groupnames[1]]

        df_A = df_A.sample(n = 40, random_state=42)
        df_B = df_B.sample(n = 40, random_state=42)

        a = df_A['S'].to_numpy()
        b = df_B['S'].to_numpy()

        stat, p = stats.mannwhitneyu(a, b)
        
    return stat, p

def make_histograms(df):
    groupnames = df.groupby("Labels")  
    groupnames = list(groupnames.groups.keys())

    df_A = df[df['Labels'] == groupnames[0]]['S'].to_numpy()
    df_B = df[df['Labels'] == groupnames[1]]['S'].to_numpy()

    bins = np.linspace(0, 0.5, 50)

    plt.hist(df_A, bins, alpha=0.5, label=f'{groupnames[0]}')
    plt.hist(df_B, bins, alpha=0.5, label=f'{groupnames[1]}')
    plt.title('Distribution of S Phasors for Cell Samples')
    plt.xlabel('Phasor Value')
    plt.legend(loc='upper right')
    plt.show()


def main():
    df = manual_grouping('MB231', 'MCF10aSet5')
    t_stats, p_values, variance = test_normality(df)
    print(t_stats)
    print(p_values)
    print(variance)
    t_stat, p = test_homogeneity(df, 1)
    print(t_stat)
    print(p)
    make_histograms(df)
    pred = classify_phasors(df)

if __name__ == '__main__':
    main()

