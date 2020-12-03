import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics


def clustering(X, method="kmeans", filt=""):
    max_nmi = 0
    k_optimal_nmi = 0
    x_nmi = []
    y_nmi = []
    for i in range(2, 20):
        if method=="kmeans":
            labels = KMeans(n_clusters=i, random_state=0).fit(X).labels_
            color_plot = "red"
        elif method=="agglomerative_complete":
            labels = AgglomerativeClustering(n_clusters=i, linkage="complete").fit(X).labels_
            color_plot = "blue"
        elif method=="agglomerative_average":
            labels = AgglomerativeClustering(n_clusters=i, linkage="average").fit(X).labels_
            color_plot = "cyan"
        elif method=="agglomerative_single":
            labels = AgglomerativeClustering(n_clusters=i, linkage="single").fit(X).labels_
            color_plot = "green"
        elif method=="agglomerative_ward":
            labels = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(X).labels_
            color_plot = "magenta"

        nmi = metrics.normalized_mutual_info_score(class_labels, labels, average_method="geometric")
        x_nmi.append(i)
        y_nmi.append(nmi)
        if nmi > max_nmi:
            max_nmi = nmi
            k_optimal_nmi = i
    plt.clf()
    plt.title(method)
    plt.xlabel("num clusters")
    plt.ylabel("nmi")
    plt.plot(x_nmi, y_nmi, color_plot)
    plt.savefig("../plots/" + filt + method + "_nmi_clusters_msdata.png")
    #plt.show()

    return(k_optimal_nmi, max_nmi)

def compute_PCA(data_set, num_components=2):
    x = data_set.values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=num_components)
    principalComponents = pca.fit_transform(x)

    return(pd.DataFrame(principalComponents))



# Data contains the initial dataset, with classes and features
data = pd.read_csv("../data/msdata.csv")


# Extract class labels
class_labels = data['class']
class_labels = np.array(class_labels)
data = data.drop(columns = ['id', 'class'])



X = data
filt = ""
#######################################################################
# Comment the following lines to run the program without performing PCA
X = compute_PCA(X)
filt = "filtered_"
#######################################################################
X = np.array(X)



#CLUSTERIZATION
#k-means
num_clusters, nmi = clustering(X, "kmeans", filt)
print("NMI kmeans:", nmi)
print("Num_clusters:", num_clusters)
print()

#Agglomerative complete
num_clusters, nmi = clustering(X, "agglomerative_complete", filt)
print("NMI agg complete:", nmi)
print("Num_clusters:", num_clusters)
print()

#Agglomerative average
num_clusters, nmi = clustering(X, "agglomerative_average", filt)
print("NMI agg avg:", nmi)
print("Num_clusters:", num_clusters)
print()

#Agglomerative single
num_clusters, nmi = clustering(X, "agglomerative_single", filt)
print("NMI agg single:", nmi)
print("Num_clusters:", num_clusters)
print()

#Agglomerative ward
num_clusters, nmi = clustering(X, "agglomerative_ward", filt)
print("NMI agg ward:", nmi)
print("Num_clusters:", num_clusters)
