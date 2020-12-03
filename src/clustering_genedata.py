import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


# Filter_data removes those highly correlated feature (above the threshold)
# Use corr_threshold = 0.8 for good results
def filter_data(file_name, corr_threshold=0.8):
    # Read the data set
    data = pd.read_csv(file_name)
    # Drop non-feature columns
    data = data.drop(columns = ['id', 'class'])
    # Correlation matrix
    corr_matrix = data.corr()
    # Corr_features contains the features to be removed
    corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j] > corr_threshold):
                colname = corr_matrix.columns[i]
                corr_features.add(colname)

    data.drop(labels=corr_features, axis=1, inplace=True)
    data.to_csv("genedata_modified.csv", index=False)
    return(data)


def clustering(X, method="kmeans"):        
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
            
        nmi = metrics.normalized_mutual_info_score(class_labels, labels, "geometric")
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
    plt.savefig(method + "_nmi_clusters.png")
    plt.show()

    return(k_optimal_nmi, max_nmi)

#data_modified = pd.read_csv("genedata_modified.csv")
data_modified = filter_data("genedata.csv")
# Data contains the initial dataset, with classes and features
data = pd.read_csv("genedata.csv")
# Extract class labels
class_labels = data['class']
class_labels = np.array(class_labels)
data = data.drop(columns = ['id', 'class'])

##############################################################################
# Uncomment the following line to run the program without performing filtering
#X = np.array(data)
##############################################################################
# Comment the following lines to run the program without performing PCA
X = np.array(data_modified)
##############################################################################



#CLUSTERIZATION
#k-means
num_clusters, nmi = clustering(X, "kmeans")
print("\nNMI kmeans:", nmi)
print("\nNum_clusters:", num_clusters)

#Agglomerative complete
num_clusters, nmi = clustering(X, "agglomerative_complete")
print("\nNMI agg complete:", nmi)
print("\nNum_clusters:", num_clusters)

#Agglomerative average
num_clusters, nmi = clustering(X, "agglomerative_average")
print("\nNMI agg avg:", nmi)
print("\nNum_clusters:", num_clusters)

#Agglomerative single
num_clusters, nmi = clustering(X, "agglomerative_single")
print("\nNMI agg single:", nmi)
print("\nNum_clusters:", num_clusters)

#Agglomerative ward
num_clusters, nmi = clustering(X, "agglomerative_ward")
print("\nNMI agg ward:", nmi)
print("\nNum_clusters:", num_clusters)

