'''
Clustering Data
'''

##load libraries
from typing import List, Dict, Union 
import pandas as pd
import numpy as np
import seaborn as sbn 
from sklearn import decomposition
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

import umap.umap_ as umap
#import hdbscan
import sklearn.cluster as cluster
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
#from jqmcvi import base

### Clustering
class Clustering():
    #def __init__(algorithm: str, **params):
    def __init__(self):
        self.params=None

    def fit(self, encodedX, **params):
        self.model = self.algorithm.fit(encodedX, **params)

    def fit_dbscan(self, encodedX, **params):
        self.model = self.algorithm.fit(encodedX, **params)
        unique, counts = np.unique(self.model.labels_, return_counts=True)
        labels =pd.DataFrame((unique, counts)).T
        labels.columns=['cluster', 'number']
        #labels.rename(columns={0: "a", 1: "c"})
        print(labels)
        fig = plt.figure(figsize = (10, 5))
        labels.bar.plot('cluster','number')
        labels.set_xlabel("DBSCAN Labels")
        plt.show()

    def predict(self, encodedX):
        self.pred = self.model.predict(encodedX)

    def fit_predict(self,encodedX):
        self.pred = self.algorithm.fit_predict(encodedX)

    def score(self, encodedX):
        #evaluate different training models using a score function
        SS= silhouette_score(encodedX, self.pred)
        DBS = davies_bouldin_score(encodedX, self.pred)
        print("SS: ",SS)
        print("DBS: ", DBS)

    def plot_label(self, encodeX, drop_df):
        drop_df['Cluster'] = pd.Series(self.pred, index=encodeX.index)
        
        for c in drop_df:
            grid= sbn.FacetGrid(drop_df, col='Cluster')
            grid.map(plt.hist, c)
        plt.show() 
        
        print(encodeX)

    def PCA_cluster(self, encodedX):
        pca = PCA(n_components = 2)
        X_principal = pca.fit_transform(encodedX)
        X_principal = pd.DataFrame(X_principal)
        X_principal.columns = ['P1', 'P2']
        plt.figure(figsize =(6, 6))
        plt.scatter(X_principal['P1'], X_principal['P2'],c = self.algorithm.fit_predict(X_principal), cmap ='rainbow')
        plt.show()
#### Estimators    
class Estimator(Clustering):

    def __init__(self):
        self.params=None
        super().__init__()
    
    def Optimal_K(self, encodedX, **params):
        distortions = []
        K = range(1,50)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(encodedX)
            distortions.append(kmeanModel.inertia_)
        ## Plot distortions vs Values of K
        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

        sse = []
        for i in range(3,50):
            kmeans = KMeans(n_clusters=i , max_iter=300, tol=0.01)
            kmeans.fit(encodedX)  # <- fit here.....
            sse.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center

        plt.plot(range(3,50),sse)
        plt.xlabel("Number of Custers")
        plt.ylabel("Inertia")
        plt.show()

    def K_Means(self, **params):   
        self.algorithm = KMeans()

    def K_mediods(self, **params):
        self.algorithm = KMedoids()

    def Aggl_Clustering(self, **params):
        self.algorithm = AgglomerativeClustering() #n_clusters=10, linkage='ward'
        

    def MeanShift_Clustering(self, **params):
        self.algorithm = MeanShift(bandwidth=None)

    def dbscan(self, **params):
        self.algorithm = DBSCAN()


import Analyzer
test=Analyzer.DataPreprocess("diamonds.csv", "price", 0.1)
test.data()

test.drop_missing_data()
test.encode_features_OH()
test.encode_label()
test.scale()
test.retrieve_data()
test.encodedX_cluster()
encodedX = test.encodedX_cluster_noOL()
print(encodedX)
# standard_embedding = umap.UMAP(random_state=42).fit_transform(encodedX)
# plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
# plt.show()


######### 
# pca = PCA(2)
# df = pca.fit_transform(encodedX)
# print(df.shape)

# pca = decomposition.PCA(n_components=2)
# X = pca.fit_transform(encodedX)
# loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=encodedX.columns) 
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=encodedX.columns)
# print(loading_matrix)

# kmeans = KMeans(n_clusters= 3)
# label = kmeans.fit_predict(df)

# centroids = kmeans.cluster_centers_
# u_labels = np.unique(label)


# for i in u_labels:
#     plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
# plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
# plt.legend()
# plt.show()


# kmeans_labels = KMeans(n_clusters=10).fit_predict(encodedX)
# # plt.scatter
# # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=0.1, cmap='Spectral')
# # plt.show()
# print(silhouette_score(encodedX, kmeans_labels))
# plt.figure(figsize=(8, 6))
# plt.scatter(encodedX[:,0], encodedX[:,1], c=kmeans_labels)
# plt.show()

# ###################################################
# ### silhouette score for optimal K
# range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9 , 10, 12, 14, 16, 18, 20]
# silhouette_avg = []
# for num_clusters in range_n_clusters:
 
#  # initialise kmeans
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(encodedX)
#     cluster_labels = kmeans.labels_
 
#  # silhouette score
#     silhouette_avg.append(silhouette_score(encodedX, cluster_labels))
# plt.plot(range_n_clusters,silhouette_avg,'bx-')
# plt.xlabel('Values of K') 
# plt.ylabel('Silhouette score') 
# plt.title('Silhouette analysis For Optimal k')
# plt.show()