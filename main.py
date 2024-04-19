import Analyzer
#import Classifier
import Regressor
#import Clustering
import pandas as pd
import numpy as np

test=Analyzer.DataPreprocess("diamonds.csv", "price", 0.1)
test.data()
#test.describe()
test.drop_missing_data()
test.encode_features_OH()
test.encode_label()
test.scale()
test.retrieve_data()
test.encodedX_cluster()
test.encodedX_cluster_noOL()

data = test.split_data()
X_train=data.get("X_train")
X_test=data.get("X_test")
y_train=data.get("y_train")
y_test=data.get("y_test")

#vis=Analyzer.Data_Visualize("diamonds.csv", "clarity")
#vis.correlation_matrix()
#vis.pair_plot()
# vis.histograms_numeric()
# vis.histograms_categorical()
# vis.Boxplot()
#vis.clarity()

##*********************************************************************************************************
########## Classifiers

#ClassierObj= Classifier.Estimator()
#ClassierObj.Logistic_regression()
#ClassierObj.Knn_Classifier(n_neighbors=3)
#ClassierObj.decision_tree(criterion='gini')
#ClassierObj.random_forest(criterion = "entropy", n_estimators =100) #criterion = "entropy", n_estimators =50
#ClassierObj.svc(C=1.0, kernel = 'linear', gamma = 'auto')
# ClassierObj.ann_MLC()

#ClassierObj.fit(X_train, y_train)
#ClassierObj.predict(X_test)
#ClassierObj.score(X_test, y_test)
#ClassierObj.losscurve() ## for ClassierObj.ann_MLC()

# ClassierObj.ann_keras()
# ClassierObj.fit(X_train, y_train, batch_size = 20, epochs = 50) # train network
# #ClassierObj.fit_keras(X_train, y_train, X_test, y_test)
# ClassierObj.predict(X_test)
# ClassierObj.score(X_test, y_test)
# ClassierObj.losscurve()

        ##********************
## Hypertune param

#print(test.split_data()[1])
#print(X_train.reshape(1, -1))
#print(y_train.reshape(-1, 1))
#ClassierObj= Classifier.HypertuneParams()
#ClassierObj.DT_optuna(X_train, y_train.reshape(-1, 1), X_test, y_test.reshape)
#ClassierObj.DT_optuna(X_train, y_train, X_test, y_test)

##*********************************************************************************************************
### Regression

RegressorObj = Regressor.Estimator()
#RegressorObj.Linear_regression()
#RegressorObj.Decision_Tree_Regressor()
RegressorObj.ANN_Sklearn()
RegressorObj.fit(X_train.values, y_train.values)
RegressorObj.importance(X_train)
RegressorObj.predict(X_test.values)
RegressorObj.score(X_test.values, y_test.values)
RegressorObj.plot_truepred(y_test.values)

#RegressorObj = Regressor.HypertuneParams()
#RegressorObj.KNN_hypertune()

##*********************************************************************************************************
### Clustering
encodedX = test.encodedX_cluster()

ClusterObj = Clustering.Estimator()

ClusterObj.K_Means(n_clusters=4, tol = 1e-5, random_state=0)
#ClusterObj.K_mediods(n_clusters=3, random_state=0)
ClusterObj.fit(encodedX)
ClusterObj.predict(encodedX)
ClusterObj.score(encodedX)

#drop_df = test.drop_missing_data()
#ClusterObj.plot_label(encodedX, drop_df)

#ClusterObj.Optimal_K(encodedX)

# ClusterObj.dbscan(eps=3, min_samples=10)
# ClusterObj.fit_dbscan(encodedX)
#ClusterObj.fit_predict(encodedX)
# ClusterObj.score(encodedX)
# ClusterObj.PCA_cluster(encodedX)
# ClusterObj.MeanShift_Clustering()

# ClusterObj.Aggl_Clustering(n_clusters=10)
# ClusterObj.fit_predict(encodedX)
# ClusterObj.score(encodedX)
# ClusterObj.PCA_cluster(encodedX)
#ClusterObj.MeanShift_Clustering()








