'''
Regression Analysis
'''
## Load Libraries
from typing import List, Dict, Union 
import numpy as np
import seaborn as sbn 
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import optuna
import time
#from sklearn.metrics import root_mean_squared_error


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from keras.models import Sequential
from keras.layers import Dense


### Regressor
class Regressor():
    #def __init__(algorithm: str, **params):
    def __init__(self):
        self.params=None
        print("testing")

    def fit(self, X_train, y_train, **params):
        self.model = self.algorithm.fit(X_train, y_train, **params)
    
    def importance(self, X_train):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots()
        ax.barh(range(len(importances)), importances[indices])
        ax.set_yticks(range(len(importances)))
        _ = ax.set_yticklabels(np.array(X_train.columns)[indices])
        plt.show()
               
    def predict(self, X_test):
        self.pred = self.model.predict(X_test)

    def score(self, X_test, y_test):
        #evaluate different training models using a score function
        score = self.model.score(X_test, y_test)
        r2score= r2_score(y_test,self.pred)
        MSE= mean_squared_error(y_test,self.pred)
        RMSE= sqrt(mean_squared_error(y_test, self.pred))
        MAE= mean_absolute_error(y_test,self.pred)
        print(score, r2score, MSE, RMSE, MAE)
        print('MAE: '+ str(mean_absolute_error(y_test,self.pred)))

    def plot_truepred(self, y_test):
        plt.figure(figsize=(10,10))
        plt.scatter(y_test, self.pred, c='crimson')

        p1 = max(max(self.pred), max(y_test))
        p2 = min(min(self.pred), min(y_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.show()
        
######
class Estimator(Regressor):
    def __init__(self):
        self.params=None
        super().__init__()

    def Linear_regression(self, **params):
        self.algorithm = LinearRegression()

    def Knn_Regressor(self, **params):
        # different values for K, optimal K, r2 score should be above 90%.
        self.algorithm = KNeighborsRegressor(n_neighbors =3)

    def Decision_Tree_Regressor(self, **params):
        # different values for criterion. , optimal criterion., r2 score should be above 90%.
        self.algorithm = DecisionTreeRegressor(criterion="absolute_error", random_state=0)

    def Random_Forest(self, **params):
        # different values for criterion and number of estimators., r2  above 90%.
        self.algorithm = RandomForestRegressor(criterion="absolute_error", n_estimators=25, random_state=0)

    def SVR(self, **params):
        #list of scores for predicting the 'price' label., r2 score should be above 90%.
        self.algorithm = SVR(kernel='rbf', C=1.0, gamma= 'auto', epsilon=0.1)

    def ANN_Sklearn(self, **params):
        self.algorithm = MLPRegressor(hidden_layer_sizes=(100,100,), activation='relu', solver='adam', batch_size=20, max_iter=1000, random_state=0)


    def ANN(self, **params):
        #Change the architecture, the learning rate, the optimizers,
        # the activation functions or the regulizers until you find the optimum accuracy.
        #list of scores for predicting the 'price' label.
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        model.fit(x_train, y_train, batch_size = 20, epochs = 1000)
        loss, accuracy = model.evaluate(x_test, y_test)


class HypertuneParams():
    def __init__(self):
        self.params=None

    def KNN_hypertune(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 10)
            }
            model = SVR(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test ,pred)
            return r2 
        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=50)
        self.params = study.best_params
        print(self.params)

    def DT_hypertune(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            params = {
                "criterion": trial.suggest_int("criterion", ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
            }
            model = SVR(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test ,pred)
            return r2 
        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=50)
        self.params = study.best_params
        print(self.params)

    def SVR_hypertune(self, X_train, y_train, X_test, y_test):
        def objective (trial):
            params ={
                "kernel": trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid']),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "C": trial.suggest_uniform("C", 0.01, 10),
                "degree": trial.suggest_discrete_uniform("degree", 1, 5, 1),  
            }
            model = SVR(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test ,pred)
            return r2 
        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=50)
        self.params = study.best_params
        print(self.params)


    def RF_hypertune(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 150, step = 5), #n_estimators changed from 10 to 100 in 0.22.
                "max_depth": trial.suggest_int("max_depth", 4, 20),                  #The maximum depth of the tree. 
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 14),  #The minimum number of samples required to split an internal node:
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 14),    #The minimum number of samples required to be at a leaf node.
                "random_state": 10, 
            }
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test ,pred)
            return r2 

        start_time = time.time()
        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=100)
        print ("total_time =", time.time()-start_time)
        self.params = study.best_params
        print(self.params)