## file responsible for implementing different classification techniques on input datasets.
import pandas as pd
from typing import List, Dict, Union 
import numpy as np
import seaborn as sbn 
from matplotlib import pyplot as plt
import optuna
import time

#from sklearn import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from livelossplot import PlotLossesKeras

print("packages good")

ALGORITHM_NAME_MAP ={
   "Logistic_regression": LogisticRegression,
   "KNN-Classifier":KNeighborsClassifier,
   "Decision_Tree": DecisionTreeClassifier,
   "Random_forest": RandomForestClassifier,
   "SVC": SVC
   #"Artificial_Neural_Network":ann
}

print(ALGORITHM_NAME_MAP['Logistic_regression'])

class Classifier():
    #def __init__(algorithm: str, **params):
    def __init__(self):
        self.params=None
        print("testing")

    def fit(self, X_train, y_train, **params):
        self.model = self.algorithm.fit(X_train, y_train, **params)

    def fit_keras(self, X_train, y_train,X_test, y_test, **params):
        self.model = self.algorithm.fit(X_train, y_train, batch_size = 20, epochs = 50)
        #self.model =self.algorithm.fit(X_train, y_train, batch_size = 20, epochs = 50, validation_data=(X_test, y_test), callbacks=[PlotLossesKeras()],verbose=0)

    def predict(self, X_test):
        self.pred = self.model.predict(X_test)

    def score(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        self.accuracy = accuracy_score(y_test, self.pred)

        cm = confusion_matrix(y_test, self.pred, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        plt.show()

        print(score, self.accuracy)
        print('precision score: '+ str(precision_score(y_test, self.pred, average='weighted')))
        print('recall score: '+ str(recall_score(y_test, self.pred, average='weighted')))
        print('F1 score: '+ str(f1_score(y_test, self.pred, average='weighted')))
        #print(confusion_matrix)
        return(self.accuracy)
    def losscurve(self):
        plt.figure(figsize=(8, 2))
        plt.title("ANN Model Loss Evolution")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(self.model.loss_curve_)
        plt.show()

class Estimator(Classifier):
    def __init__(self):
        self.params=None
        super().__init__()

    def Logistic_regression(self, **params):
        self.algorithm=LogisticRegression(**params)

    def Knn_Classifier(self, **params):
        self.algorithm=KNeighborsClassifier(**params)

    def decision_tree(self, **params):
        ## Choose different values for criterion.
        self.algorithm = DecisionTreeClassifier(criterion = "entropy") #gini, log_loss
        
    def random_forest(self, **params):
        self.algorithm = RandomForestClassifier()
        #Choose different values for criterion and number of estimators

    def svc(self, **params):
        self.algorithm=SVC(**params)
        ##SVC(C=1.0, kernel = 'linear', gamma = 'auto')

    def ann_MLC(self, **params):
        self.algorithm = MLPClassifier(hidden_layer_sizes=(100,100,50),activation='relu', solver='adam',batch_size=10,max_iter=50, random_state=0,verbose = True)

    def ann_keras(self, **params):
        self.algorithm = Sequential()
        self.algorithm.add(Dense(units=100, activation='relu', kernel_initializer='uniform')) #input layer
        self.algorithm.add(Dense(units=100, activation='relu')) # hidden layer
        self.algorithm.add(Dense(units=1, activation='sigmoid')) # O/p layer
        self.algorithm.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy']) # build 
        #self.algorithm.fit(X_train, y_train, batch_size = 20, epochs = 100)   
        #self.algorithm.fit(X_train, y_train, batch_size = 20, epochs = 100, validation_data=(X_test, y_test), callbacks=[PlotLossesKeras()],verbose=0)
        #y_pred = model.predict(x_test) ## test network
        #y_pred = (y_pred > 0.5) ## test network

class HypertuneParams():
    def __init__(self):
        self.params=None

    def LR_optuna(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            c= trial.suggest_float('C', 1e-10, 1000, log=True)
            i= trial.suggest_int('max_iter', 1, 1000, log=False)
            #r = trial.suggest_float('l1_ratio', 0, 1, log=False)
            s = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'saga'])
            
            model = LogisticRegression(random_state=48, class_weight='balanced', max_iter=i, C=c, solver=s) # l1_ratio=r
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = model.score(X_test, y_test)
            return score        

        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=5)
        self.params = study.best_params
        print(self.params)

    def KNN_optuna(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            n = trial.suggest_int('neighbors', 1, 50)
            w = trial.suggest_categorical('weights', ['uniform', 'distance'])
            a = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            model = KNeighborsClassifier(n_neighbors=n, weights=w, algorithm=a)

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = model.score(X_test, y_test)
            return score         

        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=5)
        self.params = study.best_params  
        print(self.params)

    def RF_optuna(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 250, step = 10),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "random_state": 50,
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            score=model.score(X_test, y_test)
            return score

        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=5)   
        self.params = study.best_params
        print(self.params)
    

    def DT_optuna(self, X_train, y_train, X_test, y_test):
        def objective(trial):    
            cr = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            sp = trial.suggest_categorical('splitter', ['best', 'random'])
            md = trial.suggest_int('max_depth', 2, 50)
            msl = trial.suggest_int('min_sample_leaf', 2, 32)
            msp = trial.suggest_int('min_sample_split', 2, 32)
            model = DecisionTreeClassifier(random_state=48, class_weight='balanced', criterion=cr, splitter=sp, max_depth=md, min_samples_leaf=msl, min_samples_split=msp)

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = model.score(X_test, y_test)
            return score
        
        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials=5)
        self.params = study.best_params
        print(self.params)



#         def ann(Classifier):
#              ## Change the architecture, the learning rate, the optimizers, 
#              ##the activation functions, or the regularizes until you find the optimum accuracy.
#              #self.classifier=MLPClassifier(**params)
#              pass




# class Logistic_regression(Classifier):
#     def __init__(self):
#         super().__init__()
#         self.algorithm=LogisticRegression(self.params)
#         #self.model = self.algorithm.fit(X_train, y_train)


# class Knn_Classifier(Classifier):
#     def __init__(self, **params): #, **params
#         super().__init__()
#         self.algorithm=KNeighborsClassifier(**params)
'''
Multi-class classification
For multi-class classification, the output layer must have nodes equal to the number of classes to predict. The Activation in the output layer should be ‘softmax’.
For a simple target column in the data, you should use a one-hot encoder to encode the output column into the number of needed columns for training the classification Neural Network in Keras.'''
