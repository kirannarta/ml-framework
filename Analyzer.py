### Analyser.py

# import packages
import pandas as pd
from typing import List, Dict, Union 
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
import seaborn as sbn 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class DataReader():
    def __init__(self, data_path:str):
        self.data_path = data_path
    
    def read_dataset(self):
        """name of csv file, 
        reads,save data in instance of analyser"""
        try:
            self.df = pd.read_csv(self.data_path, index_col=0) 
            print(self.df)
            return(self.df)
        except IOError as err:
            print("I/O error")

    def describe(self):
        """
        print input features, attribute types
        basic stats on numeric data 
        """        
        isNumeric = self.df.select_dtypes(include=np.number)
        numeric_data_stats =  isNumeric.describe()
        print(numeric_data_stats)


class DataPreprocess(DataReader):
    def __init__(self, data_path:str, label:str, percent:float):
        #super().__init__(data_path:str, label:str)
        self.data_path=data_path
        self.label=label
        self.percent=percent

    def data(self):
        self.df=self.read_dataset()
        return(self.df)
        
    def drop_missing_data(self):
        self.dropdf=self.df.dropna()
        return(self.dropdf)
    
    def drop_columns(self):
        return(self.df.dropna(axis=1))

    def encodedX_cluster(self):
        self.numerical= self.dropdf._get_numeric_data()

        scaler = MinMaxScaler()
        self.num_scale = pd.DataFrame(scaler.fit_transform(self.numerical))
        self.num_scale.columns = self.numerical.columns

        categorical_cols = list(set(self.dropdf.columns) - set(self.numerical.columns))
        self.cat = self.dropdf[self.dropdf.columns.intersection(categorical_cols)]
        self.cat_oh = pd.get_dummies(self.cat, dtype=int)

        self.culster_processed = pd.concat([self.num_scale, self.cat_oh.set_index(self.num_scale.index)], axis=1)
        return(self.culster_processed)

    def encodedX_cluster_noOL(self): 
        cols = ['carat', 'depth', 'price', 'x', 'y', 'z'] # The columns you want to search for outliers in

        # Calculate quantiles and IQR
        Q1 = self.culster_processed[cols].quantile(0.25) # Same as np.percentile but maps (0,1) and not (0,100)
        Q3 = self.culster_processed[cols].quantile(0.75)
        IQR = Q3 - Q1

        # Return a boolean array of the rows with (any) non-outlier column values
        condition = ~((self.culster_processed[cols] < (Q1 - 1.5 * IQR)) | (self.culster_processed[cols] > (Q3 + 1.5 * IQR))).any(axis=1)

        # Filter our dataframe based on condition
        self.culster_OL = self.culster_processed[condition] 
        return(self.culster_OL)  

    def encode_features_nominal(self):
        """
        Using data from drop missing data
        """
        self.Xdf= self.dropdf.drop(columns = self.label, axis=1)

        self.X_numerical= self.Xdf._get_numeric_data()

        categorical_cols = list(set(self.Xdf.columns) - set(self.X_numerical.columns))
        self.X_cat = self.Xdf[self.Xdf.columns.intersection(categorical_cols)]
        print(self.X_cat)

    def encode_features_OH(self):
        self.Xdf= self.dropdf.drop(columns = self.label, axis=1)
        self.X_numerical= self.Xdf._get_numeric_data()
        
        categorical_cols = list(set(self.Xdf.columns) - set(self.X_numerical.columns))
        self.X_cat = self.Xdf[self.Xdf.columns.intersection(categorical_cols)]
        self.X_cat_oh = pd.get_dummies(self.X_cat, dtype=int)
        return(self.X_cat_oh)

    def encode_label(self):
        #self.LE = LabelEncoder() #For classification:cut
        #self.Y = self.LE.fit_transform(self.dropdf[self.label])    #For classification:cut
        self.Y = self.dropdf[self.label] #For regression:price
        return(self.Y)  

    def scale(self):
        self.Xdf= self.dropdf.drop(columns = self.label, axis=1)
        self.X_numerical= self.Xdf._get_numeric_data()

        scaler = MinMaxScaler()
        self.X_num_scale = pd.DataFrame(scaler.fit_transform(self.X_numerical))
        self.X_num_scale.columns = self.X_numerical.columns
        return(self.X_num_scale)

    def retrieve_data(self):
        self.df_processed = pd.concat([self.X_num_scale, self.X_cat_oh.set_index(self.X_num_scale.index)], axis=1)
        #self.Ydf=pd.DataFrame(self.Y, columns=[self.label])
        #self.df_processed = pd.concat([self.df_processed, self.Ydf])
        #print(self.df_processed)
        return(self.df_processed)
    
        
    def sample(self, rf:float):
        self.df.sampled=self.dropdf.sample(frac=rf)
        return(self.df.sampled)
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_processed ,self.Y, test_size=self.percent, random_state=0)
        #print("self.X_train:", self.X_train, "self.X_test", self.X_test, "self.y_train:", self.y_train, "self.y_test:", self.y_test)
        d = dict();
        d['X_train']=self.X_train
        d['X_test']=self.X_test
        d['y_train']=self.y_train
        d['y_test']=self.y_test
        
        return d
        #return(self.X_train, self.X_test, self.y_train, self.y_test)


class Data_Visualize(DataPreprocess):
    # def __init__(self):
    #     self.params=None
    #     super().__init__()
    def __init__(self, data_path:str, label:str):
        self.data_path=data_path
        self.label=label
        self.df=self.read_dataset()
        self.dropdf=self.drop_missing_data()
        self.X_num_scale=self.scale()
        self.X_cat_oh=self.encode_features_OH()
        self.df_processed=self.retrieve_data()
        self.dropdf=self.drop_missing_data()
        
    def correlation_matrix(self):
        plt.figure(figsize=(16, 14))
        heatmap = sbn.heatmap(self.df_processed.corr(), annot=True)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
        plt.show()

        plt.figure(figsize=(8, 8))
        heatmap = sbn.heatmap(self.X_num_scale.corr(), annot=True)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
        plt.show()


    def pair_plot(self):
        plt.figure(figsize=(16, 14))
        sbn.pairplot(self.df_processed)
        plt.savefig('pairplot.png')
        plt.show()

        plt.figure(figsize=(10,10))
        sbn.pairplot(self.X_num_scale)
        plt.savefig('pairplot.png')
        plt.show()

    def histograms_numeric(self):
        num_df = self.dropdf._get_numeric_data()
        fig, axes = plt.subplots(ncols=len(num_df.columns), figsize=(20,5))
        for col, ax in zip(num_df, axes):
            num_df[col].hist(ax=ax)

        plt.tight_layout()    
        plt.savefig('histogram_numeric.png')
        plt.show()

    def histograms_categorical(self):
        num_df = self.dropdf._get_numeric_data()
        cat_cols = list(set(self.dropdf.columns) - set(num_df.columns))
        cat_df = self.dropdf[self.dropdf.columns.intersection(cat_cols)]
                
        fig, axes = plt.subplots(ncols=len(cat_df.columns), figsize=(10,5))
        for col, ax in zip(cat_df, axes):
            cat_df[col].value_counts().sort_index().plot.bar(ax=ax, title=col)

        plt.tight_layout()
        plt.savefig('histogram_cat.png')    
        plt.show()

    def Boxplot(self):
        num_df = self.dropdf._get_numeric_data()
        fig, axes = plt.subplots(ncols=len(num_df.columns), figsize=(10,5))
        for col, ax in zip(num_df, axes):
            sbn.boxplot(num_df[col], ax=ax)
            #num_df[col].boxplot()

        plt.tight_layout()    
        plt.savefig('boxplot_numeric.png')
        plt.show()

    def clarity(self):
        data=self.dropdf
        sbn.catplot(x="clarity", 
            kind="count",  
            data=data).set(title='4.4: The Purchase Count by Clarity')
        plt.show()