import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import os

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

class LocationClustering(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=100, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
    
    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        coords = X[['lat', 'long']]
        self.kmeans.fit(coords)
        return self
    
    def transform(self, X):
        coords = X[['lat', 'long']]
        clusters = self.kmeans.predict(coords)
        X = X.copy()

        #convertion de la colonne date en format datetime
        X['date'] = pd.to_datetime(X['date'], errors='coerce') # errors = 'coerce' gère les mauvaises dates
        X['month_sold'] = X['date'].dt.month

        X['location_cluster'] = clusters
        X['sqft_living'] = np.log(X['sqft_living'])
        X['sqft_living15'] = np.log(X['sqft_living15'])
        X['sqft_above'] = np.log(X['sqft_above'])
        X = X.drop(columns=['lat', 'long'])
        return X

class TargetEncodingWrapper(BaseEstimator, TransformerMixin):
    """
    Applique un target encoding sur plusieurs colonnes, en gérant fit et transform.
    """
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {}
    
    def fit(self, X, y):
        for col in self.cols:
            enc = TargetEncoder(cols=[col])
            enc.fit(X[[col]], y)
            self.encoders[col] = enc
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, enc in self.encoders.items():
            X[col] = enc.transform(X[[col]])
        return X

def preprocessing_pipeline(n_clusters=100, cols_to_encode=None):
    if cols_to_encode is None:
        cols_to_encode = ['zipcode', 'yr_built', 'yr_renovated', 'location_cluster', 'month_sold']
    
    pipeline = Pipeline([
        ('clustering', LocationClustering(n_clusters=n_clusters)),
        ('target_encoding', TargetEncodingWrapper(cols=cols_to_encode)),
        ('drop_cols', DropColumns(columns=['id', 'date']))
    ])
    
    return pipeline

def split(df):
    trainset, testset = train_test_split(df, train_size=0.8, shuffle=True, random_state=3)
    return trainset, testset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def preprocessing(df, trainset_output_path='data/processed/trainset_processed.csv', 
                  testset_output_path='data/processed/testset_processed.csv'):
    
    trainset_output_path= os.path.join(PROJECT_ROOT, trainset_output_path)
    testset_output_path= os.path.join(PROJECT_ROOT, testset_output_path)
    
    os.makedirs(os.path.dirname(trainset_output_path), exist_ok=True) # cree le path s'il n'existe pas encore
    os.makedirs(os.path.dirname(testset_output_path), exist_ok=True)

    trainset, testset = split(df)
    # Séparer features et target
    x_train = trainset.drop(columns=['price'])
    y_train = trainset['price']
    x_test = testset.drop(columns=['price'])
    y_test = testset['price']

    # Initialiser pipeline
    pipeline = preprocessing_pipeline(n_clusters=100)

    # Fit + transform sur train
    x_train = pipeline.fit_transform(x_train, y_train)
    trainset = x_train.copy()
    trainset['price'] = y_train
    trainset.to_csv(trainset_output_path, index=False)

    # Pour le test (pas de fit, juste transform)
    x_test = pipeline.transform(x_test)
    testset = x_test.copy()
    testset['price'] = y_test
    testset.to_csv(testset_output_path, index=False)

    return x_train, x_test, y_train, y_test

