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
        self.feature_names_in_ = None  

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()  # important pour l'API fast API
        else:
            raise ValueError("LocationClustering needs a DataFrame at fit.")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        coords = X[['lat', 'long']]
        self.kmeans.fit(coords)
        return self


    def transform(self, X):
        import pandas as pd

        # Si X est un numpy array → reconversion en DataFrame
        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is None:
                raise ValueError("Pas de noms de colonnes sauvegardés.")
            if X.shape[1] != len(self.feature_names_in_):
                raise ValueError(f"Shape mismatch: attendu {len(self.feature_names_in_)} colonnes, reçu {X.shape[1]}")  # pour avoir plus de clarté si erreur
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        coords = X[['lat', 'long']]
        clusters = self.kmeans.predict(coords)
        X = X.copy()

        X['date'] = pd.to_datetime(X['date'], errors='coerce')
        X['month_sold'] = X['date'].dt.month
        X['location_cluster'] = clusters

        for col in ['sqft_living', 'sqft_living15', 'sqft_above']:
            if col in X.columns:
                X[col] = np.log(X[col] + 1)

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


