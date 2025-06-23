import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_config, load_data
from src.preprocessing import preprocessing_pipeline, split
from src.modeling import initialize_models
from src.evaluation import evaluation, save_model, save_model_params, taux_surestimation
from sklearn.pipeline import make_pipeline 
import pandas as pd

def main():
    # chargement des donnees
    config = load_config()
    df = load_data(config['data']['raw_path'])
    
    #separation des ensembles (train/test)
    trainset, testset = split(df)
    
    #separation features, target
    X_train = trainset.drop('price', axis=1)
    y_train = trainset['price']

    X_test = testset.drop('price', axis=1)
    y_test = testset['price']

    # definition du modèle
    models = initialize_models()
    lgbm = models['LGBMRegressor'] 
    preprocessor = preprocessing_pipeline(n_clusters=100, cols_to_encode=None) 

    final_model = make_pipeline(preprocessor, lgbm)
    
    final_model = evaluation(final_model, X_train, y_train, X_test, y_test)
    save_model(final_model)
    save_model_params(final_model)
if __name__ == "__main__":
    main()