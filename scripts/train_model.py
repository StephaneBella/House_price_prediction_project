import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_config, load_data
from src.preprocessing import preprocessing
from src.modeling import initialize_models
from src.evaluation import evaluation, save_model, save_model_params, taux_surestimation

def main():
    config = load_config()
    df = load_data(config['data']['raw_path'])
    
    X_train, X_test, y_train, y_test = preprocessing(df)
    
    models = initialize_models()
    model = models['LGBMRegressor']  
    
    model = evaluation(model, X_train, y_train, X_test, y_test)
    save_model(model)
    save_model_params(model)
if __name__ == "__main__":
    main()