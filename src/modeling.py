from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler



def initialize_models():
    lgbmr = LGBMRegressor(random_state=3, verbose=-1)
    xgbr =  XGBRegressor(random_state=3)

    rp = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),LinearRegression())

    et = ExtraTreesRegressor(random_state=3)
    knn = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), KNeighborsRegressor(n_neighbors=50))
    br = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), BayesianRidge())

    # Modèles de base (stacking)
    estimators = [("lgbm", lgbmr), ("extra_trees", et)]

    # Méta-modèle
    meta_model= rp

    stacking_model = make_pipeline(StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=5, 
                                                     passthrough=True  # permet au méta-modèle de voir aussi les features originales
                                                     ))
    
    models = {
    "LGBMRegressor": lgbmr,
    "XGBRegressor": xgbr,
    " PolynomialRegression": rp,
    "ExtraTreesRegressor": et,
    "KNeighborsRegressor": knn,
    "BayesianRidge": br,
    "Stacking": stacking_model
    }
    return models

