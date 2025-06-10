from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                            root_mean_squared_log_error, make_scorer, r2_score)
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os
import json
import lightgbm as lgb


# fonction pour le taux de surestimation supérieur à 10%
def taux_surestimation(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    threshold = y_test * 0.1
    residuals = y_pred - y_test
    nbre_surestimations = 0

    for i in range (len(residuals)):
        if residuals[i] > threshold[i]:
            nbre_surestimations += 1

    rate = (nbre_surestimations * 100) / len(residuals)

    return rate 
# Récupère le chemin du dossier racine du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def evaluation(model, x_train, y_train, x_test, y_test, model_name="model",
                relative_fig_path='outputs/figures/Learning_curve_&_residuals.png',
                relative_metrics_path='outputs/metrics/metrics_{model_name}.json'):
    """Évalue un modèle, affiche et enregistre les métriques et figures"""
    # Variables communes
    fig_learning_path = os.path.join(PROJECT_ROOT, relative_fig_path)
    metrics_path = os.path.join(PROJECT_ROOT, relative_metrics_path)

    os.makedirs(os.path.dirname(fig_learning_path), exist_ok=True) # cree le path s'il n'existe pas encore
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    y_train_log = np.log(y_train)
    model.fit(x_train, y_train_log)
    y_pred_log = model.predict(x_test)
    y_pred = np.exp(y_pred_log)

    N, train_score, val_score = learning_curve(model,x_train, y_train, scoring='neg_mean_absolute_error',
                                                cv=4, train_sizes=np.linspace(0.1, 0.8, 5))


    mean_train = -train_score.mean(axis=1)  # négatif car scoring est négatif
    mean_val = -val_score.mean(axis=1)
    residus = (y_pred - y_test)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(N, mean_train, label='Training error')
    plt.plot(N, mean_val, label='Validation error')
    plt.xlabel("Taille de l'échantillon d'entraînement")
    plt.ylabel("MAE")
    plt.title("Courbe d'apprentissage")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(residus, kde=True)
    plt.title("Distribution des résidus")
    plt.savefig(fig_learning_path)
    plt.show()

    # sauvegarde des métriques
    R2 = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    RMSLE = root_mean_squared_log_error(y_test, y_pred)
    rate = taux_surestimation(y_test, y_pred) #taux de surestimations supérieurs à 10%

    results = {
        "R2": R2,
        "MAE": MAE,
        "MAPE": MAPE,
        "RMSLE": RMSLE,
        "rate": rate
    }

    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[✔] Résultats sauvegardés dans :\n- {fig_learning_path}\n- {metrics_path}")

    return model


# Récupère le chemin du dossier racine du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def save_model(model, output_path='models/model_cut_prediction.pkl'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Modèle LightGBM sauvegardé dans : {output_path}")

def load_model(relative_path= 'models/model_price_prediction.txt'):
    output_path = os.path.join(PROJECT_ROOT, relative_path)
    model = lgb.Booster(model_file=output_path)
    return model



def save_model_params(model, relative_output_path='config/model_params.json'):
    output_path = os.path.join(PROJECT_ROOT, relative_output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    params = model.get_params()

    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return str(obj)

    try:
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=4, default=default_serializer)
        print(f"✅ Paramètres enregistrés dans : {output_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du modèle : {e}")


