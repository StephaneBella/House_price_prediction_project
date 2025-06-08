# House_price_prediction_project

Prédiction du prix des maisons dans le comté de King (Seattle et environs) à l’aide de techniques de machine learning.  
Ce projet s’inscrit dans le cadre d’un cas d’usage métier pour une **entreprise immobilière privée spécialisée dans l’achat-revente de biens résidentiels**.

---

## Objectif du projet

Le but est de développer un modèle prédictif fiable permettant à une entreprise immobilière (*Seattle Property Advisors*) :

- d’estimer la valeur de revente d’une maison après rénovation,
- d’identifier les biens sous-évalués à fort potentiel,
- et de limiter les risques financiers liés à une mauvaise évaluation.

A partir des caractéristiques comme:
    - `bedrooms`: Nombre de chambres
    - `bathrooms`: Nombre de salles de bain
    - `sqft_living`: surface habitable (en pieds carrés)
    - `sqft_living15`: surface habitable moyenne des 15 maisons les plus proches (en pieds carrés)
    - `sqft_lot`: surface totale du terrain (en pieds carrés)
    - `sqft_lot15`: surface moyenne des terrains des 15 maisons les plus proches (en pieds carrés)
    - `floors`: Nombre d'étages
    - `waterfront`: Vue sur l'eau (0 ou 1)
    - `view`: Qualité de la vue (de 0 à 4)
    - `condition`: Etat général de la maison (de 1 à 5)
    - `grade`: Note globale donnée par l'administration
    - `sqft_above`: surface des étages au dessus du sol
    - `sqft_basement`: Surface du sous-sol
    - `yr_built`: Année de construction
    - `yr_renovated`: Année de rénovation (0 si jamais rénové)

---

##  Contexte métier

**Problématique :**  
> *Comment prédire avec précision les prix de revente des maisons dans le comté de King pour optimiser les décisions d’investissement immobilier ?*

**Contrainte stratégique :**  
> *Une surestimation du prix de revente peut engendrer des pertes importantes. Le modèle doit donc privilégier la prudence dans ses prédictions.*

**Bénéfices pour l’entreprise :**
- Maximisation de la marge sur les achats-reventes.
- Réduction des erreurs de pricing.
- Meilleure priorisation des investissements immobiliers.

---

## Données utilisées

- **Source** : [kc_house_data.csv] (Kaggle)
- **Taille** : 21 613 enregistrements, 21 variables.
- **Variables clés** : `price`, `sqft_living`, `bedrooms`, `bathrooms`, `zipcode`, `grade`, `condition`, `year_built`, `sqft_lot`, etc.

---

## Modèles explorés

- **Régression polynomiale**
- **Extra trees Regressor**
- **XGBoost Regressor**
- **LigGBM Regressor**
- **BayesianRidge(polynomiale)**
- **Stacking Regressor** 
---

## Pipeline de traitement

1. **Analyse exploratoire des données (EDA)**  
   → Visualisations, corrélations, outliers, distributions.

2. **Prétraitement**  
   - Encodage des variables catégorielles.
   - Transformation des dates (`year_built`, `yr_renovated`).
   - Normalisation / standardisation.

3. **Feature ingeneering**
   → Location_cluster(clustering sur la longitude et la lattitude), month_sol(mois de vente de la maison).

4. **Entraînement & évaluation**  
   → Validation croisée (K-Fold), courbes d'apprentissage, distribution des résidus.

5. **Évaluation personnalisée**  
   → **MAE**, **MAPE**, **RMSLE**, mais aussi **taux de surestimation >10%**.



