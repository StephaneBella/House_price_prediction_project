from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import uvicorn
import os


app = FastAPI()

# chargé le modèle 

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_house_price_prediction.pkl')

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class housefeatures(BaseModel):
    id: int
    date: str
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int 
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int 

@app.get("/")
def read_root():
    return {"message": "API House price Prediction ready!"}

@app.post("/predict")
def predict(data: housefeatures):
    # On convertit l’objet Pydantic en dict
    data_dict = data.dict()

    expected_cols = model.named_steps['pipeline'].named_steps['clustering'].feature_names_in_

    try:
        input_df = pd.DataFrame([[data_dict[col] for col in expected_cols]], columns=expected_cols)
    except KeyError as e:
        return {"error": f"Colonne manquante dans l'entrée : {e}"}

    try:
        log_predictions = model.predict(input_df)
        predictions = np.exp(log_predictions)
        return {"prediction": float(predictions[0])}
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)