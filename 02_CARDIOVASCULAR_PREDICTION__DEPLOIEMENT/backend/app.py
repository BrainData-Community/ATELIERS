# mettre en place une API pour le modèle de prédiction cardiovasculaire (fastapi)
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import sklearn
    
app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int

@app.post("/predict")
def predict_cardiovascular_risk(data: PatientData):
    # Charger le modèle pré-entraîné
    model = joblib.load("../models/cardio_model.pkl")

    print("Données reçues pour la prédiction :", data)
    
    # Préparer les données d'entrée pour la prédiction
    input_data = [[
        data.age,
        data.weight,
        data.ap_hi,
        data.ap_lo,
        data.cholesterol,
        data.gluc
    ]]
    
    # Faire la prédiction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  
    print("Prédiction :", prediction)
    print("Probabilité :", probability)

    # Retourner le résultat de la prédiction
    return {"cardiovascular_risk": int(prediction[0]), "probability": float(probability)}