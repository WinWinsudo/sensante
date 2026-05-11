# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP/UCAD

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Schémas Pydantic ---
class PatientInput(BaseModel):
    """Données d'entrée : les symptômes d'un patient."""
    age: int = Field(..., ge=0, le=120, description="Age en années")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Température en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Présence de toux")
    fatigue: bool = Field(..., description="Présence de fatigue")
    maux_tete: bool = Field(..., description="Présence de maux de tête")
    region: str = Field(..., description="Région du Sénégal")

class DiagnosticOutput(BaseModel):
    """Données de sortie : le résultat du diagnostic."""
    diagnostic: str = Field(..., description="Diagnostic prédit")
    probabilite: float = Field(..., description="Probabilité du diagnostic")
    confiance: str = Field(..., description="Niveau de confiance")
    message: str = Field(..., description="Recommandation")

# --- Application FastAPI ---
app = FastAPI(
    
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement du modèle ---
print("Chargement du modèle...")
model        = joblib.load("models/model.pkl")
le_sexe      = joblib.load("models/encoder_sexe.pkl")
le_region    = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modèle chargé : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")

# --- Routes ---
@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {"status": "ok", "message": "SenSante API is running"}

@app.get("/model-info")
def model_info():
    """Informations sur le modèle chargé."""
    return {
        "type":         type(model).__name__,
        "n_estimators": model.n_estimators,
        "classes":      list(model.classes_),
        "n_features":   model.n_features_in_
    }

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """Prédire un diagnostic à partir des symptômes d'un patient."""
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
                                confiance="aucune",
                                message=f"Sexe invalide : {patient.sexe}")
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
                                confiance="aucune",
                                message=f"Région inconnue : {patient.region}")

    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete), region_enc
    ]])

    diagnostic = model.predict(features)[0]
    proba_max  = float(model.predict_proba(features)[0].max())
    confiance  = ("haute" if proba_max >= 0.7
                  else "moyenne" if proba_max >= 0.4 else "faible")

    messages = {
        "palu":   "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh":   "Suspicion de typhoïde. Consultation nécessaire.",
        "sain":   "Pas de pathologie détectée."
    }
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )