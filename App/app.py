from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
import shap
import json
import os
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI(title="CVD Risk Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "output", "cost_sensitive_xgboost_model.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
# Load model
print(f"Loading model from {MODEL_PATH}...")
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']
    scale_pos_weight = model_data.get('scale_pos_weight', 2.04)
    print(f"✓ Model loaded successfully")
    print(f"✓ Features: {features}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    features = []

# SHAP explainer
if model:
    explainer = shap.TreeExplainer(model)
else:
    explainer = None

class PatientData(BaseModel):
    gender: int
    age_years: float
    height: float
    weight: float
    bmi: float
    ap_hi: float
    ap_lo: float
    pulse_pressure: float
    map: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    patient_id: Optional[str] = "unknown"



def get_feature_name(feature):
    """Convert feature code to readable name"""
    names = {
        'gender': 'Gender',
        'age_years': 'Age',
        'height': 'Height',
        'weight': 'Weight',
        'bmi': 'Body Mass Index',
        'ap_hi': 'Systolic Blood Pressure',
        'ap_lo': 'Diastolic Blood Pressure',
        'pulse_pressure': 'Pulse Pressure',
        'map': 'Mean Arterial Pressure',
        'cholesterol': 'Cholesterol Level',
        'gluc': 'Glucose Level',
        'smoke': 'Smoking Status',
        'alco': 'Alcohol Consumption',
        'active': 'Physical Activity'
    }
    return names.get(feature, feature.replace('_', ' ').title())

def get_feature_unit(feature):
    """Get unit for feature"""
    units = {
        'age_years': 'years',
        'height': 'cm',
        'weight': 'kg',
        'bmi': 'kg/m²',
        'ap_hi': 'mmHg',
        'ap_lo': 'mmHg',
        'pulse_pressure': 'mmHg',
        'map': 'mmHg',
        'cholesterol': 'category',
        'gluc': 'category',
        'smoke': 'binary',
        'alco': 'binary',
        'active': 'binary',
        'gender': 'binary'
    }
    return units.get(feature, '')

def get_feature_interpretation(feature, value):
    """Get clinical interpretation of feature value"""
    if feature == 'ap_hi':
        if value < 120: return "Normal systolic BP"
        elif value < 130: return "Elevated systolic BP"
        elif value < 140: return "Stage 1 hypertension"
        else: return "Stage 2 hypertension"
    elif feature == 'ap_lo':
        if value < 80: return "Normal diastolic BP"
        elif value < 90: return "Stage 1 hypertension"
        else: return "Stage 2 hypertension"
    elif feature == 'bmi':
        if value < 18.5: return "Underweight"
        elif value < 25: return "Normal weight"
        elif value < 30: return "Overweight"
        else: return "Obese"
    elif feature == 'pulse_pressure':
        return "Widened pulse pressure" if value > 60 else "Normal pulse pressure"
    elif feature == 'map':
        return "Elevated MAP" if value > 100 else "Normal MAP"
    elif feature == 'cholesterol':
        return {1: "Normal", 2: "Above normal", 3: "Well above normal"}.get(value, "Unknown")
    elif feature == 'gluc':
        return {1: "Normal", 2: "Above normal (prediabetes)", 3: "Well above normal"}.get(value, "Unknown")
    elif feature == 'smoke':
        return "Active smoker" if value == 1 else "Non-smoker"
    elif feature == 'alco':
        return "Consumes alcohol" if value == 1 else "No alcohol"
    elif feature == 'active':
        return "Physically active" if value == 1 else "Sedentary"
    elif feature == 'gender':
        return "Female" if value == 1 else "Male"
    elif feature == 'age_years':
        if value < 40: return "Young adult"
        elif value < 60: return "Middle-aged"
        else: return "Older adult"
    return f"{value}"

def test_new_patient_with_details(patient_data: Dict[str, Any], threshold=0.5, top_n=5):
    """
    Detailed prediction logic
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # input_data in correct feature order
    input_dict = {feature: patient_data.get(feature) for feature in features}
    
    # Check for missing values after mapping
    missing = [f for f in features if input_dict[f] is None]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        
    df = pd.DataFrame([input_dict])[features]
    
    # Make prediction
    risk_score = float(model.predict_proba(df)[0, 1])
    prediction = 1 if risk_score >= threshold else 0
    risk_category = "HIGH" if prediction == 1 else "LOW"
    
    # Calculate SHAP values
    shap_vals = explainer.shap_values(df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1][0]  # Binary classification index 1
    else:
        shap_vals = shap_vals[0]
    
    feature_importance = dict(zip(features, model.feature_importances_))
    
    all_risk_factors = []
    for i, feature in enumerate(features):
        value = input_dict[feature]
        
        risk_factor = {
            'feature': feature,
            'feature_name': get_feature_name(feature),
            'value': float(value) if isinstance(value, (np.floating, float)) else int(value),
            'unit': get_feature_unit(feature),
            'importance': float(feature_importance[feature]),
            'shap_value': float(shap_vals[i]),
            'impact': 'increases risk' if shap_vals[i] > 0 else 'decreases risk',
            'magnitude': abs(float(shap_vals[i])),
            'interpretation': get_feature_interpretation(feature, value)
        }
        all_risk_factors.append(risk_factor)
    
    top_risk_factors = sorted(all_risk_factors, key=lambda x: x['magnitude'], reverse=True)[:top_n]
    for factor in top_risk_factors:
        del factor['magnitude']
    
    result = {
        'patient_id': patient_data.get('patient_id', 'unknown'),
        'risk_score': round(risk_score, 4),
        'risk_percentage': int(risk_score * 100),
        'prediction': int(prediction),
        'risk_category': risk_category,
        'threshold_used': float(threshold),
        'top_risk_factors': top_risk_factors,
        'model_confidence': round(max(risk_score, 1 - risk_score), 4),
        'prediction_timestamp': datetime.utcnow().isoformat() + "Z"
    }
    
    return result

@app.get("/")
async def home():
    """Health check endpoint"""
    return {
        "service": "CVD Risk Prediction API",
        "status": "running",
        "model": "Cost-Sensitive XGBoost",
        "features_required": len(features),
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features": features,
        "scale_pos_weight": scale_pos_weight if model else None
    }

@app.post("/predict")
async def predict(
    data: PatientData, 
    threshold: float = 0.5, 
    top_n: int = 5
):
    """
    Main prediction endpoint
    """
    try:
        if not 0 < threshold < 1:
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
        
        # Convert Pydantic model to dict
        patient_dict = data.dict()
        
        # Run prediction
        result = test_new_patient_with_details(patient_dict, threshold=threshold, top_n=top_n)
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)