from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(title="BinaAZ Rent Price Prediction API")

# Modelləri və place_columns yüklə
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.joblib")
PLACE_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "place_columns.joblib")

all_models = joblib.load(MODEL_PATH)
PLACE_COLUMNS = joblib.load(PLACE_PATH)

# API input modeli
class ApartmentFeatures(BaseModel):
    rooms: int
    area_m2: float
    floor: int
    total_floor: int
    is_new_building: int
    place: str
    model_name: str = Field("xgboost", description="Model adı: linear_regression, random_forest, xgboost, mlp")

@app.post("/predict")
def predict_rent(features: ApartmentFeatures):
    if features.model_name not in all_models:
        raise HTTPException(status_code=400, detail=f"Model '{features.model_name}' tapılmadı.")
    
    model = all_models[features.model_name]

    # Input DataFrame hazırla
    input_df = pd.DataFrame([features.dict()])

    # place sütunlarını əlavə et
    for col in PLACE_COLUMNS:
        input_df[col] = 1 if col.split('_', 1)[1] in input_df.loc[0, 'place'] else 0

    # Artıq 'place' və 'model_name' sütunları lazım deyil
    input_df = input_df.drop(columns=['place', 'model_name'], errors='ignore')

    try:
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    
    return {"predicted_price": float(prediction[0])}