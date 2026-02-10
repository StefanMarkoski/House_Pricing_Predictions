from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import traceback
from pathlib import Path

app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = BASE_DIR / 'models' / 'xgb_model.pkl'
scaler_path = BASE_DIR / 'models' / 'scaler.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int


class PredictionResponse(BaseModel):
    predicted_price: float
    status: str


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    """
    Endpoint to predict house price
    """
    try:
        
        feature_names = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated']
        
        
        features = [
            house.bedrooms,
            house.bathrooms,
            house.sqft_living,
            house.sqft_lot,
            house.floors,
            house.waterfront,
            house.view,
            house.condition,
            house.grade,
            house.sqft_above,
            house.sqft_basement,
            house.yr_built,
            house.yr_renovated
        ]
        
        print(f"Features: {features}")
        print(f"Feature names: {feature_names}")
        
        features_df = pd.DataFrame([features], columns=feature_names)
        print(f"DataFrame shape: {features_df.shape}")
        print(f"Scaler expected features: {scaler.n_features_in_}")
        
        features_scaled = scaler.transform(features_df)
        
        prediction = model.predict(features_scaled)[0]
        
        return PredictionResponse(
            predicted_price=int(prediction),
            status="success"
        )
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {"message": "House Price Prediction API", "endpoint": "/predict"}