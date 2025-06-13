from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
import joblib
import numpy as np
import pandas as pd
import uvicorn
import os 
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("regression-predictions")

# load model
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

app = FastAPI(title="California Housing Price Predictor")

# define schema (validate)
class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: int
    total_rooms: int
    total_bedrooms: int
    population: int
    households: int
    median_income: float
    ocean_proximity: str

@app.get("/")
def root():
    return {"message": "Welcome to the California Housing Price Predictor API"}

@app.post("/predict")
def predict(input_data: HousingInput):
    try:
        # convert input to dataframe
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])

        # preprocess input
        processed_input = pipeline.transform(input_df)

        # make prediction
        prediction = model.predict(processed_input)
        predicted_value = float(prediction[0])

        # log to mlflow
        with mlflow.start_run(run_name="prediction_log", nested=True):
            mlflow.log_params(input_dict)
            mlflow.log_metric("predicted_median_house_value", predicted_value)

        return {
            "predicted median house value": predicted_value,
            "input_data": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

