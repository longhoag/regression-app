import joblib
import pandas as pd

def load_model():
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

def predict(input_json):
    model, pipeline = load_model()
    df = pd.DataFrame([input_json])
    X_proc = pipeline.transform(df)
    return model.predict(X_proc)[0]