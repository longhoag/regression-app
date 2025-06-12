import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import hashlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
import pandas as pd
import os

from src.data_loader import load_data
from src.preprocess import build_pipeline


def get_data_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def train_model(data_path):
    df = load_data(data_path)
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']

    pipeline = build_pipeline(X)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #fit and transform
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    models = {
        "rf": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [50, 100], # number of decision trees
                "max_depth": [5, 10, None] # depth of tree
            }
        },
        "xgb": {
            "model": XGBRegressor(),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.1, 0.05]
            }
        }
    }

    best_score = float('inf')
    best_model = None
    best_model_name = None

    snapshot_path = os.path.join("data", "housing_snapshot.csv")
    df.to_csv(snapshot_path, index=False)
    data_hash = get_data_hash(snapshot_path)

    for name, config in models.items():
        grid = GridSearchCV(config["model"], config["params"], cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X_train_proc, y_train)
        preds = grid.predict(X_test_proc)

        rmse = np.sqrt(mean_squared_error(y_test, preds)) #root of mse
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # MLflow logging 
        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("data_hash", data_hash)

            # infer signature and input example --> rm warning
            signature = infer_signature(X_test_proc, preds)
            input_example = pd.DataFrame(X_test_proc[:1].toarray() if hasattr(X_test_proc, 'toarray') else X_test_proc[:1])

            # log model
            mlflow.sklearn.log_model(grid.best_estimator_, "model", signature=signature, input_example=input_example)
            joblib.dump(pipeline, "pipeline.pkl")
            mlflow.log_artifact("pipeline.pkl")
            mlflow.log_artifact(snapshot_path)

        # best model selection
        if rmse < best_score:
            best_score = rmse
            best_model = grid.best_estimator_
            best_model_name = name

    # save
    joblib.dump(best_model, "model.pkl")
    joblib.dump(pipeline, "pipeline.pkl")
    print(f"Best model: {best_model_name} with RMSE: {best_score:.2f}")