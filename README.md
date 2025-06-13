# regression-app
Production-Ready Regression Pipeline

# California Housing Price Predictor API

## Installation after cloning the repo


### Set up a virtual environment
```
python -m venv venv
source venv/bin/activate
```


### Install dependencies
```
pip install fastapi uvicorn joblib pandas scikit-learn xgboost
```

### For macos (together with the previous)
```
brew info libomp
```

```
pip install scikit-learn==1.6.1
```

## Run locally
```
uvicorn main:app --reload
```

## Docker Deployment
```
docker build -t housing-api .
docker run -p 8000:8000 housing-api
```

```
docker-compose up --build
```


