import pandas as pd
import joblib
from src.features.engineer import engineer_features

def predict_new(dataframe):
    model = joblib.load("data/models/xgb_hit_predictor.pkl")
    X, _ = engineer_features(dataframe)
    probs = model.predict_proba(X)[:, 1]
    dataframe["pred_hit_prob"] = probs
    return dataframe[["appid", "name", "pred_hit_prob"]]
