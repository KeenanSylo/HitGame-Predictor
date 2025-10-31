import json
from pathlib import Path
import pandas as pd
import gradio as gr
import joblib

from src.data.fetch_storefront import get_appdetails
from src.data.fetch_steamspy import get_steamspy
from src.features.engineer import engineer_features

MODEL_PATH = Path("data/models/xgb_hit_predictor.pkl")
META_PATH = Path("data/models/model_meta.json")

def load_model_and_meta():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Train it first.")
    model = joblib.load(MODEL_PATH)
    best_thr = 0.5
    feature_cols = None
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        best_thr = float(meta.get("threshold", 0.5))
        feature_cols = meta.get("feature_columns")
    return model, best_thr, feature_cols

model, threshold, feature_cols = load_model_and_meta()

def predict_hit(appid: int):
    store = get_appdetails(appid)
    spy = get_steamspy(appid)
    if not store or not spy:
        return f"Failed to fetch data for AppID {appid}."

    df = pd.DataFrame([{**store, **spy}])
    X, _ = engineer_features(df, return_target=False)

    # align to training feature set
    if feature_cols is not None:
        # add any missing columns with 0, and drop extras
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_cols]

    prob = model.predict_proba(X)[:, 1][0]
    is_hit = prob >= float(threshold)

    name = store.get("name", "Unknown Game")
    return (
        f"**{name}**\n\n"
        f"- Predicted hit probability: **{prob:.2%}**\n"
        f"- Threshold: {float(threshold):.3f}\n"
        f"- Verdict: {'Hit' if is_hit else 'Not hit'}"
    )

demo = gr.Interface(
    fn=predict_hit,
    inputs=gr.Number(label="Enter Steam AppID"),
    outputs=gr.Markdown(),
    title="Steam Hit Predictor",
    description="Enter a Steam AppID to predict whether the game is likely to become a hit.",
    examples=[[1091500], [381210], [582010]],
)

if __name__ == "__main__":
    demo.launch()
