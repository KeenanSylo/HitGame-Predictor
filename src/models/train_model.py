import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from pathlib import Path
from src.features.engineer import engineer_features

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train():
    df = pd.read_csv("data/features/base_dataset.csv")
    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)
    print(f"ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds > 0.5))

    joblib.dump(model, MODEL_DIR / "xgb_hit_predictor.pkl")
    print("Model saved → data/models/xgb_hit_predictor.pkl")

if __name__ == "__main__":
    train()
