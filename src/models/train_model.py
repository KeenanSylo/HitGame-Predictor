import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_recall_curve, average_precision_score
)
from joblib import dump
from src.features.engineer import engineer_features


def train():
    # Load dataset
    df = pd.read_csv("data/features/base_dataset.csv")

    # Feature engineering
    X, y = engineer_features(df)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = max(1.0, neg / max(1, pos))
    print(f"scale_pos_weight = {scale_pos_weight:.2f} (neg={neg}, pos={pos})")

    # Model setup
    model = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",     # focus on PR-AUC for imbalance
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    # Evaluate
    prob_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob_val)
    ap = average_precision_score(y_val, prob_val)
    print(f"\nROC-AUC: {auc:.3f} | PR-AUC: {ap:.3f}")

    # Choose best probability threshold for positive class (max F1)
    prec, rec, thr = precision_recall_curve(y_val, prob_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1)
    best_thr = thr[max(0, best_idx - 1)]
    print(f"Chosen threshold: {best_thr:.3f} (F1+ for positives)\n")

    # Final validation report
    y_pred = (prob_val >= best_thr).astype(int)
    print(classification_report(y_val, y_pred, digits=3))

    # Save model and threshold
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, model_dir / "xgb_hit_predictor.pkl")
    pd.Series({"threshold": float(best_thr)}).to_json(model_dir / "model_meta.json")
    print("\nModel + threshold saved → data/models/")


if __name__ == "__main__":
    train()
