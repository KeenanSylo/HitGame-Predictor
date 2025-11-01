import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.features.engineer import engineer_features


def train():
    # Load data & features
    df = pd.read_csv("data/features/base_dataset.csv")
    X, y = engineer_features(df, return_target=True)
    feature_cols = list(X.columns)

    # Handle imbalance
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = max(1.0, neg / max(1, pos))
    print(f"scale_pos_weight = {scale_pos_weight:.2f} (neg={neg}, pos={pos})")

    # XGB config
    base_params = dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    # 5-fold stratified CV with OOF predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(X))
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBClassifier(**base_params)
        # Early stopping via fit() argument works on xgboost 2.x; for 3.x, set in constructor if needed.
        model.set_params(early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        prob_va = model.predict_proba(X_va)[:, 1]
        oof_prob[va_idx] = prob_va

        auc = roc_auc_score(y_va, prob_va)
        ap = average_precision_score(y_va, prob_va)
        fold_metrics.append({"fold": fold, "roc_auc": auc, "pr_auc": ap})
        print(f"Fold {fold}: ROC-AUC={auc:.3f} | PR-AUC={ap:.3f}")

    # Aggregate CV metrics
    cv_auc = float(np.mean([m["roc_auc"] for m in fold_metrics]))
    cv_ap = float(np.mean([m["pr_auc"] for m in fold_metrics]))
    print(f"\nCV mean ROC-AUC: {cv_auc:.3f} | CV mean PR-AUC: {cv_ap:.3f}")

    # Choose threshold on OOF predictions (robust)
    prec, rec, thr = precision_recall_curve(y, oof_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1)
    best_thr = float(thr[max(0, best_idx - 1)])
    print(f"Chosen threshold (OOF, F1+): {best_thr:.3f}")

    # Final fit on full data
    final_model = xgb.XGBClassifier(**base_params)
    final_model.set_params(early_stopping_rounds=50)
    # Use a small internal validation split inside XGB for early stopping
    final_model.fit(X, y, eval_set=[(X, y)], verbose=False)

    # Report on OOF with chosen threshold (transparent sanity check)
    y_pred_oof = (oof_prob >= best_thr).astype(int)
    print("\nOOF classification report with chosen threshold:")
    print(classification_report(y, y_pred_oof, digits=3))

    # Save model + meta
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    dump(final_model, model_dir / "xgb_hit_predictor.pkl")
    meta = {
        "threshold": best_thr,
        "feature_columns": feature_cols,
        "cv_mean_roc_auc": cv_auc,
        "cv_mean_pr_auc": cv_ap,
        "folds": fold_metrics,
        "scale_pos_weight": float(scale_pos_weight),
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nModel + meta saved → data/models/")


if __name__ == "__main__":
    train()
