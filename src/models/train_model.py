import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from joblib import dump
from src.features.engineer import engineer_features


def train():
    df = pd.read_csv("data/features/base_dataset.csv")
    y = df["hit_label"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros(len(y))

    # class imbalance
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = max(1.0, neg / max(1, pos))
    print(f"scale_pos_weight = {scale_pos_weight:.2f} (neg={neg}, pos={pos})")

    models = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), 1):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # feature engineering within fold
        X_tr, _ = engineer_features(df_tr, return_target=False)
        X_va, _ = engineer_features(df_va, return_target=False)
        X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

        # ---- XGBoost ----
        xgb_model = xgb.XGBClassifier(
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.2,  # L2
            reg_alpha=0.3,   # L1
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=fold
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        prob_xgb = xgb_model.predict_proba(X_va)[:, 1]

        # ---- Random Forest ----
        rf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            n_jobs=-1,
            random_state=fold
        )
        rf_model.fit(X_tr, y_tr)
        prob_rf = rf_model.predict_proba(X_va)[:, 1]

        # ---- Ensemble prediction ----
        prob_avg = (prob_xgb * 0.7 + prob_rf * 0.3)
        probs[va_idx] = prob_avg

        roc = roc_auc_score(y_va, prob_avg)
        pr = average_precision_score(y_va, prob_avg)
        print(f"Fold {fold}: ROC-AUC={roc:.3f} | PR-AUC={pr:.3f}")

        models.append({"xgb": xgb_model, "rf": rf_model})

    # ----- Final Evaluation -----
    roc_all = roc_auc_score(y, probs)
    pr_all = average_precision_score(y, probs)
    print(f"\nCV mean ROC-AUC: {roc_all:.3f} | CV mean PR-AUC: {pr_all:.3f}")

    # threshold tuning
    prec, rec, thr = precision_recall_curve(y, probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_thr = thr[np.nanargmax(f1)]
    print(f"Chosen threshold (OOF, F1+): {best_thr:.3f}\n")

    y_pred = (probs >= best_thr).astype(int)
    print("OOF classification report:")
    print(classification_report(y, y_pred, digits=3))

    # save
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(models[-1], model_dir / "ensemble_v050.pkl")
    pd.Series({"threshold": float(best_thr)}).to_json(model_dir / "meta_v050.json")
    print("\nEnsemble model + meta saved → data/models/")


if __name__ == "__main__":
    train()
