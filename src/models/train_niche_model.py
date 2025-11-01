import re
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from joblib import dump

from src.features.engineer import engineer_features


def _parse_year(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"(20\d{2}|19\d{2})", text)
    return int(m.group(1)) if m else None


def _owners_series(df):
    """Return numeric owners_est if present; else None."""
    if "owners_est" in df.columns:
        return pd.to_numeric(df["owners_est"], errors="coerce")
    return None


def _players_series(df):
    """Return numeric players_forever if present; else None."""
    if "players_forever" in df.columns:
        return pd.to_numeric(df["players_forever"], errors="coerce")
    return None


def _reviews_series(df):
    """Return numeric total reviews (pos+neg) if present; else None."""
    pos = pd.to_numeric(df.get("positive"), errors="coerce") if "positive" in df.columns else None
    neg = pd.to_numeric(df.get("negative"), errors="coerce") if "negative" in df.columns else None
    if pos is not None and neg is not None:
        return (pos.fillna(0) + neg.fillna(0))
    return None


def build_niche_subset(df, years_back, max_owners, allow_free, verbose=True):
    """Filter to 'niche' space with configurable knobs."""
    df = df.copy()

    # derive release_year
    df["release_year"] = df.get("release_date", "").apply(_parse_year)
    current_year = datetime.now().year
    df = df[df["release_year"].fillna(0) >= (current_year - years_back)]

    # apply owners cap if provided
    if max_owners is not None:
        owners = _owners_series(df)
        if owners is not None:
            df = df[owners.fillna(np.inf) <= max_owners]
        else:
            if verbose:
                print("Warning: owners_est missing; owners cap can't be applied.")

    # optionally exclude F2P
    if not allow_free and "is_free" in df.columns:
        df = df[df["is_free"] == False]

    return df.reset_index(drop=True)


def make_percentile_labels(signal: pd.Series, target_pos_min=25, start_percentile=0.85):
    """
    Make positive labels by percentile within the niche subset on the given 'signal'.
    signal: numeric pd.Series (owners_est, players_forever, or total_reviews)
    """
    s = pd.to_numeric(signal, errors="coerce").fillna(0).values.astype(float)
    for p in [start_percentile, 0.80, 0.75, 0.70, 0.65, 0.60]:
        thr = np.nanpercentile(s, p * 100.0)
        y = (s >= thr).astype(int)
        if y.sum() >= target_pos_min:
            return y, thr, p
    # fallback: ensure at least target_pos_min positives
    k = max(target_pos_min, max(5, int(len(s) * 0.1)))
    idx_sorted = np.argsort(s)
    thr = s[idx_sorted][-k] if len(s) >= k else s.min()
    y = (s >= thr).astype(int)
    return y, thr, None


def train_niche(
    data_path="data/features/base_dataset.csv",
    model_out="data/models/ensemble_niche.pkl",
    meta_out="data/models/meta_niche.json",
    years_back=2,
    max_owners=100_000,
    start_percentile=0.85,
    target_pos_min=25,
    allow_free=False,
    min_rows=80,
    n_splits=5,
):
    # 1) Load raw merged dataset
    df_full = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df_full)} rows")

    # Interpret max_owners = -1 as "no cap"
    max_cap = None if (max_owners is None or (isinstance(max_owners, int) and max_owners < 0)) else max_owners

    # 2) Try progressively relaxed filters until we have enough rows
    tries = []
    tries.append(dict(years_back=years_back, max_owners=max_cap, allow_free=allow_free))
    if not allow_free:
        tries.append(dict(years_back=years_back, max_owners=max_cap, allow_free=True))
    tries.append(dict(years_back=max(years_back, 5), max_owners=max_cap, allow_free=True))
    tries.append(dict(years_back=max(years_back, 5), max_owners=None, allow_free=True))

    niche = None
    chosen = None
    for opt in tries:
        cand = build_niche_subset(df_full, **opt)
        owners = _owners_series(cand)
        # require at least some rows with owners_est or other signals
        if len(cand) >= min_rows:
            niche = cand
            chosen = opt
            print(f"Selected niche filter: {opt} → {len(cand)} rows")
            break
        else:
            print(f"Filter {opt} → {len(cand)} rows (too small)")

    if niche is None:
        raise ValueError(
            "Niche subset still too small after relaxing filters. "
            "Consider increasing years_back, removing owners cap, or allowing F2P."
        )

    # 3) Choose the best available signal for labels
    owners = _owners_series(niche)
    players = _players_series(niche)
    reviews = _reviews_series(niche)

    label_source = None
    signal = None
    if owners is not None and owners.notna().sum() >= min_rows // 2:
        label_source = "owners_est"
        signal = owners
    elif players is not None and players.notna().sum() >= min_rows // 2:
        label_source = "players_forever"
        signal = players
    elif reviews is not None and reviews.notna().sum() >= min_rows // 2:
        label_source = "reviews_total"
        signal = reviews
    else:
        raise ValueError("No sufficient label signal (owners_est/players/reviews) in niche subset.")

    print(f"Label source: {label_source}")
    y, thr_val, used_p = make_percentile_labels(signal, target_pos_min=target_pos_min, start_percentile=start_percentile)
    niche = niche.copy()
    niche["rising_label"] = y

    # 4) Cross-validated training with feature engineering inside each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs_oof = np.zeros(len(niche))

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = max(1.0, neg / max(1, pos))
    print(f"subset size = {len(niche)} | positives = {pos}, negatives = {neg}")
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    fold_models = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(niche, y), 1):
        df_tr, df_va = niche.iloc[tr_idx], niche.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Engineer features within fold (no leakage)
        X_tr, _ = engineer_features(df_tr, return_target=False)
        X_va, _ = engineer_features(df_va, return_target=False)

        # Align columns
        X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

        # Models
        xgb_model = xgb.XGBClassifier(
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.2,
            reg_alpha=0.3,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=fold,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        prob_xgb = xgb_model.predict_proba(X_va)[:, 1]

        rf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            n_jobs=-1,
            random_state=fold,
        )
        rf_model.fit(X_tr, y_tr)
        prob_rf = rf_model.predict_proba(X_va)[:, 1]

        prob_avg = (0.7 * prob_xgb + 0.3 * prob_rf)
        probs_oof[va_idx] = prob_avg

        roc = roc_auc_score(y_va, prob_avg)
        pr = average_precision_score(y_va, prob_avg)
        print(f"Fold {fold}: ROC-AUC={roc:.3f} | PR-AUC={pr:.3f}")

        fold_models.append({"xgb": xgb_model, "rf": rf_model})

    # 5) Aggregate metrics and choose threshold on OOF
    roc_all = roc_auc_score(y, probs_oof)
    pr_all = average_precision_score(y, probs_oof)
    print(f"\nCV mean ROC-AUC: {roc_all:.3f} | CV mean PR-AUC: {pr_all:.3f}")

    prec, rec, thr = precision_recall_curve(y, probs_oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_thr = thr[np.nanargmax(f1)]
    print(f"Chosen threshold (OOF F1+): {best_thr:.3f}\n")

    y_pred = (probs_oof >= best_thr).astype(int)
    print("OOF classification report:")
    print(classification_report(y, y_pred, digits=3))

    # 6) Save last fold models + meta (includes feature columns for the UI aligner)
    model_dir = Path(model_out).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = None
    try:
        feature_columns = list(fold_models[-1]["xgb"].feature_names_in_)
    except Exception:
        pass

    dump(fold_models[-1], model_out)

    meta = {
        "threshold": float(best_thr),
        "feature_columns": feature_columns,
        "filter_chosen": chosen,
        "label_source": label_source,
        "owners_threshold_value": float(thr_val) if thr_val is not None and not math.isnan(thr_val) else None,
        "owners_percentile_used": float(used_p) if used_p is not None else None,
        "subset_size": int(len(niche)),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "cv_roc_auc_oof": float(roc_all),
        "cv_pr_auc_oof": float(pr_all),
        "note": "Rising label defined by percentile within relaxed niche subset; proxy used if owners missing.",
    }
    Path(meta_out).write_text(json.dumps(meta, indent=2))
    print(f"\nNiche model + meta saved → {model_out} and {meta_out}")


def main():
    ap = argparse.ArgumentParser(description="Train niche-mode rising popularity model")
    ap.add_argument("--data", default="data/features/base_dataset.csv")
    ap.add_argument("--out-model", default="data/models/ensemble_niche.pkl")
    ap.add_argument("--out-meta", default="data/models/meta_niche.json")
    ap.add_argument("--years-back", type=int, default=2,
                    help="Only include games released in the last N years (default: 2)")
    ap.add_argument("--max-owners", type=int, default=100_000,
                    help="Upper bound for owners_est; -1 means no cap (default: 100000)")
    ap.add_argument("--start-percentile", type=float, default=0.85,
                    help="Initial percentile for positive label within subset (default: 0.85)")
    ap.add_argument("--target-pos-min", type=int, default=25,
                    help="Relax percentile until at least this many positives (default: 25)")
    ap.add_argument("--allow-free", action="store_true",
                    help="Include free-to-play titles in the subset")
    ap.add_argument("--min-rows", type=int, default=80,
                    help="Minimum rows required for training after filtering (default: 80)")
    ap.add_argument("--folds", type=int, default=5)

    args = ap.parse_args()
    train_niche(
        data_path=args.data,
        model_out=args.out_model,
        meta_out=args.out_meta,
        years_back=args.years_back,
        max_owners=args.max_owners,
        start_percentile=args.start_percentile,
        target_pos_min=args.target_pos_min,
        allow_free=args.allow_free,
        min_rows=args.min_rows,
        n_splits=args.folds,
    )


if __name__ == "__main__":
    main()
