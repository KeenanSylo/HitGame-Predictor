import re
import math
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def _to_list(x):
    """Safely convert stringified lists or other types to Python lists."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return list(eval(s))
            except Exception:
                return []
    return []


def _parse_year(date_str):
    """Extract release year (e.g., '2019') from text like 'Aug 15, 2019'."""
    if not isinstance(date_str, str):
        return None
    m = re.search(r"(20\d{2}|19\d{2})", date_str)
    return int(m.group(1)) if m else None


def engineer_features(df: pd.DataFrame, return_target: bool = True):
    """
    Converts raw Steam game data into ML-ready numeric and categorical features.
    Works for both training (return_target=True) and inference (False).
    """
    df = df.copy()

    # Normalize list-like columns (safe defaults)
    for col in ["genres", "categories", "tags"]:
        df[col] = df[col].apply(_to_list) if col in df.columns else [[] for _ in range(len(df))]

    # One-hot encode tags
    mlb = MultiLabelBinarizer()
    tag_feats = pd.DataFrame(
        mlb.fit_transform(df["tags"]),
        columns=[f"tag_{t}" for t in mlb.classes_],
        index=df.index,
    )
    df = pd.concat([df, tag_feats], axis=1)

    # Base numeric features
    df["discount_norm"] = df.get("discount_percent", 0).fillna(0).astype(float) / 100.0
    df["price_norm"] = df.get("price_final", 0).fillna(0).astype(float)
    pos = pd.to_numeric(df.get("positive", 0), errors="coerce").fillna(0)
    neg = pd.to_numeric(df.get("negative", 0), errors="coerce").fillna(0)
    df["review_ratio"] = pos / (pos + neg + 1.0)

    # v0.3 feature enhancements
    df["num_tags"] = df["tags"].apply(len)
    df["has_discount"] = (df.get("discount_percent", 0).fillna(0).astype(float) > 0).astype(int)
    df["release_year"] = df.get("release_date", "").apply(_parse_year).fillna(0).astype(int)

    # Assemble feature matrix (no owners_est — prevents label leakage)
    numeric_cols = [
        "discount_norm", "price_norm", "review_ratio",
        "players_forever", "average_forever",
        "num_tags", "has_discount", "release_year",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    tag_cols = [c for c in df.columns if c.startswith("tag_")]

    X = df[numeric_cols + tag_cols].fillna(0)

    # Ensure unique feature names
    X = X.loc[:, ~X.columns.duplicated()]

    # Target label
    y = None
    if return_target and "hit_label" in df.columns:
        y = df["hit_label"].astype(int)

    return X, y
