import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def _to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        # stored as "['Action','RPG']" in CSV sometimes
        if s.startswith("[") and s.endswith("]"):
            try:
                return eval(s)
            except Exception:
                return []
    return []

def engineer_features(df: pd.DataFrame, return_target: bool = True):
    df = df.copy()

    # normalize list-like columns
    for col in ["genres", "categories", "tags"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # one-hot tags (fit on the given df; at inference we will reindex later)
    mlb = MultiLabelBinarizer()
    tag_features = pd.DataFrame(
        mlb.fit_transform(df["tags"]),
        columns=[f"tag_{t}" for t in mlb.classes_],
        index=df.index,
    )
    df = pd.concat([df, tag_features], axis=1)

    # simple numeric features
    df["discount_norm"] = df.get("discount_percent", 0).fillna(0) / 100
    df["price_norm"] = df.get("price_final", 0).fillna(0)
    pos = pd.to_numeric(df.get("positive", 0), errors="coerce").fillna(0)
    neg = pd.to_numeric(df.get("negative", 0), errors="coerce").fillna(0)
    df["review_ratio"] = pos / (pos + neg + 1)

    num_cols = [
        "discount_norm",
        "price_norm",
        "review_ratio",
        "players_forever",
        "average_forever",
    ]
    base = [c for c in num_cols if c in df.columns]
    tags = [c for c in df.columns if c.startswith("tag_")]

    X = df[base + tags].fillna(0)

    y = None
    if return_target and "hit_label" in df.columns:
        y = df["hit_label"]

    return X, y
