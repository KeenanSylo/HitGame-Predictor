import re, math, datetime
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
    """Extract release year (e.g., 2019) from date text."""
    if not isinstance(date_str, str):
        return None
    m = re.search(r"(20\\d{2}|19\\d{2})", date_str)
    return int(m.group(1)) if m else None


def engineer_features(df: pd.DataFrame, return_target: bool = True):
    """
    Converts raw Steam game data into ML-ready numeric/categorical features.
    Designed for both training (return_target=True) and inference (False).
    """
    df = df.copy()

    # normalize list-like columns (safe defaults)
    for col in ["genres", "categories", "tags"]:
        df[col] = df[col].apply(_to_list) if col in df.columns else [[] for _ in range(len(df))]

    # one-hot encode tags
    mlb = MultiLabelBinarizer()
    tag_feats = pd.DataFrame(
        mlb.fit_transform(df["tags"]),
        columns=[f"tag_{t}" for t in mlb.classes_],
        index=df.index,
    )
    df = pd.concat([df, tag_feats], axis=1)

    # ---- Frequency encoding for genres/categories ----
    def freq_encode(col_name):
        flat = [g for lst in df[col_name] for g in lst]
        freq = pd.Series(flat).value_counts(normalize=True)
        return df[col_name].apply(lambda lst: sum(freq.get(x, 0) for x in lst) / (len(lst) or 1))

    if "genres" in df.columns:
        df["genre_score"] = freq_encode("genres")
        df["genre_count"] = df["genres"].apply(len)
    if "categories" in df.columns:
        df["category_score"] = freq_encode("categories")
        df["category_count"] = df["categories"].apply(len)

    # ---- Base numeric features ----
    df["discount_norm"] = df.get("discount_percent", 0).fillna(0).astype(float) / 100.0
    df["price_norm"] = df.get("price_final", 0).fillna(0).astype(float)

    pos = pd.to_numeric(df.get("positive", 0), errors="coerce").fillna(0)
    neg = pd.to_numeric(df.get("negative", 0), errors="coerce").fillna(0)
    df["review_ratio"] = pos / (pos + neg + 1.0)

    df["num_tags"] = df["tags"].apply(len)
    df["has_discount"] = (df.get("discount_percent", 0).fillna(0).astype(float) > 0).astype(int)
    df["release_year"] = df.get("release_date", "").apply(_parse_year).fillna(0).astype(int)

    # ---- New v0.5 derived features ----
    df["desc_len"] = df.get("short_description", "").fillna("").astype(str).str.len()
    current_year = datetime.datetime.now().year
    df["release_age"] = (current_year - df["release_year"]).clip(lower=0)
    df["price_per_hour"] = df["price_norm"] / (pd.to_numeric(df.get("average_forever", 0), errors="coerce") / 60 + 1)

    # ---- Assemble feature matrix ----
    numeric_cols = [
        "discount_norm", "price_norm", "review_ratio",
        "players_forever", "average_forever",
        "num_tags", "has_discount", "release_year", "release_age",
        "genre_score", "category_score", "genre_count", "category_count",
        "desc_len", "price_per_hour"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    tag_cols = [c for c in df.columns if c.startswith("tag_")]

    # drop rare tags (<3%)
    tag_freq = (df[tag_cols].sum() / len(df)) if tag_cols else pd.Series()
    keep_tags = list(tag_freq[tag_freq > 0.03].index) if not tag_freq.empty else []
    X = df[numeric_cols + keep_tags].fillna(0)

    # ensure unique feature names
    X = X.loc[:, ~X.columns.duplicated()]

    y = df["hit_label"].astype(int) if return_target and "hit_label" in df.columns else None
    return X, y
