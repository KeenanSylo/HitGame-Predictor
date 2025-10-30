import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def engineer_features(df):
    # convert tags/genres to one-hots
    for col in ["genres", "categories", "tags"]:
        df[col] = df[col].fillna("").apply(
            lambda x: x if isinstance(x, list) else eval(x) if x.startswith("[") else []
        )
    mlb = MultiLabelBinarizer()
    tag_features = pd.DataFrame(
        mlb.fit_transform(df["tags"]), columns=[f"tag_{t}" for t in mlb.classes_]
    )
    df = pd.concat([df.reset_index(drop=True), tag_features], axis=1)

    df["discount_norm"] = df["discount_percent"].fillna(0) / 100
    df["price_norm"] = df["price_final"].fillna(0)
    df["review_ratio"] = df["positive"] / (df["positive"] + df["negative"] + 1)
    num_cols = ["discount_norm", "price_norm", "review_ratio", "players_forever", "average_forever"]
    X = df[num_cols + [c for c in df.columns if c.startswith("tag_")]]
    y = df["hit_label"]
    return X, y
