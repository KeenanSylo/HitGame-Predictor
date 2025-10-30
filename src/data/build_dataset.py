import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def build_dataset(storefront_csv, steamspy_csv):
    store = pd.read_csv(storefront_csv)
    spy = pd.read_csv(steamspy_csv)
    df = pd.merge(store, spy, on="appid", how="inner")

    # parse owners string "200,000 .. 500,000"
    def parse_midpoint(x):
        try:
            lo, hi = x.split("..")
            lo = int(lo.replace(",", ""))
            hi = int(hi.replace(",", ""))
            return (lo + hi) / 2
        except Exception:
            return None
    df["owners_est"] = df["owners"].apply(parse_midpoint)

    # label: hit if owners >= 200k
    df["hit_label"] = (df["owners_est"] >= 200_000).astype(int)

    df.to_csv(FEATURE_DIR / "base_dataset.csv", index=False)
    return df

if __name__ == "__main__":
    build_dataset("data/raw/storefront_games.csv", "data/raw/steamspy_games.csv")
