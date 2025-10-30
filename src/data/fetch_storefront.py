import requests, time, json, os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_appdetails(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return None
    payload = r.json().get(str(appid), {})
    if not payload.get("success"):
        return None
    data = payload["data"]
    out = {
        "appid": appid,
        "name": data.get("name"),
        "release_date": data.get("release_date", {}).get("date"),
        "is_free": data.get("is_free"),
        "price_initial": (data.get("price_overview") or {}).get("initial", 0) / 100,
        "price_final": (data.get("price_overview") or {}).get("final", 0) / 100,
        "discount_percent": (data.get("price_overview") or {}).get("discount_percent", 0),
        "genres": [g["description"] for g in data.get("genres", [])],
        "categories": [c["description"] for c in data.get("categories", [])],
        "short_description": data.get("short_description"),
    }
    return out


def fetch_many(appids, out_csv="data/raw/storefront_games.csv"):
    records = []
    for appid in tqdm(appids, desc="Fetching Storefront"):
        rec = get_appdetails(appid)
        if rec:
            records.append(rec)
        time.sleep(0.2)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    appids_df = pd.read_csv("data/raw/appids.csv")
    appids = appids_df["appid"].astype(int).tolist()

    out_path = "data/raw/storefront_games.csv"
    if Path(out_path).exists():
        existing = set(pd.read_csv(out_path)["appid"])
        appids = [a for a in appids if a not in existing]
        print(f"Resuming: {len(appids)} remaining...")

    fetch_many(appids, out_csv=out_path)