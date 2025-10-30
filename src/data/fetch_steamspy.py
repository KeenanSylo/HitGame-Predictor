import requests, time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def get_steamspy(appid):
    r = requests.get("https://steamspy.com/api.php",
                     params={"request": "appdetails", "appid": appid},
                     timeout=15)
    if r.status_code != 200:
        return None
    j = r.json()
    return {
        "appid": appid,
        "owners": j.get("owners"),
        "players_forever": j.get("players_forever"),
        "average_forever": j.get("average_forever"),
        "median_forever": j.get("median_forever"),
        "positive": j.get("positive"),
        "negative": j.get("negative"),
        "tags": list((j.get("tags") or {}).keys())
    }

def fetch_many(appids, out_csv="data/raw/steamspy_games.csv"):
    records = []
    for appid in tqdm(appids, desc="Fetching SteamSpy"):
        rec = get_steamspy(appid)
        if rec:
            records.append(rec)
        time.sleep(0.2)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    appids_df = pd.read_csv("data/raw/appids.csv")
    appids = appids_df["appid"].astype(int).tolist()
    fetch_many(appids)
