import pandas as pd
import requests
from pathlib import Path
import time

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def from_steamspy(out_csv="data/raw/appids.csv", top_n=2000):
    """Fetch top games from SteamSpy and save as appids.csv"""
    print("Fetching app list from SteamSpy...")
    r = requests.get("https://steamspy.com/api.php", params={"request": "all"}, timeout=60)
    r.raise_for_status()
    j = r.json()

    def midpoint(owners_str):
        try:
            lo, hi = owners_str.split("..")
            lo, hi = int(lo.replace(",", "")), int(hi.replace(",", ""))
            return (lo + hi) / 2
        except:
            return 0

    rows = []
    for appid, info in j.items():
        rows.append({
            "appid": int(appid),
            "name": info.get("name"),
            "owners_mid": midpoint(info.get("owners", "0..0")),
        })

    df = pd.DataFrame(rows).sort_values("owners_mid", ascending=False).head(top_n)
    df[["appid"]].to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} appids → {out_csv}")
    return out_csv

if __name__ == "__main__":
    from_steamspy(top_n=2000)
