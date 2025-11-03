import os
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

STORE_FEATURES_URL = "https://store.steampowered.com/api/featuredcategories"
CONCURRENT_URL = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"

def fetch_featuredcategories(cc="US", lang="english"):
    r = requests.get(STORE_FEATURES_URL, params={"cc": cc, "l": lang}, timeout=15)
    r.raise_for_status()
    return r.json()

def extract_items(payload, key="new_releases"):
    """
    key can be: 'new_releases', 'specials', 'topsellers', 'coming_soon', etc.
    """
    block = (payload or {}).get(key, {}) or {}
    items = block.get("items", []) or []
    out = []
    for it in items:
        out.append({
            "appid": it.get("id"),
            "name": it.get("name"),
            "release_date": it.get("release_date", ""),
            "final_price": (it.get("final_price") or 0) / 100.0,
            "discount_percent": it.get("discount_percent", 0),
        })
    # Remove rows without appid just in case
    return [x for x in out if x.get("appid")]

def get_concurrent_players(appid, api_key=None):
    params = {"appid": appid}
    if api_key:
        params["key"] = api_key
    r = requests.get(CONCURRENT_URL, params=params, timeout=15)
    if r.status_code != 200:
        return None
    try:
        return int(((r.json() or {}).get("response") or {}).get("player_count") or 0)
    except Exception:
        return None

def build_new_release_concurrents(
    out_csv=RAW_DIR / "live_new_concurrent.csv",
    cc="US",
    lang="english",
    section="new_releases",      # or 'topsellers', 'coming_soon'
    sleep_sec=0.2
):
    api_key = os.getenv("STEAM_API_KEY", "").strip() or None

    feat = fetch_featuredcategories(cc=cc, lang=lang)
    rows = extract_items(feat, key=section)
    if not rows:
        print(f"No items found in section '{section}'.")
        pd.DataFrame([], columns=[
            "appid","name","release_date","final_price","discount_percent","concurrent"
        ]).to_csv(out_csv, index=False)
        return out_csv

    enriched = []
    for r in tqdm(rows, desc=f"Concurrents for {section}"):
        c = get_concurrent_players(r["appid"], api_key=api_key)
        time.sleep(sleep_sec)
        enriched.append({
            **r,
            "concurrent": c if c is not None else 0
        })

    df = pd.DataFrame(enriched)
    df = df.sort_values("concurrent", ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv} ({len(df)} rows)")
    return out_csv

if __name__ == "__main__":
    # Example: build for New Releases (default)
    build_new_release_concurrents()
