# src/app/gradio_app.py

import json
from pathlib import Path
import re
import pandas as pd
import gradio as gr
import joblib
import requests
import numpy as np

from src.data.fetch_storefront import get_appdetails
from src.data.fetch_steamspy import get_steamspy
from src.features.engineer import engineer_features

# ---------- Paths ----------
AAA_MODEL_PATH = Path("data/models/ensemble_aaa.pkl")
AAA_META_PATH  = Path("data/models/meta_aaa.json")

NICHE_MODEL_PATH = Path("data/models/ensemble_niche.pkl")
NICHE_META_PATH  = Path("data/models/meta_niche.json")

# Backward-compatible fallbacks (earlier versions)
ENSEMBLE_FALLBACKS = [Path("data/models/ensemble_v050.pkl"),
                      Path("data/models/ensemble_v041.pkl")]
META_FALLBACKS     = [Path("data/models/meta_v050.json"),
                      Path("data/models/meta_v041.json")]

LEGACY_MODEL = Path("data/models/xgb_hit_predictor.pkl")
LEGACY_META  = Path("data/models/model_meta.json")

DATA_PATH = Path("data/features/base_dataset.csv")


# ---------- Utilities ----------
def _load_model_and_meta(primary_model: Path, primary_meta: Path,
                         fallbacks_model=None, fallbacks_meta=None,
                         legacy_model=None, legacy_meta=None):
    """Load model + meta with fallbacks; return (model, threshold, feature_cols)."""
    def _first_existing(paths):
        if not paths:
            return None
        for p in paths:
            if p and Path(p).exists():
                return Path(p)
        return None

    mpath = primary_model if primary_model and primary_model.exists() else _first_existing(fallbacks_model)
    meta  = primary_meta if primary_meta and primary_meta.exists() else _first_existing(fallbacks_meta)

    if mpath is None and legacy_model and legacy_model.exists():
        mpath = legacy_model
        meta = legacy_meta if legacy_meta and legacy_meta.exists() else None

    if mpath is None:
        return None, 0.5, None

    model = joblib.load(mpath)

    threshold = 0.5
    feature_cols = None
    if meta and meta.exists():
        try:
            j = json.loads(meta.read_text())
            threshold = float(j.get("threshold", 0.5))
            feature_cols = j.get("feature_columns", None)
        except Exception:
            pass

    # Prefer model.feature_names_in_
    try:
        if isinstance(model, dict) and "xgb" in model and hasattr(model["xgb"], "feature_names_in_"):
            feature_cols = [str(c) for c in model["xgb"].feature_names_in_]
        elif isinstance(model, dict) and "rf" in model and hasattr(model["rf"], "feature_names_in_") and feature_cols is None:
            feature_cols = [str(c) for c in model["rf"].feature_names_in_]
        elif hasattr(model, "feature_names_in_"):
            feature_cols = [str(c) for c in model.feature_names_in_]
    except Exception:
        pass

    return model, threshold, feature_cols


def _steam_search_appid(query: str):
    """Un-official storefront search: name -> appid."""
    try:
        r = requests.get(
            "https://store.steampowered.com/api/storesearch/",
            params={"term": query, "l": "english", "cc": "US"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json() or {}
        items = data.get("items", [])
        if not items:
            return None
        return int(items[0].get("id"))
    except Exception:
        return None


def _name_to_appid_local(query: str):
    """Local fuzzy match from dataset."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if "name" not in df.columns or "appid" not in df.columns:
        return None
    m = df[df["name"].astype(str).str.contains(str(query), case=False, na=False)]
    if m.empty:
        return None
    if "owners_est" in m.columns:
        m = m.sort_values("owners_est", ascending=False)
    return int(m.iloc[0]["appid"])


def _steamspy_defaults():
    """Defaults when SteamSpy is missing."""
    return {
        "owners": "0 .. 0",
        "players_forever": 0,
        "average_forever": 0,
        "median_forever": 0,
        "positive": 0,
        "negative": 0,
        "tags": [],
    }


def _align_features(X: pd.DataFrame, feature_cols):
    """Ensure columns and order match the model’s training columns."""
    X = X.copy()
    X.columns = X.columns.astype(str)
    if feature_cols:
        cols = [str(c) for c in feature_cols]
        for c in cols:
            if c not in X.columns:
                X[c] = 0
        X = X[cols]
    else:
        X = X.reindex(sorted(X.columns.astype(str)), axis=1)
    return X


def _predict_with_model(model, X):
    """Predict for a single-row DataFrame."""
    if isinstance(model, dict):
        p = 0.0
        if "xgb" in model:
            p += 0.7 * model["xgb"].predict_proba(X)[:, 1]
        if "rf" in model:
            p += 0.3 * model["rf"].predict_proba(X)[:, 1]
        return float(p[0])
    else:
        return float(model.predict_proba(X)[:, 1][0])


def _predict_batch(model, X):
    """Predict probabilities for a batch DataFrame -> np.ndarray shape (n,)."""
    if isinstance(model, dict):
        p = np.zeros(len(X))
        if "xgb" in model:
            p += 0.7 * model["xgb"].predict_proba(X)[:, 1]
        if "rf" in model:
            p += 0.3 * model["rf"].predict_proba(X)[:, 1]
        return p
    else:
        return model.predict_proba(X)[:, 1]


def _parse_year(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"(20\d{2}|19\d{2})", s)
    return int(m.group(1)) if m else None


# ---------- Load two model variants ----------
AAA_MODEL, AAA_THRESH, AAA_FEATURES = _load_model_and_meta(
    primary_model=AAA_MODEL_PATH, primary_meta=AAA_META_PATH,
    fallbacks_model=ENSEMBLE_FALLBACKS, fallbacks_meta=META_FALLBACKS,
    legacy_model=LEGACY_MODEL, legacy_meta=LEGACY_META
)

NICHE_MODEL, NICHE_THRESH, NICHE_FEATURES = _load_model_and_meta(
    primary_model=NICHE_MODEL_PATH, primary_meta=NICHE_META_PATH,
    fallbacks_model=None, fallbacks_meta=None,
    legacy_model=None, legacy_meta=None
)

AAA_MODE_READY = AAA_MODEL is not None
NICHE_MODE_READY = NICHE_MODEL is not None


# ---------- Core prediction ----------
def predict_hit(game_identifier, mode_choice):
    """
    mode_choice: 'AAA / Established' or 'New / Niche (rising)'
    """
    try:
        q = str(game_identifier).strip()
        if not q:
            return "Please enter a game name or Steam AppID."

        # Resolve to AppID
        if q.isdigit():
            appid = int(q)
            resolved_by_search = False
        else:
            appid = _name_to_appid_local(q)
            resolved_by_search = False
            if appid is None:
                appid = _steam_search_appid(q)
                resolved_by_search = appid is not None
            if appid is None:
                return f"No game found named **{q}** (not in local dataset and no online match)."

        # Store data
        store = get_appdetails(appid)
        if not store:
            return f"Failed to fetch store data for AppID **{appid}**."

        if mode_choice.startswith("AAA"):
            spy = get_steamspy(appid) or _steamspy_defaults()
            used_defaults = spy is not None and spy.get("owners") == "0 .. 0"

            df = pd.DataFrame([{**store, **spy}])
            X, _ = engineer_features(df, return_target=False)
            X = _align_features(X, AAA_FEATURES)
            if AAA_MODEL is None:
                return "AAA model not available. Please train it first."
            prob = _predict_with_model(AAA_MODEL, X)
            threshold = AAA_THRESH

            name = store.get("name", f"AppID {appid}")
            is_hit = prob >= float(threshold)
            lines = [
                f"### {name} (AAA / Established)",
                f"- Predicted hit probability: **{prob:.2%}**",
                f"- Threshold: {float(threshold):.3f}",
                f"- Verdict: {'Hit' if is_hit else 'Not hit'}",
            ]
            if resolved_by_search:
                lines.append("_Resolved game by online name search._")
            if used_defaults:
                lines.append("_SteamSpy missing; used defaults. Less certain._")
            if "release_date" in store:
                lines.append(f"- Release date: {store['release_date']}")
            if "price_final" in store:
                lines.append(f"- Price: ${store['price_final']}")
            return "\n".join(lines)

        else:
            # Niche: ignore SteamSpy (fill zeros)
            spy = _steamspy_defaults()
            df = pd.DataFrame([{**store, **spy}])
            X, _ = engineer_features(df, return_target=False)
            X = _align_features(X, NICHE_FEATURES if NICHE_MODEL else AAA_FEATURES)

            model = NICHE_MODEL if NICHE_MODEL else AAA_MODEL
            threshold = NICHE_THRESH if NICHE_MODEL else (AAA_THRESH if AAA_MODEL else 0.5)
            if model is None:
                return "No trained model available. Please train and save a model first."

            prob = _predict_with_model(model, X)
            is_hit = prob >= float(threshold)
            name = store.get("name", f"AppID {appid}")
            lines = [
                f"### {name} (New / Niche)",
                f"- Predicted popularity score: **{prob:.2%}**",
                f"- Threshold: {float(threshold):.3f}",
                f"- Verdict: {'Rising / likely hit' if is_hit else 'Not likely (yet)'}",
            ]
            if resolved_by_search:
                lines.append("_Resolved game by online name search._")
            if "release_date" in store:
                lines.append(f"- Release date: {store['release_date']}")
            if "price_final" in store:
                lines.append(f"- Price: ${store['price_final']}")
            lines.append("_Note: Niche mode ignores SteamSpy engagement; prediction based on metadata only._")
            return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


def get_leaderboard(top_n=10):
    """AAA leaderboard by owners_est from local dataset."""
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["name", "owners_est", "price_final", "release_date"])
    df = pd.read_csv(DATA_PATH)
    keep = ["name", "owners_est", "price_final", "release_date"]
    keep = [c for c in keep if c in df.columns]
    if "owners_est" in df.columns:
        df = df.sort_values("owners_est", ascending=False)
    return df[keep].head(top_n)


def get_niche_leaderboard(top_n=10, years_back=5, max_owners=200_000, allow_free=True):
    """
    Rank niche games by predicted popularity (using the niche model if available).

    Filters:
      - released within last `years_back` years
      - owners_est <= max_owners (if owners_est exists). IMPORTANT: rows with NaN owners_est are KEPT.
      - include F2P if allow_free=True
    """
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["name", "predicted_popularity", "owners_est", "release_year", "price_final"])

    df = pd.read_csv(DATA_PATH).copy()
    if df.empty:
        return pd.DataFrame(columns=["name", "predicted_popularity", "owners_est", "release_year", "price_final"])

    # derive release_year and filter by years_back
    df["release_year"] = df.get("release_date", "").apply(_parse_year)
    current_year = pd.Timestamp.now().year
    df = df[df["release_year"].fillna(0) >= (current_year - years_back)]

    # owners cap: KEEP NaNs (unknown owners) and only cap known owners
    if "owners_est" in df.columns and max_owners is not None:
        df["owners_est"] = pd.to_numeric(df["owners_est"], errors="coerce")
        known_mask = df["owners_est"].notna()
        cap_mask = ~known_mask | (df["owners_est"] <= max_owners)
        df = df[cap_mask]

    if not allow_free and "is_free" in df.columns:
        df = df[df["is_free"] == False]

    if df.empty:
        return pd.DataFrame(columns=["name", "predicted_popularity", "owners_est", "release_year", "price_final"])

    # Build features and predict in batch
    X, _ = engineer_features(df, return_target=False)
    feature_cols = NICHE_FEATURES if NICHE_MODEL is not None else (AAA_FEATURES if AAA_MODEL is not None else None)
    if feature_cols is None:
        return pd.DataFrame(columns=["name", "predicted_popularity", "owners_est", "release_year", "price_final"])

    X = _align_features(X, feature_cols)
    model = NICHE_MODEL if NICHE_MODEL is not None else AAA_MODEL
    if model is None:
        return pd.DataFrame(columns=["name", "predicted_popularity", "owners_est", "release_year", "price_final"])

    probs = _predict_batch(model, X)

    out = pd.DataFrame({
        "name": df.get("name", pd.Series(["Unknown"] * len(df))),
        "predicted_popularity": probs,
        "owners_est": df.get("owners_est", pd.Series([np.nan] * len(df))),
        "release_year": df["release_year"],
        "price_final": df.get("price_final", pd.Series([np.nan] * len(df))),
    }).sort_values("predicted_popularity", ascending=False).head(top_n).reset_index(drop=True)

    # show probability as percent
    out["predicted_popularity"] = (out["predicted_popularity"] * 100).round(2)
    return out


# ---------- UI ----------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Steam Hit Predictor\nPredict which games are likely to become hits.")

    with gr.Tabs():
        with gr.Tab("Predict Game"):
            gr.Markdown("Select the game type and enter a Steam AppID or name.")
            with gr.Row():
                mode = gr.Radio(
                    ["New / Niche (rising)", "AAA / Established"],
                    value="New / Niche (rising)",
                    label="Game type"
                )
            with gr.Row():
                query = gr.Textbox(label="Game name or AppID", placeholder="e.g., 1091500 or Waterpark Simulator")
                btn = gr.Button("Predict", variant="primary")
            out = gr.Markdown()
            btn.click(predict_hit, inputs=[query, mode], outputs=out)

            # Model availability hint
            hint = []
            if not AAA_MODE_READY:
                hint.append("- AAA model not found; using fallback if available.")
            if not NICHE_MODE_READY:
                hint.append("- Niche model not found; using AAA model as fallback.")
            if hint:
                gr.Markdown("\n".join(hint))

        with gr.Tab("Top Games"):
            gr.Markdown("Top games by owners (AAA) and by predicted popularity (Niche):")
            with gr.Row():
                table_aaa = gr.Dataframe(get_leaderboard(), interactive=False, wrap=True, label="AAA Leaderboard (by owners_est)")

            with gr.Accordion("Niche Leaderboard Filters", open=False):
                years_in = gr.Slider(1, 10, value=5, step=1, label="Years back")
                owners_in = gr.Number(value=200_000, label="Max owners (set -1 for no cap)")
                free_in = gr.Checkbox(value=True, label="Include Free-to-Play")
                topn_in = gr.Slider(5, 50, value=10, step=1, label="Top N")

            table_niche = gr.Dataframe(interactive=False, wrap=True, label="Niche Leaderboard (by predicted popularity %)")

            def _niche_lb_cb(y, o, f, n):
                cap = None if (o is None or (isinstance(o, (int, float)) and o < 0)) else int(o)
                return get_niche_leaderboard(top_n=int(n), years_back=int(y), max_owners=cap, allow_free=bool(f))

            for ctl in (years_in, owners_in, free_in, topn_in):
                ctl.change(_niche_lb_cb, inputs=[years_in, owners_in, free_in, topn_in], outputs=table_niche)

            # initialize once
            table_niche.value = get_niche_leaderboard()

        with gr.Tab("About"):
            gr.Markdown(
                f"""
**Models available:**  
- AAA: {"yes" if AAA_MODE_READY else "no"}  
- Niche: {"yes" if NICHE_MODE_READY else "no"}  

This app supports two modes:
- **AAA / Established:** uses engagement + metadata (SteamSpy + Storefront).
- **New / Niche:** ignores SteamSpy and predicts from metadata only.

To get the best results, train and save two models:
- `data/models/ensemble_aaa.pkl`, `data/models/meta_aaa.json`
- `data/models/ensemble_niche.pkl`, `data/models/meta_niche.json`
                """
            )

if __name__ == "__main__":
    demo.launch()
