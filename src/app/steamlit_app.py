import streamlit as st
import pandas as pd
from src.data.fetch_storefront import get_appdetails
from src.data.fetch_steamspy import get_steamspy
from src.models.predict import predict_new

st.title("🎮 Steam Hit Predictor")

appid = st.text_input("Enter Steam AppID (e.g. 1091500 for Cyberpunk 2077):")
if st.button("Predict"):
    if appid:
        appid = int(appid)
        store = get_appdetails(appid)
        spy = get_steamspy(appid)
        if not store or not spy:
            st.error("Could not fetch data. Try another appid.")
        else:
            df = pd.DataFrame([{**store, **spy}])
            preds = predict_new(df)
            prob = float(preds.iloc[0]["pred_hit_prob"])
            st.success(f"Predicted probability of being a hit: {prob:.2%}")
