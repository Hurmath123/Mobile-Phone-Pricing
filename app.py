import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import base64
from streamlit_lottie import st_lottie

# ---- BACKGROUND SETUP ----
def set_background(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        .stTextInput > div > input,
        .stNumberInput > div > input,
        .stSelectbox > div > div {{
            background-color: rgba(0,0,0,0.2);
            color: white;
        }}
        .stButton > button {{
            background-color: #00bfff;
            color: white;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# ---- LOAD ANIMATIONS ----
def load_lottiefile(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

# ---- LOAD ASSETS ----
set_background("background.webp")
prediction_anim = load_lottiefile("predict_anim.json")
success_anim = load_lottiefile("success_anim.json")

# ---- LOAD MODEL & SCALER ----

scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
model = joblib.load("stacking_model.pkl")

# ---- UI HEADER ----
if prediction_anim:
    st_lottie(prediction_anim, height=180, key="header")
st.title("📱 Mobile Price Category Predictor")
st.markdown("Enter mobile phone specifications to predict its **price category** based on real-world market data.")

# ---- USER INPUT ----
user_input = {}
for feat in features:
    # Improve UX: provide defaults, ranges, and hints if possible
    if "ram" in feat:
        user_input[feat] = st.slider("RAM (MB)", 256, 12000, 4000, step=256)
    elif "battery" in feat:
        user_input[feat] = st.slider("Battery Power (mAh)", 500, 7000, 3000, step=100)
    elif "weight" in feat:
        user_input[feat] = st.slider("Phone Weight (grams)", 80, 250, 150)
    else:
        user_input[feat] = st.number_input(f"{feat.replace('_', ' ').capitalize()}", min_value=0, value=1)

# ---- PREDICTION ----
if st.button("🔍 Predict Price Range"):
    try:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Convert numeric category to readable label
        price_labels = {
            0: "Low",
            1: "Medium",
            2: "High",
            3: "Very High"
        }

        st.success(f"📊 Predicted Price Category: **{price_labels.get(prediction, 'Unknown')}**")
        if success_anim:
            st_lottie(success_anim, height=150, key="success")

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check input values.")
        st.text(str(e))
