import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Use joblib to load the model
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
# ---- SMARTER USER INPUT ----
st.subheader("📋 Phone Specifications")

dropdown_features = {
    "dual_sim": ["No", "Yes"],
    "three_g": ["No", "Yes"],
    "four_g": ["No", "Yes"],
    "touch_screen": ["No", "Yes"],
    "wifi": ["No", "Yes"],
    "blue": ["No", "Yes"]
}

slider_features = {
    "ram": (512, 12000, 2048, 256),
    "battery_power": (1000, 7000, 3000, 100),
    "px_height": (0, 2000, 960, 10),
    "px_width": (0, 2000, 1280, 10),
    "mobile_wt": (80, 250, 150, 1),
    "int_memory": (4, 256, 32, 4),
    "talk_time": (2, 24, 12, 1),
    "clock_speed": (0.5, 3.0, 1.5, 0.1),
    "fc": (0, 20, 5, 1),
    "pc": (0, 30, 12, 1),
    "n_cores": (1, 8, 4, 1),
    "sc_h": (2, 20, 12, 1),
    "sc_w": (2, 10, 6, 1)
}

user_input = {}

for feat in features:
    if feat in dropdown_features:
        val = st.selectbox(f"{feat.replace('_', ' ').title()}", dropdown_features[feat])
        user_input[feat] = 1 if val == "Yes" else 0
    elif feat in slider_features:
        min_val, max_val, default, step = slider_features[feat]
        user_input[feat] = st.slider(f"{feat.replace('_', ' ').title()}", min_val, max_val, default, step=step)
    else:
        user_input[feat] = st.number_input(f"{feat.replace('_', ' ').title()}", min_value=0, value=1)


# ---- PREDICTION ----
if st.button("🔍 Predict Price Range"):
    try:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

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
