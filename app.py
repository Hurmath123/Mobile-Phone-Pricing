import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import base64
from streamlit_lottie import st_lottie

# Set custom background

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }}
    .stTextInput > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > input {{
        background-color: rgba(0,0,0,0.3);
        color: white;
    }}
    .stButton > button {{
        background-color: #00bfff;
        color: white;
        border-radius: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load Lottie animation

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Set background
set_background("background.webp")

# Load animations and model components
predict_anim = load_lottiefile("predict_anim.json")
success_anim = load_lottiefile("success_anim.json")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
model = joblib.load("stacking_model.pkl")

# Use only top 5 features (based on your selection or model importance)
important_features = ['ram', 'battery_power', 'px_width', 'px_height', 'mobile_wt']
binary_features = ['wifi', 'four_g']  # Not used since they are not in top 5

# App UI
st_lottie(predict_anim, height=180, key="predict")
st.title("Mobile Price Category Predictor")
st.markdown("Enter mobile specifications to estimate its price category.")

# Input form
user_input = {}
with st.form("input_form"):
    for feat in important_features:
        label = feat.replace('_', ' ').capitalize()
        user_input[feat] = st.number_input(label, min_value=0)
    submitted = st.form_submit_button("Predict Price Range")

# Make prediction
if submitted:
    try:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        label_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

        st.success(f"Predicted Price Category: {label_map[prediction]} ({prediction})")
        st_lottie(success_anim, height=150, key="success_done")
    except Exception as e:
        st.error("Prediction failed. Please check the inputs or model files.")
        st.text(str(e))
