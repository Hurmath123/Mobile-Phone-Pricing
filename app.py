import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
    .stSelectbox > div > div {{
        background-color: rgba(0,0,0,0.3);
        color: white;
    }}
    .stButton > button {{
        background-color: #00bfff;
        color: white;
        border: none;
        border-radius: 5px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load Lottie animations from file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load assets
set_background("background.webp")
prediction_anim = load_lottiefile("predict_anim.json")
success_anim = load_lottiefile("success_anim.json")

# Load model components (adjust file paths as needed)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("features.pkl", "rb") as f:
    features = pickle.load(f)
with open("stacking_model.pkl", "rb") as f:
    model = pickle.load(f)

st_lottie(prediction_anim, height=200, key="predict_anim")
st.title("📱 Mobile Price Category Predictor")

# Input fields
user_input = {}
for feat in features:
    user_input[feat] = st.number_input(f"{feat.replace('_', ' ').capitalize()}", min_value=0)

if st.button("Predict Price Range"):
    try:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"📊 Predicted Price Category: **{prediction}**")
        st_lottie(success_anim, height=150, key="success")
    except Exception as e:
        st.error("Prediction failed. Please check inputs.")
        st.text(str(e))
