import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, features, and labels
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("features.pkl")
label_map = joblib.load("target_labels.pkl")

st.set_page_config(page_title="Mobile Price Predictor", layout="centered")
st.title("📱 Mobile Price Range Predictor")
st.markdown("Enter mobile specifications to predict the price category.")

# Input UI
input_data = {}

input_data["battery_power"] = st.slider("Battery Power (mAh)", 500, 2000, 1000)
input_data["blue"] = st.selectbox("Bluetooth", [0, 1], format_func=lambda x: "Yes" if x else "No")
input_data["clock_speed"] = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
input_data["dual_sim"] = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: "Yes" if x else "No")
input_data["fc"] = st.slider("Front Camera (MP)", 0, 20, 5)
input_data["four_g"] = st.selectbox("4G Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
input_data["int_memory"] = st.slider("Internal Memory (GB)", 2, 128, 32)
input_data["m_deep"] = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.01)
input_data["mobile_wt"] = st.slider("Mobile Weight (grams)", 80, 250, 150)
input_data["n_cores"] = st.slider("Number of Cores", 1, 8, 4)
input_data["pc"] = st.slider("Primary Camera (MP)", 0, 20, 10)
input_data["px_height"] = st.slider("Pixel Height", 0, 1960, 800)
input_data["px_width"] = st.slider("Pixel Width", 0, 2000, 1200)
input_data["ram"] = st.slider("RAM (MB)", 256, 8000, 3000)
input_data["sc_h"] = st.slider("Screen Height (cm)", 5, 20, 12)
input_data["sc_w"] = st.slider("Screen Width (cm)", 0, 20, 5)
input_data["talk_time"] = st.slider("Talk Time (hours)", 2, 20, 10)
input_data["three_g"] = st.selectbox("3G Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
input_data["touch_screen"] = st.selectbox("Touch Screen", [0, 1], format_func=lambda x: "Yes" if x else "No")
input_data["wifi"] = st.selectbox("WiFi Support", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Predict
if st.button("Predict Price Range"):
    # Convert input to dataframe and scale
    input_df = pd.DataFrame([input_data])[feature_list]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    label = label_map[prediction]
    
    st.success(f"📊 Predicted Price Range: **{label.upper()}**")

