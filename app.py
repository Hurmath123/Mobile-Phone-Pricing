import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import json
import requests

# --- Load model and assets ---
@st.cache_resource
def load_assets():
    model = joblib.load("stacking_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    target_labels = joblib.load("target_labels.pkl")
    return model, scaler, features, target_labels

model, scaler, features, target_labels = load_assets()

# --- Lottie loader ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

prediction_anim = load_lottieurl("https://lottie.host/497fe43e-7005-4f92-b42b-fc90c0c8ef29/YjGPBwwS9K.json")
success_anim = load_lottieurl("https://lottie.host/3e452d1a-a395-487c-bb12-e4642a556a3c/ZcKoz0UItb.json")

# --- Page config ---
st.set_page_config(page_title="Mobile Price Predictor", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f2f6fa;
    }
    .main {
        background: linear-gradient(135deg, #e3f2fd, #ffffff);
        border-radius: 15px;
        padding: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #d0f0c0, #e6ffe6);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        border-left: 6px solid #4CAF50;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.6em 1.2em;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st_lottie(prediction_anim, height=200, key="predict_anim")
st.title("Mobile Price Predictor")
st.markdown("Predict the price range of a mobile phone based on its features.")

# --- Theme toggle ---
theme = st.toggle("Dark Mode", False)

# --- Form ---
with st.form("form"):
    st.subheader("Phone Specifications")
    col1, col2 = st.columns(2)
    binary_map = {"Yes": 1, "No": 0}

    with col1:
        battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1000)
        blue = st.selectbox("Bluetooth", ["Yes", "No"])
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
        dual_sim = st.selectbox("Dual SIM", ["Yes", "No"])
        fc = st.slider("Front Camera (MP)", 0, 20, 5)
        int_memory = st.slider("Internal Memory (GB)", 2, 128, 32)
        m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5)
        mobile_wt = st.slider("Weight (gm)", 80, 250, 150)
        n_cores = st.slider("Processor Cores", 1, 8, 4)

    with col2:
        pc = st.slider("Primary Camera (MP)", 0, 30, 10)
        px_height = st.slider("Pixel Height", 0, 2000, 1000)
        px_width = st.slider("Pixel Width", 0, 2000, 1000)
        ram = st.slider("RAM (MB)", 256, 8192, 2048)
        sc_h = st.slider("Screen Height (cm)", 5, 20, 10)
        sc_w = st.slider("Screen Width (cm)", 2, 10, 5)
        talk_time = st.slider("Talk Time (hrs)", 2, 20, 10)
        four_g = st.selectbox("4G", ["Yes", "No"])
        three_g = st.selectbox("3G", ["Yes", "No"])
        touch_screen = st.selectbox("Touch Screen", ["Yes", "No"])
        wifi = st.selectbox("WiFi", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'battery_power': battery_power,
        'blue': binary_map[blue],
        'clock_speed': clock_speed,
        'dual_sim': binary_map[dual_sim],
        'fc': fc,
        'four_g': binary_map[four_g],
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': binary_map[three_g],
        'touch_screen': binary_map[touch_screen],
        'wifi': binary_map[wifi],
    }

    df = pd.DataFrame([input_data])
    scaled = scaler.transform(df[features])
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred]

    label_map = {
        0: "Low (Below ₹5,000)",
        1: "Medium (₹5,000 - ₹10,000)",
        2: "High (₹10,000 - ₹15,000)",
        3: "Very High (Above ₹15,000)"
    }

    st_lottie(success_anim, height=150, key="success")
    st.markdown(f"""
        <div class="result-card">
            <h4>{label_map[pred]}</h4>
            <p><strong>Confidence:</strong> {prob:.2%}</p>
        </div>
    """, unsafe_allow_html=True)
