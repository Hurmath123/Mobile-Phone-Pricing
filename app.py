import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model assets ---
@st.cache_resource
def load_assets():
    model = joblib.load("stacking_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    target_labels = joblib.load("target_labels.pkl")
    return model, scaler, features, target_labels

model, scaler, features, target_labels = load_assets()

# --- Config ---
st.set_page_config(page_title="📱 Mobile Price Predictor", layout="centered")

# --- Custom Style ---
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f6f9fc;
}
h1, h2, h3 {
    color: #1f3b4d;
}
.sidebar .sidebar-content {
    background: #ffffff;
}
.stButton > button {
    background-color: #0066cc;
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
}
.stButton > button:hover {
    background-color: #005bb5;
    transform: scale(1.01);
}
.info-card {
    background-color: #e9f2ff;
    padding: 1rem;
    border-left: 6px solid #0066cc;
    border-radius: 8px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("Mobile Price Predictor")
st.caption("Instantly predict mobile price category based on tech specs")

# --- Tabs for input ---
tab1, tab2 = st.tabs(["Predict", "About"])

# --- Prediction Tab ---
with tab1:
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        binary_map = {"No": 0, "Yes": 1}

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
            four_g = st.selectbox("4G Support", ["Yes", "No"])
            three_g = st.selectbox("3G Support", ["Yes", "No"])
            touch_screen = st.selectbox("Touch Screen", ["Yes", "No"])
            wifi = st.selectbox("WiFi", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
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

            df_input = pd.DataFrame([input_data])
            scaled_input = scaler.transform(df_input[features])
            pred_class = model.predict(scaled_input)[0]
            pred_proba = model.predict_proba(scaled_input)[0][pred_class]

            label_map = {
                0: "Low (Below ₹5,000)",
                1: "Medium (₹5,000 - ₹10,000)",
                2: "High (₹10,000 - ₹15,000)",
                3: "Very High (Above ₹15,000)"
            }

            st.markdown(f"""
                <div class='info-card'>
                    <h4>{label_map[pred_class]}</h4>
                    <p><strong>Confidence:</strong> {pred_proba:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- About Tab ---
with tab2:
    st.subheader("About This App")
    st.write("""
        This app uses a machine learning model trained on mobile specifications to predict 
        the likely **price range** of a phone. It considers various features such as RAM, 
        battery, screen size, and connectivity options like 4G/3G/WiFi.
        
        **Model used:** Stacking Classifier (ensemble of multiple models)  
        **Accuracy:** ~90% on validation data  
        
        Built by Frough Hurmath S using Streamlit.
    """)
