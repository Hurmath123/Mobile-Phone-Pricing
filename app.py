import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Set custom background (optional)
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
    .stNumberInput > div > input,
    .stSlider > div {{
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

# Set background
set_background("background.webp")

# Load model components
scaler = joblib.load("scaler.pkl")
model = joblib.load("stacking_model.pkl")
features = joblib.load("features.pkl")

important_features = ['ram', 'battery_power', 'px_width', 'px_height', 'mobile_wt', 'm_dep', 'four_g']

name_map = {
    'ram': ("RAM (MB)", "Random Access Memory - higher means better multitasking"),
    'battery_power': ("Battery Power (mAh)", "Total battery capacity"),
    'px_width': ("Pixel Width", "Horizontal screen resolution"),
    'px_height': ("Pixel Height", "Vertical screen resolution"),
    'mobile_wt': ("Weight (g)", "Weight of the phone in grams"),
    'm_dep': ("Mobile Depth (cm)", "Thickness of the device"),
    'four_g': ("4G Support", "Whether the phone supports 4G connectivity")
}

binary_features = ['four_g']

presets = {
    "Low": {'ram': 719.5, 'battery_power': 1066.0, 'px_width': 1132.5, 'px_height': 465.5, 'mobile_wt': 142.0, 'm_dep': 0.5, 'four_g': 1.0},
    "Medium": {'ram': 1686.5, 'battery_power': 1206.0, 'px_width': 1223.0, 'px_height': 606.0, 'mobile_wt': 141.0, 'm_dep': 0.5, 'four_g': 1.0},
    "High": {'ram': 2577.0, 'battery_power': 1219.5, 'px_width': 1221.5, 'px_height': 538.5, 'mobile_wt': 145.0, 'm_dep': 0.5, 'four_g': 0.0},
    "Very High": {'ram': 3509.5, 'battery_power': 1449.5, 'px_width': 1415.5, 'px_height': 674.0, 'mobile_wt': 134.0, 'm_dep': 0.5, 'four_g': 1.0}
}

st.title("📱 Mobile Price Category Predictor")
st.markdown("### 🧮 Select a phone type or customize its specifications below:")
st.divider()

selected_category = st.selectbox("📂 Choose a predefined phone type", list(presets.keys()))
defaults = presets[selected_category]

user_input = {}
with st.container():
    st.markdown("### 📋 Input Specifications")
    st.markdown("Use the sliders and fields below to adjust your configuration.")
    with st.form("input_form"):
        col1, col2, col3 = st.columns([1, 1, 1])
        for i, feat in enumerate(important_features):
            label, help_text = name_map.get(feat, (feat.replace('_', ' ').capitalize(), ""))
            with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
                if feat in binary_features:
                    user_input[feat] = st.radio(label, [0, 1], index=1 if defaults.get(feat, 0) else 0,
                                                format_func=lambda x: "Yes" if x else "No", key=feat, help=help_text)
                elif feat == 'ram':
                    user_input[feat] = st.slider(label, min_value=256, max_value=8000,
                                                 value=int(defaults.get(feat, 0)), step=256, key=feat, help=help_text)
                elif feat == 'battery_power':
                    user_input[feat] = st.slider(label, min_value=800, max_value=5000,
                                                 value=int(defaults.get(feat, 0)), step=100, key=feat, help=help_text)
                elif feat == 'mobile_wt':
                    user_input[feat] = st.slider(label, min_value=100, max_value=250,
                                                 value=int(defaults.get(feat, 0)), step=5, key=feat, help=help_text)
                else:
                    user_input[feat] = st.number_input(label, min_value=0.0, value=float(defaults.get(feat, 0)),
                                                      step=1.0, format="%.2f" if feat == 'm_dep' else "%.0f",
                                                      key=feat, help=help_text)
        submitted = st.form_submit_button("🔍 Predict Price Range", use_container_width=True)

if submitted:
    try:
        st.markdown("---")
        st.subheader("📈 Prediction Breakdown")
        if user_input['ram'] <= 0:
            st.warning("⚠️ RAM must be greater than 0 MB.")
        else:
            input_df = pd.DataFrame([[user_input.get(col, 0) for col in features]], columns=features)
            input_scaled = scaler.transform(input_df)
            proba = model.predict_proba(input_scaled)[0]
            prediction = int(np.argmax(proba))

            label_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
            emoji_map = {0: "💰", 1: "📱", 2: "💎", 3: "🛍️"}

            st.success(f"🎯 Predicted Price Category: {label_map[prediction]} {emoji_map[prediction]}")
            st.info(f"🔢 Prediction Confidence: {proba[prediction]*100:.2f}%")

            # Visualization of all class probabilities
            class_labels = list(label_map.values())
            st.bar_chart(pd.DataFrame({'Confidence': proba}, index=class_labels))

            # Optional radar chart comparing user input to Very High preset
            st.markdown("### 📊 Feature Profile Comparison")
            import plotly.graph_objects as go
            compare_features = important_features[:-1]  # exclude 'four_g' (binary)
            user_vals = [user_input[feat] for feat in compare_features]
            high_vals = [presets['Very High'][feat] for feat in compare_features]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=user_vals, theta=compare_features, fill='toself', name='Your Input'))
            fig.add_trace(go.Scatterpolar(r=high_vals, theta=compare_features, fill='toself', name='Very High Profile'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error("Prediction failed. Please check the inputs or model files.")
        st.text(str(e))

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("**Accuracy**: 98%\n\n**Model**: Stacking Classifier")
    st.markdown("ℹ️ Optimized for ease of use — select a phone type or adjust specs manually.")
