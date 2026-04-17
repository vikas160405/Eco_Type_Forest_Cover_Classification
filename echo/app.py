# =============================================================================
# EcoType: Forest Cover Type Prediction - Streamlit App 
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoType - Forest Cover Predictor",
    page_icon="🌲",
    layout="wide"
)

# ── Cover Type Mapping ────────────────────────────────────────────────────────
COVER_TYPE_NAMES = {
    1: "🌲 Spruce/Fir",
    2: "🌲 Lodgepole Pine",
    3: "🌲 Ponderosa Pine",
    4: "🌿 Cottonwood/Willow",
    5: "🍂 Aspen",
    6: "🌲 Douglas-fir",
    7: "🏔️ Krummholz"
}

COVER_TYPE_DESC = {
    1: "Dominant in high-altitude cold zones.",
    2: "Most common cover type.",
    3: "Lower elevations, warmer conditions.",
    4: "Near water bodies.",
    5: "Transition zones.",
    6: "Medium-elevation forests.",
    7: "Highest elevation vegetation."
}

MODEL_DIR = "models"

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/top_features.pkl", "rb") as f:
        top_features = pickle.load(f)
    with open(f"{MODEL_DIR}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, top_features, encoders


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a5c2a, #2d8f4e); 
     padding: 30px; border-radius: 15px; margin-bottom: 25px; text-align:center;'>
    <h1 style='color:white;'>🌲 EcoType</h1>
    <p style='color:#c8f0d4;'>Forest Cover Type Prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
try:
    model, scaler, top_features, encoders = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error("Model not found. Run ecotype_pipeline.py first.")
    model_loaded = False

# ── Inputs ────────────────────────────────────────────────────────────────────
st.header("📋 Enter Location Details")

col1, col2, col3 = st.columns(3)

with col1:
    elevation = st.number_input("Elevation", 0, 4000, 2500)
    aspect = st.number_input("Aspect", 0, 360, 150)
    slope = st.number_input("Slope", 0, 90, 15)

with col2:
    h_dist_hydro = st.number_input("Horiz Dist Hydrology", 0, 2000, 300)
    v_dist_hydro = st.number_input("Vert Dist Hydrology", -500, 500, 50)
    h_dist_roads = st.number_input("Horiz Dist Roads", 0, 7000, 1500)
    h_dist_fire = st.number_input("Horiz Dist Fire", 0, 7000, 1200)

with col3:
    hillshade_9am = st.slider("Hillshade 9AM", 0, 255, 200)
    hillshade_noon = st.slider("Hillshade Noon", 0, 255, 230)
    hillshade_3pm = st.slider("Hillshade 3PM", 0, 255, 150)

    wilderness_area = st.selectbox(
        "Wilderness Area",
        ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]
    )

    soil_type = st.selectbox("Soil Type", [str(i) for i in range(1, 41)])

# ── Feature Engineering ───────────────────────────────────────────────────────
distance_to_hydro = np.sqrt(h_dist_hydro**2 + v_dist_hydro**2)
mean_hillshade = (hillshade_9am + hillshade_noon + hillshade_3pm) / 3
hydro_road_diff = h_dist_hydro - h_dist_roads

# ── Encode categorical (FIXED) ────────────────────────────────────────────────
try:
    wilderness_encoded = encoders['Wilderness_Area'].transform([wilderness_area])[0]
    soil_encoded = encoders['Soil_Type'].transform([soil_type])[0]
except:
    wilderness_encoded = 0
    soil_encoded = 0

# ── Input Data ────────────────────────────────────────────────────────────────
input_data = {
    "Elevation": elevation,
    "Aspect": aspect,
    "Slope": slope,
    "Horizontal_Distance_To_Hydrology": h_dist_hydro,
    "Vertical_Distance_To_Hydrology": v_dist_hydro,
    "Horizontal_Distance_To_Roadways": h_dist_roads,
    "Hillshade_9am": hillshade_9am,
    "Hillshade_Noon": hillshade_noon,
    "Hillshade_3pm": hillshade_3pm,
    "Horizontal_Distance_To_Fire_Points": h_dist_fire,
    "Wilderness_Area": wilderness_encoded,
    "Soil_Type": soil_encoded,
    "Distance_To_Hydrology": distance_to_hydro,
    "Mean_Hillshade": mean_hillshade,
    "Hydro_Road_Distance_Diff": hydro_road_diff
}

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict"):

    if not model_loaded:
        st.error("Model not loaded")
    else:
        try:
            input_df = pd.DataFrame([input_data])

            # Ensure all required features exist
            for col in top_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[top_features]

            # 🔥 Apply log transform (FIXED)
            for col in input_df.columns:
                input_df[col] = np.log1p(input_df[col] - input_df[col].min())

            # Scale
            input_scaled = scaler.transform(input_df)

            # Predict
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]

            st.success(f"🌲 Prediction: {COVER_TYPE_NAMES[pred]}")

            st.subheader("Confidence:")
            for i, val in enumerate(proba):
                st.write(f"{COVER_TYPE_NAMES[i+1]}: {val*100:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")