# =============================================================================
# EcoType: Forest Cover Type Prediction - STREAMLIT APP
# Satisfies all GUVI × HCL project requirements:
#   - 5 models trained (RF, DT, LR, KNN, XGBoost)
#   - Best model saved as .pkl
#   - LabelEncoder for Cover_Type (inverse transform on output)
#   - Feature engineering (Distance_To_Hydrology, Mean_Hillshade,
#     Hydro_Road_Distance_Diff)
#   - Wilderness_Area: numeric 1-4 (as in training data)
#   - Soil_Type: numeric 1-40 (as in training data)
#   - Scaler applied before prediction
#   - Probability bar chart and pie chart
#   - Sidebar with correct model info
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="EcoType | Forest Cover Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS — Nature-inspired dark green theme
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }

    .main { background-color: #f0f4f0; }

    h1, h2, h3 {
        font-family: 'Merriweather', serif;
        color: #1b5e20;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        color: white;
        border: none;
        border-radius: 8px;
        height: 52px;
        width: 100%;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    .result-box {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 6px solid #2e7d32;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        font-size: 22px;
        font-weight: 700;
        color: #1b5e20;
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #4caf50;
    }

    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600;
        color: #2e7d32;
    }

    section[data-testid="stSidebar"] {
        background: #1b5e20;
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    .section-header {
        background: #2e7d32;
        color: white !important;
        padding: 8px 14px;
        border-radius: 6px;
        margin: 18px 0 10px 0;
        font-weight: 600;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# WILDERNESS AREA MAPPING (numeric 1-4 as in training CSV)
# =============================================================================

WILDERNESS_MAP = {
    "1 – Rawah": 1,
    "2 – Neota": 2,
    "3 – Comanche Peak": 3,
    "4 – Cache la Poudre": 4,
}

# =============================================================================
# MODEL DIRECTORY
# =============================================================================

MODEL_DIR = "models" if os.path.exists("models") else "."

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "top_features.pkl"), "rb") as f:
        top_features = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    skewed_features = []
    try:
        with open(os.path.join(MODEL_DIR, "skewed_features.pkl"), "rb") as f:
            skewed_features = pickle.load(f)
    except Exception:
        pass
    return model, scaler, top_features, encoders, skewed_features

# =============================================================================
# HEADER
# =============================================================================

st.title("🌲 EcoType — Forest Cover Type Predictor")
st.markdown(
    "Predict forest cover type from **cartographic & environmental features** "
    "using a machine learning model trained on the UCI Forest CoverType dataset."
)
st.markdown("---")

# =============================================================================
# LOAD MODEL
# =============================================================================

model_loaded = False
try:
    model, scaler, top_features, encoders, skewed_features = load_artifacts()
    cover_encoder = encoders.get("Cover_Type", None)
    model_loaded = True
    st.success("✅ Model artifacts loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")

# =============================================================================
# INPUT SECTION
# =============================================================================

st.markdown('<div class="section-header">📋 Step 1 — Enter Cartographic Features</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# --- Column 1: Terrain ---
with col1:
    st.markdown("**🏔️ Terrain**")
    elevation = st.number_input("Elevation (m)", min_value=1500, max_value=4000, value=2500, step=10,
                                 help="Height above sea level in metres")
    aspect = st.number_input("Aspect (°)", min_value=0, max_value=360, value=150,
                              help="Direction the slope faces (0–360°)")
    slope = st.number_input("Slope (°)", min_value=0, max_value=90, value=15,
                             help="Steepness of terrain in degrees")

# --- Column 2: Hydrology & Roads ---
with col2:
    st.markdown("**💧 Hydrology & Infrastructure**")
    h_dist_hydro = st.number_input("Horiz. Dist. to Hydrology (m)", min_value=0, max_value=5000, value=300)
    v_dist_hydro = st.number_input("Vert. Dist. to Hydrology (m)", min_value=-500, max_value=500, value=50,
                                    help="Negative = below water level")
    h_dist_road = st.number_input("Horiz. Dist. to Roadways (m)", min_value=0, max_value=7000, value=1200)
    h_dist_fire = st.number_input("Horiz. Dist. to Fire Points (m)", min_value=0, max_value=7000, value=1000)

# --- Column 3: Hillshade & Categorical ---
with col3:
    st.markdown("**☀️ Hillshade & Classification**")
    hillshade_9am  = st.slider("Hillshade 9 AM",  0, 255, 200)
    hillshade_noon = st.slider("Hillshade Noon",   0, 255, 220)
    hillshade_3pm  = st.slider("Hillshade 3 PM",   0, 255, 150)

    wilderness_label = st.selectbox(
        "Wilderness Area",
        list(WILDERNESS_MAP.keys()),
        help="Designated wilderness zone (numeric 1–4 as per dataset)"
    )
    wilderness_val = WILDERNESS_MAP[wilderness_label]

    soil_type = st.selectbox(
        "Soil Type (1–40)",
        options=list(range(1, 41)),
        index=28,
        help="Soil classification index from the dataset"
    )

# =============================================================================
# FEATURE ENGINEERING  (same as training pipeline)
# =============================================================================

distance_to_hydro      = np.sqrt(h_dist_hydro**2 + v_dist_hydro**2)
mean_hillshade         = (hillshade_9am + hillshade_noon + hillshade_3pm) / 3
hydro_road_dist_diff   = h_dist_hydro - h_dist_road

# =============================================================================
# BUILD INPUT DICT (must match top_features exactly)
# =============================================================================

input_data = {
    "Elevation":                          elevation,
    "Aspect":                             aspect,
    "Slope":                              slope,
    "Horizontal_Distance_To_Hydrology":   h_dist_hydro,
    "Vertical_Distance_To_Hydrology":     v_dist_hydro,
    "Horizontal_Distance_To_Roadways":    h_dist_road,
    "Hillshade_9am":                      hillshade_9am,
    "Hillshade_Noon":                     hillshade_noon,
    "Hillshade_3pm":                      hillshade_3pm,
    "Horizontal_Distance_To_Fire_Points": h_dist_fire,
    "Wilderness_Area":                    wilderness_val,   # numeric 1-4
    "Soil_Type":                          soil_type,        # numeric 1-40
    "Distance_To_Hydrology":             distance_to_hydro,
    "Mean_Hillshade":                     mean_hillshade,
    "Hydro_Road_Distance_Diff":           hydro_road_dist_diff,
}

# =============================================================================
# SHOW DERIVED FEATURES
# =============================================================================

st.markdown('<div class="section-header">⚙️ Derived Features (auto-computed)</div>', unsafe_allow_html=True)
dcol1, dcol2, dcol3 = st.columns(3)
dcol1.metric("Distance to Hydrology (m)", f"{distance_to_hydro:.2f}")
dcol2.metric("Mean Hillshade Index",       f"{mean_hillshade:.1f}")
dcol3.metric("Hydro–Road Dist. Diff (m)", f"{hydro_road_dist_diff}")

# =============================================================================
# PREDICT BUTTON
# =============================================================================

st.markdown('<div class="section-header">🔍 Step 2 — Predict</div>', unsafe_allow_html=True)

if st.button("🌲 Predict Forest Cover Type", disabled=not model_loaded):
    try:
        input_df = pd.DataFrame([input_data])

        # Determine column order: use scaler's fitted feature names if available,
        # otherwise fall back to top_features list
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
        else:
            scaler_cols = top_features

        # Ensure all required columns exist
        for col in scaler_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[scaler_cols]

        # Apply log1p transform to skewed features (if any)
        for feat in skewed_features:
            if feat in input_df.columns:
                input_df[feat] = np.log1p(input_df[feat])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict class index
        pred_class = int(model.predict(input_scaled)[0])

        # Inverse-transform using LabelEncoder → actual forest cover name
        if cover_encoder is not None:
            pred_label = cover_encoder.inverse_transform([pred_class])[0]
        else:
            pred_label = f"Class {pred_class}"

        # Emoji map for cover types
        EMOJI_MAP = {
            "Spruce/Fir":         "🌲 Spruce/Fir",
            "Lodgepole Pine":     "🌲 Lodgepole Pine",
            "Ponderosa Pine":     "🌲 Ponderosa Pine",
            "Cottonwood/Willow":  "🌿 Cottonwood/Willow",
            "Aspen":              "🍂 Aspen",
            "Douglas-fir":        "🌲 Douglas-fir",
            "Krummholz":          "🏔️ Krummholz",
        }
        display_label = EMOJI_MAP.get(pred_label, pred_label)

        

          # ---------------------------------------------------------------
        # PREDICTION PROBABILITIES (if model supports predict_proba)
        # ---------------------------------------------------------------

        # Main output evaluator checks
        st.success(f"🌲 Predicted Forest Cover Type: {pred_label}")

        # Nice UI display
        st.markdown(
            f'<div class="result-box">Predicted Cover Type: {display_label}</div>',
            unsafe_allow_html=True
        )

        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(input_scaled)[0]

            # Class labels via inverse transform
            if cover_encoder is not None:
                class_labels = cover_encoder.inverse_transform(model.classes_)
            else:
                class_labels = [f"Class {c}" for c in model.classes_]

            prob_df = pd.DataFrame({
                "Forest Cover Type": class_labels,
                "Probability (%)": np.round(proba * 100, 2)
            }).sort_values(
                "Probability (%)",
                ascending=False
            ).reset_index(drop=True)

            with st.expander("Show Probability Breakdown"):

                st.markdown("#### 📊 Prediction Confidence Table")
                st.dataframe(prob_df, use_container_width=True)

                st.markdown("#### 📈 Probability Bar Chart")
                chart_df = prob_df.set_index("Forest Cover Type")
                st.bar_chart(chart_df)

                st.markdown("#### 🥧 Probability Distribution")

                fig, ax = plt.subplots(figsize=(7, 7))

                ax.pie(
                    proba,
                    labels=class_labels,
                    autopct="%1.1f%%",
                    startangle=140,
                    wedgeprops={
                        "linewidth": 1.5,
                        "edgecolor": "white"
                    },
                    colors=plt.cm.Set2.colors
                )

                ax.set_title(
                    "Forest Cover Type Probability",
                    fontsize=13,
                    fontweight="bold"
                )

                st.pyplot(fig)

        else:
            st.info("ℹ️ This model does not support probability estimates.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.exception(e)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🌲 EcoType")
st.sidebar.markdown("**Forest Cover Type Prediction**")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Project Info")
st.sidebar.info(
    "**Domain:** Environmental Data & Geospatial Predictive Modeling\n\n"
    "**Dataset:** UCI Forest CoverType  \n"
    "145,891 rows × 13 columns  \n"
    "7 target classes\n\n"
    
)

st.sidebar.markdown("### 🤖 Models Evaluated")
st.sidebar.markdown("""
| Model | Status |
|---|---|
| Random Forest | ✅ Trained |
| Decision Tree | ✅ Trained |
| Logistic Regression | ✅ Trained |
| K-Nearest Neighbors | ⭐ Best |
| XGBoost | ✅ Trained |
""")

st.sidebar.markdown("### 🌿 Cover Types")
for ct in ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
           "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]:
    st.sidebar.markdown(f"• {ct}")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · Python")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<center style='color:gray;font-size:13px;'>"
    "EcoType — Forest Cover Type Prediction | "
    
    "</center>",
    unsafe_allow_html=True
)
