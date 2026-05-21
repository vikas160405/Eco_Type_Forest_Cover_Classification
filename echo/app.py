# =============================================================================
# EcoType: Forest Cover Type Prediction - ADVANCED STREAMLIT APP
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="EcoType Forest Predictor",
    page_icon="🌲",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7f9;
}

h1 {
    color: #1b5e20;
    text-align: center;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# COVER TYPE LABELS
# =============================================================================

COVER_TYPE_NAMES = {
    1: "🌲 Spruce/Fir",
    2: "🌲 Lodgepole Pine",
    3: "🌲 Ponderosa Pine",
    4: "🌿 Cottonwood/Willow",
    5: "🍂 Aspen",
    6: "🌲 Douglas-fir",
    7: "🏔️ Krummholz"
}

# =============================================================================
# MODEL DIRECTORY
# =============================================================================

MODEL_DIR = "models"

# =============================================================================
# LOAD MODEL FILES
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

    with open(os.path.join(MODEL_DIR, "skewed_features.pkl"), "rb") as f:
        skewed_features = pickle.load(f)

    return (
        model,
        scaler,
        top_features,
        encoders,
        skewed_features
    )

# =============================================================================
# HEADER
# =============================================================================

st.title("🌲 EcoType Forest Cover Prediction")

st.markdown("""
### Predict Forest Cover Type using Machine Learning

This application predicts forest cover type based on
geographical and environmental features.
""")

# =============================================================================
# LOAD MODEL
# =============================================================================

try:

    (
        model,
        scaler,
        top_features,
        encoders,
        skewed_features
    ) = load_artifacts()

    model_loaded = True

except Exception as e:

    st.error("❌ Model files not found")

    st.write(e)

    model_loaded = False

# =============================================================================
# USER INPUTS
# =============================================================================

st.subheader("📋 Enter Input Features")

col1, col2, col3 = st.columns(3)

# =============================================================================
# COLUMN 1
# =============================================================================

with col1:

    elevation = st.number_input(
        "Elevation",
        min_value=0,
        max_value=5000,
        value=2500
    )

    aspect = st.number_input(
        "Aspect",
        min_value=0,
        max_value=360,
        value=150
    )

    slope = st.number_input(
        "Slope",
        min_value=0,
        max_value=90,
        value=15
    )

# =============================================================================
# COLUMN 2
# =============================================================================

with col2:

    h_dist_hydro = st.number_input(
        "Horizontal Distance To Hydrology",
        min_value=0,
        max_value=5000,
        value=300
    )

    v_dist_hydro = st.number_input(
        "Vertical Distance To Hydrology",
        min_value=-500,
        max_value=500,
        value=50
    )

    h_dist_road = st.number_input(
        "Horizontal Distance To Roadways",
        min_value=0,
        max_value=7000,
        value=1200
    )

    h_dist_fire = st.number_input(
        "Horizontal Distance To Fire Points",
        min_value=0,
        max_value=7000,
        value=1000
    )

# =============================================================================
# COLUMN 3
# =============================================================================

with col3:

    hillshade_9am = st.slider(
        "Hillshade 9AM",
        0,
        255,
        200
    )

    hillshade_noon = st.slider(
        "Hillshade Noon",
        0,
        255,
        220
    )

    hillshade_3pm = st.slider(
        "Hillshade 3PM",
        0,
        255,
        150
    )

    wilderness = st.selectbox(
        "Wilderness Area",
        [
            "Rawah",
            "Neota",
            "Comanche Peak",
            "Cache la Poudre"
        ]
    )

    soil = st.selectbox(
        "Soil Type",
        [str(i) for i in range(1, 41)]
    )

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

distance_to_hydro = np.sqrt(
    h_dist_hydro**2 +
    v_dist_hydro**2
)

mean_hillshade = (
    hillshade_9am +
    hillshade_noon +
    hillshade_3pm
) / 3

hydro_road_diff = (
    h_dist_hydro -
    h_dist_road
)

# =============================================================================
# ENCODING
# =============================================================================

try:

    wilderness_encoded = encoders[
        "Wilderness_Area"
    ].transform([wilderness])[0]

    soil_encoded = encoders[
        "Soil_Type"
    ].transform([soil])[0]

except:

    wilderness_encoded = 0
    soil_encoded = 0

# =============================================================================
# INPUT DATA
# =============================================================================

input_data = {

    "Elevation": elevation,

    "Aspect": aspect,

    "Slope": slope,

    "Horizontal_Distance_To_Hydrology":
        h_dist_hydro,

    "Vertical_Distance_To_Hydrology":
        v_dist_hydro,

    "Horizontal_Distance_To_Roadways":
        h_dist_road,

    "Hillshade_9am":
        hillshade_9am,

    "Hillshade_Noon":
        hillshade_noon,

    "Hillshade_3pm":
        hillshade_3pm,

    "Horizontal_Distance_To_Fire_Points":
        h_dist_fire,

    "Wilderness_Area":
        wilderness_encoded,

    "Soil_Type":
        soil_encoded,

    "Distance_To_Hydrology":
        distance_to_hydro,

    "Mean_Hillshade":
        mean_hillshade,

    "Hydro_Road_Distance_Diff":
        hydro_road_diff
}

# =============================================================================
# PREDICT BUTTON
# =============================================================================

if st.button("🔍 Predict Forest Cover Type"):

    if not model_loaded:

        st.error("Model not loaded")

    else:

        try:

            input_df = pd.DataFrame([input_data])

            # Ensure all required columns exist
            for col in top_features:

                if col not in input_df.columns:
                    input_df[col] = 0

            # Select required features
            input_df = input_df[top_features]

            # Apply skewness transformation
            for col in skewed_features:

                if col in input_df.columns:

                    input_df[col] = np.log1p(
                        input_df[col] -
                        input_df[col].min()
                    )

            # Scale data
            input_scaled = scaler.transform(input_df)

            # Predict
            prediction = model.predict(
                input_scaled
            )[0]

            # Predict probabilities
            probability = model.predict_proba(
                input_scaled
            )[0]

            # =============================================================================
            # RESULT
            # =============================================================================

            st.success(
                f"🌲 Predicted Forest Cover Type: "
                f"{COVER_TYPE_NAMES[prediction]}"
            )

            # =============================================================================
            # CONFIDENCE TABLE
            # =============================================================================

            st.subheader("📊 Prediction Confidence")

            prob_df = pd.DataFrame({

                "Forest Type":
                    list(COVER_TYPE_NAMES.values()),

                "Probability (%)":
                    probability * 100
            })

            st.dataframe(prob_df)

            # =============================================================================
            # BAR CHART
            # =============================================================================

            st.subheader("📈 Prediction Probability Chart")

            chart_df = prob_df.set_index(
                "Forest Type"
            )

            st.bar_chart(chart_df)

            # =============================================================================
            # PIE CHART
            # =============================================================================

            st.subheader("🥧 Probability Distribution")

            fig, ax = plt.subplots(figsize=(7, 7))

            ax.pie(
                probability,
                labels=list(COVER_TYPE_NAMES.values()),
                autopct='%1.1f%%'
            )

            st.pyplot(fig)

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🌲 About Project")

st.sidebar.info("""
EcoType is a Machine Learning application
that predicts forest cover type using
environmental and geographical features.

Models Used:
- Random Forest
- Decision Tree
- Logistic Regression
- KNN
- XGBoost
""")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.markdown(
    "Developed using Streamlit & Machine Learning"
)