import streamlit as st
import pandas as pd
import numpy as np
import pickle


FEATURES = {
    'Serial No.':(1,500,1),
    'GRE Score': (300, 340, 316),
    'TOEFL Score': (92, 120, 107),
    'University Rating': (1, 5, 3),
    'SOP': (1.0, 5.0, 3.4),
    'LOR': (1.0, 5.0, 3.5),
    'CGPA': (6.8, 9.92, 8.6),
    'Research': (0, 1, 0)
}
FEATURE_ORDER = list(FEATURES.keys())

# --- Function to Load Model ---
@st.cache_resource
def load_model():
    """Loads the pickled model from the file system."""
    try:
        with open('random_forest_regressor_pipeline.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure it is in the same directory.")
        return None

# --- Main Streamlit App ---

st.set_page_config(page_title="Admission Chance Predictor", layout="centered")
st.title("🎓 Graduate Admission Chance Prediction")
st.markdown("Enter the student's details below to predict their chance of admission.")

model = load_model()

if model:
    # --- Collect User Inputs ---
    st.header("Student Metrics")

    with st.form("admission_form"):
        # User input collection for the 7 features
        gre_score = st.slider("GRE Score", min_value=FEATURES['GRE Score'][0], max_value=FEATURES['GRE Score'][1], value=FEATURES['GRE Score'][2])
        toefl_score = st.slider("TOEFL Score", min_value=FEATURES['TOEFL Score'][0], max_value=FEATURES['TOEFL Score'][1], value=FEATURES['TOEFL Score'][2])
        university_rating = st.slider("University Rating (1-5)", min_value=FEATURES['University Rating'][0], max_value=FEATURES['University Rating'][1], value=FEATURES['University Rating'][2])
        sop = st.slider("SOP Score (1.0-5.0)", min_value=FEATURES['SOP'][0], max_value=FEATURES['SOP'][1], value=FEATURES['SOP'][2], step=0.5)
        lor = st.slider("LOR Score (1.0-5.0)", min_value=FEATURES['LOR'][0], max_value=FEATURES['LOR'][1], value=FEATURES['LOR'][2], step=0.5)
        cgpa = st.slider("CGPA", min_value=FEATURES['CGPA'][0], max_value=FEATURES['CGPA'][1], value=FEATURES['CGPA'][2], step=0.01)
        research = st.selectbox("Research Experience", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        submitted = st.form_submit_button("Predict Chance")

    if submitted:
        # --- Prepare Data for Prediction ---
        # ONLY the 7 actual predictive features are included here
        input_data = [
            gre_score, toefl_score, university_rating, sop, lor, cgpa, research
        ]

        # Create DataFrame in the exact order (7 columns)
        final_features = pd.DataFrame([input_data], columns=FEATURE_ORDER)

        # --- Make Prediction ---
        try:
            prediction = model.predict(final_features)
            chance_of_admit = round(prediction[0] * 100, 2)

            st.subheader("Prediction Result")
            st.success(f"Chance of Admission: **{chance_of_admit}%**")
            st.balloons()

        except Exception as e:
            # If the error still occurs here, it means the new model.pkl wasn't loaded correctly.
            st.error(f"An error occurred during prediction: {e}")
