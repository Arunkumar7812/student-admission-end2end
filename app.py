import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
import os # Import the os module for debugging

# --- 2. Streamlit UI Design (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Graduate Admission Chance Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load the Model Pipeline ---
# NOTE: The traceback indicates a model was saved WITH the 'Serial No.' column.
# We will load the first available pipeline and inject 'Serial No.' into the input data to match.
MODEL_PATHS = ['best_model_pipeline.pkl',               # 1. Preferred (Ridge/Poly)
    'random_forest_regressor_pipeline.pkl',  
    'linear_regression_pipeline.pkl' ]

# Use st.cache_resource to load the model once when the app starts
@st.cache_resource
def load_model(paths):
    """Tries to load the trained ML pipeline from known paths."""
    
    # Debugging step: print the current directory to help the user place the file
    st.sidebar.caption(f"Current working directory: {os.getcwd()}")
    
    loaded_pipeline = None
    loaded_path = None
    
    for path in paths:
        try:
            with open(path, 'rb') as file:
                pipeline = pickle.load(file)
                loaded_pipeline = pipeline
                loaded_path = path
                break  # Exit loop once a model is successfully loaded
        except FileNotFoundError:
            continue # Try the next path
        except Exception as e:
            # Handle other loading errors (e.g., corrupted file)
            st.error(f"Error loading model '{path}': {e}")
            continue

    if loaded_pipeline is None:
        st.error(f"Error: No model file found.")
        st.info("Please ensure one of the following files exists in your current directory:")
        for path in paths:
            st.code(path, language='text')
        st.stop()
        return None
    
    return loaded_pipeline, loaded_path # Return both the pipeline and the path it loaded

model_pipeline, loaded_model_path = load_model(MODEL_PATHS)


st.title("🎓 Graduate Admission Chance Predictor")
st.markdown("Use the sliders below to enter an applicant's profile details and predict their chance of admission.")

# Model pipeline must be loaded to proceed
if model_pipeline:
    # --- 3. Input Controls (Sidebar) ---
    st.sidebar.header("Applicant Profile Input")

    gre_score = st.sidebar.slider(
        "GRE Score (out of 340)", 
        min_value=290, max_value=340, value=320, step=1,
        help="Graduate Record Examination Score."
    )
    toefl_score = st.sidebar.slider(
        "TOEFL Score (out of 120)", 
        min_value=92, max_value=120, value=105, step=1,
        help="Test of English as a Foreign Language Score."
    )
    university_rating = st.sidebar.slider(
        "University Rating (1 to 5)", 
        min_value=1, max_value=5, value=3, step=1,
        help="Rating of the target university (1: Low, 5: High)."
    )
    sop = st.sidebar.slider(
        "SOP (Statement of Purpose) Strength", 
        min_value=1.0, max_value=5.0, value=3.5, step=0.5,
        help="Statement of Purpose rating (1.0: Poor, 5.0: Excellent)."
    )
    lor = st.sidebar.slider(
        "LOR (Letter of Recommendation) Strength", 
        min_value=1.0, max_value=5.0, value=3.5, step=0.5,
        help="Letter of Recommendation rating (1.0: Poor, 5.0: Excellent)."
    )
    cgpa = st.sidebar.slider(
        "CGPA (out of 10)", 
        min_value=6.8, max_value=9.9, value=8.5, step=0.01,
        help="Cumulative Grade Point Average (Undergrad)."
    )
    research = st.sidebar.selectbox(
        "Research Experience", 
        options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
        help="Does the applicant have prior research experience?"
    )

    # --- 4. Prediction Logic ---
    
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        # FIX 1: Keeping 'Serial No.' placeholder for the model that demands it
        'Serial No.': [1],
        'GRE Score': [gre_score], 
        'TOEFL Score': [toefl_score], 
        'University Rating': [university_rating], 
        'SOP': [sop], 
        # FIX 2: Correcting 'LOR ' (with space) to 'LOR' (without space)
        # because the model was likely trained on cleaned feature names.
        'LOR': [lor], 
        'CGPA': [cgpa], 
        'Research': [research]
    })
    
    # Ensure column order matches the training data exactly
    # We use the cleaned feature name 'LOR' here as well.
    input_data = input_data[[
        'Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'
    ]]
    
    # Make prediction using the loaded pipeline
    predicted_chance = model_pipeline.predict(input_data)[0]
    
    # Clip the prediction to the valid range [0, 1]
    final_chance = np.clip(predicted_chance, 0.0, 1.0)
    
    # --- 5. Display Results ---
    
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            label="Predicted Chance of Admit",
            value=f"{final_chance * 100:.1f}%"
        )
        
        # Display the prediction as a progress bar/gauge for visualization
        st.progress(final_chance)
        
    with col2:
        st.info(
            f"**Decision Guide:** The model is trained to recognize that high **CGPA** ({cgpa}) and **GRE Score** ({gre_score}) are typically the strongest predictors."
        )
        
        # Provide feedback based on the predicted chance
        if final_chance >= 0.85:
            st.success("This applicant has an **Excellent Chance** of admission.")
        elif final_chance >= 0.70:
            st.warning("This applicant has a **Very Good Chance** of admission.")
        elif final_chance >= 0.50:
            st.info("This applicant has a **Moderate Chance** of admission.")
        else:
            st.error("This applicant has a **Lower Chance** of admission.")

    st.markdown("---")
    st.caption(f"Model loaded: **{loaded_model_path}**")
