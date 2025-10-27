import streamlit as st
import pandas as pd
import mlflow.pyfunc

# Load Model
model = mlflow.pyfunc.load_model("models:/Student_Admission_Model/2")

st.title("Student Admission Prediction App")
st.write("Enter details to predict admission chance")

# Inputs (ensure correct names match training set)
gre = st.number_input("GRE Score", min_value=260, max_value=340)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120)
university_rating = st.selectbox("University Rating", [1,2,3,4,5])
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, step=0.5)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)
research = st.selectbox("Research Experience", [0,1])

# Construct dataframe in correct order with exact column names
input_df = pd.DataFrame([[
    gre,
    toefl,
    university_rating,
    sop,
    lor,     # NOTICE THE TRAILING SPACE WILL BE ADDED BELOW
    cgpa,
    research
]], columns=[
    'GRE Score',
    'TOEFL Score',
    'University Rating',
    'SOP',
    'LOR ',  # VERY IMPORTANT: Trailing space
    'CGPA',
    'Research'
])

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Admission Chance: {prediction * 100:.2f}%")
    except Exception as e:
        st.error("Prediction failed. Please check inputs.")
        st.exception(e)
