import streamlit as st
import pandas as pd
import mlflow.pyfunc

st.title("Student Admission Prediction App")

# Load model
model = mlflow.pyfunc.load_model("models:/Student_Admission_Model/2")

# User inputs
gre_score = st.number_input("GRE Score", 0, 340, 300)
toefl_score = st.number_input("TOEFL Score", 0, 120, 100)
cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
university_rating = st.slider("University Rating", 1, 5, 3)
research = st.radio("Research Experience", [0, 1])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "GRE Score": gre_score,
        "TOEFL Score": toefl_score,
        "University Rating": university_rating,
        "CGPA": cgpa,
        "Research": research
    }])

    prediction = model.predict(input_df)
    st.success(f"Chance of Admission: {prediction[0]:.2f}")
