import streamlit as st
import pandas as pd
import mlflow.pyfunc

# Load the model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models/best_model")
# Streamlit App UI Title
st.title("Student Admission Prediction App")
st.write("Enter the student profile details to predict admission chances")

# User Inputs
gre = st.number_input("GRE Score", min_value=260, max_value=340, step=1)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
university_rating = st.selectbox("University Rating (1-5)", [1, 2, 3, 4, 5])
sop = st.number_input("SOP Strength (1-5)", min_value=1.0, max_value=5.0, step=0.5)
lor = st.number_input("LOR Strength (1-5)", min_value=1.0, max_value=5.0, step=0.5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
research_exp = st.selectbox("Research Experience", [0, 1])

# Create dataframe for prediction
input_data = {
    'GRE Score': [gre],
    'TOEFL Score': [toefl],
    'University Rating': [university_rating],
    'SOP': [sop],
    'LOR ': [lor],   # IMPORTANT: trailing space in column name
    'CGPA': [cgpa],
    'Research': [research_exp]
}

input_df = pd.DataFrame(input_data)

# Predict Button
if st.button("Predict Admission Chance"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Chance of Admission: {prediction[0]:.2f}")
