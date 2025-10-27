import streamlit as st
import pandas as pd
import pickle

# Load saved pickle model
with open("student_admission_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Student Admission Prediction App")
st.write("Enter details to predict admission chances")

numeric_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']

gre = st.number_input("GRE Score", min_value=260, max_value=340)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120)
uni = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)
research = st.selectbox("Research Experience", [0, 1])

input_df = pd.DataFrame([[gre, toefl, uni, sop, lor, cgpa, research]],
                        columns=numeric_cols)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Admission Chance: {prediction * 100:.2f}%")
