import streamlit as st
import numpy as np
import pickle

# Load the model
MODEL_PATH = "student_admission_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Upload 'student_admission_model.pkl' to the app directory.")
    st.stop()

st.title("🎓 Student Admission Chance Prediction")
st.write("Enter the student details to predict the probability of getting admission.")

gre = st.number_input("GRE Score (260 - 340)", min_value=0.0, step=1.0)
toefl = st.number_input("TOEFL Score (0 - 120)", min_value=0.0, step=1.0)
uni_rating = st.slider("University Rating (1 - 5)", 1, 5, 3)
sop = st.slider("SOP Strength (1 - 5)", 1, 5, 3)
lor = st.slider("LOR Strength (1 - 5)", 1, 5, 3)
cgpa = st.number_input("CGPA (0 - 10)", min_value=0.0, max_value=10.0, step=0.1)
research = st.radio("Research Experience", ("No", "Yes"))

research_value = 1 if research == "Yes" else 0

input_data = np.array([[gre, toefl, uni_rating, sop, lor, cgpa, research_value]])

if st.button("Predict Admission Chance"):
    try:
        prediction = model.predict(input_data)[0]
        prediction = round(prediction * 100, 2)

        st.success(f"✅ Probability of Admission: {prediction}%")
    except Exception as e:
        st.error("Error making prediction. Please check model compatibility.")
        st.exception(e)

st.markdown("Made with ❤️ using Machine Learning & Streamlit")
