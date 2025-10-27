import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

# 1. Initialize the Flask application
app = Flask(__name__)

# 2. Load the trained model
# IMPORTANT: Ensure your model file is named 'model.pkl' and is in the same directory.
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Ensure the trained model is saved and present.")
    model = None # Set model to None to prevent crashing

# 3. Define the home page route
@app.route('/')
def home():
    # Renders the HTML template where the user enters the data
    return render_template('index.html')

# 4. Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: The prediction model is not available.")

    # Get data from POST request (form submission)
    features = [
        'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 
        'LOR', 'CGPA', 'Research'
    ]
    
    # Extract form values and convert to float/int
    int_features = [float(x) for x in request.form.values()]

    # Create a DataFrame in the correct feature order for the model
    final_features = pd.DataFrame([int_features], columns=features)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Prediction result is the "Chance of Admit" (a probability between 0 and 1)
    # Format the output as a percentage for better user display
    chance_of_admit = round(prediction[0] * 100, 2)
    
    # Return the result to the HTML template
    return render_template('index.html', 
                           prediction_text=f'Chance of Admission: {chance_of_admit}%')

# 5. Run the Flask app
if __name__ == "__main__":
    # You may need to change the port if it's already in use
    app.run(debug=True)
