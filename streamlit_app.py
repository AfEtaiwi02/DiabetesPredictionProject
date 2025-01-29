import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Streamlit header and intro
st.title('Diabetes Prediction Tool 🩸')
st.write("\n\n")
st.info('Welcome! This tool helps predict whether a patient might have diabetes based on a set of health indicators.')

# Load pre-trained model and scaler (use correct path for your files)
@st.cache_data
def load_model_and_scaler():
    # Load trained Logistic Regression model and the scaler
    model = joblib.load("logistic_regression_model.pkl")  # Load the trained model
    scaler = joblib.load("scaler.pkl")  # Load the scaler
    return model, scaler

model, scaler = load_model_and_scaler()

# Sidebar input fields (user-friendly labels and descriptions)
st.sidebar.header("Patient Details")

pregnancies = st.sidebar.number_input("Number of Pregnancies (How many times has the patient been pregnant?)", min_value=0, max_value=20, value=0, step=1)
glucose = st.sidebar.number_input("Glucose Level (Blood sugar level after fasting, in mg/dL)", min_value=0, max_value=300, value=120, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure (Systolic pressure in mm Hg)", min_value=0, max_value=200, value=80, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness (Thickness of skin fold in mm)", min_value=0, max_value=100, value=20, step=1)
insulin = st.sidebar.number_input("Insulin Level (Insulin level in blood, in µU/ml)", min_value=0, max_value=800, value=80, step=1)
bmi = st.sidebar.number_input("BMI (Body Mass Index, a measure of body fat)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function (A measure of family history of diabetes)", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.sidebar.number_input("Age (Age of the patient)", min_value=18, max_value=120, value=30, step=1)

# Display the input values
st.write(f"### Patient Input Data:")
st.write(f"- Pregnancies: {pregnancies}")
st.write(f"- Glucose: {glucose}")
st.write(f"- Blood Pressure: {blood_pressure}")
st.write(f"- Skin Thickness: {skin_thickness}")
st.write(f"- Insulin: {insulin}")
st.write(f"- BMI: {bmi}")
st.write(f"- Diabetes Pedigree Function: {dpf}")
st.write(f"- Age: {age}")

# Creating a DataFrame from the inputs
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
user_input_scaled = scaler.transform(user_input)  # Scale user input

# Adding a "Predict" button to trigger the prediction
if st.sidebar.button("Predict"):
    # Predicting the outcome using the trained model
    prediction = model.predict(user_input_scaled)

    # Display the result with a user-friendly message
    if prediction == 1:
        st.write("### Prediction Result: The model predicts that the patient **may have diabetes**.")
        st.warning("Please consult a healthcare professional for further testing and diagnosis.")
    else:
        st.write("### Prediction Result: The model predicts that the patient **does not have diabetes**.")
        st.success("The model indicates that the patient is less likely to have diabetes. However, regular health check-ups are recommended.")
