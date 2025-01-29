import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit header and info
st.title('Diabetes Prediction Machine Learning Project 🩸')
st.write("\n\n")
st.info('Hey there! This app takes a set of features and predicts whether the patient has diabetes or not.')

# Dataset Loading and Caching
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df


df = load_data()

# Display basic dataset info
st.write("### Dataset Overview")
st.write(df.head())
st.write("#### Dataset Information")
st.text(df.info())
st.write("#### Dataset Statistics")
st.text(df.describe())
st.write("#### Missing Values")
st.text(df.isnull().sum())
st.write("#### Duplicate Rows")
st.text(df.duplicated().sum())

# Split dataset into features and target
X = df.drop(columns=["Outcome"])  # Features
y = df["Outcome"]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
LR = LogisticRegression(max_iter=200)
LR.fit(X_train_scaled, y_train)

# Making predictions and calculating accuracy
y_predictions = LR.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_predictions)

st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_
st.write(f"### Best Accuracy from Grid Search: {best_accuracy * 100:.2f}%")
st.write(f"### Best Hyperparameters: {grid_search.best_params_}")

# Model performance with best hyperparameters
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Test Set Accuracy with Best Model: {test_accuracy * 100:.2f}%")

# Confusion Matrix Visualization
st.subheader("Confusion Matrix (Model's Performance)")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# User Input Section for Prediction
st.sidebar.header("Enter the patient's details to predict diabetes")

# Sidebar for user input (adjust to your dataset's columns)
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=800, value=80, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=18, max_value=120, value=30, step=1)

# Display the input values in the main page
st.write(f"### Input Data")
st.write(f"Pregnancies: {pregnancies}")
st.write(f"Glucose: {glucose}")
st.write(f"Blood Pressure: {blood_pressure}")
st.write(f"Skin Thickness: {skin_thickness}")
st.write(f"Insulin: {insulin}")
st.write(f"BMI: {bmi}")
st.write(f"Diabetes Pedigree Function: {dpf}")
st.write(f"Age: {age}")

# Creating a DataFrame from the inputs
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
user_input_scaled = scaler.transform(user_input)  # Scale user input

# Adding a "Predict" button to trigger the prediction
if st.sidebar.button("Predict"):
    # Predicting the outcome using the trained model
    prediction = best_model.predict(user_input_scaled)

    # Display the result with a user-friendly message
    if prediction == 1:
        st.write("### The model predicts that the patient has diabetes.")
        st.success("The prediction indicates that the patient may have diabetes. Please consult a doctor for further examination.")
    else:
        st.write("### The model predicts that the patient does not have diabetes.")
        st.success("The prediction indicates that the patient is less likely to have diabetes. However, maintaining a healthy lifestyle is important.")
