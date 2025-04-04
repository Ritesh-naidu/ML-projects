import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained models
with open("logistic_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

# Load the scaler (if needed)
scaler = StandardScaler()

# Streamlit UI
st.title("🩺 Cardiovascular Disease Prediction")
st.write("Enter your health details below to predict your cardiovascular disease risk.")

# User input fields
age = st.number_input("Age (in days)", min_value=10000, max_value=25000, step=100, value=18000)
gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 2 else "Female")
height = st.number_input("Height (in cm)", min_value=100, max_value=220, step=1, value=170)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, step=1, value=70)
ap_hi = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, step=1, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=180, step=1, value=80)
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
glucose = st.selectbox("Glucose Level", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
smoke = st.selectbox("Do you smoke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
alcohol = st.selectbox("Do you consume alcohol?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
active = st.selectbox("Are you physically active?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Convert user inputs into a DataFrame
user_data = pd.DataFrame([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alcohol, active]],
                         columns=["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "glucose", "smoke", "alco", "active"])

# Standardize user data
user_data_scaled = scaler.fit_transform(user_data)

# Predict function
if st.button("Predict"):
    lr_pred = lr_model.predict(user_data_scaled)[0]
    knn_pred = knn_model.predict(user_data_scaled)[0]

    result_text = "✅ No Cardiovascular Disease Detected" if lr_pred == 0 else "⚠️ High Risk of Cardiovascular Disease!"
    
    st.subheader("Prediction Result")
    st.info(f"**Logistic Regression Prediction:** {result_text}")
    
    result_text_knn = "✅ No Cardiovascular Disease Detected" if knn_pred == 0 else "⚠️ High Risk of Cardiovascular Disease!"
    st.info(f"**KNN Prediction:** {result_text_knn}")
