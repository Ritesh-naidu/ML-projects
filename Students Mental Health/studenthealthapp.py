import streamlit as st
import pickle
import numpy as np

# Load trained models
@st.cache_resource
def load_models():
    models = {}
    with open("randomforest_model.pkl", "rb") as f:
        models["RandomForest"] = pickle.load(f)
    with open("svc_model.pkl", "rb") as f:
        models["SVC"] = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        models["KNN"] = pickle.load(f)
    return models

models = load_models()

# Streamlit UI
st.title("Students' Mental Health Prediction")

st.write("This app predicts whether a student might be suffering from depression based on their features.")

# User input fields
age = st.number_input("Age", min_value=15, max_value=30, value=20)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, step=0.01, value=3.0)
study_stress = st.slider("Study Stress Level (1-10)", min_value=1, max_value=10, value=5)
sleep_quality = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=5)

# Select model
model_choice = st.selectbox("Choose a model", list(models.keys()))

# Add space for better layout
st.markdown("---")  

# Make prediction button at the bottom
if st.button("Predict", use_container_width=True):
    input_data = np.array([[age, cgpa, study_stress, sleep_quality]])
    prediction = models[model_choice].predict(input_data)
    
    if prediction[0] == 1:
        st.error("The model predicts that the student might be suffering from depression.")
    else:
        st.success("The model predicts that the student is not suffering from depression.")
