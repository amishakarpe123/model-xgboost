import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import requests

# Set page config for a professional look
st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 30px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_anim = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_hxart9lz.json") # Graduation animation

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# --- HEADER ---
with st.container():
    st_lottie(lottie_anim, height=200, key="header")
    st.title("Student Outcome Prediction")
    st.write("Fill in the details below to predict the student's performance status.")

st.markdown("---")

# --- INPUT SECTION ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("Age", min_value=10, max_value=100, value=18)
        study_hours = st.slider("Weekly Study Hours", 0, 60, 20)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
        parent_edu = st.selectbox("Parent Education Level", options=["High School", "Bachelor", "Master", "PhD"])

    with col2:
        internet = st.radio("Internet Access", ["Yes", "No"])
        extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"])
        prev_score = st.number_input("Previous Score", 0, 100, 75)
        final_score = st.number_input("Current Midterm/Final Score", 0, 100, 70)
    
    submit = st.form_submit_button("Predict Result")

# --- PREDICTION LOGIC ---
if submit:
    # Pre-processing (Ensure this matches your training label encoding)
    input_data = pd.DataFrame({
        'gender': [1 if gender == "Male" else 0],
        'age': [age],
        'study_hours_per_week': [study_hours],
        'attendance_rate': [attendance],
        'parent_education': [{"High School":0, "Bachelor":1, "Master":2, "PhD":3}[parent_edu]],
        'internet_access': [1 if internet == "Yes" else 0],
        'extracurricular': [1 if extracurricular == "Yes" else 0],
        'previous_score': [prev_score],
        'final_score': [final_score]
    })

    prediction = model.predict(input_data)
    
    st.markdown("### Result:")
    if prediction[0] == 1:
        st.balloons()
        st.success("✨ The model predicts **SUCCESS/PASS**!")
    else:
        st.error("⚠️ The model predicts **AT RISK/FAIL**.")
