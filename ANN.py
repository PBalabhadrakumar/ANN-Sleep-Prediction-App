# =============================================================================
# 1. IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np  # Make sure numpy is imported
import pickle
from tensorflow import keras
import json
from pathlib import Path
from streamlit_oauth import OAuth2Component
import requests  # For Lottie
from streamlit_lottie import st_lottie  # For Lottie

# =============================================================================
# 2. CONFIGURATION AND LOADING ASSETS
# =============================================================================

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="Sleep Quality Prediction", layout="centered")

# --- Google OAuth Configuration (Reads from .streamlit/secrets.toml) ---
CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID")
CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET")
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

# --- User Database for Simple Login ---
user_db = Path("users.json")

# --- Load BOTH ML Models and Tools ---
# Model 1: Classification (Disorder)
model_class = keras.models.load_model('sleep_model.keras')
# Model 2: Regression (Quality Score)
model_reg = keras.models.load_model('sleep_quality_model.keras')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

# =============================================================================
# 3. HELPER FUNCTIONS (LOTTIE + USERS)
# =============================================================================

# --- LOTTIE FUNCTION ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- USER AUTHENTICATION FUNCTIONS ---
def load_users():
    if user_db.exists():
        with open(user_db, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(user_db, "w") as f:
        json.dump(users, f, indent=4)

# =============================================================================
# 4. MAIN APPLICATION LOGIC
# =============================================================================
users = load_users()

if not st.session_state.get("logged_in", False):
    # --- AUTHENTICATION PAGE (IF NOT LOGGED IN) ---
    st.title("Welcome to the Sleep Quality App!")
    st.write("Please log in or sign up to continue.")
    
    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
    result = oauth2.authorize_button("Continue with Google", icon="https://www.google.com.tw/favicon.ico", redirect_uri="http://localhost:8501", scope="openid email profile", key="google", use_container_width=True)
    if result:
        st.session_state['token'] = result.get('token')
        st.session_state['logged_in'] = True
        st.rerun()

    st.write("--- OR ---")
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username").lower()
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if username in users and users[username] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Incorrect username or password")
    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("New Username").lower()
            new_password = st.text_input("New Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if new_username in users: st.error("Username already exists.")
                elif not new_username or not new_password: st.error("Username and password cannot be empty.")
                else:
                    users[new_username] = new_password
                    save_users(users)
                    st.success("Account created! Please go to the Login tab to log in.")

else:
    # --- IF THE USER IS LOGGED IN, SHOW THE MULTI-PAGE APP ---
    
    # --- SIDEBAR NAVIGATION ---
    st.sidebar.title("Navigation")
    user_name = st.session_state.get('username', 'User')
    if 'token' in st.session_state:
        user_info = st.session_state['token'].get('userinfo')
        if user_info: user_name = user_info.get('name', 'User')
    st.sidebar.write(f"Logged in as: **{user_name}**")
    
    page = st.sidebar.radio("Go to", ["Prediction App", "Project Information"])
    
    if st.sidebar.button("Logout"):
        for key in ['logged_in', 'token', 'username']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

    # --- PAGE 1: PREDICTION APP ---
    if page == "Prediction App":
        st.title("Sleep Quality Prediction Demo")
        st.write("Enter your health and lifestyle information to predict your risk for poor sleep quality.")
        
        st.sidebar.header("User Input Features")
        
        # All of your input widgets
        gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
        age = st.sidebar.slider("Age", 18, 100, 40)
        sleep_duration = st.sidebar.slider("Sleep Duration (hours per day)", 4.0, 10.0, 7.5)
        activity_level = st.sidebar.slider("Physical Activity (minutes per day)", 0, 120, 60)
        stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
        bmi_category = st.sidebar.selectbox("BMI Category", ('Normal', 'Overweight', 'Obese'))
        heart_rate = st.sidebar.slider("Resting Heart Rate (bpm)", 50, 100, 75)
        daily_steps = st.sidebar.slider("Daily Steps", 1000, 15000, 8000)
        occupation = st.sidebar.selectbox("Occupation", ('Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Lawyer', 'Salesperson', 'Scientist', 'Manager'))
        
        if st.sidebar.button("Predict"):
            user_input_df = pd.DataFrame({'Gender': [gender], 'Age': [age], 'Occupation': [occupation], 'Sleep Duration': [sleep_duration], 'Physical Activity Level': [activity_level], 'Stress Level': [stress_level], 'BMI Category': [bmi_category], 'Heart Rate': [heart_rate], 'Daily Steps': [daily_steps]})
            input_df_encoded = pd.get_dummies(user_input_df)
            input_df_reindexed = input_df_encoded.reindex(columns=model_columns, fill_value=0)
            scaled_input = scaler.transform(input_df_reindexed)
            
            # --- THIS IS THE 'nan' FIX ---
            scaled_input = np.nan_to_num(scaled_input, nan=0.0, posinf=0.0, neginf=0.0)
            
            # --- MAKE TWO PREDICTIONS ---
            # Prediction 1: Disorder (Classification)
            pred_prob_class = model_class.predict(scaled_input)
            prediction_class = np.argmax(pred_prob_class, axis=1)
            result_class = label_encoder.inverse_transform(prediction_class)[0]
            
            # Prediction 2: Quality Score (Regression)
            pred_reg = model_reg.predict(scaled_input)
            # We use max(1, min(10, ...)) to keep the score between 1 and 10
            result_reg = max(1, min(10, pred_reg[0][0]))
            
            # --- DISPLAY BOTH RESULTS ---
            st.subheader("Prediction Results")
            
            # Use columns for a nice layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Predicted Sleep Quality", value=f"{result_reg:.1f} / 10")
            
            with col2:
                st.metric(label="Predicted Disorder Risk", value=result_class)
            
            # Add the text explanation
            if pd.isna(result_class) or result_class == "Unknown":
                st.warning("Could not determine a prediction. Please check your inputs.")
            elif result_class == "None":
                st.success(f"✅ The model predicts good sleep quality with no disorder.")
            else:
                st.error(f"⚠️ The model predicts a potential risk for: {result_class}")
            
            st.info("Note: This is an AI-based prediction and not a medical diagnosis.")

    # --- PAGE 2: PROJECT INFORMATION ---
    elif page == "Project Information":
        st.title("About the Project")
        
        # --- LOTTIE CODE ---
        lottie_sleep_url = "https://assets5.lottiefiles.com/packages/lf20_0yfsb3a1.json"
        lottie_sleep = load_lottieurl(lottie_sleep_url)
        if lottie_sleep:
            st_lottie(lottie_sleep, speed=1, height=200, key="sleep_animation")
        
        st.write("This project was developed by **P Balabhadra kumar**.")
        st.write("---")
        st.subheader("Project Goal")
        st.write("The objective of this project is to build an Artificial Neural Network (ANN) to predict sleep quality based on key lifestyle and health factors.")
        
        st.subheader("Model Performance")
        st.write("The model was evaluated on a held-out test set to ensure its predictions are reliable. The results are displayed below.")
        
        st.write("#### Confusion Matrix")
        st.write("The confusion matrix shows the instances where the model predicted correctly versus where it made errors.")
        st.image("confusion_matrix.png")

        with st.expander("Click here to see the detailed Classification Report"):
            st.write("The classification report shows the precision, recall, and f1-score, which are key metrics for evaluating a classifier's performance on each category.")
            st.image("report.png")

        st.write("---")
        st.subheader("Next Steps")
        st.write("The next enhancement for this page is to add a 'Feature Importance' chart to explain which factors are the most predictive.")