# Sleep Quality Prediction App

This is an ANN-based project that predicts sleep quality and disorder risk. It features a full user login system (simple and Google OAuth) and a multi-page Streamlit interface.

## Features
- Dual-model predictions (Classification for disorder, Regression for quality score)
- Secure Google Sign-In (OAuth 2.0)
- Local user registration and login
- Multi-page app with results and project info
- Lottie animations for a polished UI

## How to Run
1. Clone the repository.
2. Create a `.streamlit/secrets.toml` file with your Google Client ID and Secret.
3. Install the required libraries: `pip install -r requirements.txt`
4. Run the app: `streamlit run ANN.py`