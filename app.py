import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Streamlit Page Config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
  html { scroll-behavior: smooth; }
  body, .stApp { background: linear-gradient(to right,#1e3c72,#2a5298)!important; color:white!important; }
  .header { width:100%; padding:30px; background:#0f172a; text-align:center; border-bottom:1px solid #334155;}
  .header h1 { color:#facc15; margin-bottom:10px;}
  .nav a { margin:0 15px; color:#93c5fd; font-size:1.1rem; text-decoration:none; }
  .nav a:hover { color:#fff; text-decoration:underline;}
  .card { background:rgba(255,255,255,0.1); padding:30px; margin:60px auto; max-width:600px; border-radius:16px;
           box-shadow:0 8px 20px rgba(0,0,0,0.3); }
  label { color:white!important; font-weight:bold!important; }
  input, select { width:100%; padding:12px; margin:12px 0 25px; border-radius:10px; border:1px solid #475569;
                  background:#1f2937!important; color:white!important; }
  .btn { background:#3b82f6; color:white; padding:14px; border:none; border-radius:10px; font-size:16px; width:100%; }
  .btn:hover { background:#2563eb; }
  .result { background:#dc2626; padding:16px; border-radius:12px; font-size:18px; color:white; }
  footer { background:#0f172a; color:#cbd5e1; text-align:center; padding:20px; margin-top:60px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
<div class="header">
  <h1>❤️ Heart Disease Predictor</h1>
  <div class="nav">
    <a href="#home">Home</a>
    <a href="#about">About</a>
    <a href="#predict">Predict</a>
  </div>
</div>
''', unsafe_allow_html=True)

# Home Section
st.markdown('''
<div id="home" class="card">
  <h3>👋 Welcome!</h3>
  <p>Enter your medical information below to see if you might be at risk for heart disease.</p>
</div>
''', unsafe_allow_html=True)

# About Section
st.markdown('''
<div id="about" class="card">
  <h4>📌 About</h4>
  <p>This application predicts the risk of heart disease using common medical parameters.</p>
  <p>Built with Python, Streamlit, and a logistic regression model trained on the UCI Cleveland dataset.</p>
</div>
''', unsafe_allow_html=True)

# Prediction Section
st.markdown('<div id="predict" class="card">', unsafe_allow_html=True)
st.markdown('<h4>📝 Enter Your Health Information:</h4>', unsafe_allow_html=True)

with st.form("heart_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120?", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina?", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0, format="%.1f")
    slope = st.selectbox("ST Slope", [1, 2, 3])
    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed", "Reversible"])

    # Center the Predict Button on the Page
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        submit = st.form_submit_button("🔮 Predict")

if submit:
    df = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol,
                        1 if fbs == "Yes" else 0, restecg, thalach,
                        1 if exang == "Yes" else 0, oldpeak, slope, ca,
                        {"Normal": 3, "Fixed": 6, "Reversible": 7}[thal]]],
                      columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                               "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
    pred = model.predict(df)[0]
    color = "#059669" if pred == 0 else "#dc2626"
    result = "No Heart Disease" if pred == 0 else "At Risk"
    st.markdown(f'<div class="result" style="background:{color}">🎯 {result}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # closes #predict

# Footer
st.markdown('<footer>Crafted with ❤️ by Future Health Innovators | © 2025</footer>', unsafe_allow_html=True)
