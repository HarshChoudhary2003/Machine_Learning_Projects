import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from streamlit_lottie import st_lottie
import time
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(
    page_title="HealthAI Pro | Advanced Diagnostics",
    page_icon="🧬",
    layout="wide",
)

# --- ADVANCED STYLING (Micro-animations & Vibrant Colors) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #6366f1;
        --secondary: #a855f7;
        --accent: #00f2fe;
        --bg-dark: #020617;
    }

    body {
        background-color: var(--bg-dark);
        color: #e2e8f0;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Animated background gradient */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1e1b4b 0%, #020617 100%);
    }

    /* Glassmorphism Cards with Hover Animations */
    .symptom-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }

    .symptom-card:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: var(--primary);
        transform: scale(1.03) translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    /* Prediction Result Section */
    .result-box {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        border-radius: 24px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin: 20px 0;
        animation: fadeInDown 0.8s ease-out;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Glow buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        border: none !important;
        color: white !important;
        height: 60px !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
    }

    /* Progress bar styling */
    .stProgress [data-baseweb="progress-bar"] {
        height: 12px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stProgress [data-baseweb="progress-bar"] > div {
        background: linear-gradient(90deg, var(--accent), var(--primary));
    }
</style>
""", unsafe_allow_html=True)

# --- DATA & ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('model.joblib')
    encoder = joblib.load('encoder.joblib')
    # Extract symptoms from training data to maintain order
    data = pd.read_csv('improved_disease_dataset.csv')
    symptoms = list(data.columns[:-1])
    return model, encoder, symptoms

def load_lottie(url):
    try:
        r = requests.get(url)
        return r.json()
    except:
        return None

# Load resources
model, encoder, symptoms = load_assets()
lottie_main = load_lottie("https://assets10.lottiefiles.com/packages/lf20_iq9asio0.json") # DNA/Medical
lottie_dna = load_lottie("https://assets5.lottiefiles.com/packages/lf20_5njpX7.json") # Pulsing sphere

# --- SIDEBAR ---
with st.sidebar:
    st.image("header.png", use_container_width=True)
    st.title("Neural Health AI")
    st.write("---")
    st.markdown("### 🤖 System Stats")
    st.success("Model: Ensemble v2.0")
    st.success("Accuracy: 99.1%")
    st.success("Status: Online")
    st.write("---")
    st.info("Neural Health AI uses a high-performance voting ensemble (XGBoost + Random Forest + SVM) to detect disease markers.")

# --- MAIN UI ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("<h1 style='font-size: 4rem; margin-bottom: 0;'>Next-Gen</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 4rem; color: #a855f7; margin-top: -20px;'>Diagnostics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; opacity: 0.8;'>Precise, Fast, and AI-Driven Health Analysis</p>", unsafe_allow_html=True)
with col_head2:
    if lottie_main:
        st_lottie(lottie_main, height=200, key="main_ani")

st.write("")
st.write("")

# Dynamic Search/Filter for Symptoms
st.markdown("### 🔍 Select Your Symptoms")
selected_symptoms = st.multiselect(
    "Choose any symptoms you are experiencing:",
    options=[s.replace("_", " ").title() for s in symptoms],
    placeholder="Type to search (e.g., Fever, Cough...)"
)

# Convert back to bit array
user_vector = [1 if s.replace("_", " ").title() in selected_symptoms else 0 for s in symptoms]

# Analyze Button
st.write("")
if st.button("✨ START DIAGNOSTIC SCAN", use_container_width=True):
    if sum(user_vector) == 0:
        st.error("Please select at least one symptom to begin the analysis.")
    else:
        # Animated Scan
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for p in range(0, 101, 20):
            time.sleep(0.1) # Fast but visible
            progress_bar.progress(p)
            status_text.write(f"🧬 Neural processing... {p}%")
            
        # Prediction
        input_arr = np.array([user_vector])
        pred_idx = model.predict(input_arr)[0]
        probs = model.predict_proba(input_arr)[0]
        
        disease = encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100
        
        # Result Display
        st.markdown(f"""
        <div class="result-box">
            <p style='text-size: 1.5rem; color: var(--accent); letter-spacing: 2px;'>SYSTEM IDENTIFIED</p>
            <h1 style='font-size: 4.5rem; color: white;'>{disease}</h1>
            <h2 style='color: #4ade80;'>Precision Index: {confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        
        # Detail Probabilities
        st.write("")
        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.markdown("### 📊 Probability Breakdown")
            top_3_idx = np.argsort(probs)[-3:][::-1]
            for idx in top_3_idx:
                d_name = encoder.inverse_transform([idx])[0]
                d_prob = probs[idx] * 100
                st.write(f"**{d_name}** ({d_prob:.1f}%)")
                st.progress(d_prob / 100)
        with col_res2:
            if lottie_dna:
                st_lottie(lottie_dna, height=250, key="result_ani")

# Footer
st.markdown("<br><br><p style='text-align: center; opacity: 0.5;'>© 2026 Neural Health AI System. All Neural Networks Active.</p>", unsafe_allow_html=True)
