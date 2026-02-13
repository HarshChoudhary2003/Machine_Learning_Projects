
import streamlit as st
import pandas as pd
import pickle
import xgboost
import os
import time
import requests
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# --------------------------
# CONFIGURATION & ASSETS
# --------------------------
st.set_page_config(
    page_title="InsurAI | Premium Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie Animations
lottie_health = load_lottieurl("https://lottie.host/02029707-3316-432d-93cc-f2d3a339a03c/JzQ17j3z7u.json") # Health shield/check
lottie_money = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json") # Flying money/coins
lottie_doctor = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_tutvdkg0.json") # Doctor/Health

# --------------------------
# CUSTOM CSS STYLING
# --------------------------
st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }

    /* Titles and Typography */
    h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: #2d3436;
    }
    .big-stat {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Custom Button */
    div.stButton > button {
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0 10px 20px -10px rgba(0, 21, 255, 0.5);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 30px -10px rgba(0, 21, 255, 0.7);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# MAIN APP LOGIC
# --------------------------
def main():
    # Load Model (with cache for speed)
    @st.cache_resource
    def load_model():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, 'insurancemodelf.pkl')
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            return None

    model = load_model()
    if not model:
        st.error("🚨 Critical Error: Model file not found. Please verify the file path.")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        if lottie_doctor:
            st_lottie(lottie_doctor, height=200, key="doctor_anim")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=150)
            
        st.write("## ⚙️ User Parameters")
        
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            age = st.slider("📅 Age", 18, 100, 25)
            sex = st.radio("👤 Gender", ["male", "female"], horizontal=True)
            smoker = st.radio("🚬 Smoker?", ["yes", "no"], horizontal=True)
            bmi = st.slider("⚖️ BMI", 10.0, 50.0, 22.5, 0.1)
            children = st.select_slider("👶 Dependents", options=list(range(6)), value=0)
            region = st.selectbox("🌍 Region", ["southwest", "southeast", "northwest", "northeast"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        predict_btn = st.button("✨ ANALYZE PREMIUM")

    # --- Main Content ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("# 🛡️ InsurAI")
        st.markdown("### Next-Gen Health Cost Estimator")
        st.write("Leveraging XGBoost algorithms to provide precision insurance forecasting.")

    with col2:
        if lottie_health:
            st_lottie(lottie_health, height=150, key="health_anim")

    st.markdown("---")

    # Prediction Block
    if predict_btn:
        # Progress Bar Simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            if i == 20: status_text.text("📥 Ingesting Data...")
            if i == 50: status_text.text("� Running XGBoost Algo...")
            if i == 80: status_text.text("� Finalizing Report...")
        
        status_text.empty()
        progress_bar.empty()

        # Data Prep
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])
        
        input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})
        input_data = input_data.drop(['sex', 'region'], axis=1)

        try:
            prediction = model.predict(input_data)[0]
            
            # --- Results Display ---
            st.markdown('<div class="glass-card" style="text-align: center; padding: 40px;">', unsafe_allow_html=True)
            st.markdown("<h3>🏥 PROJECTION RESULT</h3>", unsafe_allow_html=True)
            st.markdown(f'<div class="big-stat">${prediction:,.2f}</div>', unsafe_allow_html=True)
            st.markdown("<p style='color: #636e72;'>Estimated Annual Medical Insurance Premium</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Advanced Visuals ---
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### ⚖️ BMI Analysis")
                
                # Gauge Chart for BMI
                bmi_color = "green" if 18.5 <= bmi <= 24.9 else "orange" if bmi < 29.9 else "red"
                
                fig_bmi = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = bmi,
                    title = {'text': "Body Mass Index"},
                    gauge = {
                        'axis': {'range': [10, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': bmi_color},
                        'steps': [
                            {'range': [10, 18.5], 'color': "lightblue"},
                            {'range': [18.5, 25], 'color': "lightgreen"},
                            {'range': [25, 30], 'color': "lemonchiffon"},
                            {'range': [30, 50], 'color': "lightcoral"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30}}))
                fig_bmi.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bmi, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_viz2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🚬 Risk Factor Breakdown")
                
                # Radar Chart or simple bar chart for 'Risk'
                # Synthetic risk calculation for visual
                risk_factors = {
                    'Age Impact': (age/100) * 100,
                    'BMI Impact': (bmi/50) * 100,
                    'Smoking Impact': 100 if smoker == 'yes' else 10,
                    'Dependents': (children/5) * 100
                }
                
                fig_risk = go.Figure([go.Bar(
                    x=list(risk_factors.values()),
                    y=list(risk_factors.keys()),
                    orientation='h',
                    marker=dict(
                        color='rgba(50, 171, 96, 0.6)',
                        line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
                    )
                )])
                fig_risk.update_layout(
                    xaxis_title="Relative Cost contribution (%)",
                    height=250, 
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.balloons()

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    
    else:
        # Idle State Content
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info("👈 Please enter the patient details in the sidebar and click 'ANALYZE PREMIUM' to start the AI prediction engine.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
