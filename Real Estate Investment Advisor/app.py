
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
import requests
import time
from streamlit_lottie import st_lottie

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PropAI | Real Estate Vision",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load Lottie animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Animations
lottie_city = load_lottieurl("https://lottie.host/80e3049a-6725-4682-8929-2315d023158e/L2K8T2g3z1.json") # Cityscape
lottie_invest = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_5w2awb1i.json") # Investment/Growth
lottie_analysis = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qpwbv5gm.json") # Data Analysis
lottie_success = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_lk80fpsm.json") # Success check

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif;
    }

    /* Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: #ffffff;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .glass-card:hover {
        transform: translateY(-5px);
    }

    /* Stats */
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #fff;
    }
    .stat-label {
        color: #ddd;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Input Fields */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px;
    }
    .stSlider > div > div > div > div {
        background-color: #00c6ff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. MODEL LOADING LOGIC
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    """Load all required models and artifacts."""
    paths = {
        'models': Path('models'),
        'artifacts': Path('artifacts')
    }
    
    required_files = {
        'models': ['best_classification_model.pkl', 'best_regression_model.pkl'],
        'artifacts': ['label_encoders.pkl', 'scaler.pkl', 'feature_columns.pkl', 
                      'unique_values.pkl', 'numerical_columns.pkl']
    }
    
    # Check all files exist
    missing_files = []
    for folder, files in required_files.items():
        for file in files:
            if not (paths[folder] / file).exists():
                missing_files.append(f"{folder}/{file}")
    
    if missing_files:
        st.error(f"❌ Missing files: {missing_files}")
        return None, None, None, None, None, None, None, False
    
    try:
        # Load models
        classifier = joblib.load(paths['models'] / 'best_classification_model.pkl')
        regressor = joblib.load(paths['models'] / 'best_regression_model.pkl')
        
        # Load artifacts
        with open(paths['artifacts'] / 'label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open(paths['artifacts'] / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(paths['artifacts'] / 'feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        with open(paths['artifacts'] / 'unique_values.pkl', 'rb') as f:
            unique_values = pickle.load(f)
        with open(paths['artifacts'] / 'numerical_columns.pkl', 'rb') as f:
            numerical_columns = pickle.load(f)
        
        return classifier, regressor, encoders, scaler, feature_columns, unique_values, numerical_columns, True
        
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        return None, None, None, None, None, None, None, False

# Load everything
clf, reg, encoders, scaler, feature_cols, unique_vals, num_cols, success = load_all_artifacts()

if not success:
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR - PROPERTY INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏘️ **Property Config**")
    
    # Decorative animation in sidebar
    if lottie_city:
        st_lottie(lottie_city, height=120, key="city_anim")
    
    with st.form("property_form"):
        st.markdown("### 📍 Location & Type")
        state = st.selectbox("State", sorted(unique_vals['State']))
        city = st.selectbox("City", sorted(unique_vals['City']))
        locality = st.selectbox("Locality", sorted(unique_vals['Locality']))
        property_type = st.selectbox("Property Type", sorted(unique_vals['Property_Type']))
        
        st.markdown("### 📐 Dimensions & Owner")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            bhk = st.slider("BHK", 1, 5, 2)
            floor_no = st.slider("Floor", 0, 30, 3)
        with col_s2:
            price_lakhs = st.slider("Price (₹ Lakhs)", 10, 500, 100, step=5)
            total_floors = st.slider("Total Floors", 1, 30, 10)
            
        size_sqft = st.slider("Size (SqFt)", 400, 5000, 1000, step=50)
        year_built = st.slider("Year Built", 1990, 2024, 2015)
        
        st.markdown("### ✨ Amenities & Status")
        with st.expander("Explore Amenities", expanded=False):
            furnished_status = st.selectbox("Furnished", sorted(unique_vals['Furnished_Status']))
            security = st.selectbox("Security", sorted(unique_vals['Security']))
            facing = st.selectbox("Facing", sorted(unique_vals['Facing']))
            owner_type = st.selectbox("Owner", sorted(unique_vals['Owner_Type']))
            availability_status = st.selectbox("Availability", sorted(unique_vals['Availability_Status']))
            transport_access = st.selectbox("Transport", sorted(unique_vals['Public_Transport_Accessibility']))
            nearby_schools = st.slider("Schools Nearby", 0, 10, 2)
            nearby_hospitals = st.slider("Hospitals Nearby", 0, 10, 1)
            parking_space = st.slider("Parking", 0, 5, 1)
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Analyze Investment")

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT
# -----------------------------------------------------------------------------
col_main_1, col_main_2 = st.columns([2, 1])

with col_main_1:
    st.title("PropAI Advisor")
    st.markdown("### Intelligent Real Estate Investment Analysis")
    st.write("Leveraging advanced Machine Learning to predict property appreciation and investment viability.")

with col_main_2:
    if lottie_invest:
        st_lottie(lottie_invest, height=150, key="invest_anim")

# Logic Implementation
if submitted:
    # --- ANIMATED PROCESSING ---
    with st.status("🧠 Processing Property Data...", expanded=True) as status:
        st.write("🏗️ Structuring Feature Vectors...")
        time.sleep(0.3)
        
        # Step 1: Feature Engineering
        input_data = pd.DataFrame([{
            'State': state, 'City': city, 'Locality': locality, 'Property_Type': property_type,
            'BHK': bhk, 'Size_in_SqFt': size_sqft, 'Year_Built': year_built,
            'Furnished_Status': furnished_status, 'Floor_No': floor_no, 'Total_Floors': total_floors,
            'Nearby_Schools': nearby_schools, 'Nearby_Hospitals': nearby_hospitals,
            'Public_Transport_Accessibility': transport_access, 'Parking_Space': parking_space,
            'Security': security, 'Amenities_Score': 3, 'Facing': facing,
            'Owner_Type': owner_type, 'Availability_Status': availability_status,
            'Age_of_Property': 2024 - year_built,
            'Price_per_SqFt': (price_lakhs * 100000) / size_sqft,
            'Floor_Ratio': floor_no / (total_floors + 1),
            'School_Density_Score': nearby_schools / (size_sqft / 1000),
            'Hospital_Density_Score': nearby_hospitals / (size_sqft / 1000)
        }])
        
        st.write("🔢 Encoding & Scaling...")
        time.sleep(0.3)
        
        # Step 2: Encoding
        for col in encoders.keys():
            if col in input_data.columns:
                val = input_data[col].iloc[0]
                if val not in encoders[col].classes_:
                    input_data[col] = encoders[col].classes_[0]
                else:
                    input_data[col] = encoders[col].transform([val])[0]
        
        # Step 3: Missing Cols
        for col in feature_cols:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Step 4: Numeric Conversion
        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Step 5: Scaling
        if hasattr(scaler, 'feature_names_in_'):
            scaler_features = scaler.feature_names_in_
        else:
            scaler_features = [col for col in num_cols if col in input_data.columns]
            
        cols_to_scale = [col for col in scaler_features if col in input_data.columns]
        if cols_to_scale:
            data_to_scale = input_data[cols_to_scale].astype(np.float64)
            input_data[cols_to_scale] = scaler.transform(data_to_scale)
            
        input_data = input_data[feature_cols]

        st.write("🔮 Running Prediction Models...")
        time.sleep(0.4)
        
        # Step 6: Prediction
        investment_pred = clf.predict(input_data)[0]
        investment_prob = clf.predict_proba(input_data)[0]
        future_price_pred = reg.predict(input_data)[0]
        appreciation_rate = ((future_price_pred / price_lakhs) ** (1/5) - 1) * 100
        
        status.update(label="Analysis Complete", state="complete", expanded=False)

    # --- RESULTS DASHBOARD ---
    st.markdown("---")
    
    # Section 1: Top Level Verdict
    res_col1, res_col2 = st.columns([1.5, 1])
    
    with res_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏛️ Investment Verdict")
        if investment_pred == 1:
            st.success("## 🌟 HIGH POTENTIAL ASSET")
            st.markdown(f"**Confidence Score:** {investment_prob[1]:.2%}")
            st.markdown("Our AI models indicate strong growth potential for this property based on historical trends and feature analysis.")
            if lottie_success: st_lottie(lottie_success, height=100, key="success")
        else:
            st.warning("## ⚠️ CAUTION ADVISED")
            st.markdown(f"**Risk Probability:** {investment_prob[0]:.2%}")
            st.markdown("This property shows indicators that may limit long-term appreciation compared to market averages.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_col2:
        # Gauge Chart for Score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = investment_prob[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "AI Score", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "#00c6ff"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.3)'},
                    {'range': [50, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
                ]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(t=30,b=0,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Section 2: Financials
    st.markdown("### 💸 Financial Projection (5-Year Horizon)")
    
    fin_c1, fin_c2, fin_c3 = st.columns(3)
    with fin_c1:
        st.markdown(f'<div class="glass-card"><div class="stat-value">₹{price_lakhs}L</div><div class="stat-label">Current Value</div></div>', unsafe_allow_html=True)
    with fin_c2:
         st.markdown(f'<div class="glass-card"><div class="stat-value" style="color: #00ff88;">₹{future_price_pred:.2f}L</div><div class="stat-label">Projected Value</div></div>', unsafe_allow_html=True)
    with fin_c3:
         color = "#00ff88" if appreciation_rate > 0 else "#ff4b4b"
         st.markdown(f'<div class="glass-card"><div class="stat-value" style="color: {color};">{appreciation_rate:.2f}%</div><div class="stat-label">Annual ROI</div></div>', unsafe_allow_html=True)

    # Section 3: Charts
    chart_c1, chart_c2 = st.columns([2, 1])
    
    with chart_c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Growth Trajectory")
        years = list(range(6))
        prices = [price_lakhs * (1 + appreciation_rate/100) ** i for i in years]
        
        fig_line = px.line(x=years, y=prices, markers=True, labels={'x': 'Years from Now', 'y': 'Value (Lakhs)'})
        fig_line.update_traces(line_color='#00c6ff', line_width=4, marker_size=10, marker_color='white')
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔑 Key Drivers")
        if hasattr(clf, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False).head(8)
            
            fig_bar = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Idle State
    st.markdown("---")
    st.markdown('<div class="glass-card" style="text-align: center;"><h3>👋 Ready to Analyze?</h3><p>Configure the property details in the sidebar to generate your AI-powered investment report.</p></div>', unsafe_allow_html=True)
    if lottie_analysis:
        st_lottie(lottie_analysis, height=300, key="analysis_idle")