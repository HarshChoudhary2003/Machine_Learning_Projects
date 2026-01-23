import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        
        st.sidebar.success("✅ All models loaded successfully!")
        return classifier, regressor, encoders, scaler, feature_columns, unique_values, numerical_columns, True
        
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        return None, None, None, None, None, None, None, False

# Title and description
st.title("🏠 Real Estate Investment Advisor")
st.markdown("""
This AI-powered tool analyzes property features to:
- **Classify** investment potential (Good / Not Recommended)
- **Predict** price appreciation over 5 years
- **Identify** key factors influencing the decision
""")

# How It Works Section
with st.expander("ℹ️ **How It Works**", expanded=False):
    st.markdown("""
    ### Model Pipeline
    
    **1. Dual Model Architecture**
    - **Classification Model**: Random Forest/XGBoost determines investment quality
    - **Regression Model**: Predicts future price using historical trends
    
    **2. Feature Engineering**
    - *Density Scores*: Schools/Hospitals per 1000 sqft
    - *Price Efficiency*: Price per sqft ratio
    - *Property Age*: 2024 - Year Built
    - *Floor Ratio*: Floor / Total Floors
    
    **3. Preprocessing**
    - Categorical encoding for 10+ location/property features
    - StandardScaler for numerical features
    - Missing values handled with default fallbacks
    
    **4. Prediction Flow**
    - Input → Encode → Engineer → Scale → Predict
    - Confidence score from classification probabilities
    - 5-year ROI calculated from regression predictions
    
    **⚠️ Limitations**
    - Trained on Indian real estate market data
    - Assumes linear market growth trends
    - Does not account for legal/title issues
    """)

# Load all artifacts
clf, reg, encoders, scaler, feature_cols, unique_vals, num_cols, success = load_all_artifacts()

if not success:
    st.stop()

# Sidebar: Property Input Form
st.sidebar.header("📋 Property Details")

with st.sidebar.form("property_form"):
    st.subheader("Location & Type")
    state = st.selectbox("State", sorted(unique_vals['State']))
    city = st.selectbox("City", sorted(unique_vals['City']))
    locality = st.selectbox("Locality", sorted(unique_vals['Locality']))
    property_type = st.selectbox("Property Type", sorted(unique_vals['Property_Type']))
    
    st.subheader("Features")
    bhk = st.slider("BHK", 1, 5, 2)
    size_sqft = st.slider("Size (SqFt)", 400, 5000, 1000, step=50)
    price_lakhs = st.slider("Current Price (Lakhs)", 10, 500, 100, step=5)
    year_built = st.slider("Year Built", 1990, 2024, 2015)
    floor_no = st.slider("Floor Number", 0, 30, 3)
    total_floors = st.slider("Total Floors", 1, 30, 10)
    
    st.subheader("Amenities & Status")
    furnished_status = st.selectbox("Furnished Status", sorted(unique_vals['Furnished_Status']))
    security = st.selectbox("Security", sorted(unique_vals['Security']))
    facing = st.selectbox("Facing", sorted(unique_vals['Facing']))
    owner_type = st.selectbox("Owner Type", sorted(unique_vals['Owner_Type']))
    availability_status = st.selectbox("Availability Status", sorted(unique_vals['Availability_Status']))
    transport_access = st.selectbox("Public Transport Accessibility", sorted(unique_vals['Public_Transport_Accessibility']))
    nearby_schools = st.slider("Nearby Schools", 0, 10, 2)
    nearby_hospitals = st.slider("Nearby Hospitals", 0, 10, 1)
    parking_space = st.slider("Parking Space", 0, 5, 1)
    
    
    submitted = st.form_submit_button("🔍 Analyze Investment", type="primary")

# Main Area: Analysis Results
if submitted:
    # Step 1: Create DataFrame with ALL features (original + engineered)
    input_data = pd.DataFrame([{
        # Original features
        'State': state,
        'City': city,
        'Locality': locality,
        'Property_Type': property_type,
        'BHK': bhk,
        'Size_in_SqFt': size_sqft,
        'Year_Built': year_built,
        'Furnished_Status': furnished_status,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Public_Transport_Accessibility': transport_access,
        'Parking_Space': parking_space,
        'Security': security,
        'Amenities_Score': 3,
        'Facing': facing,
        'Owner_Type': owner_type,
        'Availability_Status': availability_status,
        # Engineered features
        'Age_of_Property': 2024 - year_built,
        'Price_per_SqFt': (price_lakhs * 100000) / size_sqft,
        'Floor_Ratio': floor_no / (total_floors + 1),
        'School_Density_Score': nearby_schools / (size_sqft / 1000),
        'Hospital_Density_Score': nearby_hospitals / (size_sqft / 1000)
    }])
    
    # Step 2: Encode categorical features
    for col in encoders.keys():
        if col in input_data.columns:
            val = input_data[col].iloc[0]
            if val not in encoders[col].classes_:
                input_data[col] = encoders[col].classes_[0]
                st.sidebar.warning(f"Unknown value '{val}' for {col}. Using default.")
            else:
                input_data[col] = encoders[col].transform([val])[0]
    
    # Step 3: Add any missing columns
    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Step 4: CRITICAL FIX - Convert ALL columns to numeric dtypes
    # This prevents XGBoost errors about object dtypes
    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
    
    # Step 5: Get scaler's expected columns (handle different sklearn versions)
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = scaler.feature_names_in_
    else:
        # Fallback: use intersection of saved num_cols and available columns
        scaler_features = [col for col in num_cols if col in input_data.columns]
    
    # Step 6: Scale ONLY the columns the scaler was trained on
    cols_to_scale = [col for col in scaler_features if col in input_data.columns]
    
    if cols_to_scale:
        try:
            # Ensure data is float64 before scaling
            data_to_scale = input_data[cols_to_scale].astype(np.float64)
            input_data[cols_to_scale] = scaler.transform(data_to_scale)
        except Exception as e:
            st.error(f"Scaling error: {str(e)}")
            st.stop()
    
    # Step 7: Reorder columns to match training data exactly
    input_data = input_data[feature_cols]
    
    # Step 8: DOUBLE-CHECK - Ensure no object columns remain before prediction
    object_cols = input_data.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        st.error(f"❌ Non-numeric columns detected: {object_cols}")
        st.stop()
    
    # Step 9: Make predictions
    with st.spinner('🤔 Analyzing investment potential...'):
        investment_pred = clf.predict(input_data)[0]
        investment_prob = clf.predict_proba(input_data)[0]
        future_price_pred = reg.predict(input_data)[0]
        appreciation_rate = ((future_price_pred / price_lakhs) ** (1/5) - 1) * 100
    
    # Display Results
    st.markdown("## 📊 Investment Analysis Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Investment Classification")
        if investment_pred == 1:
            st.success("🎯 **Good Investment!**")
            st.metric("Confidence", f"{investment_prob[1]:.2%}")
        else:
            st.warning("⚠️ **Not Recommended**")
            st.metric("Confidence", f"{investment_prob[0]:.2%}")
        
        # Investment Score Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=investment_prob[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Investment Score (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 5-Year Price Projection")
        st.info(f"**Predicted Future Price: ₹{future_price_pred:.2f} Lakhs**")
        st.metric("Expected Annual ROI", f"{appreciation_rate:.2f}%")
        
        # Price projection chart
        years = list(range(6))
        prices = [price_lakhs * (1 + appreciation_rate/100) ** i for i in years]
        
        fig = px.line(
            x=years, 
            y=prices,
            markers=True,
            title="Projected Price Growth (5 Years)",
            labels={'x': 'Years', 'y': 'Price (Lakhs)'}
        )
        fig.update_traces(line_color='#00CC96', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("## 🔍 Key Influencing Factors")
    if hasattr(clf, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Features Impacting Investment Decision',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<center>
<p>🏠 <strong>Real Estate Investment Advisor</strong> - Powered by Machine Learning</p>
<p><em>Disclaimer: Predictions are for informational purposes only. Conduct due diligence.</em></p>
</center>
""", unsafe_allow_html=True)