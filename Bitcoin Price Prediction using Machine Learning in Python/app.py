import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import requests
import warnings
import os

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR "WOW" FACTOR
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .main-header {
        animation: fadeIn 1.5s ease-out;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF9900, #F7931A); /* Bitcoin Colors */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: #B0B0B0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        border-color: #FF9900;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #fff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #161B22;
    }
    
    .sidebar .sidebar-content {
        background-color: #161B22;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF9900 0%, #F7931A 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(247, 147, 26, 0.4);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 5px;
        color: #fff;
        font-weight: 600;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 153, 0, 0.2);
        color: #FF9900;
        border-bottom: 2px solid #FF9900;
    }
    
    /* Plotly Chart Background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bitcoin.csv')
        
        # Convert Date to datetime objects
        # Handling format based on potential inconsistencies, though pandas usually handles iso well
        df['Date'] = pd.to_datetime(df['Date']) 
        
        # Feature Engineering (from main.ipynb)
        date_split = df['Date'].dt.date.astype(str).str.split('-', expand=True)

        # Ensure correct assignment based on format (YYYY-MM-DD or similar)
        # Assuming YYYY-MM-DD from notebook preview: split[0]=Year, split[1]=Month, split[2]=Day
        df['year'] = date_split[0].astype('int')
        df['month'] = date_split[1].astype('int')
        df['day'] = date_split[2].astype('int')
        
        df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
        
        df['open-close'] = df['Open'] - df['Close']
        df['low-high'] = df['Low'] - df['High']
        df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load Assets
lottie_bitcoin = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_p7ki6kjb.json") # Search for a generic crypto/chart lottie if this link is old or broken
if not lottie_bitcoin:
    lottie_bitcoin = load_lottieurl("https://lottie.host/5a5d0e2d-9323-45f8-8f1a-5f3366835165/123456.json") # Fallback dummy or real link

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=100)
    st.title("Settings")
    
    st.markdown("---")
    st.subheader("Model Configuration")
    model_choice = st.selectbox(
        "Select Model",
        ("Logistic Regression", "Support Vector Machine (SVC)", "XGBoost Classifier")
    )
    
    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application predicts whether the Bitcoin Price will close **higher** or **lower** the next day using Machine Learning."
    )
    st.markdown("Created with ❤️ using Streamlit")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------

# Header
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="main-header"><h1>Bitcoin Price Predictor</h1><p>Advanced Machine Learning Analysis & Forecast</p></div>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the ultimate Bitcoin price prediction dashboard. 
    Analyze historical trends, explore market volatility, and leverage AI to forecast future movements.
    """)

with col2:
    if lottie_bitcoin:
        st_lottie(lottie_bitcoin, height=200, key="bitcoin_anim")
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=150)

# -----------------------------------------------------------------------------
# HOW IT WORKS SECTION
# -----------------------------------------------------------------------------
with st.expander("ℹ️ How it works"):
    st.markdown("""
    ### Simple 6-Step Process
    
    1. **Data Loading:** We load the latest Bitcoin historical data.
    2. **Feature Engineering:** We create new features like 'Open-Close' difference, 'Low-High' difference, and check for quarter ends.
    3. **Data Splitting:** Data is split into training and validation sets based on the ratio you select in the sidebar.
    4. **Model Training:** The selected Machine Learning model (Logistic Regression, SVC, or XGBoost) learns patterns from the training data.
    5. **Evaluation:** The model's performance is tested on unseen validation data.
    6. **Forecasting:** The trained model is used to predict the next day's price movement.
    """)

# Load Data
df = load_data()

if df is not None:
    # -------------------------------------------------------------------------
    # TABS FOR NAVIGATION
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Data Analysis", "🤖 Price Prediction"])
    
    # -------------------------------------------------------------------------
    # TAB 1: MARKET OVERVIEW
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Key Market Metrics")
        
        # Latest Data
        latest_data = df.iloc[-1]
        prev_data = df.iloc[-2]
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            price_diff = latest_data['Close'] - prev_data['Close']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Close</div>
                <div class="metric-value">${latest_data['Close']:,.2f}</div>
                <div style="color: {'#00D26A' if price_diff > 0 else '#FF4B4B'}; font-weight: bold;">
                    {price_diff:+.2f} ({price_diff/prev_data['Close']*100:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Open Price</div>
                <div class="metric-value">${latest_data['Open']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">High Price</div>
                <div class="metric-value">${latest_data['High']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Low Price</div>
                <div class="metric-value">${latest_data['Low']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Historical Price Action")
        
        # Interactive Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Bitcoin')])
        
        fig.update_layout(
            title='Bitcoin Candlestick Chart',
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Recent Data")
        st.dataframe(df.tail(10).style.format({
            'Open': '${:.2f}', 'High': '${:.2f}', 'Low': '${:.2f}', 'Close': '${:.2f}', 'Adj Close': '${:.2f}', 'Volume': '{:,.0f}'
        }), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Exploratory Data Analysis")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Feature Distributions")
            # Using Plotly for distribution
            dist_feat = st.selectbox("Select Feature for Distribution", ['Open', 'Close', 'High', 'Low', 'Volume', 'open-close', 'low-high'])
            fig_dist = px.histogram(df, x=dist_feat, nbins=50, title=f"Distribution of {dist_feat}", template='plotly_dark', color_discrete_sequence=['#FF9900'])
            fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with c2:
            st.markdown("#### Correlation Heatmap")
            # Calculating correlation
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix", color_continuous_scale='Viridis', template='plotly_dark')
            fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        st.markdown("#### Grouped Analysis")
        group_col = st.selectbox("Group By", ['year', 'is_quarter_end'])
        grouped_df = df.groupby(group_col)[['Open', 'High', 'Low', 'Close']].mean().reset_index()
        
        fig_bar = px.bar(grouped_df, x=group_col, y=['Open', 'High', 'Low', 'Close'], barmode='group', title=f"Average Prices by {group_col}", template='plotly_dark')
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Target Balance
        st.markdown("#### Target Class Balance (Up vs Down)")
        target_counts = df['target'].value_counts()
        fig_pie = px.pie(values=target_counts.values, names=['Down/Neutral (0)', 'Up (1)'], title='Target Distribution', template='plotly_dark', color_discrete_sequence=['#FF4B4B', '#00D26A'])
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: PREDICTION
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Machine Learning Prediction")
        
        # Preparation
        features = df[['open-close', 'low-high', 'is_quarter_end']]
        target = df['target']
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            features_scaled, target, test_size=test_size, random_state=2022
        )
        
        # Training
        if st.button("🚀 Train Model and Evaluate"):
            with st.spinner(f"Training {model_choice}..."):
                if model_choice == "Logistic Regression":
                    model = LogisticRegression()
                elif model_choice == "Support Vector Machine (SVC)":
                    model = SVC(kernel='poly', probability=True)
                elif model_choice == "XGBoost Classifier":
                    model = XGBClassifier()
                
                model.fit(X_train, Y_train)
                
                train_pred = model.predict(X_train)
                valid_pred = model.predict(X_valid)
                
                train_acc = metrics.roc_auc_score(Y_train, train_pred)
                valid_acc = metrics.roc_auc_score(Y_valid, valid_pred)
                
                st.success("Model Trained Successfully!")
                
                # Results Metrics
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: #00D26A;">
                        <div class="metric-label">Training ROC-AUC Score</div>
                        <div class="metric-value">{train_acc:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res2:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: #007BFF;">
                        <div class="metric-label">Validation ROC-AUC Score</div>
                        <div class="metric-value">{valid_acc:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### Confusion Matrix")
                cm = metrics.confusion_matrix(Y_valid, valid_pred)
                fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", 
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Down (0)', 'Up (1)'], y=['Down (0)', 'Up (1)'],
                                   template='plotly_dark', color_continuous_scale='Blues')
                fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.markdown("### Classification Report")
                report = metrics.classification_report(Y_valid, valid_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.highlight_max(axis=0), use_container_width=True)

        if st.markdown("### 🔮 Predict Next Day Movement"):
            st.info("Based on the latest data available:")
            
            # Using the last row for "prediction" (though target is usually shift(-1), so let's simulate)
            # In a real app, you would input today's Open, High, Low, Close to predict tomorrow
            
            last_row = df.iloc[-1]
            last_features = pd.DataFrame([{
                'open-close': last_row['Open'] - last_row['Close'],
                'low-high': last_row['Low'] - last_row['High'],
                'is_quarter_end': last_row['is_quarter_end']
            }])
            
            # Scale
            # Note: Ideally fit on training + validation, transform on new. 
            # We are fitting on full set here just for demo of the input shape
            # In production, save scaler object.
            
            # Re-fit scaler on all features just for this one-shot prediction context display
            # (Strictly speaking, should use the fitted scaler, but re-fitting on full history is essentially similar for this demo)
            final_scaler = StandardScaler()
            final_features = final_scaler.fit_transform(features)
            
            input_scaled = final_scaler.transform(last_features)
            
            if 'model' in locals():
                prediction = model.predict(input_scaled)
                proba = model.predict_proba(input_scaled) if hasattr(model, "predict_proba") else None
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if prediction[0] == 1:
                        st.success("## 🚀 Prediction: UP")
                        st.markdown("The model predicts the price will close **higher** tomorrow.")
                    else:
                        st.error("## 🔻 Prediction: DOWN")
                        st.markdown("The model predicts the price will close **lower** tomorrow.")
                
                with res_col2:
                    if proba is not None:
                        st.markdown("#### Confidence")
                        st.progress(float(proba[0][prediction[0]]))
                        st.write(f"Probability: {proba[0][prediction[0]]*100:.2f}%")
            else:
                st.warning("Please train a model first using the button above.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Bitcoin Price Prediction Project | © 2026</div>", unsafe_allow_html=True)
