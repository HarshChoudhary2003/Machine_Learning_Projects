import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #00D4FF;
        font-weight: 700;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-color: #00D4FF;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00D4FF, #0055FF);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 5px 15px rgba(0, 85, 255, 0.4);
    }
    
    /* Plotly Charts */
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

# Load dataset with caching
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

# Load Animation
lottie_fraud = load_lottieurl("https://lottie.host/9e559902-6031-4043-847e-12822944733c/1M3g9X8q7r.json") # Generic security/scan animation

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=80)
    st.title("Fraud Guard AI")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Data Analysis", "Model Training", "Prediction"],
        icons=["speedometer2", "graph-up-arrow", "cpu", "search"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00D4FF", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#0055FF"},
        }
    )
    
    st.markdown("---")
    st.info("This application uses Random Forest to detect fraudulent credit card transactions.", icon="ℹ️")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------

# Load data once
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure 'creditcard.csv' is in the directory.")
    st.stop()

if selected == "Dashboard":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🛡️ Secure Transaction Monitor")
        st.markdown("Welcome to the **Fraud Guard AI** dashboard. Real-time analysis of transaction data to identify potential security threats.")
    with col2:
        if lottie_fraud:
            st_lottie(lottie_fraud, height=150, key="fraud_anim")
    
    st.markdown("### Key Metrics")
    
    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    outlier_fraction = len(fraud)/float(len(valid))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}", delta_color="off")
    col2.metric("Valid Transactions", f"{len(valid):,}", delta_color="normal")
    col3.metric("Fraud Cases", f"{len(fraud):,}", delta="-Alert", delta_color="inverse")
    col4.metric("Fraud Percentage", f"{outlier_fraction*100:.4f}%", delta_color="inverse")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transaction Amount Distribution (Fraud)")
        fig_fraud = px.histogram(fraud, x='Amount', nbins=50, title="Fraudulent Amount Distribution", 
                                color_discrete_sequence=['#FF4B4B'], template="plotly_dark")
        fig_fraud.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fraud, use_container_width=True)
        
    with c2:
        st.subheader("Transaction Amount Distribution (Valid)")
        # Sample for performance if needed, but histogram handles it well usually
        fig_valid = px.histogram(valid[valid['Amount'] < 2000], x='Amount', nbins=50, title="Valid Amount Distribution (< $2000)", 
                                color_discrete_sequence=['#00D4FF'], template="plotly_dark")
        fig_valid.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_valid, use_container_width=True)

elif selected == "Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Correlation Matrix")
    with st.spinner("Calculating correlations..."):
        # Subsample for heatmap performance if dataset is HUGE, but 280k is okay-ish. 
        # Better to sample for quick interaction
        sample_df = df.sample(frac=0.1, random_state=42)
        corr = sample_df.corr()
        
        fig_corr = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale='RdBu_r', 
                             title="Correlation Heatmap (10% Sample)", template="plotly_dark")
        fig_corr.update_layout(height=700)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.subheader("Feature Variance")
    # Show mean values of features for fraud vs non-fraud
    feature_means = df.groupby('Class').mean().transpose().reset_index()
    feature_means.columns = ['Feature', 'Valid', 'Fraud']
    feature_means = feature_means[feature_means['Feature'].str.contains('V')] # Only V features
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=feature_means['Feature'], y=feature_means['Valid'], name='Valid', marker_color='#00D4FF'))
    fig_comp.add_trace(go.Bar(x=feature_means['Feature'], y=feature_means['Fraud'], name='Fraud', marker_color='#FF4B4B'))
    fig_comp.update_layout(title="Average Feature Values (V1-V28) by Class", barmode='group', template='plotly_dark',
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)

elif selected == "Model Training":
    st.title("🤖 Model Training Lab")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuration")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        n_estimators = st.slider("Number of Trees (Random Forest)", 10, 100, 50)
        
        st.warning("Training on full dataset might be slow. We'll use a smart subsample for demo speed or enable full training.", icon="⚡")
        use_full = st.checkbox("Train on Full Dataset (Slower)", value=False)
        
        if st.button("🚀 Train Model"):
            with st.spinner("Preprocessing and Training..."):
                X = df.drop(['Class'], axis=1)
                Y = df["Class"]
                
                # If not full, sample to balance slightly or just reduce size?
                # Random Forest handles imbalance decently, but for speed let's just sample if requested
                if not use_full:
                    # Stratified sampling to keep class ratio
                    # Taking 20% of data
                    sample_indices = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.2))
                    X = X.loc[sample_indices.index]
                    Y = Y.loc[sample_indices.index]
                
                xTrain, xTest, yTrain, yTest = train_test_split(X.values, Y.values, test_size=test_size, random_state=42)
                
                rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                rfc.fit(xTrain, yTrain)
                yPred = rfc.predict(xTest)
                
                # Store in session state for prediction tab
                st.session_state['model'] = rfc
                st.session_state['trained'] = True
                
                # Metrics
                acc = accuracy_score(yTest, yPred)
                prec = precision_score(yTest, yPred)
                rec = recall_score(yTest, yPred)
                f1 = f1_score(yTest, yPred)
                mcc = matthews_corrcoef(yTest, yPred)
                
                # Store metrics in session state
                st.session_state['metrics'] = {
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'mcc': mcc,
                    'cm': confusion_matrix(yTest, yPred)
                }
                
                st.success("Model Trained Successfully!")
                
    with col2:
        st.markdown("### Performance Metrics")
        if 'trained' in st.session_state and 'metrics' in st.session_state:
            metrics = st.session_state['metrics']
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['acc']:.4f}", delta="Model Stats")
            m2.metric("Precision", f"{metrics['prec']:.4f}")
            m3.metric("Recall", f"{metrics['rec']:.4f}")
            m4.metric("F1 Score", f"{metrics['f1']:.4f}")
            
            st.metric("Matthews Correlation Coefficient", f"{metrics['mcc']:.4f}")
            
            st.markdown("#### Evaluation & Confusion Matrix")
            # Metrics Visual
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [metrics['acc'], metrics['prec'], metrics['rec'], metrics['f1']]
            })
            
            c1, c2 = st.columns(2)
            with c1:
                fig_met = px.bar(metrics_df, x='Metric', y='Value', color='Value', 
                                title="Model Performance", range_y=[0, 1],
                                color_continuous_scale='teal', template='plotly_dark')
                fig_met.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_met, use_container_width=True)
            
            with c2:
                cm = metrics['cm']
                fig_cm = px.imshow(cm, text_auto=True, 
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['Valid', 'Fraud'], y=['Valid', 'Fraud'],
                                title="Confusion Matrix",
                                color_continuous_scale='Blues', template='plotly_dark')
                fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("👈 Configure and click 'Train Model' to see results.")

elif selected == "Prediction":
    st.title("🔍 Fraud Prediction Engine")
    
    if 'model' not in st.session_state:
        st.warning("Please train the model related to 'Model Training' tab first!")
        st.stop()
        
    model = st.session_state['model']
    
    st.markdown("### Input Transaction Details")
    st.markdown("Enter the values for V1-V28 and Amount. (Time is excluded for this generic model)")
    
    input_data = {}
    
    # Create expanders for inputs to save space
    with st.expander("Transaction Features (V1 - V14)", expanded=True):
        c_v1 = st.columns(4)
        for i in range(1, 15):
            with c_v1[(i-1)%4]:
                input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
                
    with st.expander("Transaction Features (V15 - V28)", expanded=False):
        c_v2 = st.columns(4)
        for i in range(15, 29):
            with c_v2[(i-15)%4]:
                input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
    
    col_last = st.columns(2)
    with col_last[0]:
        input_data["Amount"] = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01, value=100.0)
    with col_last[1]:
        # Dummy Time input if needed by model shape, usually dropped or unused in simple RF
        # Based on notebook, X dropped 'Class'. Time was INCLUDED in notebook X.
        # Let's add Time input or auto-generate
        input_data["Time"] = st.number_input("Time (Seconds since first trans)", min_value=0.0, value=0.0)

    # Reorder to match: Time, V1...V28, Amount (as per notebook head print)
    # Notebook: Time, V1...V28, Amount
    ordered_input = [input_data["Time"]] + [input_data[f"V{i}"] for i in range(1, 29)] + [input_data["Amount"]]
    
    if st.button("🔎 Analyze Transaction"):
        prediction = model.predict([ordered_input])[0]
        probability = model.predict_proba([ordered_input])[0][1] # Prob of class 1 (Fraud)
        
        st.markdown("---")
        if prediction == 1:
            st.error("🚨 HIGH RISK: This transaction is likely FRAUDULENT!")
            st.progress(float(probability))
            st.write(f"Fraud Probability: {probability*100:.2f}%")
        else:
            st.success("✅ LOW RISK: This transaction appears VALID.")
            st.progress(float(probability))
            st.write(f"Fraud Probability: {probability*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Developed with Streamlit for Financial Security • © 2026</div>", unsafe_allow_html=True)
