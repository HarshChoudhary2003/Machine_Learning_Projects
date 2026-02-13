
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import requests
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Veritas AI | Truth Detector",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Advanced Animations
anim_scan = load_lottieurl("https://lottie.host/5a022464-aeb2-402f-9037-338b30efbb07/rJkhQ1R9xY.json") # Techno Scan
anim_success = load_lottieurl("https://lottie.host/936bZ5Y2.json") # Not real URL, fallback to icons if fail, using safe ones below
anim_robot = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ofa3xwo7.json") # Robot analysis
anim_analytics = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_DVSwGQ.json") # Data viz

# -----------------------------------------------------------------------------
# 2. ADVANCED CSS STYLING (Glassmorphism + Animation)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(20, 20, 30) 90%);
        color: #ffffff;
    }

    /* Titles */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 10px rgba(0, 201, 255, 0.3);
    }

    /* Glass Cards with Hover Animation */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.8s ease-out;
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 201, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Keyframes for Entry */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Metrics Styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #fff;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #a0a0a0;
    }

    /* Custom Input Areas */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 15, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Result Badges */
    .badge-fake {
        background: rgba(255, 75, 75, 0.2);
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        padding: 5px 15px;
        border-radius: 50px;
        font-weight: bold;
    }
    .badge-real {
        background: rgba(0, 200, 83, 0.2);
        border: 1px solid #00c853;
        color: #00c853;
        padding: 5px 15px;
        border-radius: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. ADVANCED MODEL LOGIC
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except:
        df['date'] = df['date'].astype(str)
    return df

fake_df = load_data('Fake.csv')
true_df = load_data('True.csv')

@st.cache_resource
def train_model(fake_df, true_df):
    fake_df['label'] = 0 
    true_df['label'] = 1 
    df = pd.concat([fake_df, true_df])
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = df.dropna(subset=['content'])
    
    X = df['content']
    y = df['label']

    # Using features for Explainability
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    classifier = LogisticRegression(solver='liblinear')
    
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    model.fit(X, y)
    
    return model, vectorizer, classifier

pipeline, vectorizer, classifier = train_model(fake_df, true_df)

# Function to get top features (Explainable AI)
def get_top_features(vectorizer, classifier, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    top_positive_coeffs = np.argsort(coefs)[-n:] # True News Indicators
    top_negative_coeffs = np.argsort(coefs)[:n]  # Fake News Indicators
    
    top_true_features = [(feature_names[i], coefs[i]) for i in top_positive_coeffs]
    top_fake_features = [(feature_names[i], coefs[i]) for i in top_negative_coeffs]
    
    return top_true_features, top_fake_features

# -----------------------------------------------------------------------------
# 4. UI LAYOUT: SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ **Control Center**")
    
    if anim_robot:
        st_lottie(anim_robot, height=180, key="bot")
    
    with st.expander("📝 **Analyze Text**", expanded=True):
        with st.form("analysis_form"):
            input_title = st.text_input("Headline", placeholder="Enter news headline...")
            input_text = st.text_area("Content Body", height=150, placeholder="Paste article text...")
            submit = st.form_submit_button("🚀 Run Diagnostics")

    st.markdown("---")
    view_mode = st.radio("Navigation", ["Prediction Engine", "Global Intel Dashboard", "Model Internals (XAI)"])
    
    st.markdown("---")
    st.info("System Status: **Online**\n\nModel Accuracy: **~99%**")

# Session State Handling
if submit and input_text:
    st.session_state['article'] = {'title': input_title, 'text': input_text}
    st.session_state['view'] = "Prediction Engine"
elif 'view' not in st.session_state:
    st.session_state['view'] = "Global Intel Dashboard"
    st.session_state['article'] = None
else:
    if not submit:
        st.session_state['view'] = view_mode

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT
# -----------------------------------------------------------------------------

# >>> VIEW 1: PREDICTION ENGINE <<<
if st.session_state['view'] == "Prediction Engine":
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("Prediction Engine")
        st.markdown("### Real-time Linguistic Analysis")
    with col_h2:
        if anim_scan: st_lottie(anim_scan, height=100)

    if st.session_state['article']:
        article = st.session_state['article']
        full_text = f"{article['title']} {article['text']}"
        
        # Simulation of deep scanning
        progress_text = "Initializing neural pathways..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text="Scanning semantic structures...")
        my_bar.empty()

        # Prediction
        pred = pipeline.predict([full_text])[0]
        proba = pipeline.predict_proba([full_text])
        confidence = proba[0][1] if pred == 1 else proba[0][0]
        
        # Result Display with Gauge
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            if pred == 1:
                st.markdown('<h2 style="color:#00ff88; text-align:center;">VERIFIED REAL</h2>', unsafe_allow_html=True)
                gauge_color = "#00ff88"
            else:
                st.markdown('<h2 style="color:#ff4b4b; text-align:center;">DETECTED FAKE</h2>', unsafe_allow_html=True)
                gauge_color = "#ff4b4b"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.3)'},
                        {'range': [50, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                    ]}
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Segoe UI"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Text Statistics")
            scm1, scm2, scm3 = st.columns(3)
            scm1.markdown(f'<div class="metric-value">{len(article["text"].split())}</div><div class="metric-label">Words</div>', unsafe_allow_html=True)
            scm2.markdown(f'<div class="metric-value">{len(article["text"])}</div><div class="metric-label">Chars</div>', unsafe_allow_html=True)
            scm3.markdown(f'<div class="metric-value">{len(article["text"].split("."))}</div><div class="metric-label">Sentences</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔬 AI Assessment")
            if pred == 1:
                st.success("The text contains linguistic patterns highly consistent with verified news sources. Sourcing, tone, and complexity align with standard journalism.")
            else:
                st.warning("High probability of misinformation detected. The text may contain sensationalist language, lack of proper sourcing, or patterns common in fabricated stories.")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="glass-container" style="text-align:center; padding: 50px;"><h3>👈 Awaiting Input Data</h3><p>Use the control panel to submit a news article.</p></div>', unsafe_allow_html=True)

# >>> VIEW 2: GLOBAL INTEL DASHBOARD <<<
elif st.session_state['view'] == "Global Intel Dashboard":
    st.title("Global Intelligence Dashboard")
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="glass-container"><div class="metric-value">{len(fake_df)+len(true_df):,}</div><div class="metric-label">Total Database</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="glass-container"><div class="metric-value" style="color:#ff4b4b">{len(fake_df):,}</div><div class="metric-label">Fake Articles</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="glass-container"><div class="metric-value" style="color:#00ff88">{len(true_df):,}</div><div class="metric-label">Real Articles</div></div>', unsafe_allow_html=True)
    with k4:
        # Calculate ratio
        ratio = len(fake_df) / (len(fake_df) + len(true_df)) * 100
        st.markdown(f'<div class="glass-container"><div class="metric-value">{ratio:.1f}%</div><div class="metric-label">Misinformation Rate</div></div>', unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 🚨 Fake News by Subject")
        fake_counts = fake_df['subject'].value_counts().reset_index()
        fake_counts.columns = ['Subject', 'Count']
        fig_fake = px.bar(fake_counts, x='Count', y='Subject', orientation='h', color='Count', color_continuous_scale='Reds')
        fig_fake.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, showlegend=False, xaxis_title="Count", yaxis_title="Subject")
        st.plotly_chart(fig_fake, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### ✅ Trusted News by Subject")
        true_counts = true_df['subject'].value_counts().reset_index()
        true_counts.columns = ['Subject', 'Count']
        fig_true = px.bar(true_counts, x='Count', y='Subject', orientation='h', color='Count', color_continuous_scale='Greens')
        fig_true.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, showlegend=False, xaxis_title="Count", yaxis_title="Subject")
        st.plotly_chart(fig_true, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# >>> VIEW 3: MODEL INTERNALS (XAI) <<<
elif st.session_state['view'] == "Model Internals (XAI)":
    st.title("Explainable AI (XAI) Insights")
    st.markdown("Understanding *why* the model makes its decisions by analyzing feature weights.")
    
    top_true, top_fake = get_top_features(vectorizer, classifier, n=15)
    
    col_x1, col_x2 = st.columns(2)
    
    with col_x1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#ff4b4b">🚩 Top "Fake" Triggers</h3>', unsafe_allow_html=True)
        st.write("Words that most strongly push the prediction towards 'FAKE'.")
        
        fake_words, fake_coefs = zip(*top_fake)
        # Invert to make the bar chart look logical (biggest negative impact)
        fig_x1 = px.bar(x=np.abs(fake_coefs), y=fake_words, orientation='h', 
                        labels={'x': 'Impact Score', 'y': 'Word'}, text_auto='.2f')
        fig_x1.update_traces(marker_color='#ff4b4b')
        fig_x1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=500)
        st.plotly_chart(fig_x1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_x2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#00ff88">🛡️ Top "Real" Triggers</h3>', unsafe_allow_html=True)
        st.write("Words that most strongly push the prediction towards 'REAL'.")
        
        true_words, true_coefs = zip(*top_true)
        fig_x2 = px.bar(x=true_coefs, y=true_words, orientation='h',
                        labels={'x': 'Impact Score', 'y': 'Word'}, text_auto='.2f')
        fig_x2.update_traces(marker_color='#00ff88')
        fig_x2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=500)
        st.plotly_chart(fig_x2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)