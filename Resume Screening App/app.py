import streamlit as st
import pandas as pd
import numpy as np
import pickle
import docx2txt
import PyPDF2
import re
import string
import requests
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Dependencies Check & Imports
# -----------------------------
try:
    from streamlit_lottie import st_lottie
    from streamlit_option_menu import option_menu
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("⚠️ Dependencies missing! Please run: pip install -r requirements.txt")
    st.stop()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Custom CSS (Glassmorphism)
# -----------------------------
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(242, 243, 248) 0%, rgb(219, 237, 242) 90%);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Glassmorphism Card Style */
    .glass-container {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        padding: 25px;
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15);
    }

    /* Headlines */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        transform: scale(1.02);
    }

    /* Metrics */
    .metric-card {
        text-align: center;
        padding: 15px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_resource
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Models
@st.cache_resource
def load_model():
    model = pickle.load(open("clf.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf.pkl", "rb"))
    return model, vectorizer

# Robust Model Loading
try:
    clf, tfidf = load_model()
except FileNotFoundError:
    st.warning("⚠️ Model files not found! Running in Demo Mode.")
    class Placeholder:
        def predict(self, X): return np.array([6])
        def predict_proba(self, X): return np.array([[0.04] * 25])
        def transform(self, X): return np.zeros((len(X), 1000))
    clf, tfidf = Placeholder(), Placeholder()

def cleanResume(txt):
    cleanTxt = re.sub(r'http\S+\s', ' ', txt)
    cleanTxt = re.sub(r'RT|S+', ' ', cleanTxt)
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)
    cleanTxt = re.sub(r'#\S+\s', ' ', cleanTxt)
    cleanTxt = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt)
    return cleanTxt.strip().lower()

def extract_text(uploaded_file):
    uploaded_file.seek(0)
    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        except: return ""
    elif "wordprocessingml" in uploaded_file.type:
        try:
            return docx2txt.process(uploaded_file)
        except: return ""
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return ""

def extract_skills(text):
    skills_db = [
        'python', 'java', 'c++', 'sql', 'machine learning', 'deep learning', 'nlp', 
        'data analysis', 'aws', 'azure', 'docker', 'kubernetes', 'react', 'angular', 
        'html', 'css', 'javascript', 'communication', 'leadership', 'teamwork',
        'project management', 'agile', 'scrum', 'git', 'linux', 'statistics', 
        'tableau', 'power bi', 'excel', 'tensorflow', 'pytorch', 'flask', 'django', 
        'pandas', 'numpy', 'scikit-learn'
    ]
    found = [s for s in skills_db if s in text.lower()]
    return list(set(found))

category_mapping = {
    0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain",
    4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science",
    7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
    10: "ETL Developer", 11: "Electrical Engineering", 12: "HR",
    13: "Hadoop", 14: "Health and Fitness", 15: "Java Developer",
    16: "Mechanical Engineer", 17: "Network Security Engineer",
    18: "Operations Manager", 19: "PMO", 20: "Python Developer",
    21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing"
}

# -----------------------------
# Navigation (Sidebar)
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80) 
    st.markdown("## Resume AI")
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Resume Scanner", "Batch Analysis", "Job Match", "Analytics"],
        icons=["house", "file-earmark-person", "files", "check-circle", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    st.caption("© 2024 AI Resume Screener")

# -----------------------------
# MAIN CONTENT
# -----------------------------

# --- HOME PAGE ---
if selected == "Home":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        st.markdown("<h1>👋 Welcome to <br><span style='color: #4b6cb7;'>AI Resume Screening</span></h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-container">
            <h3>🤖 Next-Gen Recruitment</h3>
            <p>Leverage the power of Artificial Intelligence to automate your hiring workflow.</p>
            <ul>
                <li>🚀 <b>Instant Categorization</b></li>
                <li>📊 <b>Confidence Scoring</b></li>
                <li>🧠 <b>Skill Extraction</b></li>
                <li>📈 <b>Interactive Analytics</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start Screening Now"):
            st.toast("Go to the 'Resume Scanner' tab!", icon="🚀")

    with col2:
        lottie_home = load_lottieurl("https://lottie.host/5a67b576-9634-4588-90c7-43407v87556f/5X698038.json")
        if lottie_home:
            st_lottie(lottie_home, height=450)
        else:
            st.image("https://cdn.dribbble.com/users/2069402/screenshots/5643486/resume.gif")

# --- RESUME SCANNER ---
elif selected == "Resume Scanner":
    st.markdown("<h1>📄 Intelligent Resume Scanner</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-container'>Upload a resume to instantly analyze its content, predict the job role, and extract key skills.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📂 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        col_anim, col_res = st.columns([1, 2])
        
        with col_anim:
            # Scanning Animation
            lottie_scan = load_lottieurl("https://lottie.host/9d819973-10d0-4d87-9566-107077758957/026857.json") 
            if lottie_scan:
                st_lottie(lottie_scan, height=250, key="scan")
        
        with col_res:
            if st.button("🔍 Analyze Resume"):
                with st.spinner("Processing..."):
                    time.sleep(1.5) # Simulate processing for effect
                    raw_text = extract_text(uploaded_file)
                    
                    if raw_text:
                        cleaned = cleanResume(raw_text)
                        tfidf_features = tfidf.transform([cleaned])
                        pred_id = clf.predict(tfidf_features)[0]
                        confidence = clf.predict_proba(tfidf_features)[0][pred_id] * 100
                        category = category_mapping.get(pred_id, "Unknown")
                        skills = extract_skills(raw_text)
                        
                        st.success("Analysis Complete!")
                        st.balloons()
                        
                        # --- RESULTS SECTION ---
                        st.markdown("---")
                        
                        # Gauge Chart for Confidence
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Confidence ({category})"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#4b6cb7"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#f4f4f4"},
                                    {'range': [50, 80], 'color': "#e6f2ff"},
                                    {'range': [80, 100], 'color': "#d6eaff"}],
                                'threshold': {
                                    'line': {'color': "#2ecc71", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Skills Display
                        st.markdown(f"""
                        <div class="glass-container">
                            <h3>🛠️ Extracted Skills</h3>
                            {' '.join([f"<span style='background-color: #e3f2fd; color: #1565c0; padding: 5px 12px; border-radius: 15px; margin: 5px; display: inline-block; font-weight: bold;'>{skill}</span>" for skill in skills])}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📄 View Extracted Text"):
                            st.text(raw_text)
                    else:
                        st.error("Failed to extract text.")

# --- BATCH ANALYSIS ---
elif selected == "Batch Analysis":
    st.markdown("<h1>📂 Batch Resume Analysis</h1>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Multiple Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 Analyze All"):
            progress = st.progress(0)
            results = []
            
            for i, f in enumerate(uploaded_files):
                text = extract_text(f)
                cleaned = cleanResume(text)
                if cleaned:
                    input_features = tfidf.transform([cleaned])
                    pred = clf.predict(input_features)[0]
                    conf = clf.predict_proba(input_features)[0][pred]
                    results.append({
                        "Filename": f.name,
                        "Category": category_mapping.get(pred, "Unknown"),
                        "Confidence": round(conf * 100, 2),
                        "Skills": ", ".join(extract_skills(text))
                    })
                progress.progress((i + 1) / len(uploaded_files))
            
            df = pd.DataFrame(results)
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.download_button("📥 Download Report (CSV)", df.to_csv(index=False), "resume_batch_report.csv", "text/csv")

# --- JOB MATCH ---
elif selected == "Job Match":
    st.markdown("<h1>🎯 Job Description Matcher</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1. Upload Resume")
        resume_file = st.file_uploader("", type=["pdf", "docx", "txt"], key="jd_resume")
    with c2:
        st.markdown("### 2. Job Description")
        job_desc = st.text_area("Paste text here...", height=150)
        
    if st.button("Check Compatibility"):
        if resume_file and job_desc:
            with st.spinner("Calculating Match..."):
                r_text = cleanResume(extract_text(resume_file))
                j_text = cleanResume(job_desc)
                
                vecs = tfidf.transform([r_text, j_text])
                sim = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
                
                # Visual Indicator
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sim,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Match Score"},
                    delta = {'reference': 70},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#2ecc71" if sim > 70 else "#f1c40f" if sim > 50 else "#e74c3c"},
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                if sim > 75:
                    st.success("🌟 Excellent Match!")
                elif sim > 50:
                    st.warning("⚠️ Average Match")
                else:
                    st.error("❌ Low Match")

# --- ANALYTICS ---
elif selected == "Analytics":
    st.markdown("<h1>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Upload a folder of resumes to visualize the talent pool distribution.")
    
    files = st.file_uploader("Upload Resumes for Analytics", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="analytics")
    
    if files:
        if st.button("Generate Dashboard"):
            cats = []
            for f in files:
                text = cleanResume(extract_text(f))
                if text:
                    pred = clf.predict(tfidf.transform([text]))[0]
                    cats.append(category_mapping.get(pred, "Unknown"))
            
            if cats:
                df = pd.DataFrame(cats, columns=["Category"])
                counts = df["Category"].value_counts().reset_index()
                counts.columns = ["Category", "Count"]   
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_pie = px.pie(counts, values='Count', names='Category', title='Category Distribution', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    fig_bar = px.bar(counts, x='Category', y='Count', title='Talent Count by Category', color='Count')
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No data found.")