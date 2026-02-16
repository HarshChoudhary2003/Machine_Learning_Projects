import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import pickle
import numpy as np
from collections import Counter
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="SkillSync Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THEME & ANIMATIONS ---
# --- 3. DATA & FUNCTIONS ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_resource
def load_model():
    try:
        with open('salary_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('jobs.db')
        df = pd.read_sql("SELECT * FROM jobs_cleaned", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_skill_frequency(df, top_n=10):
    if df.empty: return pd.DataFrame()
    
    # Pre-calculated optimization: Vectorized string processing is faster than loops for large DFs
    # But for simplicity and safety with 'Skills_Detected' structure, we'll keep it robust but cached.
    all_skills = []
    # Drop NAs and iterate
    series = df['Skills_Detected'].dropna()
    for skill_str in series:
        if skill_str and skill_str != "None":
            # Using simple split is efficient enough for typical job market datasets
            all_skills.extend([s.strip() for s in skill_str.split(',')])
            
    if not all_skills: return pd.DataFrame()
    # Counter is implemented in C, extremely fast
    return pd.DataFrame(Counter(all_skills).most_common(top_n), columns=['Skill', 'Count'])

# --- 4. ASSETS & STYLING ---
# Load specific animations (Cached)
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
lottie_loading = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_x62chJ.json")
lottie_analytics = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_sg18igu8.json")

# Custom CSS for Glassmorphism & High-End UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 18, 25) 0%, rgb(5, 5, 10) 90%);
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0c10;
        border-right: 1px solid #1f2833;
    }
    
    /* Metrics Card (Glassmorphism) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 198, 255, 0.2);
        border-color: rgba(0, 198, 255, 0.5);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #66fcf1, #45a29e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 1rem;
        color: #c5c6c7;
        margin-top: 5px;
    }

    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #45a29e 0%, #66fcf1 100%);
        color: #0b0c10;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(102, 252, 241, 0.4);
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    h1 {
        background: linear-gradient(90deg, #ffffff, #66fcf1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Expander & Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        color: #fff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #45a29e, #66fcf1);
        color: #0b0c10 !important;
    }
    
    /* Result Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #66fcf1;
        transition: all 0.3s;
    }
    .result-card:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. LAYOUT COMPONENTS ---
def metric_card(label, value, icon):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{icon} {label}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. MAIN APP ---
def main():
    # SIDEBAR NAVIGATION
    with st.sidebar:
        if lottie_loading:
            st_lottie(lottie_loading, height=150, key="logo_anim")
        
        st.markdown("## ⚡ SkillSync Pro")
        
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Salary AI", "Resume Matcher", "Raw Data"],
            icons=["bar-chart-fill", "currency-dollar", "file-earmark-person", "database"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#66fcf1", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#1f2833"},
                "nav-link-selected": {"background-color": "#45a29e"},
            }
        )
        
        st.markdown("---")
        
        # FILTERS
        df = load_data()
        if df.empty:
            st.error("⚠️ Database connection failed.")
            return

        st.markdown("### 🔍 Filters")
        locations = ["All"] + sorted(list(df['Location_Clean'].unique()))
        selected_loc = st.selectbox("Location", locations)
        
        min_exp = st.slider("Min Experience (Years)", 0, int(df['Min_Exp'].max()) if not df.empty else 15, 0)
        
        # Apply Filters
        df_filtered = df[df['Min_Exp'] >= min_exp]
        if selected_loc != "All":
            df_filtered = df_filtered[df_filtered['Location_Clean'] == selected_loc]

    # PAGE CONTENT
    if selected == "Dashboard":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.title("Market Insights 🚀")
            st.markdown(f"Real-time analytics for **{selected_loc}** market.")
        with col2:
            if lottie_analytics:
                st_lottie(lottie_analytics, height=120, key="dash_anim")

        # METRICS ROW
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Active Jobs", len(df_filtered), "🔥")
        with c2: metric_card("Avg Experience", f"{df_filtered['Min_Exp'].mean():.1f} Yrs", "⏳")
        with c3:
            top_skill = "N/A"
            s_df = get_skill_frequency(df_filtered)
            if not s_df.empty: top_skill = s_df.iloc[0]['Skill']
            metric_card("Top Skill", top_skill, "💡")
        with c4: metric_card("Companies", df_filtered['Company'].nunique(), "🏢")

        st.markdown("---")

        # CHARTS
        col_left, col_right = st.columns((2, 1))
        
        with col_left:
            st.markdown("### 🔥 Skill Demand Heatmap")
            if not s_df.empty:
                fig = px.bar(s_df, x='Count', y='Skill', orientation='h', 
                             title=f"Most Demanded Skills in {selected_loc}",
                             color='Count', color_continuous_scale='teal')
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for current filters.")

        with col_right:
            st.markdown("### 📍 Experience Split")
            exp_counts = df_filtered['Min_Exp'].value_counts().reset_index()
            exp_counts.columns = ['Years', 'Count']
            fig2 = px.pie(exp_counts, values='Count', names='Years', hole=0.6, 
                          color_discrete_sequence=px.colors.sequential.Teal)
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

    elif selected == "Salary AI":
        c1, c2 = st.columns([1, 1])
        with c1:
            st.title("💰 Salary Estimator")
            st.markdown("Predict your market value using our **Random Forest AI Model**.")
            if lottie_coding:
                st_lottie(lottie_coding, height=200)

        with c2:
            st.markdown('<div class="glass-container" style="padding: 2rem; border-radius: 12px; background: rgba(255,255,255,0.05);">', unsafe_allow_html=True)
            st.subheader("Candidate Profile")
            ml_exp = st.slider("Years of Experience", 0, 20, 3)
            
            cc1, cc2 = st.columns(2)
            with cc1:
                ml_python = st.checkbox("Python 🐍", value=True)
                ml_sql = st.checkbox("SQL 🗄️", value=True)
            with cc2:
                ml_aws = st.checkbox("AWS ☁️", value=False)
                ml_excel = st.checkbox("Excel 📊", value=False)
            
            if st.button("Predict Salary 🚀", use_container_width=True):
                model = load_model()
                if model:
                    input_vec = [[ml_exp, int(ml_python), int(ml_sql), int(ml_aws), int(ml_excel)]]
                    pred = model.predict(input_vec)[0]
                    
                    st.success("Prediction Successful!")
                    metric_card("Estimated Salary", f"₹{pred:.2f} LPA", "💰")
                    st.progress(min(pred/40, 1.0))
                else:
                    st.error("Model file not found. Please train the model first.")
            st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Resume Matcher":
        st.title("📝 Intelligent Resume Matcher")
        
        all_skills_df = get_skill_frequency(df)
        all_unique = sorted(all_skills_df['Skill'].unique()) if not all_skills_df.empty else []
        
        with st.expander("🛠 Configure Your Tech Stack", expanded=True):
            user_skills = st.multiselect("Select Skills", all_unique, default=["Python", "SQL"] if "Python" in all_unique else [])
        
        if st.button("Find Best Matches 🔍", type="primary"):
            def score_job(job_skills):
                if not job_skills or job_skills == "None": return 0
                j_set = set([x.strip() for x in job_skills.split(',')])
                u_set = set(user_skills)
                return len(j_set.intersection(u_set))
            
            match_df = df_filtered.copy()
            match_df['Score'] = match_df['Skills_Detected'].apply(score_job)
            match_df = match_df.sort_values('Score', ascending=False).head(5)
            
            st.markdown("### Top Recommendations")
            for _, row in match_df.iterrows():
                match_pct = 0
                if len(user_skills) > 0:
                    match_pct = int(min(row['Score'] / len(user_skills), 1.0) * 100)
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3>{row['Title']}</h3>
                        <span style="background:#45a29e; padding:5px 10px; border-radius:5px; font-weight:bold;">{match_pct}% Match</span>
                    </div>
                    <p style="color:#c5c6c7; margin:0;">🏢 {row['Company']} | 📍 {row['Location_Clean']} | ⏳ {row['Min_Exp']} Yrs</p>
                    <p style="margin-top:10px; font-size:0.9rem;">🛠 {row['Skills_Detected']}</p>
                </div>
                """, unsafe_allow_html=True)

    elif selected == "Raw Data":
        st.title("💾 Data Warehouse")
        st.dataframe(df_filtered, use_container_width=True, height=600)

if __name__ == "__main__":
    main()