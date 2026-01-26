import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import pickle
import numpy as np
from collections import Counter

# --- 1. APP CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="SkillSync Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it look "Enterprise"
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND FUNCTIONS (Cached for Speed) ---
@st.cache_data # <--- This stops the DB from reloading on every click
def load_data():
    try:
        conn = sqlite3.connect('jobs.db')
        df = pd.read_sql("SELECT * FROM jobs_cleaned", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def get_skill_frequency(df, top_n=10):
    all_skills = []
    for skill_str in df['Skills_Detected']:
        if skill_str and skill_str != "None":
            skills = [s.strip() for s in skill_str.split(',')]
            all_skills.extend(skills)
    if not all_skills: return pd.DataFrame()
    return pd.DataFrame(Counter(all_skills).most_common(top_n), columns=['Skill', 'Count'])

# --- 3. MAIN APPLICATION ---
def main():
    # Sidebar Branding
    st.sidebar.title(" SkillSync Pro")
    st.sidebar.markdown("---")

    # Load Data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ Database not found. Please run 'cleaner.py' first.")
        return

    # --- GLOBAL FILTERS (Sidebar) ---
    st.sidebar.header("🔍 Global Filters")
    
    locations = ["All"] + sorted(list(df['Location_Clean'].unique()))
    selected_loc = st.sidebar.selectbox("Target Location", locations)
    
    max_exp_val = int(df['Min_Exp'].max()) if not df.empty else 15
    min_exp = st.sidebar.slider("Minimum Experience (Years)", 0, max_exp_val, 0)

    # Apply Filters
    df_filtered = df[df['Min_Exp'] >= min_exp]
    if selected_loc != "All":
        df_filtered = df_filtered[df_filtered['Location_Clean'] == selected_loc]

    # --- MAIN LAYOUT WITH TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Insights", "🤖 Salary Predictor", "📝 Resume Matcher", "💾 Data Store"])

    # TAB 1: MARKET INSIGHTS (Visuals)
    with tab1:
        st.subheader(f"Market Overview: {selected_loc}")
        
        # Top Row Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Jobs", len(df_filtered), delta="Live Data")
        c2.metric("Avg Experience", f"{df_filtered['Min_Exp'].mean():.1f} Yrs")
        
        # Dynamic Top Skill
        top_skill_name = "N/A"
        skill_df = get_skill_frequency(df_filtered)
        if not skill_df.empty:
            top_skill_name = skill_df.iloc[0]['Skill']
        c3.metric("Top Skill", top_skill_name)
        
        # Unique Companies
        c4.metric("Hiring Companies", df_filtered['Company'].nunique())

        st.divider()

        # Charts Row
        col_left, col_right = st.columns((2, 1))
        
        with col_left:
            st.markdown("### 🔥 Skill Demand Heatmap")
            if not skill_df.empty:
                # Interactive Plotly Bar Chart
                fig = px.bar(skill_df, x='Count', y='Skill', orientation='h', 
                             title=f"Top Skills in {selected_loc}",
                             color='Count', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No skill data available.")

        with col_right:
            st.markdown("### 📍 Experience vs. Volume")
            exp_counts = df_filtered['Min_Exp'].value_counts().reset_index()
            exp_counts.columns = ['Years', 'Count']
            fig2 = px.pie(exp_counts, values='Count', names='Years', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig2, use_container_width=True)

    # TAB 2: SALARY PREDICTOR (ML Model)
    with tab2:
        st.subheader("💰 AI Salary Estimator")
        st.markdown("Using a **Random Forest Regressor** trained on market patterns.")
        
        c_input, c_result = st.columns(2)
        
        with c_input:
            st.info("Configure Candidate Profile")
            ml_exp = st.number_input("Years of Experience", 0, 20, 3)
            ml_python = st.checkbox("Proficient in Python?", value=True)
            ml_sql = st.checkbox("Proficient in SQL?", value=True)
            ml_aws = st.checkbox("Proficient in AWS?", value=False)
            ml_excel = st.checkbox("Proficient in Excel?", value=False)
            
            predict_btn = st.button("Predict Market Value", type="primary")

        with c_result:
            if predict_btn:
                try:
                    with open('salary_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    
                    input_vec = [[ml_exp, int(ml_python), int(ml_sql), int(ml_aws), int(ml_excel)]]
                    pred = model.predict(input_vec)[0]
                    
                    st.success("Prediction Complete")
                    st.metric(label="Estimated Annual Salary (LPA)", value=f"₹{pred:.2f} LPA")
                    st.progress(min(pred/30, 1.0)) # Visual bar relative to 30 LPA
                    st.caption("*Estimate based on current market data scraping.*")
                    
                except FileNotFoundError:
                    st.error("ML Model not found. Run 'model_train.py' first.")

    # TAB 3: RESUME MATCHER
    with tab3:
        st.subheader("📝 Resume Keyword Matcher")
        
        # Get all skills from DB
        all_skills_df = get_skill_frequency(df)
        all_unique = sorted(all_skills_df['Skill'].unique()) if not all_skills_df.empty else []
        
        # Defaults
        defaults = [s for s in ["Python", "SQL"] if s in all_unique]
        
        user_skills = st.multiselect("Select your Tech Stack:", all_unique, default=defaults)
        
        if st.button("Find Matching Jobs"):
            def score_job(job_skills):
                if not job_skills or job_skills == "None": return 0
                j_set = set([x.strip() for x in job_skills.split(',')])
                u_set = set(user_skills)
                return len(j_set.intersection(u_set))
            
            match_df = df_filtered.copy()
            match_df['Score'] = match_df['Skills_Detected'].apply(score_job)
            match_df = match_df.sort_values('Score', ascending=False).head(5)
            
            for _, row in match_df.iterrows():
                with st.container():
                    c1, c2 = st.columns((3, 1))
                    c1.markdown(f"**{row['Title']}** at *{row['Company']}*")
                    c1.caption(f"📍 {row['Location_Clean']} | ⏳ {row['Min_Exp']} Yrs")
                    c1.write(f"🛠 {row['Skills_Detected']}")
                    
                    # Visual Match Score
                    if len(user_skills) > 0:
                        match_pct = min(row['Score'] / len(user_skills), 1.0)
                        c2.progress(match_pct, text=f"Match: {int(match_pct*100)}%")
                    st.divider()

    # TAB 4: DATA STORE
    with tab4:
        st.subheader("💾 Raw Data Explorer")
        st.dataframe(df_filtered, use_container_width=True)
        
        # CSV Download Button
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='skillsync_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()