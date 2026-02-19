import streamlit as st
import os
import requests
import json
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Learning Projects Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/HarshChoudhary2003',
        'Report a bug': "https://github.com/HarshChoudhary2003/Machine_Learning_Projects",
        'About': "# ML Projects Hub\nCurated by Harsh Choudhary"
    }
)

# -----------------------------------------------------------------------------
# CUSTOM CSS AND ANIMATIONS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    .stApp {
        background-color: #050505;
        font-family: 'Outfit', sans-serif;
        background-image: 
            radial-gradient(at 0% 0%, rgba(0, 173, 181, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(255, 107, 107, 0.05) 0px, transparent 50%);
    }
    
    /* Advanced Typography */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #00ADB5, #D500F9, #00ADB5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite;
        background-size: 200% auto;
        letter-spacing: -2px;
    }
    
    h2, h3 {
        color: #E2E8F0;
        letter-spacing: -0.5px;
    }
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    
    /* Glassmorphism Project Card */
    .project-card {
        background: rgba(30, 35, 43, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        height: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .project-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent, rgba(0, 173, 181, 0.1), transparent);
        transform: translateX(-100%);
        transition: 0.5s;
    }
    
    .project-card:hover::before {
        transform: translateX(100%);
    }
    
    .project-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #D500F9;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(213, 0, 249, 0.3);
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .category-tag {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 4px 10px;
        border-radius: 30px;
        font-weight: 700;
        color: #000;
    }
    
    .cat-web { background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%); }
    .cat-finance { background: linear-gradient(90deg, #43E97B 0%, #38F9D7 100%); }
    .cat-nlp { background: linear-gradient(90deg, #FA709A 0%, #FEE140 100%); }
    .cat-security { background: linear-gradient(90deg, #FF0844 0%, #FFB199 100%); }
    .cat-vision { background: linear-gradient(90deg, #667EEA 0%, #764BA2 100%); }
    .cat-data { background: linear-gradient(90deg, #E0C3FC 0%, #8EC5FC 100%); }
    
    .project-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8F9FA;
        margin-bottom: 10px;
        line-height: 1.3;
    }
    
    .project-meta {
        display: flex;
        gap: 15px;
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 10px;
    }
    
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 20px;
    }
    
    .tech-pill {
        background: rgba(255, 255, 255, 0.05);
        color: #00ADB5;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        border: 1px solid rgba(0, 173, 181, 0.2);
        transition: 0.2s;
    }
    
    .tech-pill:hover {
        background: rgba(0, 173, 181, 0.2);
        color: #fff;
    }
    
    /* Interactive Button */
    .view-btn {
        display: block;
        width: 100%;
        text-align: center;
        background: transparent;
        color: #00ADB5;
        border: 1px solid #00ADB5;
        padding: 10px;
        border-radius: 12px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .view-btn::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: #00ADB5;
        border-radius: 10px;
        z-index: -2;
    }
    
    .view-btn::before {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0%;
        height: 100%;
        background-color: #00ADB5; 
        transition: all 0.3s;
        border-radius: 10px;
        z-index: -1;
    }
    
    .view-btn:hover {
        color: #fff;
        background-color: #00ADB5;
        box-shadow: 0 5px 15px rgba(0, 173, 181, 0.4);
    }
    
    /* Stats Metric */
    div[data-testid="stMetric"] {
        background-color: rgba(30, 35, 43, 0.5);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 15px;
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00ADB5;
        transform: scale(1.03);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #2D3748;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00ADB5;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
if 'projects_loaded' not in st.session_state:
    st.session_state['projects_loaded'] = False

@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def get_project_stats(path):
    # Calculate file count, size, and last modified
    total_files = 0
    total_lines = 0
    last_modified = 0
    
    try:
        for root, dirs, files in os.walk(path):
            if '.git' in root or '.venv' in root: continue
            for file in files:
                filepath = os.path.join(root, file)
                total_files += 1
                last_modified = max(last_modified, os.path.getmtime(filepath))
                # Simple line count for code files
                if file.endswith(('.py', '.js', '.html', '.css', '.ipynb')):
                    try:
                        with open(filepath, 'r', errors='ignore') as f:
                            total_lines += len(f.readlines())
                    except: pass
    except: pass
    
    return total_files, total_lines, datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d')

@st.cache_data
def get_all_projects(root_dir="."):
    projects = []
    # Project directories to ignore
    ignore_dirs = {'.git', '.venv', '__pycache__', '.streamlit', '.vscode', '.idea', 'node_modules'}
    
    try:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path) and item not in ignore_dirs and not item.startswith('.'):
                
                tech = detect_tech_stack(item_path)
                category = categorize_project(item, tech)
                files_count, lines_count, last_updated = get_project_stats(item_path)
                
                # Complexity score (heuristic)
                complexity = "Beginner"
                if len(tech) > 4 or lines_count > 500: complexity = "Intermediate"
                if "Deep Learning" in item or "Agentic" in item or lines_count > 1500: complexity = "Advanced"
                
                projects.append({
                    "name": item,
                    "path": item_path,
                    "tech": tech,
                    "category": category,
                    "files": files_count,
                    "lines": lines_count,
                    "updated": last_updated,
                    "complexity": complexity,
                    "link": f"https://github.com/HarshChoudhary2003/Machine_Learning_Projects/tree/main/{item.replace(' ', '%20')}"
                })
    except Exception as e:
        st.error(f"Error scanning: {e}")
        
    # Sort by 'updated' descending
    projects.sort(key=lambda x: x['updated'], reverse=True)
    return projects

def detect_tech_stack(path):
    tech = set()
    try:
        for root, dirs, files in os.walk(path):
            if '.venv' in root: continue
            if "requirements.txt" in files:
                with open(os.path.join(root, "requirements.txt"), 'r', errors='ignore') as f:
                    c = f.read().lower()
                    if "streamlit" in c: tech.add("Streamlit")
                    if "tensorflow" in c or "keras" in c: tech.add("TensorFlow")
                    if "torch" in c: tech.add("PyTorch")
                    if "scikit-learn" in c or "sklearn" in c: tech.add("Scikit-Learn")
                    if "transformers" in c or "langchain" in c: tech.add("LangChain/LLM")
                    if "openai" in c: tech.add("OpenAI")
                    if "pandas" in c: tech.add("Pandas")
                    if "plotly" in c: tech.add("Plotly")
                    if "opencv" in c or "cv2" in c: tech.add("OpenCV")
                    if "flask" in c: tech.add("Flask")

            if any(f.endswith('.ipynb') for f in files): tech.add("Jupyter")
    except: pass
    
    if not tech: tech.add("Python")
    return list(tech)

def categorize_project(name, tech):
    name_l = name.lower()
    tech_s = " ".join([t.lower() for t in tech])
    
    if "fraud" in name_l or "spam" in name_l or "security" in name_l: return "Security & Fraud"
    if "stock" in name_l or "price" in name_l or "bitcoin" in name_l or "predict" in name_l: return "Finance & Prediction"
    if "detect" in name_l or "recognition" in name_l or "opencv" in tech_s: return "Computer Vision"
    if "nlp" in tech_s or "sentiment" in name_l or "langchain" in tech_s or "agentic" in name_l or "voice" in name_l: return "NLP & LLMs"
    if "app" in name_l or "streamlit" in tech_s or "web" in name_l: return "Web Apps"
    return "Data Science & ML"

# -----------------------------------------------------------------------------
# ASSETS LOADING
# -----------------------------------------------------------------------------
lottie_brain = load_lottieurl("https://lottie.host/6e6440e2-8b63-4416-b847-798725964d78/Fv4yXl7Q8Z.json")
lottie_analy = load_lottieurl("https://lottie.host/93386004-94e8-48b4-9276-8084a9198642/s5kQ0s0qE3.json") 
lottie_dev = load_lottieurl("https://lottie.host/4b827725-aa8c-4573-8683-1456d2ba3da4/Kx8R4k7Z2H.json")

# -----------------------------------------------------------------------------
# MAIN APP STRUCTURE
# -----------------------------------------------------------------------------

# Load Projects
projects = get_all_projects()

# --- NAVBAR ---
selected = option_menu(
    menu_title=None,
    options=["Home", "Portfolio", "Analytics", "About"],
    icons=["house-door", "collection", "graph-up", "person-circle"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#161B22", "border-radius": "10px"},
        "icon": {"color": "#00ADB5", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#262730"},
        "nav-link-selected": {"background-color": "#00ADB5", "color": "white"},
    }
)

# --- HOME TAB ---
if selected == "Home":
    # Hero Section
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("# Machine Learning<br>Projects Hub", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 1.2rem; color: #A0AEC0; margin-top: -20px; margin-bottom: 30px;">
        Explore a universe of <b>45+ AI solutions</b>. From autonomous agents to real-time financial dashboards, this portfolio showcases the power of modern Machine Learning.
        </div>
        """, unsafe_allow_html=True)
        
        # Call to Action
        if st.button("Browse Portfolio 🚀", type="primary"):
            st.toast("Navigating to Portfolio...")
            # Ideally switch tabs via session state or just direct user

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick Highlight Cards
        hc1, hc2, hc3 = st.columns(3)
        hc1.info(f"**{len(projects)}** Projects")
        hc2.success(f"**{sum([p['lines'] for p in projects]):,}** Lines of Code")
        hc3.warning(f"**{len([p for p in projects if 'Advanced' in p['complexity']])}** Advanced")

    with c2:
        # Display the AI Hexagon Circuit Image
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=450)

    st.markdown("---")
    
    # Featured Projects (Latest 3)
    st.subheader("🔥 Recently Updated")
    
    fc1, fc2, fc3 = st.columns(3)
    for i, p in enumerate(projects[:3]):
        with [fc1, fc2, fc3][i]:
            technologies = p['tech'][:3]
            tech_html = "".join([f"<span class='tech-pill'>{t}</span>" for t in technologies])
            
            cat_class = "cat-data"
            if "Web" in p['category']: cat_class = "cat-web"
            elif "Finance" in p['category']: cat_class = "cat-finance"
            elif "NLP" in p['category']: cat_class = "cat-nlp"
            elif "Security" in p['category']: cat_class = "cat-security"

            st.markdown(f"""
            <div class="project-card">
                <div class="card-header">
                    <span class="category-tag {cat_class}">{p['category']}</span>
                    <span style="font-size:0.8rem; color:#6B7280;">{p['updated']}</span>
                </div>
                <div class="project-title">{p['name'][:22] + '...' if len(p['name'])>22 else p['name']}</div>
                <div class="project-meta">
                    <span>📂 {p['files']} Files</span>
                    <span>⚡ {p['complexity']}</span>
                </div>
                <div class="tech-stack">{tech_html}</div>
                <a href="{p['link']}" target="_blank" class="view-btn">View Code</a>
            </div>
            """, unsafe_allow_html=True)

# --- PORTFOLIO TAB ---
elif selected == "Portfolio":
    st.title("📂 Project Portfolio")
    
    # Filters
    col_search, col_filter, col_sort = st.columns([2, 1, 1])
    with col_search:
        search_q = st.text_input("Search", placeholder="Search by name, tech, or topic...", label_visibility="collapsed")
    with col_filter:
        cat_filter = st.selectbox("Category", ["All"] + sorted(list(set([p['category'] for p in projects]))), label_visibility="collapsed")
    with col_sort:
        sort_by = st.selectbox("Sort By", ["Newest", "Oldest", "Complexity", "Name"], label_visibility="collapsed")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filter Logic
    filtered = projects
    if search_q:
        filtered = [p for p in filtered if search_q.lower() in p['name'].lower() or any(search_q.lower() in t.lower() for t in p['tech'])]
    if cat_filter != "All":
        filtered = [p for p in filtered if p['category'] == cat_filter]
        
    if sort_by == "Newest": filtered.sort(key=lambda x: x['updated'], reverse=True)
    elif sort_by == "Oldest": filtered.sort(key=lambda x: x['updated'])
    elif sort_by == "Name": filtered.sort(key=lambda x: x['name'])
    elif sort_by == "Complexity": filtered.sort(key=lambda x: (x['complexity'] == "Advanced", x['complexity'] == "Intermediate"), reverse=True)

    # Grid Display
    rows = [filtered[i:i+3] for i in range(0, len(filtered), 3)]
    for row in rows:
        cols = st.columns(3)
        for idx, p in enumerate(row):
            with cols[idx]:
                technologies = p['tech'][:4]
                tech_html = "".join([f"<span class='tech-pill'>{t}</span>" for t in technologies])
                if len(p['tech']) > 4: tech_html += f"<span class='tech-pill'>+{len(p['tech'])-4}</span>"
                
                cat_class = "cat-data"
                if "Web" in p['category']: cat_class = "cat-web"
                elif "Finance" in p['category']: cat_class = "cat-finance"
                elif "NLP" in p['category']: cat_class = "cat-nlp"
                elif "Security" in p['category']: cat_class = "cat-security"
                elif "Vision" in p['category']: cat_class = "cat-vision"

                st.markdown(f"""
                <div class="project-card">
                    <div class="card-header">
                        <span class="category-tag {cat_class}">{p['category']}</span>
                        <div style="text-align:right;">
                            <div style="font-size:0.7rem; color:#00ADB5;">{p['complexity']}</div>
                        </div>
                    </div>
                    <div class="project-title">{p['name'][:30] + '...' if len(p['name'])>30 else p['name']}</div>
                    <div class="project-meta">
                        <span>🗓️ {p['updated']}</span>
                        <span>📝 {p['lines']} Lines</span>
                    </div>
                    <div class="tech-stack">{tech_html}</div>
                    <a href="{p['link']}" target="_blank" class="view-btn">View Code</a>
                </div>
                """, unsafe_allow_html=True)

# --- ANALYTICS TAB ---
elif selected == "Analytics":
    st.title("📊 Architecture insights")
    
    col1, col2 = st.columns([2, 1])
    
    # 1. Tech Stack Distribution
    all_tech = []
    for p in projects: all_tech.extend(p['tech'])
    tech_counts = pd.Series(all_tech).value_counts().reset_index()
    tech_counts.columns = ['Tech', 'Count']
    
    with col1:
        st.subheader("Tech Stack Dominance")
        fig_tech = px.bar(tech_counts.head(10), x='Tech', y='Count', color='Count', 
                          color_continuous_scale='teal', template='plotly_dark',
                          title="Top 10 Technologies Used")
        fig_tech.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(30, 35, 43, 0.5)', height=400)
        st.plotly_chart(fig_tech, use_container_width=True)
        
    with col2:
        st.subheader("Project Complexity")
        comp_counts = pd.DataFrame([p['complexity'] for p in projects], columns=['Level']).value_counts().reset_index()
        comp_counts.columns = ['Level', 'Count']
        
        fig_pie = px.pie(comp_counts, values='Count', names='Level', hole=0.6, 
                         color_discrete_sequence=px.colors.qualitative.Prism, template="plotly_dark")
        fig_pie.update_layout(showlegend=True, legend=dict(orientation="h"), height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 2. Timeline Activity
    st.subheader("📅 Activity Timeline")
    # Convert dates
    timeline_data = pd.DataFrame(projects)
    timeline_data['updated'] = pd.to_datetime(timeline_data['updated'])
    timeline_agg = timeline_data.groupby(pd.Grouper(key='updated', freq='M')).size().reset_index(name='Projects')
    
    fig_line = px.area(timeline_agg, x='updated', y='Projects', title="Project Updates Over Time",
                       line_shape='spline', template='plotly_dark')
    fig_line.update_traces(line_color='#00ADB5', fillcolor='rgba(0, 173, 181, 0.2)')
    fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(30, 35, 43, 0.5)')
    st.plotly_chart(fig_line, use_container_width=True)

# --- ABOUT TAB ---
elif selected == "About":
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    
    with c1:
        if os.path.exists("profile.png"):
            st.image("profile.png", width=350, caption="Harsh Choudhary")
        elif os.path.exists("profile.jpg"):
            st.image("profile.jpg", width=350, caption="Harsh Choudhary")
        else:
            # Fallback to a sleek futuristic avatar
            st.image("https://cdn3d.iconscout.com/3d/premium/thumb/man-working-on-laptop-2996954-2492508.png", width=350, caption="Harsh Choudhary")
        st.markdown("""
        <div style="display:flex; justify-content:center; gap:15px; margin-top:10px;">
            <a href="https://github.com/HarshChoudhary2003"><img src="https://img.shields.io/badge/GitHub-black?style=flat&logo=github" height="30"></a>
            <a href="https://linkedin.com"><img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin" height="30"></a>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("# Harsh Choudhary")
        st.markdown("### 🤖 Systems & AI Engineer")
        st.info("Building intelligent systems that bridge the gap between data and decision-making.")
        
        st.markdown("""
        Highly motivated and results-oriented Machine Learning Engineer with a passion for developing scalable AI solutions. 
        Specializing in:
        
        - **Deep Learning Architectures** (Transformers, CNNs, RNNs)
        - **Generative AI & LLMs** (LangChain, OpenAI, RAG)
        - **Full Stack Integration** (Streamlit, FastAPI, React)
        - **Data Engineering** (ETL, Cloud Pipelines)
        
        This portfolio represents years of continuous learning and experimentation. Each project is a stepping stone towards mastering the art of AI.
        """)
        
        st.markdown("---")
        st.markdown("### 📬 Get in Touch")
        
        contact_form = """
        <form action="https://formsubmit.co/harshchoudhary1612@gmail.com" method="POST">
             <input type="hidden" name="_captcha" value="false">
             <input type="text" name="name" placeholder="Your Name" required style="width:100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; background: rgba(255,255,255,0.1); border: 1px solid #444; color: white;">
             <input type="email" name="email" placeholder="Your Email" required style="width:100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; background: rgba(255,255,255,0.1); border: 1px solid #444; color: white;">
             <textarea name="message" placeholder="Your Message" required style="width:100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; background: rgba(255,255,255,0.1); border: 1px solid #444; color: white; height: 100px;"></textarea>
             <button type="submit" style="background: #00ADB5; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">Send Message</button>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)

# Footer
st.markdown("""
<style>
    .footer {
        position: relative;
        margin-top: 50px;
        padding-top: 40px;
        padding-bottom: 20px;
        background: linear-gradient(180deg, rgba(5,5,5,0) 0%, rgba(22, 27, 34, 0.4) 100%);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #94A3B8;
        font-family: 'Outfit', sans-serif;
    }
    
    .footer-content {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    .footer-brand {
        flex: 1;
        min-width: 250px;
        margin-bottom: 20px;
    }
    
    .footer-brand h3 {
        color: #F8F9FA;
        font-size: 1.5rem;
        margin-bottom: 10px;
        background: -webkit-linear-gradient(45deg, #00ADB5, #D500F9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .footer-brand p {
        font-size: 0.9rem;
        max-width: 300px;
        line-height: 1.6;
        color: #A0AEC0;
    }
    
    .footer-links-col {
        flex: 0.5;
        min-width: 150px;
        margin-bottom: 20px;
    }
    
    .footer-links-col h4 {
        color: #E2E8F0;
        font-size: 1.1rem;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .footer-links-col ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .footer-links-col li {
        margin-bottom: 8px;
    }
    
    .footer-links-col a {
        color: #64748B;
        text-decoration: none;
        font-size: 0.9rem;
        transition: color 0.3s;
    }
    
    .footer-links-col a:hover {
        color: #00ADB5;
    }
    
    .footer-social {
        display: flex;
        gap: 15px;
        margin-top: 20px;
    }
    
    .social-icon {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(255,255,255,0.03);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-decoration: none;
        transition: all 0.3s;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .social-icon:hover {
        background: #00ADB5;
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 173, 181, 0.3);
        border-color: #00ADB5;
    }
    
    .social-icon img {
        width: 18px;
        height: 18px;
        filter: invert(1);
    }

    .footer-bottom {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.85rem;
        color: #4A5568;
    }
    
    .heart-icon {
        color: #D500F9;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
</style>

<div class="footer">
    <div class="footer-content">
        <div class="footer-brand">
            <h3>Machine Learning Hub</h3>
            <p>Exploring the frontiers of Artificial Intelligence, one project at a time. Built for developers, researchers, and enthusiasts.</p>
            <div class="footer-social">
                <a href="https://github.com/HarshChoudhary2003" target="_blank" class="social-icon">
                    <img src="https://simpleicons.org/icons/github.svg" alt="GitHub"/>
                </a>
                <a href="https://linkedin.com" target="_blank" class="social-icon">
                    <img src="https://simpleicons.org/icons/linkedin.svg" alt="LinkedIn"/>
                </a>
                <a href="mailto:harshchoudhary1612@gmail.com" class="social-icon">
                    <img src="https://simpleicons.org/icons/gmail.svg" alt="Email"/>
                </a>
                <a href="#" class="social-icon">
                    <img src="https://simpleicons.org/icons/twitter.svg" alt="Twitter"/>
                </a>
            </div>
        </div>
        
        <div class="footer-links-col">
            <h4>Quick Links</h4>
            <ul>
                <li><a href="#">Home</a></li>
                <li><a href="#">Portfolio</a></li>
                <li><a href="#">Analytics</a></li>
                <li><a href="#">About Me</a></li>
            </ul>
        </div>
        
        <div class="footer-links-col">
            <h4>Resources</h4>
            <ul>
                <li><a href="https://github.com/HarshChoudhary2003?tab=repositories" target="_blank">Documentation</a></li>
                <li><a href="#" target="_blank">Case Studies</a></li>
                <li><a href="#" target="_blank">Community Hub</a></li>
                <li><a href="#" target="_blank">Report Issue</a></li>
            </ul>
        </div>
        
        <div class="footer-links-col">
            <h4>Tech Stack</h4>
            <ul>
                <li><a href="#">Streamlit</a></li>
                <li><a href="#">TensorFlow</a></li>
                <li><a href="#">PyTorch</a></li>
                <li><a href="#">LangChain</a></li>
            </ul>
        </div>
    </div>
    
    <div class="footer-bottom">
        <p>© 2026 Machine Learning Projects Hub. Made with <span class="heart-icon">♥</span> by Harsh Choudhary.</p>
    </div>
</div>
""", unsafe_allow_html=True)
