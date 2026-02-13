
import streamlit as st
import joblib
import pandas as pd
import psutil
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_autorefresh import st_autorefresh
import os

# Custom Module Imports
try:
    from recommend_actions import recommend_actions
    from energy_utils import compute_efficiency_score, forecast_battery_curve
    from logger import log_snapshot, load_history
    from report import build_battery_report
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from recommend_actions import recommend_actions
    from energy_utils import compute_efficiency_score, forecast_battery_curve
    from logger import log_snapshot, load_history
    from report import build_battery_report

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoWatt | High-Performance Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Optimized Asset Loading
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=2) # Add timeout for speed
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Animations (Cached)
lottie_battery = load_lottieurl("https://lottie.host/9e530b53-4638-468e-a20c-77293527230b/Y1j0a0v0c1.json")
lottie_cpu = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ne6krq7s.json")

# Model Loading (Cached)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "battery_model.pkl")

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return joblib.load("models/battery_model.pkl")

model = load_model()

# -----------------------------------------------------------------------------
# 2. OPTIMIZED CSS (Modern "Electric Navy" Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Performance & Font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* High-Performance Static Background */
    .stApp {
        background-color: #0a192f;
        background-image: 
            radial-gradient(at 0% 0%, rgba(100, 255, 218, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(33, 150, 243, 0.1) 0px, transparent 50%);
        color: #e6f1ff;
    }

    /* Modern Card Design - Flat & Clean */
    .glass-card {
        background: #112240;
        border: 1px solid #233554;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: #64ffda;
        box-shadow: 0 10px 30px -10px rgba(2, 12, 27, 0.7);
    }

    /* Typography Highlighting */
    h1, h2, h3 {
        color: #ccd6f6 !important;
        font-weight: 800;
    }
    .highlight-text {
        color: #64ffda;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e6f1ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 4px;
    }

    /* Buttons - Neo-Mint */
    div.stButton > button {
        background-color: transparent;
        color: #64ffda;
        border: 1px solid #64ffda;
        border-radius: 4px;
        padding: 12px 28px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: rgba(100, 255, 218, 0.1);
        color: #64ffda;
        border-color: #64ffda;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.2);
    }
    div.stButton > button:active {
        transform: translateY(1px);
    }

    /* Inputs */
    .stSlider > div > div > div > div {
        background-color: #64ffda !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #112240 !important;
        color: #a8b2d1 !important;
        border-color: #233554 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #112240;
        border-right: 1px solid #233554;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. FAST SYSTEM UTILS
# -----------------------------------------------------------------------------
def get_system_stats():
    # Use psutil with minimal overhead
    try:
        battery = psutil.sensors_battery()
        percent = battery.percent if battery else 100
        plugged = battery.power_plugged if battery else False
    except:
        percent = 50
        plugged = False

    return {
        "battery_percent": percent,
        "is_plugged_in": plugged,
        # Reduce interval for faster feedback loop
        "cpu_usage": psutil.cpu_percent(interval=0), 
        "ram_usage": psutil.virtual_memory().percent,
        "num_apps_running": len(psutil.pids()),
    }

# -----------------------------------------------------------------------------
# 4. DASHBOARD COMPONENTS
# -----------------------------------------------------------------------------

def render_simulator():
    col_sim_1, col_sim_2 = st.columns([2, 1])
    
    with col_sim_1:
        st.markdown('<div class="glass-card"><h3>🛠️ Configuration Lab</h3>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            battery_percent = st.slider("🔋 Battery Level (%)", 5, 100, 60)
            time_since_unplugged = st.slider("⏱️ Minutes Unplugged", 0, 300, 30)
            cpu_usage = st.slider("⚙️ CPU Load (%)", 0, 100, 40)
            gpu_usage = st.slider("🎮 GPU Load (%)", 0, 100, 10)
            brightness = st.slider("🔆 Brightness (%)", 10, 100, 70)
            
        with c2:
            num_apps = st.slider("📱 Active Processes", 1, 40, 12)
            heavy = st.selectbox("🔥 Intensive Task?", ["No", "Yes"])
            wifi = st.selectbox("📶 WiFi?", ["No", "Yes"])
            bt = st.selectbox("🎧 Bluetooth?", ["No", "Yes"])
            power_mode = st.selectbox("⚡ System Mode", ["battery_saver", "balanced", "performance"])
            fan_speed = st.slider("💨 Fan Speed", 1200, 4500, 2500)
            temperature = st.slider("🌡️ Temp (°C)", 35, 95, 65)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("RUN SIMULATION"):
            # Instant execution - No artificial sleeps
            features = {
                "battery_percent": battery_percent,
                "time_since_unplugged_min": time_since_unplugged,
                "cpu_usage": cpu_usage,
                "gpu_usage": gpu_usage,
                "screen_brightness": brightness,
                "num_apps_running": num_apps,
                "heavy_app_running": 1 if heavy == "Yes" else 0,
                "wifi_on": 1 if wifi == "Yes" else 0,
                "bluetooth_on": 1 if bt == "Yes" else 0,
                "power_mode": power_mode,
                "fan_speed_rpm": fan_speed,
                "device_temperature": temperature,
            }

            prediction = float(model.predict(pd.DataFrame([features]))[0])
            hours = int(prediction // 60)
            mins = int(prediction % 60)
            score, label = compute_efficiency_score(features, prediction)
            
            # --- FAST RESULTS ---
            r1, r2, r3 = st.columns(3)
            
            with r1:
                st.markdown(f"""
                <div class="glass-card metric-container">
                    <div class="metric-value highlight-text">{hours}h {mins}m</div>
                    <div class="metric-label">Estimated Runtime</div>
                </div>
                """, unsafe_allow_html=True)
                
            with r2:
                 st.markdown(f"""
                <div class="glass-card metric-container">
                    <div class="metric-value">{score}</div>
                    <div class="metric-label">Efficiency Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with r3:
                color = "#64ffda" if score > 70 else "#f44336"
                st.markdown(f"""
                <div class="glass-card metric-container">
                    <div class="metric-value" style="color: {color}; font-size: 1.8rem; padding-top:5px;">{label}</div>
                    <div class="metric-label">Rating</div>
                </div>
                """, unsafe_allow_html=True)

            # Chart
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📉 Discharge Analysis")
            forecast_df = forecast_battery_curve(battery_percent, prediction)
            fig = px.area(forecast_df, x="minute", y="battery_percent")
            fig.update_traces(line_color='#64ffda', fillcolor='rgba(100, 255, 218, 0.1)')
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font={'color': "#a8b2d1"},
                margin=dict(l=20, r=20, t=10, b=20),
                height=250,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#233554')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Log
            log_snapshot(features, prediction)

    with col_sim_2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        if lottie_battery: st_lottie(lottie_battery, height=150, key="sim_batt")
        st.markdown("### Energy Lab")
        st.caption("Adjust parameters to see immediate impact on battery life.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Quick Tips")
        st.markdown("""
        - **Brightness:** Reducing by 20% can add 30m runtime.
        - **Background Apps:** Close unused Chrome tabs.
        - **WiFi/BT:** Turn off if strictly offline.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def render_monitor():
    st.markdown("## 📡 Live Telemetry")
    # Increase interval to 3s to reduce render load, improved UX
    st_autorefresh(interval=3000, key="auto_refresh_3s")
    
    stats = get_system_stats()
    
    # --- METRICS GRID ---
    m1, m2, m3, m4 = st.columns(4)
    
    def metric_card(label, value, suffix=""):
        return f"""
        <div class="glass-card" style="padding: 15px; text-align: center;">
            <div class="metric-value" style="font-size: 1.8rem;">{value}{suffix}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    
    with m1: st.markdown(metric_card("Battery", stats["battery_percent"], "%"), unsafe_allow_html=True)
    with m2: st.markdown(metric_card("CPU Load", stats["cpu_usage"], "%"), unsafe_allow_html=True)
    with m3: st.markdown(metric_card("RAM Usage", stats["ram_usage"], "%"), unsafe_allow_html=True)
    with m4: st.markdown(metric_card("Processes", stats["num_apps_running"]), unsafe_allow_html=True)

    # --- LIVE PREDICTION ---
    features = {
        "battery_percent": stats["battery_percent"],
        "time_since_unplugged_min": 30, # Estimated
        "cpu_usage": stats["cpu_usage"],
        "gpu_usage": 0,
        "screen_brightness": 70,
        "num_apps_running": stats["num_apps_running"],
        "heavy_app_running": 1 if stats["cpu_usage"] > 60 else 0,
        "wifi_on": 1,
        "bluetooth_on": 1,
        "power_mode": "balanced",
        "fan_speed_rpm": 2000,
        "device_temperature": 60,
    }
    
    # Fast Prediction
    prediction = float(model.predict(pd.DataFrame([features]))[0])
    hours = int(prediction // 60)
    mins = int(prediction % 60)
    
    c_main, c_side = st.columns([2, 1])
    
    with c_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚡ Real-Time Projection")
        
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown(f'<div class="metric-value highlight-text" style="font-size: 3rem;">{hours}h {mins}m</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Time Remaining</div>', unsafe_allow_html=True)
            
            # Simple Progress Bar
            progress_html = f"""
            <div style="background-color: #233554; border-radius: 10px; height: 10px; width: 100%; margin-top: 20px;">
                <div style="background-color: #64ffda; height: 100%; border-radius: 10px; width: {stats['battery_percent']}%;"></div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
            
            status_text = "🔌 Charging" if stats["is_plugged_in"] else "🔋 Discharging"
            st.caption(f"Status: {status_text}")

        with col_p2:
            # Minimalist Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Minutes Left", 'font': {'color': '#a8b2d1', 'size': 12}},
                gauge = {
                    'axis': {'range': [None, 600], 'visible': False},
                    'bar': {'color': "#64ffda"},
                    'bgcolor': "#112240",
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=140, margin=dict(t=0,b=0,l=20,r=20), font={'color': "#e6f1ff"})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
        log_snapshot(features, prediction)

    with c_side:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 AI Insights")
        recs = recommend_actions(features, prediction)
        for r in recs[:3]: 
            st.markdown(f"<div style='margin-bottom:8px; font-size:0.9rem; color:#ccd6f6;'>🔹 {r}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_history_tab():
    df = load_history()
    if df is None or df.empty:
        st.info("No logs available yet.")
        return

    st.markdown("## 📜 Historical Data")
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = px.area(df, x="timestamp", y="predicted_minutes", title="Battery Estimates Trend")
        fig.update_traces(line_color='#64ffda', fillcolor='rgba(100, 255, 218, 0.2)')
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font={'color': "#a8b2d1"},
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#233554')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

# -----------------------------------------------------------------------------
# 5. MAIN APP STRUCTURE
# -----------------------------------------------------------------------------
def main():
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("EcoWatt | AI Energy Master")
        st.caption("Advanced Power Management System v2.2")
    with c2:
        if lottie_cpu: st_lottie(lottie_cpu, height=80, key="head_anim")
    
    st.markdown("---")

    # Fast Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["🧪 SIMULATOR", "📡 LIVE MONITOR", "📜 HISTORY"])
    
    with tab1:
        render_simulator()
    
    with tab2:
        render_monitor()
        
    with tab3:
        render_history_tab()

if __name__ == "__main__":
    main()
