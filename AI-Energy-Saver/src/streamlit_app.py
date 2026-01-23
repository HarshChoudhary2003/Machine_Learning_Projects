import streamlit as st
import joblib
import pandas as pd
import psutil
import plotly.express as px
import plotly.graph_objects as go

from recommend_actions import recommend_actions
from energy_utils import compute_efficiency_score, forecast_battery_curve
from logger import log_snapshot, load_history
from report import build_battery_report
from streamlit_autorefresh import st_autorefresh

MODEL_PATH = "models/battery_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ===============================================
# SIMULATOR PAGE
# ===============================================
def page_simulator(model, template):
    st.header("🧪 Battery Usage Simulator")

    col1, col2 = st.columns(2)

    with col1:
        battery_percent = st.slider("Battery (%)", 5, 100, 60)
        time_since_unplugged = st.slider("Minutes since unplugged", 0, 300, 30)
        cpu_usage = st.slider("CPU usage (%)", 0, 100, 40)
        gpu_usage = st.slider("GPU usage (%)", 0, 100, 10)
        brightness = st.slider("Screen Brightness (%)", 10, 100, 70)
        num_apps = st.slider("Running apps", 1, 40, 12)

    with col2:
        heavy = st.selectbox("Heavy workload (game / editing / IDE)?", ["No", "Yes"])
        wifi = st.selectbox("WiFi on?", ["No", "Yes"])
        bt = st.selectbox("Bluetooth on?", ["No", "Yes"])
        power_mode = st.selectbox("Power mode", ["battery_saver", "balanced", "performance"])
        fan_speed = st.slider("Fan speed (RPM)", 1200, 4500, 2500)
        temperature = st.slider("Temperature (°C)", 35, 95, 65)

    if st.button("Predict (Simulator)"):
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

        st.subheader("🔋 Estimated Remaining Time")
        st.success(f"≈ {hours} hours {mins} minutes")

        score, label = compute_efficiency_score(features, prediction)
        st.subheader("📊 Energy Efficiency Score")
        st.write(f"**{score}/100 — {label}**")

        st.subheader("📈 Animated Battery Forecast")
        forecast_df = forecast_battery_curve(battery_percent, prediction)
        fig = px.line(forecast_df, x="minute", y="battery_percent", markers=True)
        fig.update_traces(line_shape="spline")
        fig.update_layout(template=template, transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Smart Recommendations")
        recs = recommend_actions(features, prediction)
        for r in recs:
            st.markdown(f"- {r}")

        log_snapshot(features, prediction)
        st.success("Scenario logged to history.")

        st.subheader("📄 Export Report")
        pdf = build_battery_report({
            "title": "AI Energy Saver – Simulator Report",
            "battery_percent": battery_percent,
            "predicted_minutes": prediction,
            "efficiency_score": score,
            "efficiency_label": label,
            "recommendations": recs,
            "context": "Simulator",
        })
        st.download_button("⬇ Download PDF", data=pdf, file_name="simulator_report.pdf", mime="application/pdf")


# ===============================================
# LIVE MONITOR PAGE
# ===============================================
def get_safe_system_stats():
    try:
        battery = psutil.sensors_battery()
        percent = battery.percent if battery else None
    except:
        percent = None

    return {
        "battery_percent": percent if percent is not None else 50,
        "cpu_usage": psutil.cpu_percent(interval=0.5),
        "ram_usage": psutil.virtual_memory().percent,
        "num_apps_running": len(psutil.pids()),
    }


def page_live_monitor(model, template):
    st.header("📡 Live System Monitor")
    st.caption("Auto refresh every 5 seconds")
    st_autorefresh(interval=5000, key="auto_refresh_5s")

    stats = get_safe_system_stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("🔋 Battery", f"{stats['battery_percent']}%")
    c2.metric("⚙ CPU", f"{stats['cpu_usage']}%")
    c3.metric("💾 RAM", f"{stats['ram_usage']}%")

    st.write("---")
    st.json(stats)

    features = {
        "battery_percent": stats["battery_percent"],
        "time_since_unplugged_min": 30,
        "cpu_usage": stats["cpu_usage"],
        "gpu_usage": 0,
        "screen_brightness": 70,
        "num_apps_running": stats["num_apps_running"],
        "heavy_app_running": 1 if stats["cpu_usage"] > 65 else 0,
        "wifi_on": 1,
        "bluetooth_on": 1,
        "power_mode": "balanced",
        "fan_speed_rpm": 2000,
        "device_temperature": 60,
    }

    prediction = float(model.predict(pd.DataFrame([features]))[0])
    hours = int(prediction // 60)
    mins = int(prediction % 60)

    st.subheader("⏳ Predicted Remaining Time")
    st.success(f"≈ {hours} hours {mins} minutes")

    score, label = compute_efficiency_score(features, prediction)
    st.write(f"📊 **{score}/100 — {label}**")

    st.subheader("📈 Animated Battery Forecast")
    forecast_df = forecast_battery_curve(stats["battery_percent"], prediction)
    fig = px.line(forecast_df, x="minute", y="battery_percent", markers=True)
    fig.update_traces(line_shape="spline")
    fig.update_layout(template=template, transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Recommendations")
    recs = recommend_actions(features, prediction)
    for r in recs:
        st.markdown(f"- {r}")

    log_snapshot(features, prediction)
    st.success("Live snapshot logged.")

    pdf = build_battery_report({
        "title": "AI Energy Saver – Live Monitor Report",
        "battery_percent": stats["battery_percent"],
        "predicted_minutes": prediction,
        "efficiency_score": score,
        "efficiency_label": label,
        "recommendations": recs,
        "context": "Live Monitor",
    })
    st.download_button("⬇ Download PDF", data=pdf, file_name="live_report.pdf", mime="application/pdf")


# ===============================================
# HISTORY PAGE
# ===============================================
def page_history(template):
    st.header("📚 Battery History & Insights")

    df = load_history()
    if df is None or df.empty:
        st.info("No history found. Use Simulator or Live Monitor first.")
        return

    st.subheader("📝 Recent Logs")
    st.dataframe(df.tail(50), use_container_width=True)

    st.subheader("⬇ Download Complete History")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="battery_history.csv", mime="text/csv")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        daily = df.groupby(df["timestamp"].dt.date)["predicted_minutes"].mean().reset_index()
        daily["predicted_hours"] = daily["predicted_minutes"] / 60

        st.subheader("📈 Battery Life Trend Over Time")
        fig1 = px.bar(daily, x="timestamp", y="predicted_hours", text_auto=True)
        fig1.update_layout(template=template)
        st.plotly_chart(fig1, use_container_width=True)

    if "cpu_usage" in df.columns:
        st.subheader("⚙ CPU vs Predicted Battery Life (Animated)")
        fig2 = px.scatter(
            df,
            x="cpu_usage",
            y="predicted_minutes",
            animation_frame=df["timestamp"].dt.strftime("%H:%M:%S"),
        )
        fig2.update_layout(template=template)
        st.plotly_chart(fig2, use_container_width=True)


# ===============================================
# MAIN
# ===============================================
def main():
    st.set_page_config(page_title="AI Energy Saver", page_icon="⚡", layout="wide")
    model = load_model()

    st.title("⚡ AI Energy Saver – Advanced Dashboard")
    st.caption("Real-time battery intelligence • Prediction • Forecasting • Insights • Reports")
    st.write("---")

    theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
    template = "plotly_dark" if theme == "Dark" else "plotly_white"

    # >>>>>>>>>>> DARK & LIGHT UI CSS FIX <<<<<<<<<<<<
    if theme == "Dark":
        st.markdown("""
        <style>
            .stApp, body { background-color: #0d1117 !important; }
            section[data-testid="stSidebar"] { background-color: #161b22 !important; }
            h1, h2, h3, h4, h5, h6, p, label, span, div { color: #e6edf3 !important; }

            /* Dropdown + Input fields */
            div[data-baseweb="select"], input, textarea,
            .stTextInput>div>div>input, .stNumberInput input {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
                border: 1px solid #30363d !important;
            }

            /* Dropdown options */
            ul[role="listbox"] li {
                background-color: #0d1117 !important;
                color: #e6edf3 !important;
            }
            ul[role="listbox"] li:hover {
                background-color: #238636 !important;
                color: white !important;
            }

            /* Buttons */
            .stButton>button {
                background-color: #238636 !important;
                color: white !important;
                border-radius: 6px;
                border: none;
                padding: 8px 16px;
            }
            .stButton>button:hover {
                background-color: #2ea043 !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp, body { background-color: #ffffff !important; }
            section[data-testid="stSidebar"] { background-color: #f3f3f3 !important; }
            h1, h2, h3, h4, h5, h6, p, label, span, div { color: #000000 !important; }
        </style>
        """, unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.radio("Navigate", ["Simulator", "Live Monitor", "History"])

    if page == "Simulator":
        page_simulator(model, template)
    elif page == "Live Monitor":
        page_live_monitor(model, template)
    else:
        page_history(template)


if __name__ == "__main__":
    main()
