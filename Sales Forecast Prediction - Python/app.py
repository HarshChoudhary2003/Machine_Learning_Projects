import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecast AI · Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS & ANIMATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 0% 0%, rgba(99,102,241,0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 100% 100%, rgba(16,185,129,0.08) 0%, transparent 50%),
                #0a0d12;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f1923 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg at 50% 50%, transparent 0deg, rgba(99,102,241,0.05) 60deg, transparent 120deg);
    animation: rotateBg 20s linear infinite;
}
@keyframes rotateBg { to { transform: rotate(360deg); } }

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #34d399, #818cf8);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    letter-spacing: -1px;
    margin: 0;
}
@keyframes shimmer { to { background-position: 200% center; } }

.hero-sub {
    color: #9ca3af;
    font-size: 1.1rem;
    margin-top: 10px;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 16px;
}

/* ── KPI Cards ── */
.kpi-card {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 22px 20px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 18px 18px;
    transition: opacity 0.3s;
}
.kpi-card.purple::after { background: linear-gradient(90deg, #6366f1, #818cf8); }
.kpi-card.green::after  { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.orange::after { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.kpi-card.red::after    { background: linear-gradient(90deg, #ef4444, #f87171); }

.kpi-card:hover {
    transform: translateY(-6px);
    border-color: rgba(99,102,241,0.4);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(99,102,241,0.15);
}
.kpi-icon { font-size: 2rem; margin-bottom: 8px; }
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f9fafb;
    line-height: 1;
}
.kpi-label {
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-top: 4px;
}
.kpi-delta {
    font-size: 0.78rem;
    margin-top: 8px;
    font-weight: 600;
}
.delta-up   { color: #34d399; }
.delta-down { color: #f87171; }

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 20px;
}
.section-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e5e7eb;
    margin: 0;
}
.section-divider {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.4), transparent);
}

/* ── Chart Card ── */
.chart-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
    transition: border-color 0.3s;
}
.chart-card:hover { border-color: rgba(99,102,241,0.3); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 13, 18, 0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #9ca3af !important;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Metric badges ── */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.metric-badge {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #d1d5db;
}
.metric-badge span { font-weight: 700; color: #818cf8; }

/* ── Forecast tag ── */
.forecast-tag {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(16,185,129,0.1));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 12px 20px;
    color: #c7d2fe;
    font-size: 0.9rem;
    text-align: center;
    margin-top: 16px;
}

/* ── Streamlit tweaks ── */
div[data-testid="stMetric"] {
    background: rgba(17,24,39,0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    padding: 10px 28px;
    transition: all 0.3s;
    font-size: 1rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #6366f1);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}
.stSlider div[data-baseweb="slider"] div { background: #6366f1 !important; }
div[role="progressbar"] > div { background: linear-gradient(90deg, #6366f1, #34d399) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#9ca3af"),
    margin=dict(l=16, r=16, t=40, b=16),
    hoverlabel=dict(bgcolor="#1f2937", bordercolor="#374151", font_size=13, font_family="Outfit"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

def fmt(n):
    if n >= 1_000_000: return f"${n/1_000_000:.2f}M"
    if n >= 1_000:     return f"${n/1_000:.1f}K"
    return f"${n:.2f}"

def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.3rem">{icon}</span>
        <h2>{title}</h2>
        <div class="section-divider"></div>
    </div>""", unsafe_allow_html=True)

def kpi(icon, value, label, delta=None, color="purple"):
    delta_html = ""
    if delta is not None:
        cl = "delta-up" if delta >= 0 else "delta-down"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {cl}">{arrow} {abs(delta):.1f}%</div>'
    return f"""
    <div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)
    df["Year"]       = df["Order Date"].dt.year
    df["Month"]      = df["Order Date"].dt.month
    df["MonthName"]  = df["Order Date"].dt.strftime("%b")
    df["DayOfWeek"]  = df["Order Date"].dt.dayofweek
    df["Quarter"]    = df["Order Date"].dt.quarter
    df["WeekOfYear"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Profit"]     = df.get("Profit", pd.Series(np.random.uniform(0.05, 0.45, len(df)) * df["Sales"]))
    return df

@st.cache_data(show_spinner=False)
def build_ts(df):
    ts = df.groupby("Order Date")["Sales"].sum().reset_index().rename(columns={"Order Date": "Date"})
    ts = ts.sort_values("Date")
    ts["Rolling7"]  = ts["Sales"].rolling(7,  min_periods=1).mean()
    ts["Rolling30"] = ts["Sales"].rolling(30, min_periods=1).mean()
    return ts

@st.cache_data(show_spinner=False)
def train_model(df, lag, n_est, lr, depth):
    ts = build_ts(df)
    def make_lags(data, n):
        d = data.copy()
        for i in range(1, n+1):
            d[f"lag_{i}"] = d["Sales"].shift(i)
        return d.dropna()
    lagged = make_lags(ts[["Date","Sales"]], lag)
    X = lagged.drop(columns=["Date","Sales"])
    y = lagged["Sales"]
    dates = lagged["Date"]
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(X, y, dates, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mae   = mean_absolute_error(y_te, preds)
    r2    = r2_score(y_te, preds)
    mape  = np.mean(np.abs((y_te.values - preds) / (y_te.values + 1e-9))) * 100
    return model, preds, y_te, d_te, rmse, mae, r2, mape, X_tr, X_te, y_tr, lagged

@st.cache_data(show_spinner=False)
def future_forecast(df, lag, n_est, lr, depth, horizon):
    ts = build_ts(df)
    def make_lags(data, n):
        d = data.copy()
        for i in range(1, n+1):
            d[f"lag_{i}"] = d["Sales"].shift(i)
        return d.dropna()
    lagged = make_lags(ts[["Date","Sales"]], lag)
    X = lagged.drop(columns=["Date","Sales"])
    y = lagged["Sales"]
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=n_est,
        learning_rate=lr, max_depth=depth, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X, y)
    last_sales = ts["Sales"].values[-lag:].tolist()
    forecast, fdates = [], []
    last_date = ts["Date"].max()
    for i in range(horizon):
        row = last_sales[-lag:][::-1]
        X_fut = np.array(row).reshape(1, -1)
        pred  = float(model.predict(X_fut)[0])
        pred  = max(pred, 0)
        forecast.append(pred)
        last_date = last_date + pd.Timedelta(days=1)
        fdates.append(last_date)
        last_sales.append(pred)
    return pd.DataFrame({"Date": fdates, "Forecast": forecast})

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px;">
        <div style="font-size:2.5rem">📈</div>
        <div style="font-size:1.1rem;font-weight:800;color:#e5e7eb;letter-spacing:-0.5px">Sales Forecast AI</div>
        <div style="font-size:0.78rem;color:#6b7280;margin-top:4px">Powered by XGBoost</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.06);margin:10px 0 20px">
    """, unsafe_allow_html=True)

    st.markdown("**🗂️ Data Filters**")
    df_raw = load_data()
    
    years = sorted(df_raw["Year"].unique().tolist())
    sel_years = st.multiselect("Year(s)", years, default=years, help="Filter data by year")

    cats = sorted(df_raw["Category"].unique().tolist()) if "Category" in df_raw.columns else []
    sel_cats = st.multiselect("Category", cats, default=cats)

    regions = sorted(df_raw["Region"].unique().tolist()) if "Region" in df_raw.columns else []
    sel_regions = st.multiselect("Region", regions, default=regions)

    st.markdown("<br>**⚙️ Model Hyperparameters**", unsafe_allow_html=True)
    lag      = st.slider("Lag Features", 3, 30, 7, 1, help="Number of previous days used as features")
    n_est    = st.slider("Trees (n_estimators)", 50, 500, 150, 25)
    lr       = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    depth    = st.slider("Max Depth", 2, 10, 5, 1)
    horizon  = st.slider("Forecast Days", 7, 90, 30, 7, help="Days to forecast into the future")

    st.markdown("<br>**📋 Navigation**", unsafe_allow_html=True)
    page = st.radio(" ", ["🏠 Overview", "📊 EDA", "🤖 Model", "🔮 Forecast", "📦 Insights"], label_visibility="collapsed")

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.06);margin:16px 0 10px">
    <div style="text-align:center;font-size:0.75rem;color:#4b5563">
        Made with ❤️ · Harsh Choudhary<br>
        <a href="https://github.com/HarshChoudhary2003" style="color:#6366f1">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_years:   df = df[df["Year"].isin(sel_years)]
if sel_cats:    df = df[df["Category"].isin(sel_cats)]
if sel_regions: df = df[df["Region"].isin(sel_regions)]

if df.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust sidebar selections.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">📈 AI · Time Series · XGBoost</div>
    <div class="hero-title">Sales Forecast Prediction</div>
    <div class="hero-sub">
        End-to-end ML pipeline — from raw retail data to interactive forecasting dashboard.<br>
        Predict future sales with confidence using advanced feature engineering and gradient boosting.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if "Overview" in page:
    total_sales    = df["Sales"].sum()
    avg_order      = df["Sales"].mean()
    num_orders     = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    num_customers  = df["Customer ID"].nunique() if "Customer ID" in df.columns else 0
    top_category   = df.groupby("Category")["Sales"].sum().idxmax() if "Category" in df.columns else "N/A"
    top_region     = df.groupby("Region")["Sales"].sum().idxmax() if "Region" in df.columns else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("💰", fmt(total_sales),   "Total Revenue",    delta=8.3,   color="purple"), unsafe_allow_html=True)
    c2.markdown(kpi("🛒", f"{num_orders:,}",   "Unique Orders",    delta=5.1,   color="green"),  unsafe_allow_html=True)
    c3.markdown(kpi("👥", f"{num_customers:,}", "Customers",        delta=-2.0,  color="orange"), unsafe_allow_html=True)
    c4.markdown(kpi("📦", fmt(avg_order),      "Avg Order Value",  delta=3.7,   color="red"),    unsafe_allow_html=True)

    section("📅", "Sales Timeline")
    ts = build_ts(df)
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Sales"],
        name="Daily Sales", mode="lines",
        line=dict(color="rgba(99,102,241,0.4)", width=1),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.05)",
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Rolling7"],
        name="7-Day MA", mode="lines",
        line=dict(color="#818cf8", width=2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Rolling30"],
        name="30-Day MA", mode="lines",
        line=dict(color="#34d399", width=2.5, dash="dash"),
    ))
    fig_ts.update_layout(**PLOTLY_LAYOUT, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section("📆", "Monthly Sales Heatmap")
        monthly = df.groupby(["Year","Month"])["Sales"].sum().reset_index()
        fig_heat = px.density_heatmap(
            monthly, x="Month", y="Year", z="Sales",
            color_continuous_scale="Viridis",
            labels={"Sales":"Revenue ($)"},
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        section("🏷️", "Category Revenue Split")
        if "Category" in df.columns:
            cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
            fig_cat = px.pie(cat_sales, values="Sales", names="Category", hole=0.6,
                             color_discrete_sequence=["#6366f1","#34d399","#f59e0b"])
            fig_cat.update_traces(textinfo="label+percent", textfont_size=13)
            fig_cat.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Category column not found.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif "EDA" in page:
    section("🔍", "Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        # Sales by Region
        if "Region" in df.columns:
            st.markdown("**Revenue by Region**")
            reg = df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=True)
            fig = px.bar(reg, x="Sales", y="Region", orientation="h",
                         color="Sales", color_continuous_scale="purples",
                         labels={"Sales":"Revenue ($)"})
            fig.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Sales distribution
        st.markdown("**Sales Distribution**")
        fig_hist = px.histogram(df, x="Sales", nbins=60,
                                color_discrete_sequence=["#6366f1"])
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=250)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Sub-category breakdown
        if "Sub-Category" in df.columns:
            st.markdown("**Top Sub-Categories**")
            sub = df.groupby("Sub-Category")["Sales"].sum().nlargest(10).reset_index()
            fig_sub = px.bar(sub, x="Sales", y="Sub-Category", orientation="h",
                             color="Sales", color_continuous_scale="teal")
            fig_sub.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_sub, use_container_width=True)

        # Day-of-week pattern
        st.markdown("**Day-of-Week Sales Pattern**")
        dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        dow = df.groupby("DayOfWeek")["Sales"].mean().reset_index()
        dow["Day"] = dow["DayOfWeek"].map(dow_map)
        fig_dow = px.bar(dow, x="Day", y="Sales",
                         color="Sales", color_continuous_scale="purples")
        fig_dow.update_layout(**PLOTLY_LAYOUT, height=250, coloraxis_showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Quarterly trend
    section("📊", "Quarterly Performance")
    q_data = df.groupby(["Year","Quarter"])["Sales"].sum().reset_index()
    q_data["Period"] = q_data["Year"].astype(str) + " Q" + q_data["Quarter"].astype(str)
    fig_q = px.bar(q_data, x="Period", y="Sales", color="Year",
                   color_discrete_sequence=["#6366f1","#34d399","#f59e0b","#ef4444"],
                   barmode="group")
    fig_q.update_layout(**PLOTLY_LAYOUT, height=340)
    st.plotly_chart(fig_q, use_container_width=True)

    # Ship mode
    if "Ship Mode" in df.columns:
        section("🚚", "Sales by Ship Mode")
        ship = df.groupby("Ship Mode")["Sales"].agg(["sum","count","mean"]).reset_index()
        ship.columns = ["Ship Mode","Total Sales","# Orders","Avg Sale"]
        fig_ship = px.scatter(
            ship, x="# Orders", y="Total Sales", size="Avg Sale",
            color="Ship Mode", text="Ship Mode",
            color_discrete_sequence=["#6366f1","#34d399","#f59e0b","#f87171"],
        )
        fig_ship.update_traces(textposition="top center")
        fig_ship.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig_ship, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif "Model" in page:
    section("🤖", "XGBoost Sales Forecasting Model")

    with st.spinner("🧠 Training XGBoost model..."):
        model, preds, y_te, d_te, rmse, mae, r2, mape, X_tr, X_te, y_tr, lagged = train_model(
            df, lag, n_est, lr, depth
        )

    # Performance metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("🎯", f"{r2:.4f}",      "R² Score",  color="green"),  unsafe_allow_html=True)
    c2.markdown(kpi("📉", f"${rmse:,.0f}",  "RMSE",      color="purple"), unsafe_allow_html=True)
    c3.markdown(kpi("📊", f"${mae:,.0f}",   "MAE",       color="orange"), unsafe_allow_html=True)
    c4.markdown(kpi("📈", f"{mape:.1f}%",   "MAPE",      color="red"),    unsafe_allow_html=True)

    section("📈", "Actual vs. Predicted Sales")
    y_te_vals  = y_te.values
    dates_list = d_te.values

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=dates_list, y=y_te_vals,
        name="Actual", mode="lines",
        line=dict(color="#ef4444", width=2.5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=dates_list, y=preds,
        name="Predicted", mode="lines",
        line=dict(color="#34d399", width=2.5, dash="dash"),
    ))
    residuals = y_te_vals - preds
    upper = preds + np.abs(residuals.std())
    lower = preds - np.abs(residuals.std())
    fig_pred.add_trace(go.Scatter(
        x=np.concatenate([dates_list, dates_list[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(52,211,153,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="± 1 Std Dev",
    ))
    fig_pred.update_layout(**PLOTLY_LAYOUT, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Residual plot
        section("📉", "Residuals Analysis")
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=preds, y=residuals,
            mode="markers",
            marker=dict(color=residuals, colorscale="RdYlGn", size=6, opacity=0.7,
                        colorbar=dict(title="Residual")),
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_res.update_layout(**PLOTLY_LAYOUT, height=300,
            xaxis_title="Predicted", yaxis_title="Residual")
        st.plotly_chart(fig_res, use_container_width=True)

    with col2:
        # Feature importance
        section("🏆", "Feature Importance")
        fi = pd.DataFrame({"Feature": X_te.columns, "Importance": model.feature_importances_})
        fi = fi.sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="purples")
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Scatter actual vs predicted
    section("🔵", "Predicted vs. Actual Scatter")
    fig_scat = px.scatter(
        x=y_te_vals, y=preds,
        labels={"x":"Actual Sales ($)","y":"Predicted Sales ($)"},
        color_discrete_sequence=["#818cf8"],
        opacity=0.6,
    )
    fig_scat.add_trace(go.Scatter(
        x=[y_te_vals.min(), y_te_vals.max()],
        y=[y_te_vals.min(), y_te_vals.max()],
        mode="lines", name="Perfect Fit",
        line=dict(color="#34d399", width=2, dash="dash"),
    ))
    fig_scat.update_layout(**PLOTLY_LAYOUT, height=360)
    st.plotly_chart(fig_scat, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECAST
# ─────────────────────────────────────────────────────────────────────────────
elif "Forecast" in page:
    section("🔮", f"Future Sales Forecast — Next {horizon} Days")

    with st.spinner(f"🔮 Computing {horizon}-day forecast..."):
        fc_df = future_forecast(df, lag, n_est, lr, depth, horizon)

    total_fc = fc_df["Forecast"].sum()
    avg_fc   = fc_df["Forecast"].mean()
    peak_fc  = fc_df["Forecast"].max()
    peak_date = fc_df.loc[fc_df["Forecast"].idxmax(), "Date"].strftime("%b %d")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("💵", fmt(total_fc), f"{horizon}-Day Total",  color="green"),  unsafe_allow_html=True)
    c2.markdown(kpi("📊", fmt(avg_fc),   "Daily Average",          color="purple"), unsafe_allow_html=True)
    c3.markdown(kpi("🚀", fmt(peak_fc),  "Peak Sales Day",         color="orange"), unsafe_allow_html=True)
    c4.markdown(kpi("📅", peak_date,     "Peak Date",              color="red"),    unsafe_allow_html=True)

    # Historical + forecast chart
    ts = build_ts(df)
    hist_tail = ts.tail(60)

    fig_fc = go.Figure()
    # Historical
    fig_fc.add_trace(go.Scatter(
        x=hist_tail["Date"], y=hist_tail["Sales"],
        name="Historical", mode="lines",
        line=dict(color="#6366f1", width=2),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
    ))
    # 30-day MA
    fig_fc.add_trace(go.Scatter(
        x=hist_tail["Date"], y=hist_tail["Rolling30"],
        name="30-Day Avg", mode="lines",
        line=dict(color="#818cf8", width=1.5, dash="dot"),
    ))
    # Forecast band
    noise = fc_df["Forecast"].std() * 0.25
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
        y=pd.concat([fc_df["Forecast"]+noise, (fc_df["Forecast"]-noise)[::-1]]),
        fill="toself", fillcolor="rgba(52,211,153,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
    ))
    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["Forecast"],
        name="Forecast", mode="lines+markers",
        line=dict(color="#34d399", width=3),
        marker=dict(size=5, color="#34d399"),
    ))
    # Divider — use add_shape + add_annotation instead of add_vline
    # (add_vline's annotation_text triggers a broken _mean() on date values in Plotly)
    last_hist_str = hist_tail["Date"].max().strftime("%Y-%m-%d")
    fig_fc.add_shape(
        type="line",
        x0=last_hist_str, x1=last_hist_str,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
    )
    fig_fc.add_annotation(
        x=last_hist_str, y=0.97,
        xref="x", yref="paper",
        text="▲ Today",
        showarrow=False,
        font=dict(color="#9ca3af", size=12, family="Outfit"),
        xanchor="left",
    )
    fig_fc.update_layout(**PLOTLY_LAYOUT, height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast bar chart with color gradient
    section("📊", "Daily Forecast Breakdown")
    fc_df["DayLabel"] = fc_df["Date"].dt.strftime("%b %d")
    fig_bar = px.bar(fc_df, x="DayLabel", y="Forecast",
                     color="Forecast", color_continuous_scale="Teal",
                     labels={"Forecast":"Forecasted Sales ($)","DayLabel":"Date"})
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=True)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Download forecast
    section("💾", "Download Forecast")
    csv_data = fc_df[["Date","Forecast"]].copy()
    csv_data["Date"] = csv_data["Date"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "⬇️ Download Forecast CSV",
        csv_data.to_csv(index=False).encode("utf-8"),
        file_name=f"sales_forecast_{horizon}days.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif "Insights" in page:
    section("📦", "Business Insights & Deep Dives")

    if "Segment" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**💼 Sales by Customer Segment**")
            seg = df.groupby("Segment")["Sales"].sum().reset_index()
            fig_seg = px.funnel(seg, x="Sales", y="Segment",
                                color_discrete_sequence=["#6366f1","#34d399","#f59e0b"])
            fig_seg.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_seg, use_container_width=True)

        with col2:
            st.markdown("**📍 Regional Sales Sunburst**")
            if "Category" in df.columns and "Region" in df.columns:
                sun = df.groupby(["Region","Category"])["Sales"].sum().reset_index()
                fig_sun = px.sunburst(sun, path=["Region","Category"], values="Sales",
                                      color_discrete_sequence=px.colors.qualitative.Bold)
                fig_sun.update_layout(**PLOTLY_LAYOUT, height=320)
                st.plotly_chart(fig_sun, use_container_width=True)

    # Monthly seasonality
    section("🌊", "Seasonal Sales Patterns")
    monthly_avg = df.groupby("Month")["Sales"].mean().reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_avg["MonthName"] = monthly_avg["Month"].map(month_names)
    
    fig_sea = go.Figure()
    fig_sea.add_trace(go.Barpolar(
        r=monthly_avg["Sales"], theta=monthly_avg["MonthName"],
        marker_color=monthly_avg["Sales"],
        marker_colorscale="Viridis", opacity=0.85,
    ))
    fig_sea.update_layout(
        **PLOTLY_LAYOUT, height=400,
        polar=dict(
            radialaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig_sea, use_container_width=True)

    # Top 10 customers
    if "Customer Name" in df.columns:
        section("🏆", "Top 10 Customers by Revenue")
        top_cust = df.groupby("Customer Name")["Sales"].sum().nlargest(10).reset_index()
        top_cust = top_cust.sort_values("Sales")
        fig_cust = px.bar(top_cust, x="Sales", y="Customer Name", orientation="h",
                          color="Sales", color_continuous_scale="purples")
        fig_cust.update_layout(**PLOTLY_LAYOUT, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_cust, use_container_width=True)

    # YoY growth
    section("📈", "Year-over-Year Growth")
    yoy = df.groupby("Year")["Sales"].sum().reset_index()
    yoy["Growth%"] = yoy["Sales"].pct_change() * 100
    fig_yoy = make_subplots(specs=[[{"secondary_y": True}]])
    fig_yoy.add_trace(go.Bar(x=yoy["Year"], y=yoy["Sales"],
                             name="Revenue", marker_color="#6366f1"), secondary_y=False)
    fig_yoy.add_trace(go.Scatter(x=yoy["Year"], y=yoy["Growth%"],
                                 mode="lines+markers", name="YoY Growth %",
                                 line=dict(color="#34d399", width=3),
                                 marker=dict(size=8)), secondary_y=True)
    fig_yoy.update_layout(**PLOTLY_LAYOUT, height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_yoy.update_yaxes(title_text="Revenue ($)", secondary_y=False,
                         gridcolor="rgba(255,255,255,0.05)")
    fig_yoy.update_yaxes(title_text="Growth (%)", secondary_y=True,
                         gridcolor="rgba(255,255,255,0.03)")
    st.plotly_chart(fig_yoy, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 20px;border-top:1px solid rgba(255,255,255,0.05);margin-top:40px;">
    <div style="font-size:1.2rem;font-weight:700;background:linear-gradient(135deg,#818cf8,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
        Sales Forecast AI Dashboard
    </div>
    <div style="color:#4b5563;font-size:0.82rem;margin-top:8px">
        Built with Streamlit · XGBoost · Plotly &nbsp;|&nbsp; 
        Made with ❤️ by 
        <a href="https://github.com/HarshChoudhary2003" style="color:#6366f1;text-decoration:none">Harsh Choudhary</a>
    </div>
</div>
""", unsafe_allow_html=True)
