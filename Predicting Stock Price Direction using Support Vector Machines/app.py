import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVM Stock Direction · AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at 0% 0%, rgba(234,179,8,0.08) 0%, transparent 45%),
                radial-gradient(ellipse at 100% 100%, rgba(239,68,68,0.06) 0%, transparent 45%),
                #080b10;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #eab308; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
    border: 1px solid rgba(234,179,8,0.25);
    border-radius: 22px;
    padding: 44px 40px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60%; left: -60%;
    width: 220%; height: 220%;
    background: conic-gradient(from 0deg at 50% 50%, transparent 0deg,
        rgba(234,179,8,0.04) 60deg, transparent 120deg);
    animation: spin 25s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.hero-tag {
    display: inline-block;
    background: rgba(234,179,8,0.15);
    border: 1px solid rgba(234,179,8,0.35);
    color: #fbbf24;
    border-radius: 50px; padding: 4px 14px;
    font-size: 0.75rem; letter-spacing: 1.5px;
    text-transform: uppercase; font-weight: 700;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; margin: 0;
    background: linear-gradient(135deg, #fbbf24, #ef4444, #fbbf24);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
}
@keyframes shimmer { to { background-position: 200% center; } }
.hero-sub { color: #6b7280; font-size: 1.05rem; margin-top: 10px; }

/* KPI Cards */
.kcard {
    background: rgba(13,17,23,0.9);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 20px 18px;
    text-align: center; position: relative; overflow: hidden;
    transition: all 0.35s cubic-bezier(0.175,0.885,0.32,1.275);
}
.kcard::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 3px; border-radius: 0 0 16px 16px;
}
.kcard.gold::after   { background: linear-gradient(90deg,#d97706,#fbbf24); }
.kcard.red::after    { background: linear-gradient(90deg,#dc2626,#f87171); }
.kcard.green::after  { background: linear-gradient(90deg,#059669,#34d399); }
.kcard.blue::after   { background: linear-gradient(90deg,#2563eb,#60a5fa); }
.kcard:hover { transform: translateY(-6px); border-color: rgba(234,179,8,0.3);
               box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 20px rgba(234,179,8,0.1); }
.kcard-icon  { font-size: 1.9rem; margin-bottom: 6px; }
.kcard-val   { font-size: 2rem; font-weight: 800; color: #f9fafb; line-height: 1; }
.kcard-lbl   { font-size: 0.75rem; color: #6b7280; text-transform: uppercase;
               letter-spacing: 1px; font-weight: 600; margin-top: 4px; }

/* Section header */
.shdr { display:flex; align-items:center; gap:10px; margin:32px 0 18px; }
.shdr h2 { font-size:1.3rem; font-weight:700; color:#e5e7eb; margin:0; }
.shdr-line { flex:1; height:1px; background:linear-gradient(90deg,rgba(234,179,8,0.35),transparent); }

/* Signal badge */
.sig-up   { background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4);
            color:#34d399; border-radius:12px; padding:16px 24px; text-align:center; }
.sig-down { background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4);
            color:#f87171; border-radius:12px; padding:16px 24px; text-align:center; }
.sig-val  { font-size:2.5rem; font-weight:800; }
.sig-lbl  { font-size:0.85rem; color:#9ca3af; margin-top:4px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#d97706,#b45309);
    color:white; border:none; border-radius:12px;
    font-weight:700; font-family:'Outfit',sans-serif;
    padding:10px 28px; transition:all 0.3s; font-size:1rem;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#fbbf24,#d97706);
    transform:translateY(-2px); box-shadow:0 8px 25px rgba(234,179,8,0.35);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(8,11,16,0.97);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Streamlit metric */
div[data-testid="stMetric"] {
    background: rgba(13,17,23,0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 12px;
}

/* Table */
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── THEME ─────────────────────────────────────────────────────────────────────
# Base layout — NO xaxis/yaxis keys to avoid duplicate-kwarg errors
PL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#9ca3af"),
    margin=dict(l=16, r=16, t=44, b=16),
    hoverlabel=dict(bgcolor="#1f2937", bordercolor="#374151",
                    font_size=13, font_family="Outfit"),
)
GRID = dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False)

def apply_pl(fig, **layout_kwargs):
    """Apply PL base layout + default grid axes, then any extra kwargs."""
    fig.update_layout(**PL, **layout_kwargs)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig

def section(icon, title):
    st.markdown(f"""<div class="shdr"><span style="font-size:1.2rem">{icon}</span>
    <h2>{title}</h2><div class="shdr-line"></div></div>""", unsafe_allow_html=True)

def kcard(icon, val, lbl, color="gold"):
    return f"""<div class="kcard {color}">
    <div class="kcard-icon">{icon}</div>
    <div class="kcard-val">{val}</div>
    <div class="kcard-lbl">{lbl}</div></div>"""

# ── DATA & MODEL ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("RELIANCE.csv")
    df.index = pd.to_datetime(df["Date"])
    df = df.drop("Date", axis=1)
    df["Open-Close"] = df["Open"] - df["Close"]
    df["High-Low"]   = df["High"] - df["Low"]
    df["Return"]     = df["Close"].pct_change()
    df["MA20"]       = df["Close"].rolling(20).mean()
    df["MA50"]       = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(20).std()
    df["RSI"]        = compute_rsi(df["Close"])
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(show_spinner=False)
def train_svm(df, split_pct, kernel, C, gamma_str, scale):
    X = df[["Open-Close", "High-Low"]]
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    split = int(split_pct * len(df))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    gamma = "scale" if gamma_str == "scale" else float(gamma_str)
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    test_acc  = accuracy_score(y_te, y_pred)
    cm        = confusion_matrix(y_te, y_pred)
    return model, y_pred, y_te, train_acc, test_acc, cm, split

@st.cache_data(show_spinner=False)
def kernel_comparison(df, split_pct, scale):
    kernels = ["linear", "rbf", "poly", "sigmoid"]
    results = {}
    X = df[["Open-Close", "High-Low"]]
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    split = int(split_pct * len(df))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
    for k in kernels:
        m = SVC(kernel=k).fit(X_tr, y_tr)
        results[k] = {
            "Train": accuracy_score(y_tr, m.predict(X_tr)),
            "Test":  accuracy_score(y_te, m.predict(X_te)),
        }
    return results

@st.cache_data(show_spinner=False)
def backtest(df, split_pct, kernel, C, gamma_str, scale):
    X = df[["Open-Close", "High-Low"]]
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    split = int(split_pct * len(df))
    X_all = X.copy()
    if scale:
        sc = StandardScaler()
        sc.fit(X.iloc[:split])
        X_all = pd.DataFrame(sc.transform(X_all), index=X.index, columns=X.columns)
    gamma = "scale" if gamma_str == "scale" else float(gamma_str)
    m = SVC(kernel=kernel, C=C, gamma=gamma)
    m.fit(X_all.iloc[:split], y[:split])
    df2 = df.copy()
    df2["Signal"]          = m.predict(X_all)
    df2["Strategy_Return"] = df2["Return"] * df2["Signal"].shift(1)
    df2["Cum_Return"]      = (1 + df2["Return"]).cumprod()
    df2["Cum_Strategy"]    = (1 + df2["Strategy_Return"]).cumprod()
    return df2, split

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 10px">
        <div style="font-size:2.4rem">📊</div>
        <div style="font-size:1.05rem;font-weight:800;color:#e5e7eb">SVM Stock AI</div>
        <div style="font-size:0.75rem;color:#6b7280;margin-top:3px">Reliance · BSE</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.05);margin:8px 0 18px">
    """, unsafe_allow_html=True)

    st.markdown("**🎛️ Model Settings**")
    kernel    = st.selectbox("SVM Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
    C         = st.select_slider("C (Regularization)", options=[0.01,0.1,0.5,1.0,5.0,10.0,50.0,100.0], value=1.0)
    gamma_str = st.selectbox("Gamma", ["scale", "auto", "0.001", "0.01", "0.1", "1.0"], index=0)
    split_pct = st.slider("Train Split %", 60, 90, 80, 5) / 100
    scale     = st.toggle("Feature Scaling (StandardScaler)", value=True)

    st.markdown("<br>**📋 Page**", unsafe_allow_html=True)
    page = st.radio(" ", ["🏠 Overview", "📈 Price Analysis",
                           "🤖 SVM Model", "📊 Kernel Compare", "💰 Backtest"],
                    label_visibility="collapsed")

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.05);margin:14px 0 8px">
    <div style="text-align:center;font-size:0.72rem;color:#4b5563">
        Made with ❤️ · Harsh Choudhary<br>
        <a href="https://github.com/HarshChoudhary2003" style="color:#d97706">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading Reliance data..."):
    df = load_data()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">📊 SVM · Classification · Backtesting</div>
  <div class="hero-title">Stock Price Direction Predictor</div>
  <div class="hero-sub">
    Predict whether <b>Reliance Industries</b> stock will go UP or DOWN next day
    using Support Vector Machines — with live kernel tuning, signal analysis &amp; strategy backtesting.
  </div>
</div>
""", unsafe_allow_html=True)

# ── LATEST SIGNAL (always visible) ───────────────────────────────────────────
latest = df.iloc[-1]
signal_input = np.array([[latest["Open-Close"], latest["High-Low"]]])
_X = df[["Open-Close","High-Low"]]
_y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
_sp = int(split_pct * len(df))
_Xtr = _X.iloc[:_sp]
if scale:
    _sc = StandardScaler(); _sc.fit(_Xtr)
    signal_input_s = _sc.transform(signal_input)
else:
    signal_input_s = signal_input
_gm = "scale" if gamma_str == "scale" else float(gamma_str)
_m  = SVC(kernel=kernel, C=C, gamma=_gm, probability=True)
_Xfit = _sc.transform(_Xtr) if scale else _Xtr
_m.fit(_Xfit, _y[:_sp])
live_prob  = _m.predict_proba(signal_input_s)[0]
live_pred  = _m.predict(signal_input_s)[0]
conf_pct   = max(live_prob) * 100

cols_sig = st.columns([1,1,1,1,1])
cols_sig[0].markdown(kcard("💵", f"₹{latest['Close']:.1f}", "Last Close", "gold"), unsafe_allow_html=True)
cols_sig[1].markdown(kcard("📅", str(df.index[-1].date()), "Last Date", "blue"), unsafe_allow_html=True)
cols_sig[2].markdown(kcard("📊", f"{len(df):,}", "Total Days", "gold"), unsafe_allow_html=True)
cols_sig[3].markdown(kcard("🎯", f"{conf_pct:.1f}%", "Model Confidence", "green" if live_pred==1 else "red"), unsafe_allow_html=True)
cols_sig[4].markdown(kcard("🚀" if live_pred==1 else "🔻",
                            "BUY ↑" if live_pred==1 else "SELL ↓",
                            "Tomorrow Signal", "green" if live_pred==1 else "red"),
                     unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    section("📈", "Reliance Close Price History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#fbbf24", width=1.5),
                             fill="tozeroy", fillcolor="rgba(234,179,8,0.06)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA 20",
                             line=dict(color="#60a5fa", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA 50",
                             line=dict(color="#f87171", width=1.5, dash="dash")))
    fig.update_layout(**PL, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("🕯️", "Candlestick (Last 120 Days)")
        tail = df.tail(120)
        fig2 = go.Figure(go.Candlestick(
            x=tail.index, open=tail["Open"], high=tail["High"],
            low=tail["Low"], close=tail["Close"],
            increasing_line_color="#34d399", decreasing_line_color="#f87171",
            name="RELIANCE"))
        fig2.update_layout(**PL, height=340)
        fig2.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        section("📊", "Volume (Last 120 Days)")
        colors = ["#34d399" if r >= 0 else "#f87171" for r in tail["Return"]]
        fig3 = go.Figure(go.Bar(x=tail.index, y=tail["Volume"],
                                marker_color=colors, name="Volume"))
        fig3.update_layout(**PL, height=340)
        st.plotly_chart(fig3, use_container_width=True)

    section("📉", "Daily Return Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig4 = px.histogram(df, x="Return", nbins=80,
                            color_discrete_sequence=["#fbbf24"],
                            labels={"Return": "Daily Return"})
        fig4.update_layout(**PL, height=280)
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        fig5 = px.box(df, y="Return", color_discrete_sequence=["#fbbf24"])
        fig5.update_layout(**PL, height=280)
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Price Analysis" in page:
    section("📡", "Technical Indicators")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Price + MAs", "RSI (14)", "Volatility (20D)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#fbbf24", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
                             line=dict(color="#60a5fa", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                             line=dict(color="#f87171", width=1.2, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.4)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(52,211,153,0.4)", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["Volatility"]*100,
                             name="Volatility %", fill="tozeroy",
                             line=dict(color="#f59e0b", width=1.5),
                             fillcolor="rgba(245,158,11,0.1)"), row=3, col=1)

    fig.update_layout(**PL, height=650, showlegend=True,
                      legend=dict(orientation="h", y=1.03, x=1, xanchor="right"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    section("🔬", "Feature Space — SVM Inputs")
    c1, c2 = st.columns(2)
    with c1:
        fig6 = px.scatter(df, x="Open-Close", y="High-Low",
                          color=np.where(df["Close"].shift(-1) > df["Close"], "UP ↑", "DOWN ↓"),
                          color_discrete_map={"UP ↑": "#34d399", "DOWN ↓": "#f87171"},
                          opacity=0.55, labels={"color": "Next Day"},
                          title="Feature Space (Open-Close vs High-Low)")
        fig6.update_layout(**PL, height=380)
        st.plotly_chart(fig6, use_container_width=True)

    with c2:
        up_pct = (np.where(df["Close"].shift(-1) > df["Close"], 1, 0).sum() / len(df)) * 100
        fig7 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=up_pct,
            title={"text": "% UP Days", "font": {"family": "Outfit", "size": 18, "color": "#9ca3af"}},
            number={"suffix": "%", "font": {"size": 42, "color": "#fbbf24"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                "bar":  {"color": "#fbbf24"},
                "steps": [{"range": [0,40], "color": "rgba(239,68,68,0.15)"},
                           {"range": [40,60], "color": "rgba(234,179,8,0.1)"},
                           {"range": [60,100],"color": "rgba(52,211,153,0.12)"}],
                "threshold": {"line": {"color": "#34d399", "width": 3},
                              "thickness": 0.9, "value": 50},
                "bgcolor": "rgba(0,0,0,0)",
            }
        ))
        fig7.update_layout(**PL, height=380)
        st.plotly_chart(fig7, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SVM MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif "SVM Model" in page:
    section("🤖", f"SVM Classifier — Kernel: {kernel.upper()} | C={C}")
    with st.spinner("🧠 Training SVM..."):
        model, y_pred, y_te, tr_acc, te_acc, cm, split = train_svm(
            df, split_pct, kernel, C, gamma_str, scale)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kcard("🎯", f"{tr_acc:.2%}", "Train Accuracy", "gold"), unsafe_allow_html=True)
    c2.markdown(kcard("🏆", f"{te_acc:.2%}", "Test Accuracy", "green"), unsafe_allow_html=True)
    c3.markdown(kcard("📋", f"{split:,}", "Train Rows", "blue"),   unsafe_allow_html=True)
    c4.markdown(kcard("🔍", f"{len(df)-split:,}", "Test Rows", "red"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("📊", "Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="YlOrRd",
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=["DOWN (0)","UP (1)"], y=["DOWN (0)","UP (1)"])
        fig_cm.update_traces(textfont_size=20)
        fig_cm.update_layout(**PL, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        section("📈", "Accuracy Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=te_acc * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#fbbf24"}},
            title={"text": "Test Accuracy", "font": {"family": "Outfit", "color": "#9ca3af"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#fbbf24"},
                "steps": [{"range": [0,50],  "color": "rgba(239,68,68,0.15)"},
                           {"range": [50,65], "color": "rgba(234,179,8,0.1)"},
                           {"range": [65,100],"color": "rgba(52,211,153,0.12)"}],
                "bgcolor": "rgba(0,0,0,0)",
            }
        ))
        fig_g.update_layout(**PL, height=340)
        st.plotly_chart(fig_g, use_container_width=True)

    section("📉", "Prediction Signal Overlay (Test Set)")
    test_df      = df.iloc[split:].copy()
    test_df["Pred"] = y_pred
    up_idx   = test_df[test_df["Pred"] == 1].index
    down_idx = test_df[test_df["Pred"] == 0].index

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=test_df.index, y=test_df["Close"],
                                 name="Close", line=dict(color="rgba(234,179,8,0.5)", width=1.2)))
    fig_sig.add_trace(go.Scatter(x=up_idx, y=test_df.loc[up_idx, "Close"],
                                 mode="markers", name="BUY Signal",
                                 marker=dict(color="#34d399", size=7, symbol="triangle-up")))
    fig_sig.add_trace(go.Scatter(x=down_idx, y=test_df.loc[down_idx, "Close"],
                                 mode="markers", name="SELL Signal",
                                 marker=dict(color="#f87171", size=7, symbol="triangle-down")))
    fig_sig.update_layout(**PL, height=380,
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right"))
    st.plotly_chart(fig_sig, use_container_width=True)

    section("📋", "Classification Report")
    from sklearn.metrics import classification_report
    report = classification_report(y_te, y_pred, target_names=["DOWN","UP"], output_dict=True)
    st.dataframe(
        pd.DataFrame(report).transpose().round(4).style
          .highlight_max(axis=0, color="rgba(234,179,8,0.25)")
          .set_table_styles([{"selector":"th","props":[("background","#161b22"),("color","#e5e7eb")]}]),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KERNEL COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif "Kernel Compare" in page:
    section("🔬", "All SVM Kernels — Side-by-Side Comparison")
    with st.spinner("⚙️ Running all kernels..."):
        kres = kernel_comparison(df, split_pct, scale)

    kdf = pd.DataFrame(kres).T.reset_index().rename(columns={"index": "Kernel"})
    kdf["Gap"] = (kdf["Train"] - kdf["Test"]).abs()

    c1, c2 = st.columns(2)
    with c1:
        fig_k = go.Figure()
        fig_k.add_trace(go.Bar(name="Train Acc", x=kdf["Kernel"], y=kdf["Train"],
                               marker_color="#fbbf24", text=kdf["Train"].apply(lambda v: f"{v:.2%}"),
                               textposition="outside"))
        fig_k.add_trace(go.Bar(name="Test Acc",  x=kdf["Kernel"], y=kdf["Test"],
                               marker_color="#60a5fa", text=kdf["Test"].apply(lambda v: f"{v:.2%}"),
                               textposition="outside"))
        apply_pl(fig_k, height=380, barmode="group")
        fig_k.update_yaxes(range=[0, 1.15], tickformat=".0%")
        st.plotly_chart(fig_k, use_container_width=True)

    with c2:
        fig_r = px.scatter(kdf, x="Train", y="Test", text="Kernel",
                           color="Gap", color_continuous_scale="RdYlGn_r",
                           size=[50]*4, size_max=40)
        fig_r.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                        line=dict(color="rgba(255,255,255,0.2)", dash="dash"))
        fig_r.update_traces(textposition="top center", textfont_size=14)
        apply_pl(fig_r, height=380, coloraxis_showscale=False)
        fig_r.update_xaxes(tickformat=".0%", range=[0.4, 1.05])
        fig_r.update_yaxes(tickformat=".0%", range=[0.4, 1.05])
        st.plotly_chart(fig_r, use_container_width=True)

    section("🏆", "Best Kernel Leaderboard")
    kdf["Train %"] = (kdf["Train"]*100).round(2)
    kdf["Test %"]  = (kdf["Test"]*100).round(2)
    kdf["Overfit Gap"] = (kdf["Gap"]*100).round(2)
    kdf = kdf.sort_values("Test %", ascending=False)
    st.dataframe(
        kdf[["Kernel","Train %","Test %","Overfit Gap"]].style
          .background_gradient(subset=["Test %"], cmap="YlOrRd")
          .highlight_min(subset=["Overfit Gap"], color="rgba(52,211,153,0.3)")
          .set_table_styles([{"selector":"th","props":[("background","#161b22"),("color","#e5e7eb")]}]),
        use_container_width=True, hide_index=True
    )

    section("🎯", "C Hyperparameter Sweep (RBF Kernel)")
    C_vals = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
    X_s = df[["Open-Close","High-Low"]]; y_s = np.where(df["Close"].shift(-1)>df["Close"],1,0)
    sp = int(split_pct*len(df))
    Xtr2, Xte2 = X_s.iloc[:sp], X_s.iloc[sp:]
    ytr2, yte2 = y_s[:sp], y_s[sp:]
    if scale:
        sc2 = StandardScaler(); Xtr2 = sc2.fit_transform(Xtr2); Xte2 = sc2.transform(Xte2)
    c_tr, c_te = [], []
    for cv in C_vals:
        m2 = SVC(kernel="rbf", C=cv).fit(Xtr2, ytr2)
        c_tr.append(accuracy_score(ytr2, m2.predict(Xtr2)))
        c_te.append(accuracy_score(yte2, m2.predict(Xte2)))
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=[str(c) for c in C_vals], y=c_tr,
                               name="Train", line=dict(color="#fbbf24", width=2), mode="lines+markers"))
    fig_c.add_trace(go.Scatter(x=[str(c) for c in C_vals], y=c_te,
                               name="Test",  line=dict(color="#60a5fa", width=2), mode="lines+markers"))
    apply_pl(fig_c, height=320, xaxis_title="C Value")
    fig_c.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_c, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
elif "Backtest" in page:
    section("💰", "Strategy Backtesting — SVM vs. Buy & Hold")
    with st.spinner("⚙️ Running backtest..."):
        bt_df, split = backtest(df, split_pct, kernel, C, gamma_str, scale)

    # Metrics
    test_bt       = bt_df.iloc[split:]
    final_bh      = test_bt["Cum_Return"].iloc[-1]
    final_strat   = test_bt["Cum_Strategy"].iloc[-1]
    outperform    = final_strat - final_bh
    total_ret_bh  = (final_bh - 1) * 100
    total_ret_st  = (final_strat - 1) * 100
    win_days      = (test_bt["Strategy_Return"] > 0).sum()
    win_rate      = win_days / len(test_bt) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kcard("🏦", f"{total_ret_bh:+.1f}%", "Buy & Hold Return", "blue"), unsafe_allow_html=True)
    c2.markdown(kcard("🤖", f"{total_ret_st:+.1f}%", "SVM Strategy Return",
                       "green" if total_ret_st > total_ret_bh else "red"), unsafe_allow_html=True)
    c3.markdown(kcard("⚡", f"{outperform*100:+.1f}%", "Alpha vs B&H",
                       "green" if outperform > 0 else "red"), unsafe_allow_html=True)
    c4.markdown(kcard("🎯", f"{win_rate:.1f}%", "Win Rate", "gold"), unsafe_allow_html=True)

    section("📈", "Cumulative Returns — Test Period")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt["Cum_Return"],
                                name="Buy & Hold", line=dict(color="#f87171", width=2.2)))
    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt["Cum_Strategy"],
                                name="SVM Strategy", line=dict(color="#34d399", width=2.5)))
    fig_bt.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig_bt.update_layout(**PL, height=420,
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right"))
    st.plotly_chart(fig_bt, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("📊", "Daily Strategy Returns")
        colors = ["#34d399" if v > 0 else "#f87171" for v in test_bt["Strategy_Return"]]
        fig_dr = go.Figure(go.Bar(x=test_bt.index, y=test_bt["Strategy_Return"]*100,
                                  marker_color=colors, name="Strategy Daily Return"))
        fig_dr.update_layout(**PL, height=320, yaxis_title="Return %")
        st.plotly_chart(fig_dr, use_container_width=True)

    with c2:
        section("🧭", "Drawdown Analysis")
        roll_max = test_bt["Cum_Strategy"].cummax()
        drawdown = (test_bt["Cum_Strategy"] - roll_max) / roll_max * 100
        fig_dd = go.Figure(go.Scatter(x=test_bt.index, y=drawdown,
                                      fill="tozeroy", line=dict(color="#f87171", width=1.5),
                                      fillcolor="rgba(239,68,68,0.12)", name="Drawdown %"))
        fig_dd.update_layout(**PL, height=320, yaxis_title="Drawdown %")
        st.plotly_chart(fig_dd, use_container_width=True)

    section("📋", "Recent Signals (Last 20 Days)")
    show = bt_df.tail(20)[["Open","High","Low","Close","Signal","Return","Strategy_Return"]].copy()
    show["Signal"] = show["Signal"].map({1:"🟢 UP", 0:"🔴 DOWN"})
    show["Return"] = (show["Return"]*100).round(2).astype(str) + "%"
    show["Strategy_Return"] = (show["Strategy_Return"]*100).round(2).astype(str) + "%"
    st.dataframe(show, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:36px 0 16px;border-top:1px solid rgba(255,255,255,0.05);margin-top:40px">
  <div style="font-size:1.15rem;font-weight:700;
              background:linear-gradient(135deg,#fbbf24,#ef4444);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    SVM Stock Direction · AI Dashboard
  </div>
  <div style="color:#4b5563;font-size:0.8rem;margin-top:6px">
    Built with Streamlit · Scikit-Learn · Plotly &nbsp;|&nbsp;
    Made with ❤️ by
    <a href="https://github.com/HarshChoudhary2003" style="color:#d97706;text-decoration:none">Harsh Choudhary</a>
  </div>
</div>
""", unsafe_allow_html=True)
