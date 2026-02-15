import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
from datetime import datetime

# ----------------------- PAGE CONFIGURATION -----------------------
st.set_page_config(
    page_title="EMI Precision AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- CUSTOM STYLING -----------------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 18, 30) 0%, rgb(10, 10, 15) 90.2%);
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 20, 35, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Headers */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(90deg, #00dffd, #007cf0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Input Fields */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00dffd 0%, #007cf0 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 223, 253, 0.3);
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 223, 253, 0.5);
    }

    /* Success/Error Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- HELPER FUNCTIONS -----------------------
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Animations
lottie_finance = load_lottieurl("https://lottie.host/5a8e268d-8a53-4830-9755-90033c46d305/D91yq4Q9fP.json")
lottie_success = load_lottieurl("https://lottie.host/93380961-4648-4720-9467-c1074e5033c9/Wv3vYjY3gP.json")
lottie_warning = load_lottieurl("https://lottie.host/28080f53-294c-47fc-8f64-84577f8007a1/wZHKXk2b3P.json")
lottie_loading = load_lottieurl("https://lottie.host/d461011e-0e2f-4c54-8e2b-f35c24949219/V0zQv2y9b0.json")

# ----------------------- SIDEBAR NAVIGATION -----------------------
with st.sidebar:
    if lottie_finance:
        st_lottie(lottie_finance, height=150, key="logo")
    else:
        st.markdown("<h1 style='text-align: center;'>💳</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>EMI Precision AI</h2>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Predictor", "Analytics", "About"],
        icons=["speedometer2", "calculator", "graph-up-arrow", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00dffd", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "10px", "--hover-color": "rgba(255,255,255,0.1)"},
            "nav-link-selected": {"background-color": "rgba(0, 223, 253, 0.2)", "color": "#00dffd", "border-left": "4px solid #00dffd"},
        }
    )
    
    st.markdown("---")
    st.caption("v2.1.0 • Powered by XGBoost & MLflow")

# ----------------------- MODEL LOADING -----------------------
CLASSIFIER_URI = "models:/EMIClassifier/Production"
REGRESSOR_URI = "models:/EMIRegressor/Production"

@st.cache_resource
def load_models():
    try:
        clf = mlflow.pyfunc.load_model(CLASSIFIER_URI)
        reg = mlflow.pyfunc.load_model(REGRESSOR_URI)
        return clf, reg
    except Exception:
        return None, None

classifier, regressor = load_models()

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------------- PAGE LOGIC -----------------------

if selected == "Predictor":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Financial Risk Assessment")
        st.markdown("Enter your financial details below to get an instant AI-powered evaluation.")
    
    if not classifier or not regressor:
        st.warning("⚠️ AI Models not found. Showing demo interface but predictions will be simulated or unavailable.")

    # Input Form within a Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.form("emi_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("👤 Profile")
            age = st.number_input("Age", 18, 70, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox("Education", ["Graduate", "Post Graduate", "High School", "Professional"])
            marital = st.selectbox("Marital Status", ["Single", "Married"])
            
        with c2:
            st.subheader("💼 Career")
            employment = st.selectbox("Employment", ["Private", "Government", "Self-employed"])
            company = st.selectbox("Company Type", ["Private", "MNC", "Government", "Startup"])
            exp_years = st.slider("Experience (Years)", 0, 40, 5)
            salary = st.number_input("Monthly Salary (₹)", 10000, 1000000, 50000, step=1000)

        with c3:
            st.subheader("💰 Finances")
            credit_score = st.slider("Credit Score", 300, 900, 750)
            bank_balance = st.number_input("Bank Balance (₹)", 0, 5000000, 50000, step=5000)
            current_loans = st.selectbox("Existing Loans?", ["No", "Yes"])
            current_emi = st.number_input("Current EMI (₹)", 0, 200000, 0, step=1000)

        st.markdown("---")
        
        c4, c5 = st.columns(2)
        with c4:
            st.subheader("🏠 Lifestyle")
            house_type = st.selectbox("Residence", ["Owned", "Rented", "Family"])
            rent = st.number_input("Rent (₹)", 0, 100000, 0, step=1000)
            expenses = st.number_input("Monthly Expenses (₹)", 0, 200000, 15000, step=1000)
            dependents = st.slider("Dependents", 0, 5, 1)

        with c5:
            st.subheader("🎯 Loan Request")
            purpose = st.selectbox("Purpose", ["Personal", "Vehicle", "Home", "Education", "Medical"])
            loan_amt = st.number_input("Loan Amount (₹)", 10000, 5000000, 200000, step=10000)
            tenure = st.slider("Tenure (Months)", 6, 84, 24)

        submit = st.form_submit_button("🚀 Evaluate Eligibility")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        # Simulate loading
        with st.spinner("Analyzing creditworthiness..."):
            import time
            time.sleep(1.5)
            
            # --- Logic to process features matching the model ---
            # (Simplified feature/mock logic for robustness if models missing)
            try:
                # Prepare data dictionary
                data = {
                    "age": age, "gender": gender, "marital_status": marital, "education": education,
                    "monthly_salary": salary, "employment_type": employment, 
                    "years_of_employment": exp_years, "company_type": company,
                    "house_type": house_type, "monthly_rent": rent,
                    "family_size": dependents + 1, "dependents": dependents,
                    "school_fees": 0, "college_fees": 0, "groceries_utilities": expenses * 0.4,
                    "travel_expenses": expenses * 0.2, "other_monthly_expenses": expenses * 0.4,
                    "existing_loans": current_loans, "current_emi_amount": current_emi,
                    "credit_score": credit_score, "bank_balance": bank_balance,
                    "emergency_fund": bank_balance * 0.5, "emi_scenario": f"{purpose} EMI",
                    "requested_amount": loan_amt, "requested_tenure": tenure
                }
                
                df_input = pd.DataFrame([data])
                
                # --- Preprocessing (Mirroring training logic) ---
                df_input["total_monthly_expenses"] = rent + expenses
                df_input["dti_ratio"] = (current_emi + df_input["total_monthly_expenses"]) / (salary + 1)
                df_input["affordability_ratio"] = (salary - df_input["total_monthly_expenses"]) / (salary + 1)
                df_input["log_bank_balance"] = np.log1p(bank_balance)
                df_input["log_emergency_fund"] = np.log1p(df_input["emergency_fund"])
                
                # Mock encoding for scenario just in case
                scenario_map = {"E-commerce Shopping EMI":0, "Home Appliances EMI":1, "Vehicle EMI":2, "Personal Loan EMI":3, "Education EMI":4}
                mapped_scen = scenario_map.get(f"{purpose} EMI", 3)
                df_input["emi_scenario_code"] = mapped_scen
                
                # Align schema
                # Align schema
                numeric_cols = df_input.select_dtypes(include=np.number).columns.tolist()
                
                # These columns must be integers according to MLflow schema
                int_cols = ["family_size", "dependents"]
                # This column must be int32
                int32_cols = ["emi_scenario_code"]
                
                for col in numeric_cols:
                    if col in int_cols:
                        df_input[col] = df_input[col].astype("int64")
                    elif col in int32_cols:
                        df_input[col] = df_input[col].astype("int32")
                    else:
                        df_input[col] = df_input[col].astype("float64")

                if classifier and regressor:
                    pred_raw = classifier.predict(df_input)[0]
                    pred_class = {0: "Not Eligible", 1: "High Risk", 2: "Eligible"}.get(int(pred_raw), "Unknown")
                    max_emi = float(regressor.predict(df_input)[0])
                else:
                    # Fallback Simulation
                    pred_class = "Eligible" if credit_score > 700 and df_input["dti_ratio"].iloc[0] < 0.4 else "High Risk"
                    max_emi = (salary * 0.4) - current_emi

                # Store history
                rec = data.copy()
                rec.update({"prediction": pred_class, "max_emi": max_emi, "date": datetime.now()})
                st.session_state["history"].append(rec)

                # --- Results Display ---
                st.markdown("### Evaluation Results")
                r1, r2 = st.columns([1, 1])
                
                with r1:
                    if pred_class == "Eligible":
                        st.markdown('<div class="glass-card" style="border-left: 5px solid #00dffd;">', unsafe_allow_html=True)
                        st.success(f"### 🎉 Approved")
                        st.write("You are eligible for this loan structure.")
                        if lottie_success:
                            st_lottie(lottie_success, height=150, key="success_anim")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="glass-card" style="border-left: 5px solid #ff4b4b;">', unsafe_allow_html=True)
                        st.warning(f"### ⚠️ {pred_class}")
                        st.write("Consider reducing the loan amount or increasing tenure.")
                        if lottie_warning:
                            st_lottie(lottie_warning, height=150, key="warn_anim")
                        st.markdown('</div>', unsafe_allow_html=True)

                with r2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.metric("Max Affordable EMI", f"₹ {max_emi:,.2f}", delta=f"{max_emi - (loan_amt/tenure):,.0f} vs Requested")
                    st.progress(min(1.0, max_emi / (loan_amt / tenure) if loan_amt > 0 else 0))
                    st.caption(f"Requested EMI: ₹ {loan_amt/tenure:,.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

elif selected == "Dashboard":
    st.title("📊 Financial Dashboard")
    
    if len(st.session_state["history"]) > 0:
        df = pd.DataFrame(st.session_state["history"])
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Assessments", len(df))
        m2.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
        m3.metric("Avg Requested Loan", f"₹ {df['requested_amount'].mean()/1000:.0f}k")
        m4.metric("Approval Rate", f"{len(df[df['prediction']=='Eligible'])/len(df)*100:.0f}%")
        
        st.markdown("---")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Credit Score Distribution")
            fig = px.histogram(df, x="credit_score", color="prediction", nbins=20, 
                               color_discrete_map={"Eligible": "#00dffd", "High Risk": "#ff9f43", "Not Eligible": "#ff4b4b"},
                               template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Income vs Loan Amount")
            fig = px.scatter(df, x="monthly_salary", y="requested_amount", color="prediction",
                             size="credit_score", hover_data=["age", "education"],
                             color_discrete_map={"Eligible": "#00dffd", "High Risk": "#ff9f43", "Not Eligible": "#ff4b4b"},
                             template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
             if lottie_finance:
                st_lottie(lottie_finance, height=300)
             else:
                st.info("Finance Visualization Placeholder")
        with col2:
             st.info("No data available yet. Go to the Predictor tab to run your first assessment!")

elif selected == "Analytics":
     st.title("📈 Deep Dive Analytics")
     st.info("Advanced analytics module coming soon.")
     if lottie_loading:
        st_lottie(lottie_loading, height=200)

elif selected == "About":
    st.title("ℹ️ About")
    st.markdown("""
    ### EMI Precision AI
    This application leverages advanced Gradient Boosting Machines (XGBoost) orchestrated via MLflow to provide real-time financial eligibility assessments.
    
    **Key Features:**
    - **Real-time Inference:** sub-millisecond predictions.
    - **Explainable Metrics:** DTI, Affordability Ratio breakdown.
    - **Interactive Dashboard:** Track all your simulations.
    
    Developed by **Harsh Choudhary**
    """)
