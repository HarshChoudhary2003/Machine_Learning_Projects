# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import io
from datetime import datetime
import plotly.express as px

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(
    page_title=" EMIPredict AI - Financial Risk Assessment",
    page_icon="💳",
    layout="wide",
)

# ----------------------- NAVIGATION MENU -----------------------
menu = st.sidebar.radio(
    "📂 Main Menu",
    ["🏠 Home", "📊 Dashboard", "🧠 Insights", "⚙️ About & Help"],
)

# ----------------------- MODEL LOADING -----------------------
CLASSIFIER_MODEL_URI = "models:/EMIClassifier/Production"
REGRESSOR_MODEL_URI = "models:/EMIRegressor/Production"

@st.cache_resource
def load_models():
    clf = mlflow.pyfunc.load_model(CLASSIFIER_MODEL_URI)
    reg = mlflow.pyfunc.load_model(REGRESSOR_MODEL_URI)
    return clf, reg

# maintain prediction history
if "pred_history" not in st.session_state:
    st.session_state["pred_history"] = []

# ----------------------- HOME PAGE (Prediction UI) -----------------------
if menu == "🏠 Home":
    st.title(" EMIPredict AI - Intelligent EMI Risk Assessment")
    st.caption("AI-powered financial eligibility assessment using MLflow + XGBoost")

    try:
        classifier_model, regressor_model = load_models()
        st.success("✅ Models loaded successfully from MLflow Registry.")
    except Exception as e:
        st.error("❌ Failed to load models. Please check your MLflow URIs.")
        st.exception(e)
        st.stop()

    with st.expander("📘 How to Use (Step-by-Step)", expanded=False):
        st.markdown("""
        1. Enter customer’s financial details.  
        2. Click **Predict EMI Eligibility**.  
        3. View **AI prediction**, **max EMI**, and **financial indicators**.  
        4. Switch to the **Dashboard tab** to see your past predictions visually.
        """)

    # ----------------------- INPUT FORM -----------------------
    st.header("📋 Financial Details")
    with st.form("emi_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.slider("Years of Employment", 0, 40, 5)
            company_type = st.selectbox("Company Type", ["Private", "Government", "Startup", "Other"])
        with c2:
            monthly_salary = st.number_input("Monthly Salary (INR)", 5000, 300000, 45000, step=1000)
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
            current_emi_amount = st.number_input("Current EMI Amount (INR)", 0, 100000, 0, step=500)
            credit_score = st.slider("Credit Score (300–850)", 300, 850, 700)
            bank_balance = st.number_input("Bank Balance (INR)", 0, 1000000, 20000, step=1000)
            emergency_fund = st.number_input("Emergency Fund (INR)", 0, 500000, 10000, step=1000)

        st.subheader("🏠 Household & Expenses")
        house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])
        monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100000, 5000, step=1000)
        family_size = st.slider("Family Size", 1, 10, 4)
        dependents = st.slider("Dependents", 0, 5, 1)
        school_fees = st.number_input("School Fees (INR)", 0, 50000, 0, step=1000)
        college_fees = st.number_input("College Fees (INR)", 0, 100000, 0, step=1000)
        groceries_utilities = st.number_input("Groceries & Utilities (INR)", 0, 100000, 10000, step=1000)
        travel_expenses = st.number_input("Travel Expenses (INR)", 0, 50000, 2000, step=500)
        other_monthly_expenses = st.number_input("Other Expenses (INR)", 0, 50000, 3000, step=500)

        st.subheader("💳 Loan Application")
        emi_scenario = st.selectbox(
            "EMI Type", 
            ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]
        )
        requested_amount = st.number_input("Requested Loan Amount (INR)", 10000, 2000000, 100000, step=10000)
        requested_tenure = st.slider("Requested Tenure (months)", 3, 84, 24)

        submit = st.form_submit_button("🔍 Predict EMI Eligibility & Amount")

    # ----------------------- PREDICTION -----------------------
    if submit:
        st.info("🔄 Processing input data...")

        # Base input
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "groceries_utilities": groceries_utilities,
            "travel_expenses": travel_expenses,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        }])

        # ----------------- Derived Features & Schema Alignment -----------------
        try:
            # Derived totals
            input_data["total_monthly_expenses"] = (
                input_data["monthly_rent"]
                + input_data["school_fees"]
                + input_data["college_fees"]
                + input_data["groceries_utilities"]
                + input_data["travel_expenses"]
                + input_data["other_monthly_expenses"]
            )
            input_data["dti_ratio"] = (
                (input_data["current_emi_amount"] + input_data["total_monthly_expenses"])
                / (input_data["monthly_salary"] + 1)
            )
            input_data["affordability_ratio"] = (
                (input_data["monthly_salary"] - input_data["total_monthly_expenses"])
                / (input_data["monthly_salary"] + 1)
            )
            input_data["log_bank_balance"] = np.log1p(input_data["bank_balance"])
            input_data["log_emergency_fund"] = np.log1p(input_data["emergency_fund"])

            # Add encoded EMI scenario
            scenario_map = {
                "E-commerce Shopping EMI": 0,
                "Home Appliances EMI": 1,
                "Vehicle EMI": 2,
                "Personal Loan EMI": 3,
                "Education EMI": 4,
            }
            input_data["emi_scenario_code"] = (
                input_data["emi_scenario"].map(scenario_map).fillna(0)
            )

            # Convert all numerics to float64 except emi_scenario_code
            numeric_cols = input_data.select_dtypes(include=["int", "float"]).columns.tolist()
            if "emi_scenario_code" in numeric_cols:
                numeric_cols.remove("emi_scenario_code")
            input_data[numeric_cols] = input_data[numeric_cols].astype("float64")

            # Set integer-based schema columns
            input_data["family_size"] = input_data["family_size"].astype("int64")
            input_data["dependents"] = input_data["dependents"].astype("int64")
            input_data["emi_scenario_code"] = input_data["emi_scenario_code"].astype("int32")

            st.success("✅ Input data validated and schema aligned with MLflow model.")
        except Exception as e:
            st.error("⚙️ Feature computation or type conversion failed.")
            st.exception(e)
            st.stop()

        with st.expander("🧾 Input Data Sent to Model"):
            st.write(input_data.dtypes)
            st.dataframe(input_data)

        # ----------------- PREDICTION -----------------
        try:
            label_map = {0: "Not_Eligible", 1: "High_Risk", 2: "Eligible"}
            pred_raw = classifier_model.predict(input_data)[0]
            pred_class = label_map.get(int(pred_raw), str(pred_raw))
            pred_emi = float(regressor_model.predict(input_data)[0])
        except Exception as e:
            st.error("❌ Model prediction failed. Check MLflow schema alignment.")
            st.exception(e)
            st.stop()

        # Save results
        record = input_data.copy()
        record["pred_class"] = pred_class
        record["pred_emi"] = pred_emi
        record["timestamp"] = datetime.now()
        st.session_state["pred_history"].append(record.iloc[0].to_dict())

        # Display
        st.success(f"🎯 Predicted EMI Eligibility: **{pred_class}**")
        st.info(f"💵 Maximum Affordable EMI: ₹{pred_emi:,.2f}")
        st.metric("Debt-to-Income Ratio", f"{input_data['dti_ratio'][0]:.2f}")
        st.metric("Affordability Ratio", f"{input_data['affordability_ratio'][0]:.2f}")
        st.progress(min(1.0, pred_emi / requested_amount))

        if pred_class == "Eligible":
            st.balloons()
            st.success("✅ You are eligible for this EMI plan!")
        elif pred_class == "High_Risk":
            st.warning("⚠️ You’re a borderline case — consider reducing EMI or tenure.")
        else:
            st.error("❌ You are not eligible for this EMI under the current conditions.")

# ----------------------- DASHBOARD -----------------------
elif menu == "📊 Dashboard":
    st.title("📊 EMI Prediction Dashboard")
    if not st.session_state["pred_history"]:
        st.warning("No predictions yet! Go to the Home tab first.")
    else:
        df = pd.DataFrame(st.session_state["pred_history"])
        st.dataframe(df.tail(10))

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x="pred_emi", color="pred_class", nbins=10, title="Distribution of Predicted EMI")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.scatter(df, x="credit_score", y="pred_emi", color="pred_class", title="Credit Score vs Predicted EMI", trendline="ols")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x="pred_class", y="affordability_ratio", color="pred_class", title="Affordability Ratio by Eligibility Category")
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------- INSIGHTS PAGE -----------------------
elif menu == "🧠 Insights":
    st.title("🧠 Financial Insights & Trends")
    if st.session_state["pred_history"]:
        df = pd.DataFrame(st.session_state["pred_history"])
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.scatter_3d(df, x="monthly_salary", y="total_monthly_expenses", z="pred_emi", color="pred_class", title="Salary vs Expenses vs EMI")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.line(df, x="timestamp", y="pred_emi", color="pred_class", title="EMI Predictions Over Time")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Run a few predictions first to see insights here.")

# ----------------------- ABOUT PAGE -----------------------
elif menu == "⚙️ About & Help":
    st.title("⚙️ About EMIPredict AI")
    st.markdown("""
    **EMIPredict AI** uses machine learning to:
    - Predict EMI eligibility  
    - Estimate maximum affordable EMI  
    - Provide explainable financial metrics  

    **Tech Stack:** Streamlit, MLflow, XGBoost, Plotly  
    **Created by:** Harsh Chouhary © 2025  

    """)
