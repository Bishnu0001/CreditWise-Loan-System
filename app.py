"""
CreditWise Loan System
------------------------------------------------
Streamlit app that predicts whether a loan application should be
Approved or Rejected, based on a Logistic Regression model trained
on SecureTrust Bank's historical loan data.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CreditWise Loan System",
    page_icon="🏦",
    layout="centered",
)

MODEL_PATH = "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


bundle = load_model()
pipeline = bundle["pipeline"]
model_name = bundle["model_name"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏦 CreditWise Loan System")
st.caption(
    "An intelligent loan approval assistant for SecureTrust Bank — "
    "predicts whether an application is likely to be **Approved** or "
    "**Rejected**, before final human verification."
)

with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown(
        f"""
        This app uses a **{model_name}** model trained on historical
        loan applications (income, credit score, debt ratio, collateral,
        employment, and more) to flag applications for review.

        **This is a decision-support tool, not a final decision.**
        All predictions should be reviewed by a loan officer before
        any action is taken.
        """
    )

st.divider()

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Applicant Details")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input(
            "Applicant Monthly Income (₹)", min_value=0, value=8000, step=500
        )
        coapplicant_income = st.number_input(
            "Co-applicant Monthly Income (₹)", min_value=0, value=1500, step=500
        )
        age = st.number_input("Applicant Age", min_value=18, max_value=100, value=35)
        dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=0
        )
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=650
        )
        existing_loans = st.number_input(
            "Existing Loans (count)", min_value=0, max_value=10, value=1
        )
        dti_ratio = st.slider(
            "Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01
        )

    with col2:
        savings = st.number_input(
            "Savings Balance (₹)", min_value=0, value=10000, step=500
        )
        collateral_value = st.number_input(
            "Collateral Value (₹)", min_value=0, value=20000, step=500
        )
        loan_amount = st.number_input(
            "Loan Amount Requested (₹)", min_value=0, value=15000, step=500
        )
        loan_term = st.number_input(
            "Loan Term (months)", min_value=6, max_value=360, value=60, step=6
        )
        employment_status = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-employed", "Contract", "Unemployed"],
        )
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Home", "Education", "Personal", "Business", "Car"]
        )

    col3, col4 = st.columns(2)
    with col3:
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"]
        )
        education_level = st.selectbox(
            "Education Level", ["Graduate", "Not Graduate"]
        )
    with col4:
        gender = st.selectbox("Gender", ["Male", "Female"])
        employer_category = st.selectbox(
            "Employer Category", ["Government", "MNC", "Private", "Business", "Unemployed"]
        )

    submitted = st.form_submit_button("🔍 Check Loan Eligibility", use_container_width=True)

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    input_dict = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }

    input_df = pd.DataFrame([input_dict])[numeric_cols + categorical_cols]

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.success(f"✅ **Loan Likely Approved** (confidence: {probability:.1%})")
    else:
        st.error(f"❌ **Loan Likely Rejected** (confidence: {1 - probability:.1%})")

    st.progress(float(probability), text=f"Approval probability: {probability:.1%}")

    st.caption(
        "This prediction is generated automatically and is intended to support, "
        "not replace, final review by a loan officer."
    )
