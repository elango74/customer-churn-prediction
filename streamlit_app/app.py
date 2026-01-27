import sys
import os

#PATH SETUP
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

#IMPORTS
import streamlit as st
import pandas as pd
import joblib

from ui.styles import load_background
from ui.components import header, footer
from src.action_recommendation import recommend_action

#RISK CLASSIFICATION
def churn_risk_mapping(churn_prob):
    if churn_prob >= 0.40:
        return "HIGH RISK", "#ff4b4b"
    elif churn_prob >= 0.25:
        return "MEDIUM RISK", "#f7b731"
    else:
        return "LOW RISK", "#2ecc71"

#PAGE CONFIG 
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CACHE MODEL
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

#UI LOAD
load_background()
header()
st.write("")

#LOAD MODEL
MODEL_PATH = os.path.join(ROOT_DIR, "models", "churn_model.pkl")
model = load_model(MODEL_PATH)

#SIDEBAR INPUTS
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider(
    "Tenure (months)",
    0, 100, 12,
    help="Number of months the customer has stayed with the company"
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0.0, 200.0, 70.0,
    help="Average monthly billing amount"
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0,
    max_value=100000,
    value=840,
    step=100,
    help="Total amount billed to the customer"
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    help="Customer contract duration"
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
    help="Preferred payment method"
)

#PREDICT BUTTON 
st.write("")
if st.button("Predict Churn", use_container_width=True):

    with st.spinner("Analyzing customer behavior..."):

        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,

            "Contract_Month-to-month": 0,
            "Contract_One year": 0,
            "Contract_Two year": 0,

            "PaymentMethod_Electronic check": 0,
            "PaymentMethod_Mailed check": 0,
            "PaymentMethod_Bank transfer (automatic)": 0,
            "PaymentMethod_Credit card (automatic)": 0,
        }

        input_data[f"Contract_{contract}"] = 1
        input_data[f"PaymentMethod_{payment}"] = 1

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        churn_prob = model.predict_proba(input_df)[0][1]
        action = recommend_action(churn_prob)

        risk, color = churn_risk_mapping(churn_prob)

#VISUAL INDICATORS SECTION
        st.subheader("Churn Risk Indicator")
        st.progress(min(churn_prob, 1.0))

        st.markdown(
            f"""
            <div style="margin-top:30px; padding:30px;
                        border-radius:20px;
                        background:rgba(255,255,255,0.15);
                        box-shadow:0 10px 30px rgba(0,0,0,0.4);">
                <h2 style="color:white;">Prediction Result</h2>
                <p style="font-size:24px; color:{color};">
                    Churn Probability: <b>{churn_prob*100:.1f}%</b>
                </p>
                <p style="font-size:22px; color:{color};">
                    Risk Level: <b>{risk}</b>
                </p>
                <p style="font-size:20px; color:white;">
                    Risk Interpretation:
                    <b>
                        {"Immediate action required" if "HIGH" in risk else
                         "Monitor and engage" if risk == "MEDIUM RISK" else
                         "Customer is stable"}
                    </b>
                </p>
                <p style="font-size:22px; color:white;">
                    Recommended Action: <b>{action}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

#EXPLANATION SECTION
        st.subheader("Why this prediction?")

        if churn_prob >= 0.40:
            st.error(
                "🚨 Very high churn likelihood detected. "
                "Customer shows multiple strong churn indicators."
            )

        if churn_prob < 0.40 and tenure < 6 and contract == "Month-to-month":
            st.warning(
                "⚠️ Early churn behavior detected: "
                "New customer on a flexible contract with high monthly charges."
            )

        reasons = []

        if tenure < 12:
            reasons.append("Customer has a short tenure")
        if contract == "Month-to-month":
            reasons.append("Month-to-month contracts show higher churn rates")
        if payment == "Electronic check":
            reasons.append("Electronic check users churn more frequently")
        if monthly_charges > 80:
            reasons.append("High monthly charges increase churn probability")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("• Customer profile indicates stable behavior")

#MODEL DETAILS SECTION
with st.expander("Model Details"):
    st.write("""
    • Model: Random Forest Classifier  
    • Dataset: Telco Customer Churn  
    • Output: Churn Probability  
    • Business Goal: Customer Retention  
    • Evaluation Focus: Recall over Accuracy
    """)

footer()
