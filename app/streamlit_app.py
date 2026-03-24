# app/streamlit_app.py

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_single


st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📉",
    layout="wide"
)

FINAL_MODEL_NAME = "Logistic Regression"


def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --text-main: var(--text-color);
            --text-soft: color-mix(in srgb, var(--text-color) 72%, transparent);
            --text-muted: color-mix(in srgb, var(--text-color) 55%, transparent);
            --card-bg: var(--secondary-background-color);
            --page-bg: var(--background-color);
            --border-color: color-mix(in srgb, var(--text-color) 14%, transparent);
            --shadow-color: rgba(0, 0, 0, 0.10);
            --risk-low: #22c55e;
            --risk-moderate: #f59e0b;
            --risk-high: #ef4444;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .main-title {
            font-size: 2.05rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            line-height: 1.15;
            color: var(--text-main) !important;
        }

        .sub-title {
            font-size: 0.98rem;
            color: var(--text-soft) !important;
            margin-bottom: 1.25rem;
        }

        .section-title {
            font-size: 1.12rem;
            font-weight: 650;
            margin-top: 0.25rem;
            margin-bottom: 0.8rem;
            color: var(--text-main) !important;
        }

        .summary-card {
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: var(--card-bg);
            box-shadow: 0 2px 10px var(--shadow-color);
            min-height: 120px;
        }

        .summary-label {
            font-size: 0.82rem;
            color: var(--text-soft) !important;
            margin-bottom: 0.35rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .summary-value {
            font-size: 1.65rem;
            font-weight: 750;
            line-height: 1.1;
            margin-bottom: 0.18rem;
            color: var(--text-main) !important;
        }

        .summary-note {
            font-size: 0.9rem;
            color: var(--text-muted) !important;
            margin-top: 0.2rem;
        }

        .risk-low {
            color: var(--risk-low) !important;
        }

        .risk-moderate {
            color: var(--risk-moderate) !important;
        }

        .risk-high {
            color: var(--risk-high) !important;
        }

        .action-box {
            border-left: 5px solid var(--text-main);
            background: var(--card-bg);
            padding: 0.95rem 1rem;
            border-radius: 12px;
            margin-top: 0.25rem;
            margin-bottom: 0.75rem;
            border-top: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
        }

        .action-title {
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.22rem;
            color: var(--text-main) !important;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .action-text {
            font-size: 1rem;
            color: var(--text-main) !important;
            line-height: 1.45;
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 1rem 1rem 0.25rem 1rem;
            background: var(--card-bg) !important;
        }

        /* labels */
        label,
        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label,
        .stTextArea label {
            color: var(--text-main) !important;
            font-weight: 600 !important;
        }

        /* native inputs */
        input,
        textarea,
        select {
            color: var(--text-main) !important;
            background: var(--card-bg) !important;
        }

        /* baseweb select control */
        div[data-baseweb="select"] > div {
            color: var(--text-main) !important;
            background: var(--card-bg) !important;
            border-color: var(--border-color) !important;
        }

        /* selected value in dropdown */
        div[data-baseweb="select"] span {
            color: var(--text-main) !important;
        }

        /* dropdown menu */
        ul[role="listbox"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
        }

        ul[role="listbox"] li {
            color: var(--text-main) !important;
            background: var(--card-bg) !important;
        }

        ul[role="listbox"] li:hover {
            background: color-mix(in srgb, var(--text-color) 10%, var(--card-bg)) !important;
        }

        /* number input wrapper */
        div[data-baseweb="input"] > div {
            background: var(--card-bg) !important;
            border-color: var(--border-color) !important;
        }

        div[data-baseweb="input"] input {
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
        }

        ::placeholder {
            color: var(--text-muted) !important;
        }

        thead tr th {
            color: var(--text-main) !important;
            font-weight: 700 !important;
            background-color: var(--card-bg) !important;
        }

        tbody tr td {
            color: var(--text-main) !important;
            background-color: transparent !important;
        }

        .stDataFrame div {
            color: var(--text-main) !important;
        }

        div[data-testid="stExpander"] {
            margin-top: 0.45rem;
            color: var(--text-main) !important;
        }

        div[data-testid="stExpander"] details {
            background: transparent !important;
        }

        div[data-testid="stExpander"] summary {
            color: var(--text-main) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def get_recommendation(risk, prediction):
    if prediction == 1 and risk == "High Risk":
        return "Prioritize this customer for retention review."
    elif prediction == 1 and risk == "Moderate Risk":
        return "Monitor this customer and consider light-touch engagement."
    return "No immediate action needed."


def risk_class_name(risk):
    if risk == "High Risk":
        return "risk-high"
    elif risk == "Moderate Risk":
        return "risk-moderate"
    return "risk-low"


def load_threshold_report():
    path = "reports/threshold_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_model_comparison():
    path = "reports/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_feature_importance():
    path = "reports/feature_importance_logistic.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_retention_scenarios():
    path = "reports/retention_scenarios.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def validate_inputs(tenure, monthly, total):
    warnings = []

    expected_upper = monthly * max(tenure, 1) * 1.5
    expected_lower = monthly * max(tenure, 1) * 0.3

    if tenure > 0 and total > 0:
        if total > expected_upper:
            warnings.append(
                "Total Charges looks unusually high compared to Monthly Charges and Tenure Months."
            )
        elif total < expected_lower:
            warnings.append(
                "Total Charges looks unusually low compared to Monthly Charges and Tenure Months."
            )

    return warnings


def render_summary_card(label, value, note="", value_class=""):
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{label}</div>
            <div class="summary-value {value_class}">{value}</div>
            <div class="summary-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()

st.markdown('<div class="main-title">📉 Telco Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict churn risk using the final selected model and review supporting decision reports.</div>',
    unsafe_allow_html=True
)

threshold_report = load_threshold_report()
model_comparison_df = load_model_comparison()
retention_scenarios_df = load_retention_scenarios()
feature_importance_df = load_feature_importance()

suggested_threshold = 0.60
if threshold_report is not None and not threshold_report.empty:
    if "f1_score" in threshold_report.columns:
        best_row = threshold_report.sort_values(by="f1_score", ascending=False).iloc[0]
        suggested_threshold = float(best_row["threshold"])

st.sidebar.header("Prediction Settings")
st.sidebar.write(f"**Final model:** {FINAL_MODEL_NAME}")

selected_threshold = st.sidebar.slider(
    "Choose threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(suggested_threshold),
    step=0.01
)

st.sidebar.info(f"Suggested threshold from report: {suggested_threshold:.2f}")

with st.form("input_form"):
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure Months", 0, 100, 12)

    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    with col3:
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 100000.0, 1000.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_warnings = validate_inputs(tenure, monthly, total)

    for warning in input_warnings:
        st.warning(warning)

    data = {
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multi,
        "Internet Service": internet,
        "Online Security": security,
        "Online Backup": backup,
        "Device Protection": device,
        "Tech Support": tech,
        "Streaming TV": tv,
        "Streaming Movies": movies,
        "Contract": contract,
        "Paperless Billing": billing,
        "Payment Method": payment,
        "Monthly Charges": monthly,
        "Total Charges": total
    }

    result = predict_single(
        input_dict=data,
        threshold=selected_threshold
    )

    st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        render_summary_card(
            label="Probability",
            value=f"{result['probability']:.2%}",
            note="Estimated churn likelihood based on the selected model."
        )

    with c2:
        render_summary_card(
            label="Prediction",
            value=result["label"],
            note=f"Decision based on threshold = {result['threshold']:.2f}"
        )

    with c3:
        render_summary_card(
            label="Risk Level",
            value=result["risk_segment"],
            note="Risk label updates with the selected threshold.",
            value_class=risk_class_name(result["risk_segment"])
        )

    st.markdown("### Suggested next step")
    st.markdown(
        f"""
        <div class="action-box">
            <div class="action-title">Suggested next step</div>
            <div class="action-text">{get_recommendation(result["risk_segment"], result["prediction"])}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Why this prediction?"):
        explanation = result.get("explanation")

        if explanation:
            st.write("### Top factors increasing churn risk")
            positive_df = pd.DataFrame(explanation["top_positive"])
            if not positive_df.empty:
                st.dataframe(positive_df, use_container_width=True)
            else:
                st.write("No strong positive churn drivers found for this customer.")

            st.write("### Top factors reducing churn risk")
            negative_df = pd.DataFrame(explanation["top_negative"])
            if not negative_df.empty:
                st.dataframe(negative_df, use_container_width=True)
            else:
                st.write("No strong negative churn drivers found for this customer.")
        else:
            st.info("Detailed local explanation is not available for this prediction.")

    with st.expander("Suggested threshold and report view"):
        st.write(f"**Your selected threshold:** {selected_threshold:.2f}")
        st.write(f"**Suggested threshold from report:** {suggested_threshold:.2f}")
        st.write(
            "The suggested threshold comes from the saved evaluation report. "
            "It is meant to reflect the best trade-off between retention impact and campaign cost, "
            "not just the default 0.50 cutoff."
        )
        if threshold_report is not None:
            st.dataframe(threshold_report, use_container_width=True)
        else:
            st.info("Threshold report not found yet. Run the training pipeline first.")

    with st.expander("Feature importance / model weights"):
        if feature_importance_df is not None:
            st.dataframe(feature_importance_df, use_container_width=True)
        else:
            st.info("Feature importance report not found yet. Run the training pipeline first.")

    with st.expander("Retention scenario analysis"):
        if retention_scenarios_df is not None:
            st.dataframe(retention_scenarios_df, use_container_width=True)
        else:
            st.info("Retention scenario report not found yet. Run the training pipeline first.")

    with st.expander("Model info"):
        st.write(f"**Model used:** {FINAL_MODEL_NAME}")
        st.write(
            "This app uses Logistic Regression as the final decision model. "
            "It provides decision support by estimating churn risk and suggesting who may need review."
        )
        st.write(
            "Alternative models were evaluated during development, but only the selected model output is shown here for consistency."
        )
        st.write(f"**Decision threshold:** {result['threshold']:.2f}")

    with st.expander("Development / reviewer view"):
        st.write(
            "Other candidate models were evaluated during development and are summarized in the saved reports. "
            "That comparison is intentionally kept out of the main prediction view to avoid user confusion."
        )

        if model_comparison_df is not None:
            st.dataframe(model_comparison_df, use_container_width=True)
        else:
            st.info("Model comparison report not found yet. Run the training pipeline first.")