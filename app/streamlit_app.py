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

st.title("📉 Telco Customer Churn Prediction")
st.caption("Predict churn risk using the selected final model")


FINAL_MODEL_NAME = "Logistic Regression"


def get_recommendation(risk, prediction):
    if prediction == 1 and risk == "High Risk":
        return "Customer shows high churn risk. Retention action should be considered."
    elif prediction == 1 and risk == "Medium Risk":
        return "Customer shows moderate churn risk. Targeted engagement or a smaller retention incentive may help."
    return "Customer looks relatively stable. No immediate retention action is needed."


def risk_color(risk):
    if risk == "High Risk":
        return "red"
    elif risk == "Medium Risk":
        return "orange"
    return "green"


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
    """
    Basic sanity checks for input consistency.
    """
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


def estimate_customer_impact(prediction, risk_segment):
    """
    Very simple customer-level action framing.
    """
    if prediction == 1 and risk_segment == "High Risk":
        return {
            "action_cost": 500,
            "potential_value_saved": 5000,
            "message": "High-priority retention candidate."
        }
    elif prediction == 1:
        return {
            "action_cost": 500,
            "potential_value_saved": 5000,
            "message": "Possible retention candidate."
        }
    else:
        return {
            "action_cost": 0,
            "potential_value_saved": 0,
            "message": "No immediate retention spend suggested."
        }


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
    st.subheader("Customer Profile")

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
        model_name=FINAL_MODEL_NAME,
        threshold=selected_threshold
    )

    st.divider()
    st.subheader("Prediction")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Probability", f"{result['probability']:.2%}")
    col2.metric("Prediction", result["label"])
    col3.metric("Confidence", result["confidence"])
    col4.markdown(
        f"**Risk:** <span style='color:{risk_color(result['risk_segment'])}'>{result['risk_segment']}</span>",
        unsafe_allow_html=True
    )

    st.markdown("### Recommendation")
    st.write(get_recommendation(result["risk_segment"], result["prediction"]))

    impact = estimate_customer_impact(result["prediction"], result["risk_segment"])

    st.markdown("### Estimated customer-level impact")
    impact_col1, impact_col2, impact_col3 = st.columns(3)
    impact_col1.metric("Suggested Action Cost", f"{impact['action_cost']}")
    impact_col2.metric("Potential Value Saved", f"{impact['potential_value_saved']}")
    impact_col3.metric("Action View", impact["message"])

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
            "Alternative models were evaluated during development, but only the selected model output is shown here for consistency."
        )
        st.write(
            "Logistic Regression was kept as the final model because it offers stable performance, "
            "clear interpretability, and cleaner deployment behavior."
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