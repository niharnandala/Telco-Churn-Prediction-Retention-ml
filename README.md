# Telco Customer Churn Prediction App

End-to-end machine learning project to predict customer churn and support retention decisions.

This project builds a complete pipeline from raw data to deployment, including model comparison, threshold optimization, and business-oriented retention strategy simulation.

---

## Problem

Customer churn directly impacts revenue in subscription-based businesses.

The goal of this project is to:
- identify customers likely to churn
- prioritize retention efforts
- balance prediction performance with business cost

---

## Solution Overview

The system predicts churn probability for each customer and classifies them into risk segments.

It also simulates how different decision thresholds impact:
- model performance
- campaign cost
- expected retained value

---

## Project Structure


telco_churn_app/
│
├── app/ # Streamlit application
├── artifacts/ # schema and default input configs
├── data/ # raw dataset (Excel)
├── data_processed/ # cleaned dataset
├── models/ # trained models
├── notebooks/ # analysis and experimentation
├── reports/ # metrics, comparisons, simulations
├── src/ # core pipeline code
├── screenshots/ # app UI previews
├── README.md
└── requirements.txt


---

## Pipeline


Raw Data
↓
Data Cleaning
↓
Feature Engineering
↓
Preprocessing (scaling + encoding)
↓
Model Training (Logistic + XGBoost)
↓
Evaluation + Threshold Analysis
↓
Retention Strategy Simulation
↓
Saved Model + Streamlit App


---

## Feature Engineering

Key engineered features include:

- `num_services` → number of active add-on services  
- `Tenure_group` → customer lifecycle stage  
- `AvgCharges` → normalized cost behavior  
- `HighRisk_Combo` → month-to-month + high charges  
- `ServiceIntensity` → normalized service usage  

---

## Models

Two models were trained and compared:

- Logistic Regression  
- XGBoost  

Evaluation included:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The final deployed model is **Logistic Regression** due to:
- competitive performance
- interpretability
- simpler deployment

---

## Threshold Optimization

Instead of using a fixed 0.5 threshold, multiple thresholds were evaluated.

Metrics were compared across:
- 0.30, 0.40, 0.50, 0.60, 0.70

The selected threshold balances:
- recall of churn customers
- unnecessary targeting cost

---

## Business Simulation

A retention strategy was simulated to evaluate real-world impact.

Assumptions:
- cost per targeted customer = 500  
- retained customer value = 5000  
- retention success rate = 35%  

Outputs include:
- number of customers targeted
- campaign cost
- expected retained value
- net business impact

---

## Streamlit App

The app allows real-time prediction for a single customer.

### Outputs:
- churn probability  
- predicted label  
- risk segment (Low / Medium / High)  
- recommended action  

An additional section provides model details for transparency.

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt


### 2. Run pipeline

python -m src.run_pipeline


### 3. Launch app

streamlit run app/streamlit_app.py


---

## Example Use Case

- Identify high-risk customers  
- Target them with retention offers  
- Reduce churn-related revenue loss  

---

## Key Takeaways

- Proper preprocessing and feature engineering significantly improve model performance  
- Threshold selection should be aligned with business goals, not just metrics  
- Prediction alone is not enough — it must connect to action  

---

## Future Improvements

- add uplift modeling for targeted interventions  
- include customer segmentation strategies  
- integrate API-based deployment  
- refine business assumptions using real cost data  

---

## Author

This project demonstrates a structured approach to building and deploying a machine learnin