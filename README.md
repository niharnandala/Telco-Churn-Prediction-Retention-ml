# 🚀 End-to-End Customer Churn Prediction & Retention Decision System

**Built by Nihar Nandala**

---

## 🌐 Live Application

👉 https://end-to-end-customer-churn-prediction-retention-decision-system.streamlit.app/

---

## 📌 What this project is

Most churn projects stop at prediction.  
This one doesn’t.

This system answers:

> “Given this customer, should the business act or not?”

It combines:
- Machine learning  
- Decision-making logic  
- Business context  

To create a practical, deployable system.

---

## ⭐ Why this project stands out

- Not just a model → full pipeline + deployment  
- Not just accuracy → decision + business trade-offs  
- Not just code → clean product experience  

Built like something usable by a business team.

---

## 🧠 Core Idea

Instead of just predicting churn, the system:

1. Predicts churn probability  
2. Applies a configurable threshold  
3. Assigns a decision-aware risk level  
4. Suggests a next step  

So instead of:

```text
Churn = 0.77
```

You get:

```text
High Risk → prioritize for retention review
```

---

## ⚙️ System Overview

### Pipeline
- Data cleaning & preprocessing  
- Feature engineering  
- Model training  
- Evaluation & reporting  
- Deployment via Streamlit  

---

### Models Used

| Model               | Role                  |
|--------------------|-----------------------|
| Logistic Regression | Final deployed model |
| XGBoost            | Benchmark comparison |

---

### Why Logistic Regression is used in production

Even though XGBoost was tested:

- Logistic Regression performed competitively  
- More stable predictions  
- Fully interpretable  
- Easier to explain in a business setting  

This decision was intentional.

---

## 🎯 Decision System (Not Just Prediction)

### Threshold-based logic
- Not fixed at 0.5  
- Multiple thresholds evaluated  
- Trade-offs analyzed  
- Adjustable in the app  

---

### Risk segmentation (dynamic)

- Low Risk  
- Moderate Risk  
- High Risk  

Risk depends on the selected threshold.

---

## 💼 Business Layer

This system simulates:

- Who would be targeted in a retention campaign  
- Cost of targeting  
- Expected retained value  
- Net business impact  

Includes scenario analysis:
- Pessimistic  
- Expected  
- Optimistic  

This makes the model usable for decision-making.

---

## 🖥️ Streamlit App

### What you see
- Customer input form  
- Real-time churn prediction  
- Adjustable threshold slider  
- Risk classification  
- Suggested next action  

### What you can explore
- Why the model predicted churn  
- Feature contributions  
- Threshold trade-offs  
- Model comparison (hidden from main UI)

---

## 📸 Demo

![App Screenshot](screenshots/Screenshot%20(86).png)

---

## 🎥 App Walkthrough

[▶️ Watch Demo Video](App_videos/App%20Recording.webm)

---

## 📁 Project Structure

```text
telco-churn-decision-system/
├── app/                  # Streamlit UI
├── src/                  # Core pipeline
│   ├── cleaning.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── predict.py
│   └── run_pipeline.py
├── models/               # Saved model
├── reports/              # Evaluation outputs
├── data/                 # Raw dataset
├── data_processed/       # Cleaned data
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Project

```bash
pip install -r requirements.txt
python -m src.run_pipeline
streamlit run app/streamlit_app.py
```

---

## ⚠️ Important Design Decisions

- Churn Reason NOT used → avoids data leakage  
- Only one model shown in UI → avoids confusion  
- Confidence removed → replaced with clearer risk logic  

---

## 📌 What this project demonstrates

- End-to-end ML system design  
- Clean modular code structure  
- Model comparison and selection  
- Threshold-based decision making  
- Business-aware evaluation  
- Deployment with usable UI  

---

## 🧠 Final Note

This project is not about building the most complex model.

It is about building the right system:

> One that connects predictions to decisions.

---

## 👤 Author

**Nihar Nandala**

Focused on building practical ML systems that bridge the gap between models and real-world decisions.