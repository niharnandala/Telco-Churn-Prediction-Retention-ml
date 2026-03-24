# 🚀 End-to-End Customer Churn Prediction & Retention Decision System

<p align="center">
  <b>Built for real-world decision making — not just predictions</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
  <img src="https://img.shields.io/badge/ML-ScikitLearn-orange">
  <img src="https://img.shields.io/badge/Deployment-Streamlit-red">
  <img src="https://img.shields.io/badge/Status-Production--Ready-success">
</p>

---

## 📌 Problem

Most churn models answer:

> “Will this customer churn?”

That’s useless in isolation.

Businesses actually need:

> **“Should we act on this customer?”**

---

## 💡 Solution

This project turns churn prediction into a **decision system**.

Instead of:
```
Churn = 0.77
```

You get:
```
High Risk → Target for retention campaign
```

---

## 🔥 Key Features

- ✅ End-to-end ML pipeline  
- ✅ Threshold-based decision system  
- ✅ Business impact simulation  
- ✅ Real-time Streamlit app  
- ✅ Interpretable model (Logistic Regression)  
- ✅ Scenario analysis (pessimistic / expected / optimistic)  

---

## 🧠 How It Works

### Step 1 — Prediction
- Model outputs churn probability  

### Step 2 — Decision Logic
- Apply configurable threshold  
- Convert probability → risk category  

### Step 3 — Action Layer
- Suggest business action  
- Estimate ROI impact  

---

## ⚙️ Tech Stack

| Category        | Tools Used |
|----------------|-----------|
| Language       | Python |
| ML             | Scikit-learn, XGBoost |
| Data           | Pandas, NumPy |
| Visualization  | Matplotlib, Seaborn |
| Deployment     | Streamlit |

---

## 📊 Model Strategy

| Model               | Purpose                |
|--------------------|------------------------|
| Logistic Regression | Final production model |
| XGBoost            | Benchmark comparison   |

### Why Logistic Regression?

- More stable  
- Fully interpretable  
- Business-friendly explanations  
- Comparable performance  

👉 Chosen deliberately, not by default.

---

## 🎯 Decision System

### 🔹 Threshold Optimization
- Multiple thresholds evaluated  
- Trade-offs analyzed  
- User-adjustable in app  

### 🔹 Risk Segmentation
- 🟢 Low Risk  
- 🟡 Moderate Risk  
- 🔴 High Risk  

---

## 💼 Business Layer

This is where most projects fail — this one doesn’t.

The system simulates:

- 🎯 Who to target  
- 💰 Cost of targeting  
- 📈 Expected retention value  
- 📊 Net business impact  

### Scenario Analysis
- Worst case  
- Expected  
- Best case  

---

## 🖥️ Live App Features

- Customer input form  
- Real-time predictions  
- Threshold slider  
- Risk classification  
- Suggested actions  
- Feature importance insights  

---

## 📁 Project Structure

```
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
├── models/
├── reports/
├── data/
├── data_processed/
├── requirements.txt
└── README.md
```

---

## ⚡ Installation

```bash
git clone https://github.com/your-username/telco-churn-decision-system.git
cd telco-churn-decision-system
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python -m src.run_pipeline
streamlit run app/streamlit_app.py
```

---

## 📸 Demo (Add Screenshot Here)

> Replace this with your actual app screenshot

```
![App Screenshot](assets/demo.png)
```

---

## 🧩 Key Design Decisions

- ❌ No "Churn Reason" → prevents data leakage  
- 🎯 Single model in UI → avoids confusion  
- 🔁 No confidence score → replaced with decision logic  

---

## 📌 What This Proves

- You can build **production-ready ML systems**  
- You understand **business impact, not just models**  
- You can design **decision-driven AI systems**  

---

## 🧠 Final Thought

> This is not about building the most complex model.  
> It’s about building the **right system**.

---

## 👤 Author

**Nihar Nandala**

> Building ML systems that connect predictions → decisions