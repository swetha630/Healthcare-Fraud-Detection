# 🏥 Healthcare Insurance Claim Fraud Detection

An end-to-end Machine Learning project that detects potentially fraudulent healthcare insurance claims using an explainable XGBoost model. The system provides an interactive Streamlit dashboard for investigators to analyze claims, predict fraud risk, and monitor model performance.

---

## 📌 Project Overview

Healthcare insurance fraud results in significant financial losses every year. Detecting fraudulent claims early helps insurers reduce unnecessary payouts and improve operational efficiency.

This project develops an intelligent fraud detection system using machine learning techniques to classify healthcare claims as **Fraudulent** or **Genuine**.

The final solution includes:

- Data preprocessing and feature engineering
- Class imbalance handling using SMOTE
- XGBoost-based fraud detection
- Model evaluation using multiple metrics
- SHAP explainability for model interpretation
- Interactive Streamlit dashboard deployment

---

## 🎯 Objectives

- Detect fraudulent healthcare insurance claims.
- Minimize false investigations while maximizing fraud detection.
- Handle highly imbalanced healthcare claim data.
- Provide interpretable predictions for investigators.
- Build an interactive dashboard for real-time claim investigation.

---

## 📂 Project Structure

```
Healthcare-Fraud-Detection/
│
├── app.py
├── requirements.txt
├── readme.md
│
├── models/
│   ├── xgb_realistic.pkl
│   └── encoders.pkl
│
├── dataset/
│   └── healthcare_claims.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
└── assets/
```

---

## 📊 Dataset Information

The dataset contains **10,000 healthcare insurance claims** with the following features:

| Feature | Description |
|-----------|-------------|
| Provider_ID | Provider identifier |
| Claim_ID | Claim identifier |
| Patient_Age | Age of patient |
| Patient_Gender | Gender of patient |
| Diagnosis_Code | Diagnosis code |
| Procedure_Code | Medical procedure code |
| Claim_Amount | Total claim amount |
| Approved_Amount | Approved reimbursement amount |
| Insurance_Type | Type of insurance |
| Claim_Submission_Date | Claim submission date |
| Days_Between_Service_and_Claim | Delay between service and claim |
| Number_of_Claims_Per_Provider_Monthly | Provider activity level |
| Provider_Specialty | Medical specialty |
| Patient_State | State of patient |
| Claim_Status | Claim processing status |
| Is_Fraud | Target variable |
| Length_of_Stay | Hospital stay duration |
| Visit_Type | Emergency/Inpatient/Outpatient |
| Chronic_Condition_Flag | Presence of chronic condition |
| Prior_Visits_12m | Previous visits in 12 months |

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

### Missing Value Handling

Missing values were observed in:

- Insurance_Type
- Provider_Specialty
- Prior_Visits_12m

These were handled appropriately before training.

---

### Feature Engineering

Date features were extracted from Claim Submission Date:

- Claim_Year
- Claim_Month
- Claim_Day
- Claim_DayOfWeek

---

### Categorical Encoding

Label Encoding was used for:

- Patient_Gender
- Diagnosis_Code
- Insurance_Type
- Provider_Specialty
- Patient_State
- Visit_Type

The encoders were saved for deployment.

---

### Class Imbalance Handling

Original class distribution:

| Class | Count |
|---------|-------|
| Genuine Claims | 9,171 |
| Fraud Claims | 829 |

Fraud Rate:

```
8.29%
```

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the training data.

---

## 🤖 Machine Learning Models

### 1. Random Forest

Used as a baseline model.

Performance:

- ROC-AUC: 98.49%
- Recall: 89%
- F1 Score: 78%

---

### 2. XGBoost (Final Model)

The final model used:

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
```

---

## 📈 Final Model Performance

### XGBoost (Realistic Model)

| Metric | Score |
|----------|--------|
| Precision | 91% |
| Recall | 80% |
| F1 Score | 85% |
| ROC-AUC | 98.91% |

Confusion Matrix:

```
[[1821   13]
 [  34  132]]
```

---

## 🔍 Explainability Using SHAP

SHAP (SHapley Additive exPlanations) was used to understand model predictions.

Key findings:

### Most Influential Features

- Claim_Amount
- Approved_Amount
- Prior_Visits_12m
- Visit_Type
- Number_of_Claims_Per_Provider_Monthly
- Insurance_Type

SHAP helped explain why a claim was flagged as fraudulent, improving trust and transparency.

---

## 🚨 Leakage Detection

During experimentation, two features showed potential data leakage:

### Claim_Status

Fraudulent claims had a significantly higher proportion of rejected claims.

### Days_Between_Service_and_Claim

This feature became highly dominant and unrealistically boosted performance.

To improve real-world applicability:

- Claim_Status was excluded.
- Days_Between_Service_and_Claim was removed from the final realistic model.

---

## 🖥️ Streamlit Dashboard

The deployed dashboard contains three sections:

### Executive Dashboard

Displays:

- Total Claims
- Fraud Claims
- Fraud Rate
- ROC-AUC

---

### Fraud Prediction

Allows investigators to:

- Enter claim details
- Predict fraud probability
- Assess risk levels

Risk Categories:

- Low Risk
- Medium Risk
- High Risk

---

### Model Performance

Displays:

- Precision
- Recall
- F1 Score
- ROC-AUC

---


## 📦 Requirements

Major libraries used:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Streamlit
- Matplotlib
- Joblib

---

## 💼 Business Impact

This system can help healthcare organizations:

- Reduce financial losses caused by fraud.
- Prioritize suspicious claims for investigation.
- Improve claim processing efficiency.
- Support investigators with explainable AI insights.
- Enhance trust in automated fraud detection systems.

---
