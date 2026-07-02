# 🏥 Healthcare Insurance Claim Fraud Detection

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Fraud%20Detection-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)

</p>

<p align="center">

An end-to-end Explainable AI system that detects fraudulent healthcare insurance claims using Machine Learning, XGBoost, SHAP, and an interactive Streamlit dashboard.

</p>

---

# 📖 Overview

Healthcare insurance fraud leads to billions of dollars in financial losses every year, making early fraud detection essential for insurers.

This project presents an **end-to-end fraud detection pipeline** that analyzes healthcare insurance claims, predicts fraudulent activity using machine learning, and explains every prediction using **SHAP Explainable AI**.

Unlike traditional black-box models, the system not only predicts fraud but also provides interpretable explanations that help investigators understand *why* a claim was flagged.

---

#  Problem Statement

Insurance providers process thousands of healthcare claims daily, making manual fraud investigation expensive and time-consuming.

Key challenges include:

- 🚨 Highly imbalanced fraud data
- 🔍 Difficulty identifying suspicious claims
- ⚖️ Need for explainable AI predictions
- ⏳ Long investigation cycles
- 💰 Financial losses due to fraudulent reimbursements

---

# 💡 Solution

The proposed solution combines Machine Learning and Explainable AI to automatically identify fraudulent healthcare claims.

The system:

- 📂 Processes healthcare claim records
- 🧹 Cleans and preprocesses data
- ⚖️ Balances classes using SMOTE
- 🤖 Detects fraud using XGBoost
- 📊 Explains predictions with SHAP
- 📈 Displays insights through a Streamlit dashboard

---

# ✨ Key Features

- 🏥 Healthcare Insurance Fraud Detection
- 🤖 XGBoost Classification Model
- ⚖️ SMOTE-based Class Balancing
- 📊 Fraud Risk Prediction
- 🧠 SHAP Explainability
- 📈 Interactive Streamlit Dashboard
- 📉 Performance Metrics Visualization
- ⚡ Real-time Claim Prediction

---

# 🏗️ System Architecture

```text
Healthcare Claims Dataset
          │
          ▼
Data Cleaning & Preprocessing
          │
          ▼
Feature Engineering
          │
          ▼
SMOTE Class Balancing
          │
          ▼
XGBoost Fraud Detection Model
          │
          ▼
SHAP Explainability Engine
          │
          ▼
Interactive Streamlit Dashboard
```

---

# 🛠 Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Model Serialization | Joblib |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
Healthcare-Fraud-Detection/

│── app.py
│── requirements.txt
│── models/
│── dataset/
│── notebooks/
│── assets/
│── README.md
```

---

# 📊 Dataset Overview

The dataset contains **10,000 healthcare insurance claims** with demographic, provider, diagnosis, procedure, claim amount, insurance, and medical history information.

Important attributes include:

- Claim Amount
- Approved Amount
- Insurance Type
- Diagnosis Code
- Provider Specialty
- Visit Type
- Patient State
- Prior Visits
- Length of Stay

Target Variable:

**Is_Fraud**

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Missing Value Handling
- Date Feature Extraction
- Label Encoding
- Feature Engineering
- Data Cleaning
- Class Balancing using SMOTE

Special attention was given to preventing **data leakage**, ensuring realistic model performance for deployment.

---

# 🤖 Machine Learning Pipeline

Models Evaluated

| Model | Purpose |
|---------|----------|
| Random Forest | Baseline Model |
| XGBoost | Final Production Model |

The XGBoost classifier was selected due to its superior performance and robustness on imbalanced healthcare data.

---

# 📈 Model Performance

| Metric | Score |
|----------|--------|
| Precision | **91%** |
| Recall | **80%** |
| F1 Score | **85%** |
| ROC-AUC | **98.91%** |

### Confusion Matrix

```text
[[1821   13]
 [  34  132]]
```

---

# 🧠 Explainable AI with SHAP

To improve transparency, SHAP was integrated into the prediction pipeline.

The model explains each prediction by highlighting the contribution of individual features.

Most influential features include:

- Claim Amount
- Approved Amount
- Prior Visits
- Visit Type
- Provider Activity
- Insurance Type

This enables investigators to understand why a claim was classified as fraudulent.

---

# ⚠️ Preventing Data Leakage

During experimentation, two features produced unrealistically high accuracy:

- Claim Status
- Days Between Service and Claim

Since these features would not always be available during real-time prediction, they were excluded from the final production model to improve generalization.

---

# 🖥️ Dashboard Features

### 📊 Executive Dashboard

- Total Claims
- Fraud Rate
- ROC-AUC
- Fraud Distribution

### 🚨 Fraud Prediction

- Claim Risk Prediction
- Fraud Probability
- Risk Categorization
- Real-time Inference

### 📈 Model Analytics

- Precision
- Recall
- F1 Score
- ROC-AUC

---

---

# 💼 Business Impact

The solution helps healthcare insurers by:

- 💰 Reducing financial losses
- 🚨 Prioritizing suspicious claims
- ⚡ Accelerating claim investigation
- 🧠 Supporting investigators with explainable AI
- 📈 Improving operational efficiency

---

# 🚀 Future Enhancements

- Deep Learning Models
- Graph Neural Networks for Fraud Detection
- Real-time API Deployment
- LLM-assisted Fraud Investigation
- Cloud Deployment (AWS/Azure)
- Continuous Model Retraining
- Fraud Network Visualization

---

# 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- End-to-End Machine Learning Pipelines
- Fraud Detection
- Explainable AI (SHAP)
- Handling Imbalanced Datasets
- Feature Engineering
- Data Leakage Prevention
- Streamlit Dashboard Development
- Model Deployment

---

# 👩‍💻 Author

## Swetha Mandapuri

**B.E. Artificial Intelligence & Machine Learning**  
Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad

📧 Connect with me

- GitHub: https://github.com/swetha630
- LinkedIn: https://www.linkedin.com/in/swetha-mandapuri-3346042a3

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

Your support motivates me to build more impactful AI-powered applications.
