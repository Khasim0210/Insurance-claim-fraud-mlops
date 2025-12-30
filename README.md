# 🛡️ Insurance Claim Fraud Detection (MLOps)

[![Live App](https://img.shields.io/badge/Live-App-brightgreen)](https://insurance-claim-fraud-mlops.onrender.com)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Render](https://img.shields.io/badge/Deployed%20on-Render-purple)

An **end-to-end Machine Learning & MLOps project** for detecting fraudulent insurance claims.  
This project covers **data engineering, model training, experiment tracking, threshold tuning, and cloud deployment** with a live interactive UI.

---

## 🚀 Live Demo
👉 **https://insurance-claim-fraud-mlops.onrender.com**

---

## 📌 Project Overview

Insurance fraud causes significant financial losses.  
This project builds a **production-ready ML system** that predicts whether an insurance claim is **fraudulent (Y/N)** using structured claim and policy data.

### Key highlights:
- Fully normalized **SQL (3NF) data model**
- Multiple ML experiments tracked using **MLflow**
- **Threshold tuning** for business-aligned decision making
- **Streamlit web app** for real-time predictions
- Deployed on **Render Cloud**

---

## 🧠 Machine Learning Pipeline

1. **Data Ingestion**
   - Raw CSV ingested into a normalized SQLite database (3NF)

2. **Feature Engineering**
   - SQL joins → Pandas dataframe
   - Categorical encoding + numeric scaling

3. **Model Training**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Histogram Gradient Boosting

4. **Experiment Tracking**
   - MLflow + DagsHub
   - Metrics: F1-score (macro), ROC-AUC

5. **Threshold Optimization**
   - Business-driven fraud threshold tuning

6. **Final Model**
   - Best model + threshold saved as a reusable artifact

---

## 🧪 Experiments & Tracking

- All experiments tracked in **MLflow**
- Parameters, metrics, and artifacts logged
- Final model selected based on **macro F1 + recall tradeoff**

📊 **MLflow UI (via DagsHub)**  
https://dagshub.com/Khasim0210/insurance-claim-fraud-mlops.mlflow

---

## 🌐 Web Application (Streamlit)

The app supports two modes:

### 🔹 Manual Input
- Enter a single insurance claim
- Get fraud probability + prediction

### 🔹 CSV Upload
- Upload a CSV with matching features
- Batch predictions
- Download results as CSV

---

## 📂 Repository Structure

