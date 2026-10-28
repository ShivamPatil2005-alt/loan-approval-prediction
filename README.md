# 🏦 Loan Approval Prediction

A complete end-to-end Machine Learning classification project to evaluate loan applicant profiles and predict approval status based on credit history, financial metrics, and demographic indicators.

---

## 📌 Project Overview
Financial institutions need reliable, data-driven systems to assess creditworthiness while minimizing default risk. This project performs:
- **Data Cleaning & Preprocessing:** Null value auditing, missing value imputation, and nominal/ordinal encoding.
- **Exploratory Data Analysis (EDA):** Univariate distributions, class balance analysis, and risk factor correlations.
- **Machine Learning Modeling:** Baseline classification comparing Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes with standardized feature scaling.

---

## 📁 Dataset Details
- **Source:** Synthetic banking and credit assessment benchmark dataset (`loan_approval_data.csv`).
- **Size:** 1,000 records × 20 features
- **Key Features:**
  - **Demographics:** `Age`, `Gender`, `Marital_Status`, `Dependents`, `Education_Level`
  - **Financial Profile:** `Applicant_Income`, `Coapplicant_Income`, `Employment_Status`, `Employer_Category`, `Savings`
  - **Credit Risk:** `Credit_Score`, `DTI_Ratio` (Debt-to-Income), `Existing_Loans`, `Collateral_Value`
  - **Loan Terms:** `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, `Property_Area`
  - **Target Variable:** `Loan_Approved` (`No`: 70.2%, `Yes`: 29.8% after imputation)

---

## 🔍 Preprocessing & EDA Findings
1. **Target Distribution:** Evaluated class balance. Loan rejections (`No`) form ~70.2% of the dataset, while approvals (`Yes`) make up ~29.8%.
2. **Missing Value Handling:**
   - Numerical features imputed using mean strategy (`SimpleImputer(strategy='mean')`).
   - Categorical features imputed using mode strategy (`SimpleImputer(strategy='most_frequent')`).
3. **Leakage & Feature Selection:** Dropped `Applicant_ID` prior to modeling to avoid arbitrary index-based feature bias.
4. **Encoding:**
   - Binary/Ordinal: `LabelEncoder` for `Education_Level` and `Loan_Approved`.
   - Multi-class Nominal: `OneHotEncoder(drop='first')` for `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, and `Employer_Category`.
5. **Feature Scaling:** Applied `StandardScaler` to prevent distance-sensitive algorithms (KNN, Logistic Regression) from being dominated by high-magnitude currency fields.

---

## 📈 Model Performance & Comparison

Evaluated on an 80/20 stratified train-test split:

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.5%** | **64.8%** | **76.7%** | **70.2%** | **0.888** |
| **Naive Bayes (Gaussian)** | 80.0% | 66.7% | 66.7% | 66.7% | **0.892** |
| **K-Nearest Neighbors (k=5)** | 74.5% | 57.6% | 56.7% | 57.1% | 0.755 |

---

## 🛠️ Tech Stack
- **Language:** Python 3
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`

---

## 🚀 Getting Started Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ShivamPatil2005-alt/loan-approval-prediction.git](https://github.com/ShivamPatil2005-alt/loan-approval-prediction.git)
   cd loan-approval-prediction
