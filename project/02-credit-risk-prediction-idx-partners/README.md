
# Credit Risk Prediction  
### Loan Default Classification — ID/X Partners Case Study

## 📌 Project Overview
This project builds an **end-to-end machine learning pipeline** to predict **loan default risk** using historical lending data.

The case study is inspired by **ID/X Partners**, focusing on how machine learning can support **credit approval decisions** by balancing risk, recall, and business impact.

---

## 🎯 Business Problem
Financial institutions face two major risks:
- ❌ Approving high-risk borrowers (default risk)
- ❌ Rejecting low-risk customers (lost revenue)

This project aims to:
- Predict loan default probability
- Compare **Logistic Regression vs Random Forest**
- Optimize decision threshold for **business-aligned performance**

---

## 📊 Dataset
- Source: Public Lending Club dataset (2007–2014)
- Records: ~466,000 loans
- Target:
  - `1` → Bad Loan (Default / Charged Off / Late)
  - `0` → Good Loan (Fully Paid / Current)

> ⚠️ This project is for **educational and portfolio purposes only**.

---

## ⚙️ Methodology

### 1️⃣ Data Preparation
- Removed data leakage features (post-loan information)
- Handled missing values (>50% dropped)
- Safe feature parsing (interest rate, employment length, term)

### 2️⃣ Feature Engineering
- Numerical scaling (`StandardScaler`)
- Ordinal encoding (`grade`, `sub_grade`)
- One-hot encoding for low-cardinality categorical features

### 3️⃣ Modeling
| Model | Description |
|------|------------|
| Logistic Regression | Interpretable baseline model |
| Random Forest | Non-linear ensemble model |

Both models are wrapped in **production-grade sklearn Pipelines**.

---

## 📈 Model Performance

### ROC-AUC Comparison
| Model | ROC-AUC |
|------|--------|
| Logistic Regression | **0.917** |
| Random Forest | **0.957** |

![ROC Curve](images/roc_curve.png)

---

## 🎯 Threshold Optimization
Instead of using the default threshold (0.5), the model was optimized for **F1-score**, balancing precision and recall.

**Optimal Threshold:** `0.52`

### Final Classification Report (Random Forest)
