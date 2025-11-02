# Home Credit Default Risk Prediction

A machine learning project to predict credit default risk using Logistic Regression and LightGBM.

## 📋 Problem Statement

Home Credit serves customers without traditional credit history. The challenge:
- **Good customers** are often rejected due to lack of credit history (false negatives)
- **Bad customers** sometimes get approved and default (false positives)

**Objective:** Build an ML model to predict default probability and optimize lending decisions.

## 📊 Dataset

- **Training Data:** 307,511 applications, 122 features
- **Target:** Binary (0 = Good customer, 1 = Default)
- **Class Imbalance:** 92% Good vs 8% Default (11.5:1 ratio)
- **Supporting Tables:** Bureau, Previous Applications, Payment History, POS Cash Balance, Credit Card Balance, Installments

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)
- Target distribution analysis (8% default rate)
- Missing values handling (dropped features with >60% missing)
- Correlation analysis with target variable
- Key insights visualization (age, income, credit amount vs default)

**Key Findings:**
- External source scores (EXT_SOURCE_2, EXT_SOURCE_3) are strongest predictors
- Younger customers have higher default risk
- Credit-to-income ratio is a critical indicator
- Previous credit history strongly influences default probability

### 2. Feature Engineering
Created **150+ features** from multiple data sources:

**Basic Features:**
- Age in years (from DAYS_BIRTH)
- Employment length (from DAYS_EMPLOYED)
- Financial ratios: Credit/Income, Annuity/Income, Credit term
- External source aggregations: mean, max, min, std

**Bureau Features (Credit History):**
- Credit day overdue statistics
- Active credit ratio
- Total credit sum and debt
- Credit type count

**Previous Applications:**
- Approval rate from past applications
- Average credit amounts and annuities
- Decision timeline patterns

**Payment Behavior:**
- Installment payment ratios
- Payment delays and patterns

### 3. Data Preprocessing
- Dropped columns with >60% missing values
- Label encoding for categorical variables
- Median imputation for remaining missing values
- Train/validation split (80/20) with stratification

### 4. Model Development

**Models Trained:**

| Model | ROC-AUC | Description |
|-------|---------|-------------|
| **Logistic Regression** | 0.72 | Baseline model with class balancing |
| **LightGBM** | **0.78** | Primary model with early stopping |

**LightGBM Configuration:**
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'is_unbalance': True
}
```

**Training Details:**
- Early stopping: 50 rounds
- Best iteration: ~800 rounds
- Cross-validation: Stratified split

## 📈 Results Summary

### Model Performance

| Metric | Baseline (LogReg) | LightGBM | Improvement |
|--------|-------------------|----------|-------------|
| **ROC-AUC** | 0.72 | **0.78** | +8.3% |
| **Precision** | 0.35 | **0.42** | +20% |
| **Recall** | 0.68 | **0.70** | +2.9% |

### Business Impact (Threshold = 0.65)

| Metric | Value | Baseline |
|--------|-------|----------|
| **Expected Profit** | **$72.6M** per 60k apps | -$10M (no model) |
| **Approval Rate** | **85.2%** | 100% or manual |
| **Default Rate** | **5.1%** | 8.0% |
| **Improvement** | **36% reduction** in defaults | - |

### Threshold Optimization Analysis

Optimal threshold: **0.65**
- Maximizes profit while maintaining high approval rate
- Balance between risk (5.1% default) and opportunity (85% approval)
- Tested thresholds from 0.20 to 0.65 in 0.05 increments

## 🎯 Business Recommendations

### 1. Model Implementation
- **Deploy:** LightGBM with threshold 0.65
- **Expected ROI:** $72.6M per 60,000 applications batch
- **Approval Rate:** Maintain 85.2% for customer satisfaction
- **Default Rate:** Target <6% with continuous monitoring

### 2. Customer Segmentation (3-Tier Risk-Based)

**🔴 High Risk (Probability > 0.65)**
- **Action:** REJECT or require 30-40% down payment + higher interest
- **Terms:** Maximum 12-month maturity, co-signer required
- **Rationale:** Above optimal threshold, profit declines

**🟡 Medium Risk (0.40 - 0.65)**
- **Action:** MANUAL REVIEW by senior loan officers
- **Terms:** Standard interest, 12-24 month maturity
- **Requirements:** Additional documentation (employment, income proof)

**🟢 Low Risk (< 0.40)**
- **Action:** AUTO-APPROVE with favorable terms
- **Terms:** Lower interest rates, 24-36 month maturity
- **Benefits:** Fast-track processing (24-48 hours)

### 3. Key Risk Factors to Validate
Based on feature importance analysis:
- ✅ External credit scores (EXT_SOURCE_2, EXT_SOURCE_3) - **MANDATORY**
- ✅ Credit-to-income ratio (must be <50%)
- ✅ Previous credit history verification
- ✅ Age and employment stability (min 6 months)
- ✅ Bureau active credit ratio

### 4. Monitoring & Continuous Improvement

**Weekly Monitoring:**
- Default rate (alert if >6%)
- Approval rate (alert if <80%)

**Monthly Analysis:**
- Profit per batch tracking
- False positive/negative rates by segment
- Customer satisfaction metrics

**Quarterly Actions:**
- Model retraining with latest 3-month data
- Feature importance review
- A/B testing of thresholds by region

### 5. Implementation Timeline
- **Month 1:** System integration & UAT testing
- **Month 2:** Pilot program (1,000 applications)
- **Month 3:** Gradual rollout (10% → 50% → 100%)
- **Month 4+:** Full production with monitoring

## 🛠️ Tech Stack

**Programming & Tools:**
- Python 3.12
- Google Colab
- Git & GitHub

**Libraries:**
- **Data Processing:** Pandas 1.5+, NumPy 1.23+
- **Visualization:** Matplotlib 3.6+, Seaborn 0.12+
- **Machine Learning:** Scikit-learn 1.2+, LightGBM 3.3+
- **Metrics:** ROC-AUC, Classification Report, Confusion Matrix


## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Steps
1. **Download dataset** from Kaggle (see Dataset section)
2. **Upload to Google Colab** or local Jupyter
3. **Update data paths** in notebook (DATA_DIR variable)
4. **Run all cells** sequentially
5. **Check outputs:** Visualizations, model results, submission.csv

### Expected Runtime
- Data loading: ~2 minutes
- Feature engineering: ~3 minutes
- Model training: ~5 minutes
- **Total:** ~10-15 minutes on Colab (free tier)

## 📊 Key Features

### Top 20 Most Important Features (by LightGBM)

1. **EXT_SOURCE_2** (25%) - External credit score
2. **EXT_SOURCE_3** (18%) - Alternative credit rating
3. **CREDIT_INCOME_RATIO** (12%) - Financial burden indicator
4. **AGE_YEARS** (8%) - Customer maturity
5. **BUREAU_ACTIVE_RATIO** (6%) - Credit activity level
6. EXT_SOURCE_MEAN (5%)
7. DAYS_BIRTH (4%)
8. EMPLOYMENT_YEARS (3%)
9. AMT_CREDIT (3%)
10. BUREAU_DAYS_CREDIT_MEAN (2%)
... (and 140+ more engineered features)

### Feature Engineering Highlights
- Created domain-specific ratios (credit/income, annuity/income)
- Aggregated credit bureau history (6 statistical measures)
- Extracted temporal features (age, employment duration)
- Combined external sources (mean, product, std)

## 📝 Insights & Learnings

### Key Takeaways
1. **Feature engineering is crucial:** From 122 → 150+ features boosted AUC by 3%
2. **Imbalanced data handling:** class_weight='balanced' and threshold tuning essential
3. **Business context matters:** ROC-AUC alone doesn't tell the full story - need profit analysis
4. **Tree models excel:** LightGBM outperformed linear models for complex interactions
5. **External validation:** External credit scores are the strongest single predictors

### Challenges Faced
- **Class imbalance:** 92:8 ratio required careful handling
- **Missing data:** 40%+ missing in some features
- **Feature alignment:** Ensuring train/test consistency after engineering
- **Business translation:** Converting ML metrics to dollar impact

### Future Improvements
- [ ] Hyperparameter tuning (GridSearch/Optuna)
- [ ] Try XGBoost and CatBoost for comparison
- [ ] Implement SMOTE for synthetic oversampling
- [ ] Add SHAP values for model explainability
- [ ] Explore deep learning approaches
- [ ] Incorporate more alternative data sources

## 🎓 Skills Demonstrated

- ✅ **Binary Classification** with imbalanced data
- ✅ **Feature Engineering** from multiple relational tables
- ✅ **Model Comparison** (Linear vs Tree-based)
- ✅ **Threshold Optimization** for business objectives
- ✅ **Business Impact Analysis** (profit calculation)
- ✅ **Data Visualization** and storytelling
- ✅ **End-to-end ML Pipeline** development

## 📚 References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Handling Imbalanced Data](https://imbalanced-learn.org/)

---

⭐ **If you find this project helpful, please consider giving it a star!**

*Last updated: November 2025*
