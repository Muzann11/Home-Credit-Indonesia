---

# 🏦 Credit Default Risk Prediction using Logistic Regression & XGBoost

This project tackles the **Home Credit Default Risk** classification problem using real-world data from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk). The objective is to build robust machine learning models that can predict a client's probability of defaulting on a loan.

---

## 🎯 Objectives

* Handle missing data and outliers effectively
* Engineer new features from time-related fields
* Apply binning and **Weight of Evidence (WoE)** for feature selection
* Balance imbalanced target classes with **SMOTE** and **undersampling**
* Train and evaluate:

  * **Logistic Regression** (with and without resampling)
  * **XGBoost Classifier**
* Implement and test the final model on new (unlabeled) test data

---

## 📁 Dataset

* `application_train.csv`
* `application_test.csv`
* Plus supporting tables like `bureau.csv`, `credit_card_balance.csv`, etc.

From Kaggle competition: **Home Credit Default Risk**

---

## ⚙️ Workflow Overview

1. **Data Cleaning**

   * Remove high-missing-value columns
   * Treat missing numerical with median, categorical with mode
   * Handle outliers and rare categories
2. **Feature Engineering**

   * Create features like `AGE`, `YEAR_EMPLOYED`, `TOTAL_CREDIT_INQUIRIES`
   * Apply **binning** on numeric columns
   * Replace raw columns with meaningful binned versions
3. **Feature Selection**

   * Calculate **Weight of Evidence (WoE)**
   * Select features with acceptable **Information Value (IV)**
4. **Preprocessing for Modeling**

   * One-hot encode categorical bins
   * Concatenate with binary features
   * Finalize dataset for modeling
5. **Modeling**

   * Train/test split (stratified)
   * Try multiple versions of **Logistic Regression**:

     * Imbalanced baseline
     * SMOTE
     * Random undersampling (chosen)
   * Tune **XGBoost** via RandomizedSearchCV
6. **Evaluation**

   * Metrics: Confusion Matrix, Classification Report, ROC AUC
   * ROC Curve and Feature Importance Plots
7. **Implementation**

   * Predict new application data using best model
   * Output results to CSV

---

## 🧪 Model Performance

| Model Variant               | AUC Score  | Notes                   |
| --------------------------- | ---------- | ----------------------- |
| Logistic Regression (imbal) | 0.7335     | Imbalanced base model   |
| Logistic Regression (SMOTE) | 0.7334     | Minor difference        |
| Logistic Regression (RUS)   | ⭐ **Best** | Best recall on class 1  |
| XGBoost                     | High AUC   | Strong but more complex |

---

## 🗃️ Key Files

* `building_model.py`: Full end-to-end pipeline from raw data to final prediction
* `final_data.csv`: Cleaned and engineered training data
* `result.csv`: Output prediction result for application test data

---

## 📊 Feature Importance (Top 5)

```
1. EXT_SOURCE_3
2. EXT_SOURCE_2
3. REGION_POPULATION_RELATIVE
4. AMT_INCOME_TOTAL
5. AGE
```

---

## 📦 Libraries Used

* `pandas`, `numpy`, `seaborn`, `matplotlib`
* `scikit-learn`
* `xgboost`
* `imbalanced-learn`

---
