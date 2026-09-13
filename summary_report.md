# 📊 Telco Customer Churn — Summary Report

> **Supervised Learning | Customer Retention & Churn Prediction**

---

## 🎯 1. Business Problem & Dataset

Customer churn is an important business problem for telecom companies because losing existing customers can reduce recurring revenue and increase the cost of acquiring new customers.

The objective of this project is to build a **supervised-learning classification system** that predicts whether a customer is likely to churn.

### 📁 Dataset Overview

| 📌 Item | Details |
|---|---|
| Dataset | Telco Customer Churn |
| 👥 Total Customers | **7,043** |
| 📊 Total Columns | **21** |
| 🎯 Target | `Churn` |
| 🚫 No Churn | **5,174** |
| ⚠️ Churn | **1,869** |
| 🔀 Train/Test Split | **80% / 20%** |
| 🎲 Random State | **42** |
| 🧩 Stratification | **Yes** |

Important features include `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `OnlineSecurity`, and `Churn`.

---

## 🧹 2. Preprocessing & Class Imbalance

The preprocessing workflow:

- 🗑️ Removes `customerID`
- 🔢 Converts `TotalCharges` into numeric format
- 🧹 Handles missing values using the median
- 🛠️ Creates `tenure_group`
- 🛠️ Creates `num_services`
- 💳 Creates the `AutoPay` feature
- 🔄 Converts selected Yes/No fields into 1/0
- 🏷️ Applies one-hot encoding to categorical variables
- 📐 Applies standard scaling to selected numerical features
- ⚖️ Uses **SMOTE only on training data** to reduce class-imbalance effects

### ⚖️ Class Balance Results

| Strategy | Recall |
|---|---:|
| SMOTE | **0.8850** |
| Balanced Decision Tree | **0.7968** |

**SMOTE was selected for the recall-focused workflow.**

---

## 🤖 3. Models Evaluated

The project evaluates four supervised-learning algorithms:

- 🔵 **KNN**
- 🟢 **Naive Bayes**
- 🟠 **SVM**
- 🌳 **Decision Tree**

Hyperparameter tuning was also performed for KNN, SVM, and Decision Tree.

---

## 🏆 4. Recommended Model

### 🌳 Decision Tree

The final model comparison reports:

| 📈 Metric | Score |
|---|---:|
| Accuracy | **0.4372** |
| Precision | **0.3062** |
| Recall | **0.8850** |
| F1-Score | **0.4550** |
| AUC-ROC | **0.5390** |

### ⭐ Why Decision Tree?

The project focuses strongly on **Recall**, because missing a real churn customer can reduce the effectiveness of a retention campaign.

The Decision Tree achieved the highest reported final **Recall = 0.8850**, so it is recommended for this recall-focused workflow.

> ⚠️ The tuning results and final model-comparison results are separate notebook stages and should not be treated as the same evaluation experiment.

---

## 🔍 5. Top Churn Signals

The Decision Tree feature-importance analysis identified the following top five signals:

| 🥇 Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Contract_Two year` | **0.3430** |
| 2 | `Contract_One year` | **0.2456** |
| 3 | `tenure` | **0.0987** |
| 4 | `InternetService_Fiber optic` | **0.0790** |
| 5 | `MonthlyCharges` | **0.0749** |

### 💼 Recommended Retention Actions

The retention team should:

1. 📋 Monitor customers according to contract type.
2. ⏳ Track customers at different tenure stages.
3. 🌐 Review the experience of fiber-optic customers.
4. 💰 Identify customers with higher monthly charges.
5. 🚨 Combine these signals into a customer-level churn-risk score.

---

## 🔬 6. Error Analysis

The final model analysis reports:

- 🎯 **Recall:** 0.8850
- 🎯 **Precision:** 0.3062
- 🎯 **F1-Score:** 0.4550
- 📈 **AUC-ROC:** 0.5390
- ❌ **False Negatives:** 43

The missed churners had an average:

- ⏳ **Tenure:** 25.33 months
- 💰 **Monthly Charges:** 67.87

Most of these missed churners were on **one-year contracts**.

This indicates that retention monitoring should not focus only on very new or high-charge customers; longer-tenure customers can also require attention.

---

## 🚀 7. Deployment & Next Steps

The project creates a saved model pipeline:

```text
churn_model.pkl
```

The pipeline can be used to generate churn probabilities for new customers.

### 🔮 Future Improvements

- 📅 Collect more recent customer data
- 🎧 Add customer-support interaction features
- 💳 Add billing and payment-history trends
- 📡 Add service-usage behaviour
- 🤝 Test stronger ensemble models
- 🔁 Apply consistent cross-validation
- 🎯 Calibrate churn probabilities
- ⚡ Build a real-time scoring API
- 📊 Connect predictions to a retention dashboard

---

## ✅ Final Conclusion

This project demonstrates a complete supervised-learning workflow for **telecom customer churn prediction**, from data preprocessing and class balancing to model comparison, error analysis, business interpretation, and pipeline deployment.

**Recommended Model: 🌳 Decision Tree**

**Primary reason: Highest reported final Recall — 0.8850**

The model can support a retention team by identifying customers who are more likely to churn and enabling proactive customer-retention actions.
