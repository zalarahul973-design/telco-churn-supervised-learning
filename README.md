# 🚀 Telco Customer Churn Prediction

### Supervised Learning — Customer Churn Classification

<p align="center">
  <b>KNN</b> • <b>SVM</b> • <b>Decision Tree</b> • <b>SMOTE</b> •
  <b>Hyperparameter Tuning</b> • <b>Error Analysis</b> • <b>Pipeline</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Imbalanced--learn-SMOTE-orange">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
</p>

---

## 📌 Project Overview

**Telco Customer Churn Prediction** is a Machine Learning project that predicts whether a telecom customer is likely to **churn (leave the service)** or **stay**.

The project follows a complete supervised-learning workflow:

```text
Dataset → EDA → Data Cleaning → Feature Engineering
       → Encoding → Scaling → Train/Test Split
       → SMOTE → KNN / SVM / Decision Tree
       → Hyperparameter Tuning → Evaluation
       → Error Analysis → Final Pipeline
```

---

## 🎯 Project Objectives

- Understand the Telco customer churn dataset.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess customer data.
- Perform feature engineering.
- Encode categorical variables and scale numerical variables.
- Handle class imbalance using **SMOTE**.
- Build KNN, SVM and Decision Tree classifiers.
- Tune KNN `k`, SVM `C`, and Decision Tree depth.
- Compare Accuracy, Precision, Recall, F1 Score and ROC-AUC.
- Perform false-negative error analysis.
- Identify important churn-prediction features.
- Create and save a final prediction pipeline.

---

# 📊 Dataset

The uploaded dataset contains **7,043 customer records and 21 columns**.

### 🎯 Target Variable

`Churn`

- `Yes` → Customer churned
- `No` → Customer did not churn

### 🔍 Important Features

`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`,
`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
`Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

> `customerID` is treated as an identifier and is dropped before model training.

---

# 🧹 Data Cleaning & Feature Engineering

### `TotalCharges`
Converted from object/text to numeric. Blank values are treated as missing and filled with the median.

### `tenure_group`

```text
New      → 0–12 months
Mid      → 13–36 months
Senior   → 37–60 months
Loyal    → 61–72 months
```

### `num_services`
Counts the customer's subscribed services across OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies.

### `AutoPay`
Created from the customer's payment method.

---

# 🔢 Encoding & Scaling

Binary Yes/No fields are mapped to:

```text
Yes → 1
No  → 0
```

Categorical variables are one-hot encoded.

StandardScaler is applied to:

```text
tenure
MonthlyCharges
TotalCharges
num_services
```

---

# ✂️ Train-Test Split

```text
80% → Training Data
20% → Testing Data
```

The notebook uses:

```python
test_size=0.20
random_state=42
stratify=y
```

---

# ⚖️ Class Imbalance — SMOTE

The project applies **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data.

```text
Training Data
     ↓
   SMOTE
     ↓
Balanced Training Data
     ↓
   Models
```

The uploaded notebook reports:

| Method | Recall |
|---|---:|
| SMOTE | **0.8850** |
| class_weight='balanced' | **0.7968** |

**SMOTE achieved the higher recall in the notebook experiment.**

---

# 🧠 Machine Learning Models

## 1️⃣ KNN — K-Nearest Neighbors

The project tests:

```text
k = 1, 3, 5, 7, 9, 11, 15
```

Notebook result:

```text
Best k = 1
Best F1 Score = 0.2781
```

## 2️⃣ SVM — Support Vector Machine

RBF-kernel SVM is used and `C` is tuned:

```text
C = 0.1, 1, 10, 100
```

Notebook tuning result:

```text
Best C = 10
Best CV F1 = 0.8071
```

## 3️⃣ Decision Tree Classifier

Decision Tree depth is tuned using:

```text
3, 4, 5, 6, 7, 8, None
```

Notebook result:

```text
Best Depth = 7
Best F1 Score = 0.7967
```

---

# 📊 Exploratory Data Analysis

## 🎯 Customer Churn Distribution

![Customer Churn Distribution](docs/screenshots/01-churn-distribution.png)

## 📄 Churn Rate by Contract Type

![Churn Rate by Contract](docs/screenshots/02-contract-churn.png)

## 💰 Monthly Charges vs Churn

![Monthly Charges vs Churn](docs/screenshots/03-monthly-charges-churn.png)

## 📈 Tenure Distribution by Churn

![Tenure Distribution](docs/screenshots/04-tenure-churn.png)

## 🔥 Correlation Heatmap

![Correlation Heatmap](docs/screenshots/05-correlation-heatmap.png)

---

# 📈 Model Evaluation

The project evaluates classification models using:

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Correctness of positive churn predictions |
| Recall | Ability to identify actual churners |
| F1 Score | Balance between Precision and Recall |
| ROC-AUC | Ability to distinguish churn from non-churn |

For churn prediction, **Recall is especially important** because false negatives are customers who churn but were not identified.

---

# 🏆 Notebook Model Results

The uploaded notebook reports:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN (k=5) | 73.17% | 0.0000 | 0.0000 | 0.0000 | 0.5010 |
| SVM (C=1) | 73.46% | 0.0000 | 0.0000 | 0.0000 | 0.4990 |
| Decision Tree | 44.22% | 0.2731 | **0.6631** | 0.3869 | 0.4627 |

> These values are the outputs recorded in the uploaded notebook. The later error-analysis section reports a SMOTE-based Decision Tree recall of **0.8850**.

### ⭐ SMOTE-based Decision Tree Result

```text
Best Model: Decision Tree
Recall:    0.8850
Precision: 0.3062
F1 Score: 0.4550
ROC-AUC:   0.5390
```

---

# 🌟 Top Decision Tree Features

The uploaded notebook reports:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Contract_Two year` | **0.3430** |
| 2 | `Contract_One year` | **0.2456** |
| 3 | `tenure` | **0.0987** |
| 4 | `InternetService_Fiber optic` | **0.0790** |
| 5 | `MonthlyCharges` | **0.0749** |

![Feature Importance](docs/screenshots/06-feature-importance.png)

### 💡 Business Interpretation

Contract type, tenure, fiber-optic service and monthly charges are among the strongest Decision Tree features in the notebook's reported feature-importance analysis.

---

# 🔍 Error Analysis

The notebook reports:

```text
Total False Negatives = 43

Average Tenure = 25.33 months
Average MonthlyCharges = 67.87
```

False-negative contract distribution:

```text
One year → 95.35%
Two year → 4.65%
```

Overall churner profile:

```text
Average Tenure = 17.98 months
Average MonthlyCharges = 74.44
```

### Key Observation

The notebook concludes that the model mainly misses churners with **longer tenure and lower MonthlyCharges**. Most missed customers have one-year contracts.

---

# 📊 Recall Comparison

![SMOTE Recall Comparison](docs/screenshots/07-smote-recall.png)

---

# 🔄 Machine Learning Workflow

```text
                    TELCO DATASET
                          ↓
                 Data Understanding
                          ↓
                    EDA & Cleaning
                          ↓
                Feature Engineering
                          ↓
              Encoding + StandardScaler
                          ↓
                  Train-Test Split
                          ↓
                       SMOTE
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
             KNN         SVM    Decision Tree
              ↓           ↓           ↓
          Hyperparameter Tuning
                    ↓
              Model Evaluation
                    ↓
               Error Analysis
                    ↓
              Feature Importance
                    ↓
               Final Pipeline
                    ↓
              Churn Prediction
```

---

# 📌 Topics Covered

| Topic | Description |
|---|---|
| Dataset Loading | Load and inspect Telco data |
| EDA | Univariate and bivariate analysis |
| Data Cleaning | Clean `TotalCharges` |
| Feature Engineering | `tenure_group`, `num_services`, `AutoPay` |
| Encoding | Binary and one-hot encoding |
| Scaling | StandardScaler |
| Train-Test Split | 80/20 stratified split |
| SMOTE | Handle class imbalance |
| KNN | Baseline and K tuning |
| SVM | RBF model and C tuning |
| Decision Tree | Classification and depth tuning |
| SMOTE vs Balanced | Recall comparison |
| Model Comparison | Accuracy, Precision, Recall, F1, ROC-AUC |
| Error Analysis | False-negative analysis |
| Feature Importance | Top Decision Tree features |
| Final Pipeline | Save and reload prediction pipeline |

---

# 📁 Project Structure

```text
Telco_Customer_Churn/
│
├── README.md
├── CustomerChurn_SupervisedLearning.ipynb
├── Telco-Customer-Churn(3).csv
│
└── docs/
    └── screenshots/
        ├── 01-churn-distribution.png
        ├── 02-contract-churn.png
        ├── 03-monthly-charges-churn.png
        ├── 04-tenure-churn.png
        ├── 05-correlation-heatmap.png
        ├── 06-feature-importance.png
        └── 07-smote-recall.png
```

---

# 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming |
| 🐼 Pandas | Data Analysis |
| 🔢 NumPy | Numerical Computing |
| 📊 Matplotlib | Data Visualization |
| 🎨 Seaborn | Visualization |
| 🤖 Scikit-learn | Machine Learning |
| ⚖️ Imbalanced-learn | SMOTE |
| 📓 Jupyter Notebook | Development |

---

# ⚙️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
CustomerChurn_SupervisedLearning.ipynb
```

---

# ▶️ How to Run

```text
1. Load the dataset
2. Inspect and understand the data
3. Perform EDA
4. Clean TotalCharges
5. Create engineered features
6. Encode categorical variables
7. Split train and test data
8. Scale numerical features
9. Apply SMOTE to training data
10. Train KNN, SVM and Decision Tree
11. Tune model parameters
12. Evaluate models
13. Perform error analysis
14. Identify important features
15. Save/load the final pipeline
```

---

# 💡 Key Learnings

- Customer churn can be treated as a supervised binary classification problem.
- Feature scaling is important for KNN and SVM.
- SMOTE can improve minority-class recall.
- Decision Trees provide interpretable feature importance.
- Recall is important when the business wants to identify as many potential churners as possible.
- False-negative analysis helps identify customer groups that the model is missing.

---

# 🚀 Future Improvements

- 🔹 GridSearchCV / RandomizedSearchCV
- 🔹 Cross-Validation
- 🔹 ROC and Precision-Recall threshold tuning
- 🔹 SHAP Explainability
- 🔹 Model saving using Joblib
- 🔹 Streamlit Web Application
- 🔹 REST API
- 🔹 Cloud Deployment
- 🔹 Customer retention recommendation system

---

# 🏁 Conclusion

The **Telco Customer Churn Prediction** project demonstrates a complete supervised-learning workflow for identifying customers who may leave a telecom service.

It covers data cleaning, EDA, feature engineering, encoding, scaling, SMOTE, KNN, SVM, Decision Tree, hyperparameter tuning, evaluation, error analysis, feature importance and final pipeline creation.

The uploaded notebook reports **0.8850 recall for the SMOTE-based Decision Tree workflow**, making minority-class identification a key result of the project.

---

# 👨‍💻 Author

## Rahul Zala

**Python • Machine Learning • Data Science • Customer Churn Prediction**

---

## ⭐ Support

If you found this project useful, please give the repository a **Star ⭐**.

<p align="center">

**Made with ❤️ by Rahul Zala**

</p>

