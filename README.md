<h1 align="center">🚀 Telco Customer Churn Prediction</h1>

<p align="center">
  <b>📊 Supervised Learning • Churn Detection • Customer Retention</b>
</p>



<img width="1672" height="940" alt="019c6379-3459-438e-b7cf-e19326ba860c" src="https://github.com/user-attachments/assets/a9f17197-e983-4454-983f-488daefd97f2" />

### 🖱️ Click the Image Above

<p align="center">

<a href="#-project-objectives">
🎯 <b>CHURN PREDICTION</b>
</a>
&nbsp;&nbsp; ➜ &nbsp;&nbsp;

<a href="#-class-imbalance--smote">
⚖️ <b>CLASS BALANCING</b>
</a>
&nbsp;&nbsp; ➜ &nbsp;&nbsp;

<a href="#-model-evaluation">
📊 <b>MODEL EVALUATION</b>
</a>
&nbsp;&nbsp; ➜ &nbsp;&nbsp;

<a href="#-error-analysis">
🔎 <b>ERROR ANALYSIS</b>
</a>

</p>

## 🏷️ Skills Badges
<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge\&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge\&logo=pandas)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge\&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge\&logo=jupyter)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced%20Learning-green?style=for-the-badge)

</p>

---

## 📌 Project Goal

**Telco Customer Churn Prediction** is a Machine Learning project that predicts whether a telecom customer is likely to **churn (leave the service)** or **stay**.

The project follows a complete supervised-learning workflow:

<img width="1312" height="1199" alt="f2ef1b72-0fb4-438a-aec2-c903df641356" src="https://github.com/user-attachments/assets/edb7be02-503b-417c-975e-bfd7f7db4205" />



---

## 🎯 Project Objectives

The main objectives of this **Telco Customer Churn Prediction** project are:

* 📂 Understand the **Telco Customer Churn dataset**.
* 🔍 Perform **Exploratory Data Analysis (EDA)**.
* 🧹 Clean and preprocess customer data.
* 🛠️ Perform **feature engineering**.
* 🔢 Encode categorical variables and scale numerical features.
* ⚖️ Handle class imbalance using **SMOTE**.
* 🤖 Build **KNN, SVM, and Decision Tree** classification models.
* 🎯 Tune **KNN `k`**, **SVM `C`**, and **Decision Tree depth**.
* 📊 Compare **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.
* 🔎 Perform **False-Negative Error Analysis**.
* 🌟 Identify important features for **customer churn prediction**.
* 🚀 Create and save a **final prediction pipeline**.


```

```
# 📊 Dataset

<table bgcolor="#f3f4f6">
<tr>
<td>

The **Telco Customer Churn** dataset contains **7,043 customer records and 21 columns**.

| Category | Details |
|---|---|
| 👥 Total Customers | **7,043** |
| 📊 Total Columns | **21** |
| 🎯 Target Variable | `Churn` |
| 🏷️ Problem Type | Binary Classification |
| 🤖 Learning Type | Supervised Learning |
| 🔴 Positive Class | `Yes` — Customer Churned |
| 🟢 Negative Class | `No` — Customer Stayed |
| 🆔 Identifier | `customerID` |
| 💰 Numerical Features | `tenure`, `MonthlyCharges`, `TotalCharges` |
| 📝 Categorical Features | `Contract`, `InternetService`, `PaymentMethod`, etc. |

### 🎯 Target Variable

| Value | Meaning |
|---|---|
| 🟢 `No` | Customer did **not** churn |
| 🔴 `Yes` | Customer **churned** |

### 🔍 Main Features

| Feature | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `gender` | Customer gender |
| `SeniorCitizen` | Senior citizen indicator |
| `Partner` | Whether customer has a partner |
| `Dependents` | Whether customer has dependents |
| `tenure` | Number of months with the company |
| `PhoneService` | Phone service subscription |
| `InternetService` | Internet service type |
| `Contract` | Contract type |
| `PaymentMethod` | Customer payment method |
| `MonthlyCharges` | Monthly service charges |
| `TotalCharges` | Total customer charges |
| `Churn` | 🎯 Target — customer churn status |

</td>
</tr>
</table>

# 🧹 Data Cleaning & Feature Engineering

## `TotalCharges`

* Converted `TotalCharges` from **object/text** to **numeric**.
* Blank values were treated as **missing values**.
* Missing values were filled using the **median**.

## `tenure_group`

Customer tenure was divided into four groups:

| Group        | Tenure       |
| ------------ | ------------ |
| 🆕 **New**   | 0–12 months  |
| 🔄 **Mid**   | 13–36 months |
| ⭐ **Senior** | 37–60 months |
| 💎 **Loyal** | 61–72 months |

## `num_services`

Created `num_services` to count the total number of subscribed services for each customer.

The following services were included:

* 🔐 OnlineSecurity
* 💾 OnlineBackup
* 🛡️ DeviceProtection
* 🛠️ TechSupport
* 📺 StreamingTV
* 🎬 StreamingMovies

## `AutoPay`

Created the `AutoPay` feature based on the customer's **PaymentMethod**.

* **Yes** → Automatic payment method
* **No** → Non-automatic payment method


# 🔢 Encoding & Scaling

## 🔘 Binary Encoding

Binary **Yes/No** features are converted into numerical values:

```text
✅ Yes → 1
❌ No  → 0
```

This converts binary categorical data into a format that can be easily used by machine learning models.

---

## 🏷️ One-Hot Encoding

Categorical variables with multiple categories are converted using **One-Hot Encoding**.

🔹 Each category is represented as a separate binary column.
🔹 This prevents the model from assuming any numerical relationship between categories.

---

## 📏 Feature Scaling

`StandardScaler` is applied to the following numerical features:

```text
📊 tenure
💰 MonthlyCharges
💵 TotalCharges
🔢 num_services
```

### ⚙️ Standardization

`StandardScaler` transforms the numerical features to have:

```text
📌 Mean ≈ 0
📌 Standard Deviation ≈ 1
```

This ensures that features are on a similar scale and helps improve the performance of **scale-sensitive machine learning algorithms**.


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

## 🎯 Customer Churn Count
<img width="549" height="393" alt="image" src="https://github.com/user-attachments/assets/1f8cd9ba-ab7b-41df-9a3e-567b39d72601" />


## 📄 Tenure Distribution,Monthly Charges Distribution and Total Charges Distribution

<img width="1589" height="390" alt="image" src="https://github.com/user-attachments/assets/6233d652-85ca-41e8-9ed7-c6874af8d9e7" />


## Contract,Internet Service and Payment Method

<img width="1778" height="489" alt="image" src="https://github.com/user-attachments/assets/d063975f-430e-43ec-9da3-f26439d9f4c0" />


##  📊 Churn Rate by Contract Type

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/56769635-f8f5-4eed-a398-9607a611b125" />



## 📊 Churn Rate by Tenure Bucket

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/583e946c-40de-4953-8464-59d3aef9e2e0" />



## 💰 Monthly Charges vs Churn

<img width="618" height="470" alt="image" src="https://github.com/user-attachments/assets/d1e18f42-b292-4e4a-ae44-eb7f47b8076a" />




## 🔥 Correlation Heatmap

<img width="910" height="690" alt="image" src="https://github.com/user-attachments/assets/acdf0ca2-0134-4b81-a2f9-113bbd43109e" />


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



## 🌳 Decision Tree - First 3 Levels
<img width="1570" height="812" alt="image" src="https://github.com/user-attachments/assets/5dee80e3-ef59-40f9-a76d-13fa9867df41" />


# 📊 Precision vs Recall - All 4 Models

<img width="889" height="590" alt="image" src="https://github.com/user-attachments/assets/1c16c724-dee2-49f9-95ce-788011273f6b" />


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
CustomerChurn_SupervisedLearning/
│
├── README.md
├── CustomerChurn_SupervisedLearning.ipynb
├── Telco-Customer-Churn.csv
├── project_theory.md
├── churn_model.pkl
├── reduirements(2).txt
├── summary_report.md
└── video1033299810.mp4
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

The **Telco Customer Churn Prediction** project demonstrates an end-to-end **Supervised Learning workflow** for identifying customers who are likely to leave a telecom service.

### 📌 Project Covers

* 🧹 **Data Cleaning**
* 🔍 **Exploratory Data Analysis (EDA)**
* 🛠️ **Feature Engineering**
* 🔢 **Categorical Encoding**
* 📏 **Feature Scaling**
* ⚖️ **SMOTE for Class Imbalance**
* 🤖 **KNN Classification**
* 🎯 **SVM Classification**
* 🌳 **Decision Tree Classification**
* 🎚️ **Hyperparameter Tuning**
* 📊 **Model Evaluation**
* 🔎 **False-Negative Error Analysis**
* 🌟 **Feature Importance**
* 🚀 **Final Prediction Pipeline**

### ⭐ Key Result

The uploaded notebook reports a **Recall of 0.8850** for the **SMOTE-based Decision Tree workflow**.

This highlights the importance of **class-imbalance handling and recall** when the main objective is to identify as many potential churn customers as possible.

### 🚀 Final Takeaway

> **Better churn identification → Earlier customer retention action → Reduced customer loss**

---

<p align="center">
  <b>📡 Predict Churn • 📊 Analyze Data • 🤖 Build Models • 🚀 Improve Retention</b>
</p>


# 👨‍💻 Author

## Rahul Zala

**Python • Machine Learning • Data Science • Customer Churn Prediction**

---

## ⭐ Support

If you found this project useful, please give the repository a **Star ⭐**.

<p align="center">

**Made with ❤️ by Rahul Zala**

</p>

