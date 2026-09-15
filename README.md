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

The dataset is divided into **Training** and **Testing** sets to evaluate the machine learning model effectively.

### 📊 Data Distribution

```text
🟢 80% → Training Data
🔵 20% → Testing Data
```

### ⚙️ Parameters Used

```python
test_size=0.20
random_state=42
stratify=y
```

### 🔍 Parameter Explanation

* 📚 **`test_size=0.20`** → Uses **20% of the data for testing** and 80% for training.
* 🎯 **`random_state=42`** → Ensures the same train-test split every time.
* ⚖️ **`stratify=y`** → Maintains the same class distribution in both training and testing datasets.

This helps create a **reliable and balanced evaluation** of the machine learning models.

---

# ⚖️ Class Imbalance — SMOTE

Customer churn datasets can contain an **imbalanced target variable**, where one class has significantly more samples than the other.

To address this problem, the project applies **SMOTE (Synthetic Minority Over-sampling Technique)** to the **training data**.

## 🔄 SMOTE Workflow

```text
📊 Training Data
       ↓
   ⚙️ SMOTE
       ↓
⚖️ Balanced Training Data
       ↓
🤖 Machine Learning Models
```

### 📈 Recall Comparison

The notebook experiment compares **SMOTE** with `class_weight='balanced'`:

| 🧪 Method                    |  🎯 Recall |
| ---------------------------- | ---------: |
| ⚙️ **SMOTE**                 | **0.8850** |
| ⚖️ `class_weight='balanced'` | **0.7968** |

### 🏆 Result

**SMOTE achieved the higher recall (0.8850)** in the notebook experiment.

Higher recall is particularly useful for churn prediction because it helps the model identify a larger number of customers who are actually likely to **churn**.


# 🧠 Machine Learning Models

The project evaluates multiple machine learning algorithms to identify the model that performs best for **customer churn prediction**.

---

## 1️⃣ KNN — K-Nearest Neighbors

**K-Nearest Neighbors (KNN)** is tested with different values of `k` to find the optimal number of neighboring samples.

### 🔢 Values Tested

```text
k = 1, 3, 5, 7, 9, 11, 15
```

### 🏆 Best Result

| 📌 Parameter     |  📊 Result |
| ---------------- | ---------: |
| 🔢 Best `k`      |      **1** |
| 🎯 Best F1 Score | **0.2781** |

---

## 2️⃣ SVM — Support Vector Machine

An **RBF-kernel Support Vector Machine (SVM)** is used, and the regularization parameter `C` is tuned to improve model performance.

### ⚙️ C Values Tested

```text
C = 0.1, 1, 10, 100
```

### 🏆 Best Result

| 📌 Parameter        |  📊 Result |
| ------------------- | ---------: |
| ⚙️ Best `C`         |     **10** |
| 🎯 Best CV F1 Score | **0.8071** |

---

## 3️⃣ Decision Tree Classifier

A **Decision Tree Classifier** is evaluated using different maximum tree-depth values to find the best-performing configuration.

### 🌳 Depth Values Tested

```text
3, 4, 5, 6, 7, 8, None
```

### 🏆 Best Result

| 📌 Parameter     |  📊 Result |
| ---------------- | ---------: |
| 🌳 Best Depth    |      **7** |
| 🎯 Best F1 Score | **0.7967** |

---

## 📊 Model Summary

| 🤖 Model             | ⚙️ Best Parameter |    🎯 Best Score |
| -------------------- | ----------------- | ---------------: |
| 🔢 **KNN**           | `k = 1`           |    **0.2781 F1** |
| 🧠 **SVM (RBF)**     | `C = 10`          | **0.8071 CV F1** |
| 🌳 **Decision Tree** | `Depth = 7`       |    **0.7967 F1** |

### 🏅 Best Performing Model

Based on the reported notebook tuning results, **SVM with an RBF kernel and `C = 10` achieved the highest F1 score of 0.8071** among these three models.

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

The following table presents the **model performance results recorded in the notebook**.

## 📊 Model Performance Comparison

| 🤖 Model             | 🎯 Accuracy | 🎯 Precision |  🔄 Recall | ⭐ F1 Score | 📈 ROC-AUC |
| -------------------- | ----------: | -----------: | ---------: | ---------: | ---------: |
| 🔢 **KNN (k=5)**     |  **73.17%** |       0.0000 |     0.0000 |     0.0000 |     0.5010 |
| 🧠 **SVM (C=1)**     |  **73.46%** |       0.0000 |     0.0000 |     0.0000 |     0.4990 |
| 🌳 **Decision Tree** |      44.22% |       0.2731 | **0.6631** |     0.3869 |     0.4627 |

---

## 🔍 Key Observations

* 🔢 **KNN** achieved an accuracy of **73.17%**, but its Precision, Recall, and F1 Score were **0.0000**.
* 🧠 **SVM** achieved the highest accuracy among the three models at **73.46%**, but also produced **0.0000 Recall and F1 Score**.
* 🌳 **Decision Tree** achieved a lower accuracy of **44.22%**, but obtained the highest Recall of **0.6631** among these three recorded results.
* 📈 The ROC-AUC values indicate that these baseline models did not perform strongly in separating the two classes.

---


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



## 🌳 Decision Tree — First 3 Levels

<div style="background-color: black; padding: 20px; text-align: center;">

<img width="1570" height="812" alt="Decision Tree - First 3 Levels" src="https://github.com/user-attachments/assets/5dee80e3-ef59-40f9-a76d-13fa9867df41" />

</div>


# 📊 Precision vs Recall - All 4 Models

<img width="889" height="590" alt="image" src="https://github.com/user-attachments/assets/1c16c724-dee2-49f9-95ce-788011273f6b" />


---

# 🔄 Machine Learning Workflow

```text
                    📊 TELCO DATASET
                          ↓
                 🔍 Data Understanding
                          ↓
                    🧹 EDA & Cleaning
                          ↓
                ⚙️ Feature Engineering
                          ↓
              🔢 Encoding + StandardScaler
                          ↓
                  ✂️ Train-Test Split
                          ↓
                       ⚖️ SMOTE
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
             🔢 KNN      🧠 SVM     🌳 Decision Tree
              ↓           ↓           ↓
             ⚙️ Hyperparameter Tuning
                          ↓
                  📊 Model Evaluation
                          ↓
                   🔎 Error Analysis
                          ↓
                  ⭐ Feature Importance
                          ↓
                   🚀 Final Pipeline
                          ↓
                  🎯 Churn Prediction
```

> 🖤 **End-to-End Machine Learning Workflow for Customer Churn Prediction**


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

Follow the steps below to install the required Python libraries and run the project.

## 📦 Install Required Libraries

Open **Command Prompt / Terminal** and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 📚 Libraries Used

* 🐼 **Pandas** — Data manipulation and analysis
* 🔢 **NumPy** — Numerical computing
* 📊 **Matplotlib** — Data visualization
* 🎨 **Seaborn** — Statistical visualization
* 🤖 **Scikit-learn** — Machine learning algorithms
* ⚖️ **Imbalanced-learn** — SMOTE and imbalanced-data handling
* 📓 **Jupyter** — Running the notebook

---

## ▶️ Start Jupyter Notebook

After installation, start Jupyter Notebook using:

```bash
jupyter notebook
```

## 📓 Open the Project

Open the following notebook:

```text
CustomerChurn_SupervisedLearning.ipynb
```


---
# ▶️ How to Run

Follow the steps below to run the **Customer Churn Prediction** project from start to finish.

### 🚀 Project Execution Steps

```text
1️⃣  Load the dataset
2️⃣  Inspect and understand the data
3️⃣  Perform Exploratory Data Analysis (EDA)
4️⃣  Clean and convert TotalCharges
5️⃣  Create engineered features
6️⃣  Encode categorical variables
7️⃣  Split data into training and testing sets
8️⃣  Scale numerical features
9️⃣  Apply SMOTE to the training data
🔟  Train KNN, SVM and Decision Tree models
1️⃣1️⃣ Tune model hyperparameters
1️⃣2️⃣ Evaluate model performance
1️⃣3️⃣ Perform error analysis
1️⃣4️⃣ Identify important features
1️⃣5️⃣ Save and load the final ML pipeline
```

### ✅ Final Output

After completing all the steps, the trained pipeline can be used to:

🎯 **Predict whether a customer is likely to churn or stay.**

---

# 💡 Key Learnings

Through this **Customer Churn Prediction** project, the following key concepts and practical insights were learned:

* 🎯 **Binary Classification:** Customer churn can be modeled as a **supervised binary classification problem**.
* 📏 **Feature Scaling:** Feature scaling is especially important for **KNN and SVM**, as these algorithms are sensitive to feature magnitude.
* ⚖️ **SMOTE:** **SMOTE** can help improve **minority-class recall** by balancing the training data.
* 🌳 **Interpretability:** **Decision Trees** provide useful and easy-to-understand **feature importance**.
* 🔍 **Recall:** Recall is important when the business goal is to identify as many **potentially churning customers** as possible.
* 🚨 **False-Negative Analysis:** Analyzing false negatives helps identify **customer groups that the model fails to detect**.
* 📊 **Model Evaluation:** Comparing multiple evaluation metrics provides a better understanding of overall model performance.


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

