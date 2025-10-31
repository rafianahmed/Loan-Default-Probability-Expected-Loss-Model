# 🏦 Loan Default Probability & Expected Loss Model

## 📘 Overview
This project implements a **Logistic Regression model** to estimate the **Probability of Default (PD)** for borrowers using financial ratios such as **Debt-to-Income**, **Payment-to-Income**, **FICO score**, and **Credit Lines Outstanding**. Using a **10% recovery rate**, the model can calculate **Expected Loss (EL)**:

\[
EL = PD \times (1 - \text{Recovery Rate})
\]

---

## ⚙️ Methodology
1. **Data Preparation**
   - Load borrower data from `loan_data_created.csv`
   - Compute key ratios:
     - `payment_to_income = loan_amt_outstanding / income`
     - `debt_to_income = total_debt_outstanding / income`

2. **Feature Selection**
   - Features used: `credit_lines_outstanding`, `debt_to_income`, `payment_to_income`, `years_employed`, `fico_score`

3. **Model Training**
   - Train **Logistic Regression** using scikit-learn

4. **Evaluation**
   - Predict default outcomes
   - Measure **error rate** and **ROC-AUC score**

---

## 💻 Example Code

```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

# Load dataset
df = pd.read_csv('loan_data_created.csv')

# Feature engineering
df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

# Define model features
features = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']

# Train Logistic Regression
clf = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000)
clf.fit(df[features], df['default'])

# Print model coefficients
print("Coefficients:", clf.coef_)
print("Intercept:", clf.intercept_)

# Predict on dataset
y_pred = clf.predict(df[features])

# Evaluate model
fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
error_rate = (abs(df['default'] - y_pred).sum()) / len(df)
auc_score = metrics.auc(fpr, tpr)

print("Error Rate:", error_rate)
print("AUC Score:", auc_score)
Tech Stack

Python

scikit-learn

pandas, numpy

📊 Outcome

Estimates Probability of Default (PD) for borrowers

Enables Expected Loss (EL) calculation using a 10% recovery rate

Model performance evaluated via ROC-AUC and error rate
