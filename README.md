⚙️ Methodology

Data Preparation

Read borrower data from loan_data_created.csv

Engineered key credit risk features:

payment_to_income = loan_amt_outstanding / income

debt_to_income = total_debt_outstanding / income

Model Training

Trained a Logistic Regression model using features:

credit_lines_outstanding

debt_to_income

payment_to_income

years_employed

fico_score

Model Evaluation

Evaluated model accuracy using ROC Curve and AUC score

Computed model error rate and discriminative power

Expected Loss Calculation

Expected Loss (EL)
=
PD
×
(
1
−
Recovery Rate
)
Expected Loss (EL)=PD×(1−Recovery Rate)

With a 10% recovery rate, the model outputs expected loss per borrower.

🧠 Example Usage
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

# Read and prepare data
df = pd.read_csv('loan_data_created.csv')
df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

# Define model features
features = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']

# Train logistic regression
clf = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000).fit(df[features], df['default'])

# Evaluate
y_pred = clf.predict(df[features])
fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
print("Model Error:", (abs(df['default'] - y_pred).sum()) / len(df))
print("AUC Score:", metrics.auc(fpr, tpr))

🧰 Tech Stack

Language: Python

Libraries: scikit-learn, pandas, numpy

Techniques: Logistic Regression, ROC-AUC, Credit Risk Modeling

📊 Outcome

Successfully modeled borrower default probability using key financial ratios.

Provided a quantifiable metric for expected loan loss based on PD and recovery rate.

Can serve as the foundation for credit scoring systems or risk management dashboards.
