import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("credit_risk_dataset.csv")


df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())


df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})


df['loan_grade'] = df['loan_grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

df = pd.get_dummies(df, columns=['loan_intent', 'person_home_ownership'], drop_first=True)

df = df.dropna()

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("=" * 45)
print("        CREDIT SCORING MODEL RESULTS")
print("=" * 45)

print(f"\nAccuracy:       {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print("-" * 45)
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
print("=" * 45)
