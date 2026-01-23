import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.action_recommendation import recommend_action

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -----------------------------
# Data cleaning
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop("customerID", axis=1, inplace=True)

# -----------------------------
# Encoding
# -----------------------------
df = pd.get_dummies(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")

# -----------------------------
# Predict churn probability
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

X_test = X_test.copy()
X_test["churn_probability"] = y_prob
X_test["recommended_action"] = X_test["churn_probability"].apply(recommend_action)

# -----------------------------
# Save output
# -----------------------------
os.makedirs("outputs", exist_ok=True)
X_test.to_csv("outputs/churn_with_actions.csv", index=False)

print("✅ Model trained, saved, and actions generated successfully")
