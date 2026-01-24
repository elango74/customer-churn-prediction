import joblib
import pandas as pd

pipeline = joblib.load("../models/churn_pipeline.pkl")

def predict_churn(user_input: dict) -> str:
    input_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(input_df)[0]
    return "Churn" if prediction == 1 else "No Churn"
