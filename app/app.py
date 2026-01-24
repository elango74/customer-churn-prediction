from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import sys

# ---------- Fix import path for src/ ----------
# Add project root to system path to import src modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.action_recommendation import recommend_action

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (allows React to connect)

# ---------- Load Model ----------
MODEL_PATH = os.path.join(ROOT_DIR, "models", "churn_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}")
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "churn-prediction-api"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        
        # Extract features (Expecting raw inputs, same as Streamlit app)
        # We need to reconstruct the dataframe as the model expects it
        
        tenure = data.get('tenure', 0)
        monthly_charges = data.get('MonthlyCharges', 0.0)
        total_charges = data.get('TotalCharges', 0.0)
        contract = data.get('Contract', 'Month-to-month')
        payment = data.get('PaymentMethod', 'Electronic check')
        
        # Initialize input dictionary with zeros for all one-hot encoded cols
        # Ideally, we should know all columns. 
        # For robustness, we use the model's feature_names_in_ if available.
        # Construct base dictionary
        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract_Month-to-month": 0,
            "Contract_One year": 0,
            "Contract_Two year": 0,
            "PaymentMethod_Electronic check": 0,
            "PaymentMethod_Mailed check": 0,
            "PaymentMethod_Bank transfer (automatic)": 0,
            "PaymentMethod_Credit card (automatic)": 0,
        }
        
        # Set One-Hot Encoded bits
        if f"Contract_{contract}" in input_data:
            input_data[f"Contract_{contract}"] = 1
            
        if f"PaymentMethod_{payment}" in input_data:
            input_data[f"PaymentMethod_{payment}"] = 1
            
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Align with model features
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
            
        # Predict
        churn_prob = model.predict_proba(input_df)[0][1]
        action = recommend_action(churn_prob)
        
        return jsonify({
            "churn_probability": float(churn_prob),
            "recommended_action": action
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
