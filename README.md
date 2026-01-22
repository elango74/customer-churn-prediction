# customer-churn-prediction
Customer churn occurs when customers stop using a company’s product or service, and retaining existing customers is more cost-effective than acquiring new ones. This project builds a machine learning model using structured customer data to predict churn, helping businesses identify at-risk customers early and take preventive actions.

# Problem Statement
Customer churn is a major challenge for the subscription based businesses
Predicting churn in advance helps companies:
  - Retain customers
  - improve customer satisfaction
  - reduce revenue loss
This project predicts churn based on customer demographics,services used, and billing information

# DataSet
 - **Source**:Telco Customer churn DataSet
 - **Target Variable**:churn(YES/NO)
 - **Feature Types**:
     - Categorical (Contract,PaymentMethod,InterService,etc)
     - Numerical(tenure,MonthlyCharges,TotalCharges)
# Machine Learning Approach
  🔹 **Preprocessing**
   - Missing Value handling
   - Numerical feature passthrough
   - Categorical feature encoding using **OneHotEncoder**
   - Unified preprocessing using **ColumnTransformer**

  🔹 Model
   - churn_pipeline.pkl(Preprocess + model)
   - logistic Regression
   - Random Forest(experimental)
