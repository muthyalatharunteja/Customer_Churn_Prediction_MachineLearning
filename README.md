# Customer Churn Prediction (Machine Learning)

## Overview
End-to-end ML project simulating a production workflow. Conducted EDA to analyze churn behavior, engineered and scaled features, trained and tuned multiple models using GridSearchCV, selected the best model, and deployed it via Streamlit for real-time predictions.

---

## Problem Statement
Customer churn is a key business challenge in the telecom industry. This project aims to predict whether a customer is likely to churn based on demographic and usage-related features, helping businesses take proactive retention actions.

---

## Dataset
- **Source:** Kaggle – Telecom Customer Churn Dataset  
- **Target Variable:** `Churn` (Yes = 1, No = 0)

### Features Used
- Age  
- Gender (Female = 1, Male = 0)  
- Tenure  
- Monthly Charges  

---

## Approach
1. Data loading and cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature engineering and encoding  
4. Train-test split and feature scaling  
5. Model training and comparison  
6. Hyperparameter tuning using GridSearchCV  
7. Model selection and persistence  
8. Deployment using Streamlit  

---

## Models Trained
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  

The best-performing model was selected based on validation performance.

---

## Deployment
A Streamlit web application was built to allow users to input customer details and receive churn predictions in real time. The trained model and scaler are loaded using serialized `.pkl` files to ensure consistency between training and inference.

---

## Configuration

Clone the repository:
```
git clone https://github.com/yourusername/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the application:
```
streamlit run app.py
```


---

  You can view " Streamlit app " in your browser.

  Local URL: http://localhost:8501
  
  Network URL: http://192.168.178.23:8501

---


## Project Structure
```
customer-churn-prediction-ml/
│
├── data/
│ └── customer_churn_data.csv
│
├── notebooks/
│ └── Churn_Analysis_Modelling.ipynb
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore
```


---


## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib


---


## Future Improvements
- Add ROC-AUC and confusion matrix evaluation
- Handle class imbalance
- Add feature importance analysis
- Containerize the application using Docker
- Deploy on Streamlit Cloud or Hugging Face Spaces


---


## Author
Muthyala Tharun Teja
[LinkedIn](https://linkedin.com/in/muthyalatharunteja)
