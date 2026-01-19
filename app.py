#Gender -> 1 Female 0 Male #Churn 1 Yes 0 No 
#Scaler is exported as scaler.pkl 
# Model is exported as model.pkl 
# Order of the X-> 'Age', 'Gender', 'Tenure', 'MonthlyCharges
import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title('Churn Prediction App')
st.divider()
st.write('Please enter the values and hit Predict button')
st.divider()

age = st.number_input('Enter Age', min_value=10, max_value=100, value=30)

gender = st.selectbox('Enter Gender', ['Male', 'Female'])

tenure = st.number_input('Enter Tenure', min_value=0, max_value=130, value=10)

monthlycharge = st.number_input('Enter Monthly Charge', min_value=30, max_value=150)

st.divider()

predictbutton = st.button('Predict')

st.divider()

if predictbutton:
    gender_selected = 1 if gender == 'Female' else 0

    X = [age, gender_selected, tenure, monthlycharge]
    X = np.array(X).reshape(1, -1)

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    predicted = 'Yes' if prediction == 1 else 'No'
    st.success(f'Predicted Churn: {predicted}')
else:
    st.info('Please enter the values and use Predict button')
