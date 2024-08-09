import joblib
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = joblib.load('scaler.pkl')
# Load the trained model
model = joblib.load('randomforest_model.pkl')
df = pd.read_csv('date_proper.csv')
# Load the trained model
st.title("Productivity Prediction")

# Input features from the user
# date = st.date_input("Date")
quarter = st.selectbox("Quarter", [1, 2, 3, 4])
department = st.selectbox("Department", df['department'].unique())
# day = st.selectbox("Day", df['day'].unique())
team = st.selectbox("Team",df['team'].unique() )
targeted_productivity = st.number_input("Targeted Productivity")
smv = st.number_input("SMV")
# wip = st.number_input("WIP")
over_time = st.number_input("Over Time")
incentive = st.number_input("Incentive")
idle_time = st.number_input("Idle Time")
idle_men = st.number_input("Idle Men")
no_of_style_change = st.number_input("No. of Style Changes")
no_of_workers = st.number_input("No. of Workers")
target_achieved = st.selectbox("Target Achieved", df['target_achieved'].unique())

if department == 'sewing':
    department=1
else:
    department=0

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[quarter, team, department, targeted_productivity, smv, over_time, incentive, 
                            idle_time, idle_men, no_of_style_change, no_of_workers, target_achieved]])

    # # Scale the data (if scaling was used)
    input_data = scaler.transform(input_data)

    # Predict the productivity
    prediction = model.predict(input_data)
    
    st.write(f"Predicted Actual Productivity: {prediction[0]:.2f}")
