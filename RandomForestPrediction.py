import pandas as pd
import numpy as np

df=pd.read_csv("date_proper.csv")
median_wip = df['wip'].median()
df['wip'].fillna(median_wip, inplace=True)

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply the outlier removal function to all numerical columns
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df_out = remove_outliers(df, col)
    # print(col)
    
from sklearn.preprocessing import LabelEncoder

le_quarter = LabelEncoder()
le_department = LabelEncoder()
le_day = LabelEncoder()

df_out['quarter'] = le_quarter.fit_transform(df_out['quarter'])
df_out['department'] = le_department.fit_transform(df_out['department'])
df_out['day'] = le_day.fit_transform(df_out['day'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dropping columns and splitting the data
X = df_out.drop(columns=['actual_productivity', 'date', 'day', 'wip']).values
y = df_out['actual_productivity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the StandardScaler
scaler = StandardScaler()

# Fitting the scaler on the training data and transforming it
X_train = scaler.fit_transform(X_train)

# Transforming the test data using the same scaler
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}, MSE: {mse}')

from xgboost import XGBRegressor

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost - MAE: {mae_xgb}, MSE: {mse_xgb}')

import streamlit as st
import numpy as np

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
