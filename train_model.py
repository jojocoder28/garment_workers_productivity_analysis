import pandas as pd
import numpy as np
import joblib

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
joblib.dump(scaler, 'scaler.pkl')

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

joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(model, 'randomforest_model.pkl')