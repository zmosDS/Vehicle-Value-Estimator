# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Load data
df_cars = pd.read_csv('cleaned_data_july_21st.csv')

# Define features and target
features = ['Year', 'Model', 'State', 'Mileage', 'Trim', 'Make', 'Body Style']
X = df_cars[features].copy()
y = df_cars['Price']

# Define and format categorical features
categorical_features = ['Model', 'State', 'Trim', 'Make', 'Body Style']
X[categorical_features] = X[categorical_features].astype('category')

# Scale numerical features
scaler = StandardScaler()
X_categorical[['Year', 'Mileage']] = scaler.fit_transform(X_categorical[['Year', 'Mileage']])

# Split data 80/20
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

# Define final/best hyperparameters
best_hyperparams = {
    'subsample': 1.0, 
    'reg_lambda': 0.5, 
    'reg_alpha': 0.001, 
    'num_leaves': 100, 
    'n_estimators': 1000, 
    'max_depth': 5, 
    'learning_rate': 0.1, 
    'colsample_bytree': 0.9
}

# Train LightGBM model
final_model = lgb.LGBMRegressor(**best_hyperparams, verbose=-1, n_jobs=-1)
final_model.fit(train_X, train_y, categorical_feature=categorical_features)

# Predict test data
final_pred = final_model.predict(test_X)

# Print final optimized LightGBM model performance
print('\033[1mOptimized LightGBM Model Performance (Final Model):\033[0m')
print(f'Mean Absolute Error (MAE): $ {metrics.mean_absolute_error(test_y, final_pred):,.2f}')
print(f'Mean Squared Error  (MSE): {int(metrics.mean_squared_error(test_y, final_pred)):,}')
print(f'R² Score             (R²): {metrics.r2_score(test_y, final_pred):.4f}')
