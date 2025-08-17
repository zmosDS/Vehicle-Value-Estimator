# Necessary libraries
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load both datasets and combine for biggest range of features
julyData   = pd.read_csv(REPO_ROOT/'data'/'cleaned_data_july_21st.csv')
augustData = pd.read_csv(REPO_ROOT/'data'/'cleaned_data_aug_16th.csv')
df_cars    = pd.concat([julyData, augustData], ignore_index=True)

# Capitalize the first letter of the make, model, trim, and body style
df_cars['Make']       = df_cars['Make'].str.title()
df_cars['Model']      = df_cars['Model'].str.title()
df_cars['Trim']       = df_cars['Trim'].str.title()
df_cars['Body Style'] = df_cars['Body Style'].str.title()

# Sort unique values for make and state
unique_makes  = sorted(df_cars['Make'].unique())
unique_states = sorted(df_cars['State'].unique())

# Load model and scaler
final_model = joblib.load(REPO_ROOT/'models'/'final_model.pkl')
scaler      = joblib.load(REPO_ROOT/'models'/'final_scaler.pkl')

# Welcome message on sidebar
st.sidebar.title('Welcome to the Vehicle Value Estimator!')
st.sidebar.write('''
This tool helps you estimate vehicle value based on various features like make, model, year, and mileage.
Input your vehicle details to get an accurate price estimation.
''')

# Dashboard title
st.title('Vehicle Value Estimator')

# Split layout for inputs
col1, col2 = st.columns(2)

with col1:
    # Make
    make = st.selectbox('Make', unique_makes)
    
    # Model, filter based on make
    filtered_models = df_cars[df_cars['Make'] == make]['Model'].unique() if make else []
    model_name      = st.selectbox('Model', sorted(filtered_models), disabled=not make)

    # Trim, filter based on model
    filtered_trims = df_cars[df_cars['Model'] == model_name]['Trim'].unique() if model_name else []
    trim           = st.selectbox('Trim', sorted(filtered_trims), disabled=not model_name)

with col2:

    # Body style, filter based on model and trim 
    filtered_body_styles = df_cars[(df_cars['Model'] == model_name) & (df_cars['Trim'] == trim)]['Body Style'].unique() if (model_name and trim) else []
    body_style           = st.selectbox('Body Style', sorted(filtered_body_styles), disabled=not (model_name and trim))

    # Mileage
    mileage = st.number_input('Mileage', min_value=1000, max_value=300000, value=25000)

    # State
    state = st.selectbox('State', unique_states)

# Year
year = st.slider('Year', min_value=2010, max_value=2024, value=2012)

# Define dataframe from user inputs
input_data = pd.DataFrame({
    'Year':       [year],
    'Mileage':    [mileage],
    'Trim':       [trim],
    'Model':      [model_name],
    'State':      [state],
    'Make':       [make],
    'Body Style': [body_style],
})

# Ensure categorical columns are 'category'
cat_columns             = ['Model', 'State', 'Trim', 'Make', 'Body Style']
input_data[cat_columns] = input_data[cat_columns].astype('category')

# Scale numerical features
input_data[['Year', 'Mileage']] = scaler.transform(input_data[['Year', 'Mileage']])

# Predict value with final_model
predicted_price = final_model.predict(input_data)[0]
st.write(f'### Estimated Value: ${predicted_price:,.2f}')

# Price distribution for similar cars
st.write('### Price Distribution for Similar Vehicles')
similar_cars       = df_cars[(df_cars['Make'] == make) & (df_cars['Model'] == model_name) & (df_cars['Trim'] == trim) & (df_cars['Body Style'] == body_style)]
price_distribution = similar_cars['Price']

# Plot price distribution for similar cars
plt.figure(figsize=(10, 6))
sns.histplot(price_distribution, kde=True, color='coral')
plt.xlabel('Price')
plt.ylabel('# of Similar Vehicles')
st.pyplot(plt)

# Feature importance title
st.write('### Feature Importance')

# Extract feature importances
importance    = final_model.feature_importances_
importance_df = pd.DataFrame({'Feature': input_data.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)
