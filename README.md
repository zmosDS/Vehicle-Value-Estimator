# Car Value Predictor

## Overview

This project demonstrates the application of data science techniques to build a **Vehicle Value Estimator**. The tool predicts car prices based on various input features, including make, model, year, mileage, and trim, providing an accurate estimate to assist users in making data-driven decisions about vehicle pricing.

The project showcases a full end-to-end data science pipeline, from data collection and exploration to model development, tuning, and deployment. The data was collected from Cars.com of the top 10 selling car manufacturers in 2023 ([US Auto Sales '23](https://www.carpro.com/blog/national-auto-sales-numbers-for-all-automakers-in-2023/)), ensuring a diverse yet focused dataset. The final output is a user-friendly dashboard where users can select vehicle details to receive an estimated price, along with visual insights into how different factors influence vehicle value.

## Project Structure

- **`data_collection/`**: Python scripts used for web scraping from Cars.com.  
- **`data/`**: Contains raw and cleaned datasets used for training and validation.  
- **`data_analysis/`**: Includes EDA and feature engineering notebooks.  
- **`modeling/`**: Jupyter notebooks and scripts documenting model comparison, optimization, evaluation, and validation processes.  
- **`models/`**: Saved model files (`.pkl`) used for deployment.  
- **`estimator_dashboard/`**: Code for the deployment-ready Streamlit dashboard.  

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/zmosDS/vehicle-value-estimator.git
```

### 2. Install dependencies
```bash
pip install -r vehicle-value-estimator/estimator_dashboard/requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run vehicle-value-estimator/estimator_dashboard/estimator_dashboard.py
```
The dashboard will be available locally at `http://localhost:8501` in your browser.

## Key Technologies

- **Data Scraping**: `BeautifulSoup`, `requests`
- **Data Processing and EDA**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning**: `LightGBM`, `XGBoost`, `scikit-learn`, `CatBoost`
- **Dashboard**: `Dash`, `Plotly`

### Problem Statement

Pricing vehicles accurately is a common challenge. Underpricing leads to lost revenue, while overpricing can result in slow inventory turnover. This project provides a data-driven solution that predicts vehicle prices with a high degree of accuracy, enabling individuals or businesses to optimize their vehicle pricing strategies.

## Project Highlights

### 1. Data Collection and Preprocessing
- **Data Source**: Vehicle listings scraped from **cars.com**.
- **Preprocessing**: Data was cleaned to handle missing values, outliers, and categorical features. Features like **make**, **model**, **trim**, **mileage**, **year**, and **body style** were extracted and used for predictions.

### 2. Exploratory Data Analysis (EDA)
- **Goal**: Understand how various features (e.g., mileage, year) affect vehicle prices.
- **Key Insights**: Visualizations were created to show trends, such as how prices decline with higher mileage or older years.
  
### 3. Model Development
- **Initial Model**: The project started with **XGBoost** using manually encoded categorical features.
- **Final Model**: The final model was developed using **LightGBM**, which achieved the best performance, natively handling categorical features.
- **Performance**:
  - Mean Absolute Error (MAE): `~$1,700`
  - R² Score: `~0.96`
  
### 4. Dashboard Development
- **Framework**: The dashboard was built using **Dash**.
- **Features**: 
  - Users input vehicle details such as **make**, **model**, **trim**, **mileage**, **year**, and **body style**.
  - The dashboard outputs an **estimated price** and provides visualizations showing price distributions for similar vehicles and feature importance in the prediction.
  
### 5. Business Impact
- **Optimized Pricing**: Users can accurately price their vehicles, balancing profitability and competitiveness.
- **Actionable Insights**: Visualizations help users understand how various factors (e.g., mileage, trim) affect the value.

## Results and Conclusion

The Vehicle Value Estimator successfully predicts car prices with an MAE of `$1,696`. Given that the vehicle prices in the dataset range from `$2,000` to `$150,000`, the MAE represents a relatively small error in comparison to the overall price range. The average vehicle price is approximately `$26,739`, meaning the MAE accounts for around `6.3%` of the average price, which indicates that the model provides reasonably accurate predictions.

The LightGBM model's ability to handle categorical data contributes to its high performance, making the predictions both accurate and robust across a variety of vehicle types. Additionally, the dashboard provides an intuitive interface, making the tool accessible even to non-technical users.

## Dashboard Screenshot

<img width="1136" alt="Screenshot 2024-09-19 at 3 25 59 AM" src="https://github.com/user-attachments/assets/c553479c-f95a-4d91-abe6-8c5deb7cebb0">

### Target Audience
This tool can be utilized by:
- **Car dealerships** seeking to optimize pricing strategies.
- **Private sellers** wanting to price their vehicles competitively.
- **Buyers** interested in estimating the value of a car based on specific features.

## **By building this project, I have demonstrated my ability to:**
- Collect and preprocess data.
- Explore and analyze the relationships within the dataset.
- Build, tune, and evaluate machine learning models.
- Deploy a user-friendly tool that offers actionable insights.

This project showcases my end-to-end data science skills, from data acquisition to deployment, and is a valuable tool for optimizing vehicle pricing strategies.
