import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path , encoding="windows-1252")
    # Convert date columns to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

# Feature engineering
def feature_engineering(df):
    # Extract date features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Day'] = df['Order Date'].dt.day
    df['DayOfWeek'] = df['Order Date'].dt.dayofweek
    df['Quarter'] = df['Order Date'].dt.quarter
    
    # Calculate shipping days
    df['ShippingDays'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Group by Product Name and Month to calculate monthly demand
    monthly_demand = df.groupby(['Product Name', 'Year', 'Month']).agg({
        'Quantity': 'sum',
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    return df, monthly_demand

# Prepare data for training
def prepare_training_data(monthly_demand):
    # Create lag features (previous month's demand)
    monthly_demand_sorted = monthly_demand.sort_values(['Product Name', 'Year', 'Month'])
    
    # Create product-specific features
    product_avg_demand = monthly_demand.groupby('Product Name')['Quantity'].mean().reset_index()
    product_avg_demand.columns = ['Product Name', 'AvgDemand']
    
    monthly_demand_with_avg = pd.merge(monthly_demand_sorted, product_avg_demand, on='Product Name')
    
    # Create a feature matrix for the model
    X = monthly_demand_with_avg[['Year', 'Month', 'AvgDemand']].copy()
    y = monthly_demand_with_avg['Quantity'].values
    
    # Add one-hot encoding for product categories
    X['ProductName'] = monthly_demand_with_avg['Product Name']
    
    return X, y

# Train the model
def train_model(X, y):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing for numerical and categorical features
    numerical_features = ['Year', 'Month', 'AvgDemand']
    categorical_features = ['ProductName']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f'Training R² score: {train_score:.4f}')
    print(f'Testing R² score: {test_score:.4f}')
    
    return model, X_test, y_test

# Generate predictions for future months
def predict_future_demand(model, monthly_demand, products, num_months=6):
    # Get the last month and year in the data
    last_date = monthly_demand.sort_values(['Year', 'Month']).iloc[-1]
    last_year = last_date['Year']
    last_month = last_date['Month']
    
    # Create future dates
    future_dates = []
    for i in range(1, num_months + 1):
        future_month = last_month + i
        future_year = last_year
        
        if future_month > 12:
            future_month = future_month % 12
            future_year += 1
            if future_month == 0:
                future_month = 12
                future_year -= 1
                
        future_dates.append((future_year, future_month))
    
    # Create prediction data
    prediction_data = []
    
    for product in products:
        product_avg = monthly_demand[monthly_demand['Product Name'] == product]['Quantity'].mean()
        
        for year, month in future_dates:
            prediction_data.append({
                'Year': year,
                'Month': month,
                'AvgDemand': product_avg,
                'ProductName': product
            })
    
    prediction_df = pd.DataFrame(prediction_data)
    
    # Make predictions
    predictions = model.predict(prediction_df)
    
    # Create a result dataframe
    result_df = prediction_df.copy()
    result_df['PredictedDemand'] = np.round(predictions).astype(int)
    result_df['MonthName'] = result_df['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%b'))
    
    return result_df

# Save the model
def save_model(model, file_path='product_demand_model.pkl'):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

# Main function to run the pipeline
def run_pipeline(file_path, output_path='prediction_results.csv'):
    df = load_data(file_path)
    df, monthly_demand = feature_engineering(df)
    
    # Get unique products
    unique_products = df['Product Name'].unique()
    
    X, y = prepare_training_data(monthly_demand)
    model, X_test, y_test = train_model(X, y)
    
    # Select top 10 products by sales volume for prediction
    top_products = df.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False).head(10).index.tolist()
    
    future_demand = predict_future_demand(model, monthly_demand, top_products)
    
    # Save results
    future_demand.to_csv(output_path, index=False)
    save_model(model)
    
    return future_demand, model

# Example usage
if __name__ == "__main__":
    # For demonstration, we're assuming the file is named "sales_data.csv"
    file_path = "ml_model/data/dataset.csv"
    prediction_results, trained_model = run_pipeline(file_path)
    
    # Print sample of the prediction results
    print("\nPrediction Results Sample:")
    print(prediction_results.head(10))
    
    # Generate data for API
    api_data = []
    for product in prediction_results['ProductName'].unique():
        product_data = prediction_results[prediction_results['ProductName'] == product]
        product_dict = {
            'product': product,
            'months': {}
        }
        
        for _, row in product_data.iterrows():
            month_key = f"{row['MonthName']}-{row['Year']}"
            product_dict['months'][month_key] = int(row['PredictedDemand'])
            
        api_data.append(product_dict)
    
    # Convert to JSON and save for the React UI
    import json
    with open('prediction_api_data.json', 'w') as f:
        json.dump(api_data, f, indent=2)
    
    print("\nAPI data generated and saved to prediction_api_data.json")