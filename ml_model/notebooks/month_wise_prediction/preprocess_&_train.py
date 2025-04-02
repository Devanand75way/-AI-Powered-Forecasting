import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the dataset
def preprocess_data(file_path, external_data_path=None):
    # Read the dataset
    df = pd.read_csv(file_path, encoding="Windows-1252")
    
    # Convert Order Date to datetime
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        # Extract date features
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Day'] = df['Order Date'].dt.day
        df['DayOfWeek'] = df['Order Date'].dt.dayofweek
        df['Quarter'] = df['Order Date'].dt.quarter
    else:
        raise ValueError("Order Date column is required for time-based prediction")
    
    # Load and merge external data if provided
    if external_data_path:
        external_df = pd.read_csv(external_data_path)
        # Assuming external data has a date column to join on
        external_df['Date'] = pd.to_datetime(external_df['Date'])
        # Convert to same date format for merging
        df['Date_Key'] = df['Order Date'].dt.strftime('%Y-%m-%d')
        external_df['Date_Key'] = external_df['Date'].dt.strftime('%Y-%m-%d')
        # Merge external data
        df = pd.merge(df, external_df, on='Date_Key', how='left')
        # Handle missing values from the merge
        # Fill missing external data with means or appropriate values
        for col in external_df.columns:
            if col != 'Date' and col != 'Date_Key' and col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)
    
    # Feature selection - keep only relevant columns
    # Dropping columns as specified in requirements
    drop_columns = ['Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']
    # Check if columns exist before dropping
    drop_columns = [col for col in drop_columns if col in df.columns]
    df = df.drop(drop_columns, axis=1)
    
    # Feature selection for the model
    features = ['Product Name', 'Sales', 'Discount', 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']
    
    # Add external features if they exist
    if external_data_path:
        external_features = [col for col in external_df.columns 
                            if col != 'Date' and col != 'Date_Key']
        features.extend(external_features)
    
    # Make sure all selected features exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    # Create working dataframe with selected features
    working_df = df[features + ['Quantity', 'Order Date']].copy()
    
    # Encode categorical variables
    categorical_columns = ['Product Name']
    # Add any categorical external features
    categorical_columns = [col for col in categorical_columns if col in working_df.columns]
    
    # Label Encoding for categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        working_df[col] = le.fit_transform(working_df[col].astype(str))
        label_encoders[col] = le
    
    # Save the original dataframe with date for later filtering
    original_df = working_df.copy()
    
    # Drop Order Date for model training
    if 'Order Date' in working_df.columns:
        working_df = working_df.drop('Order Date', axis=1)
    
    # Split features and target
    X = working_df.drop('Quantity', axis=1)
    y = working_df['Quantity']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'features': X.columns.tolist(),
        'original_df': original_df
    }

# Train the model
def train_demand_prediction_model(preprocessed_data):
    # Use Random Forest Regressor for demand prediction
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    
    # Train the model
    model.fit(
        preprocessed_data['X_train'], 
        preprocessed_data['y_train']
    )
    
    # Evaluate the model
    train_score = model.score(
        preprocessed_data['X_train'], 
        preprocessed_data['y_train']
    )
    test_score = model.score(
        preprocessed_data['X_test'], 
        preprocessed_data['y_test']
    )
    
    print(f"Train R² Score: {train_score}")
    print(f"Test R² Score: {test_score}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': preprocessed_data['features'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

# Function to predict demand for specific date ranges
def predict_demand_for_period(model, preprocessed_data, start_date, end_date):
    """
    Predict demand for products between start_date and end_date
    """
    df = preprocessed_data['original_df']
    
    # Filter data for the given date range
    mask = (df['Order Date'] >= start_date) & (df['Order Date'] <= end_date)
    period_df = df[mask].copy()
    
    if period_df.empty:
        print(f"No data available for the period {start_date} to {end_date}")
        return None
    
    # Drop Order Date for prediction
    period_df = period_df.drop('Order Date', axis=1)
    
    # Prepare data for prediction
    X = period_df.drop('Quantity', axis=1)
    scaler = preprocessed_data['scaler']
    X_scaled = scaler.transform(X)
    
    # Predict demand
    predictions = model.predict(X_scaled)
    
    # Create a DataFrame with predictions
    results = pd.DataFrame({
        'Product': period_df.index,
        'Actual_Quantity': period_df['Quantity'].values,
        'Predicted_Quantity': predictions
    })
    
    # Calculate metrics
    results['Difference'] = results['Predicted_Quantity'] - results['Actual_Quantity']
    results['Abs_Difference'] = abs(results['Difference'])
    results['Percentage_Error'] = (results['Abs_Difference'] / results['Actual_Quantity']) * 100
    
    return results

# Function to predict future demand
def predict_future_demand(model, preprocessed_data, product_info, future_dates):
    """
    Predict future demand based on product info and dates
    """
    # Prepare product information
    label_encoders = preprocessed_data['label_encoders']
    scaler = preprocessed_data['scaler']
    features = preprocessed_data['features']
    
    # Create a DataFrame for future dates
    future_data = []
    
    for date in future_dates:
        data_point = product_info.copy()  # Copy base product info
        # Add date features
        data_point['Year'] = date.year
        data_point['Month'] = date.month
        data_point['Day'] = date.day
        data_point['DayOfWeek'] = date.dayofweek
        data_point['Quarter'] = (date.month - 1) // 3 + 1
        future_data.append(data_point)
    
    future_df = pd.DataFrame(future_data)
    
    # Encode categorical variables
    for col in label_encoders:
        if col in future_df.columns:
            le = label_encoders[col]
            future_df[col] = le.transform(future_df[col].astype(str))
    
    # Make sure all features are present
    for feature in features:
        if feature not in future_df.columns:
            future_df[feature] = 0  # Default value if feature is missing
    
    # Reorder columns to match training data
    future_df = future_df[features]
    
    # Scale the features
    future_df_scaled = scaler.transform(future_df)
    
    # Predict demand
    predictions = model.predict(future_df_scaled)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Quantity': predictions
    })
    
    return results

# Find months with high and low demand
def analyze_monthly_demand_trends(preprocessed_data):
    """
    Analyze monthly demand trends to identify high and low demand periods
    """
    df = preprocessed_data['original_df']
    
    # Group by month and calculate total quantity
    monthly_demand = df.groupby(df['Order Date'].dt.strftime('%Y-%m'))['Quantity'].sum()
    monthly_demand = monthly_demand.reset_index()
    monthly_demand.columns = ['Month', 'Total_Quantity']
    
    # Sort months by demand
    high_demand_months = monthly_demand.sort_values(by='Total_Quantity', ascending=False)
    low_demand_months = monthly_demand.sort_values(by='Total_Quantity', ascending=True)
    
    return {
        'high_demand_months': high_demand_months.head(3),  # Top 3 high demand months
        'low_demand_months': low_demand_months.head(3)     # Top 3 low demand months
    }

# Main execution
def main():
    # Preprocess the data
    preprocessed_data = preprocess_data(
        'ml_model/data/dataset.csv',
        external_data_path='ml_model/data/external_factors.csv'
    )
    
    # Train the model
    demand_model = train_demand_prediction_model(preprocessed_data)
    
    # Save the model and preprocessing artifacts
    joblib.dump(demand_model, 'demand_prediction_model.joblib')
    joblib.dump(preprocessed_data['label_encoders'], 'label_encoders.joblib')
    joblib.dump(preprocessed_data['scaler'], 'scaler.joblib')
    joblib.dump(preprocessed_data['features'], 'features.joblib')
    
    # Example: Generate monthly demand trends
    trends = analyze_monthly_demand_trends(preprocessed_data)
    print("\nHigh Demand Months:")
    print(trends['high_demand_months'])
    print("\nLow Demand Months:")
    print(trends['low_demand_months'])

if __name__ == '__main__':
    main()