# Import required libraries
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io
import base64
from datetime import datetime
import seaborn as sns

app = Flask(__name__)

# Load and preprocess the data
def load_data():
    # In a real scenario, you'd load from a CSV or database
    try:
        df = pd.read_csv('New/preprocessed_smartphone_sales.csv')
        
        # Define consistent column names that will be used throughout the application
        column_mapping = {
            'Product Name': 'Product_Name',
            'Units Sold': 'Units_Sold',
            'Price': 'Price',
            'Competitor Price': 'Competitor_Price',
            'Stock Available': 'Stock_Available', 
            'Marketing Spend': 'Marketing_Spend',
            'Holiday/Seasonal Indicator': 'Holiday_Seasonal_Indicator',
            'Weather Condition': 'Weather_Condition',
            'Economic Indicator': 'Economic_Indicator',
            'Social Media Trend Score': 'Social_Media_Trend_Score',
            'Market Sentiment Score': 'Market_Sentiment_Score',
            'Competitor Activity Score': 'Competitor_Activity_Score'
        }
        
        # Rename columns that exist in the dataframe
        for original_col, new_col in column_mapping.items():
            if original_col in df.columns:
                df.rename(columns={original_col: new_col}, inplace=True)
        
        # Convert date columns to datetime
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        
        # Feature engineering
        df['Month_Name'] = df['Date'].dt.month_name()
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
        df['Revenue'] = df['Units_Sold'] * df['Price']
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
# Function to generate plots as base64 encoded images
def get_plot_as_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

@app.route('/api/sales-analysis-by-date', methods=['POST'])
def sales_analysis_by_date():
    """
    Endpoint for Past Sales Analysis by Date Interval
    
    Expected input:
    {
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD"
    }
    """
    try:
        data = request.get_json()
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])
        
        # Load data
        df = load_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500
        # print("df", df)
        # Filter data for date range
        date_filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if len(date_filtered_df) == 0:
            return jsonify({"error": "No data found for the specified date range"}), 404
        
        # Calculate total sales per product
        product_sales = date_filtered_df.groupby('Product_Name').agg({
            'Units_Sold': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        # Monthly sales trends
        monthly_sales = date_filtered_df.groupby(['Year', 'Month']).agg({
            'Units_Sold': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        # Find best and worst performing months
        monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)
        best_month = monthly_sales.loc[monthly_sales['Revenue'].idxmax()]
        worst_month = monthly_sales.loc[monthly_sales['Revenue'].idxmin()]
        
        # # Create visualizations
        # plt.figure(figsize=(12, 8))
        
        # # Monthly sales trend visualization
        # plt.subplot(2, 1, 1)
        # plt.plot(monthly_sales['YearMonth'], monthly_sales['Revenue'], marker='o')
        # plt.title('Monthly Sales Revenue')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        
        # # Product sales comparison
        # plt.subplot(2, 1, 2)
        # product_sales = product_sales.sort_values('Revenue', ascending=False)
        # plt.bar(product_sales['Product_Name'], product_sales['Revenue'])
        # plt.title('Total Revenue by Product')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        
        # # Convert plot to base64
        # sales_trend_img = get_plot_as_base64(plt.gcf())
        # plt.close()
        
        response = {
            "total_sales_by_product": product_sales.to_dict(orient='records'),
            "monthly_sales_trends": monthly_sales.to_dict(orient='records'),
            "best_performing_month": {
                "year": int(best_month['Year']),
                "month": int(best_month['Month']),
                "revenue": float(best_month['Revenue']),
                "units_sold": int(best_month['Units_Sold'])
            },
            "worst_performing_month": {
                "year": int(worst_month['Year']),
                "month": int(worst_month['Month']),
                "revenue": float(worst_month['Revenue']),
                "units_sold": int(worst_month['Units_Sold'])
            },
            # "visualization": sales_trend_img
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sales-analysis-by-product', methods=['POST'])
def sales_analysis_by_product():
    """
    Endpoint for Past Sales Report by Product Name
    
    Expected input:
    {
        "product_name": "Product XYZ"
    }
    """
    try:
        data = request.get_json()
        product_name = data['product_name']
        print("Product name: {}".format(product_name))
        # Load data
        df = load_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500
        
        # Filter data for the specific product
        product_df = df[df['Product_Name'] == product_name]
        
        if len(product_df) == 0:
            return jsonify({"error": f"No data found for product: {product_name}"}), 404
        
        # Calculate monthly sales for the product
        monthly_product_sales = product_df.groupby(['Year', 'Month', 'Month_Name']).agg({
            'Units_Sold': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        monthly_product_sales['YearMonth'] = monthly_product_sales['Year'].astype(str) + '-' + monthly_product_sales['Month'].astype(str).str.zfill(2)
        
        # Find best selling months for this product
        best_months = monthly_product_sales.nlargest(3, 'Units_Sold')
        
        # Create visualization
        # plt.figure(figsize=(12, 6))
        
        # plt.subplot(1, 1, 1)
        # plt.plot(monthly_product_sales['YearMonth'], monthly_product_sales['Units Sold'], marker='o')
        # plt.title(f'Sales Trend Over Time for {product_name}')
        # plt.xlabel('Month')
        # plt.ylabel('Units Sold')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        
        # product_trend_img = get_plot_as_base64(plt.gcf())
        # plt.close()
        
        response = {
            "product_name": product_name,
            "sales_over_time": monthly_product_sales.to_dict(orient='records'),
            "best_selling_months": best_months.to_dict(orient='records'),
            # "visualization": product_trend_img
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sales-forecast', methods=['POST'])
def sales_forecast():
    """
    Endpoint for Future Sales Forecasting
    
    Expected input:
    {
        "product_name": "Product01",  # The specific product to forecast
        "start_date": "2023-04-01",   # Start date for forecast
        "end_date": "2023-08-01",     # End date for forecast
        "external_factors": {
            "competitor_activity": 0.7,
            "weather_condition": 0.8,
            "market_sentiment": 0.6,
            "holiday_indicator": 1  # 1 for holiday period, 0 for non-holiday
        }
    }
    """
    try:
        data = request.get_json()
        product_name = data.get('product_name')
        start_date = pd.to_datetime(data.get('start_date'))
        end_date = pd.to_datetime(data.get('end_date'))
        external_factors = data.get('external_factors', {})
        
        # Validate input
        if not product_name:
            return jsonify({"error": "Product name is required"}), 400
        if not start_date or not end_date:
            return jsonify({"error": "Both start_date and end_date are required"}), 400
        if start_date >= end_date:
            return jsonify({"error": "end_date must be after start_date"}), 400
        
        # Load data
        df = load_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500
        
        # Rename columns to match the actual dataset columns
        # This is crucial - match the exact column names from your dataset
        column_mapping = {
            'Product Name': 'Product_Name',
            'Units Sold': 'Units_Sold',
            'Price': 'Price',
            'Competitor Price': 'Competitor_Price',
            'Stock Available': 'Stock_Available', 
            'Marketing Spend': 'Marketing_Spend',
            'Holiday/Seasonal Indicator': 'Holiday_Seasonal_Indicator',
            'Weather Condition': 'Weather_Condition',
            'Economic Indicator': 'Economic_Indicator',
            'Social Media Trend Score': 'Social_Media_Trend_Score',
            'Market Sentiment Score': 'Market_Sentiment_Score',
            'Competitor Activity Score': 'Competitor_Activity_Score'
        }
        
        # Rename columns if they exist in the dataframe
        for original_col, new_col in column_mapping.items():
            if original_col in df.columns:
                df.rename(columns={original_col: new_col}, inplace=True)
        
        # Check if product exists in the dataset
        if product_name not in df['Product_Name'].unique():
            return jsonify({"error": f"Product '{product_name}' not found in dataset"}), 404
        
        # Filter data for the specific product
        product_df = df[df['Product_Name'] == product_name].sort_values('Date')
        
        # Prepare the training data
        # Define features for the model - use the renamed columns
        numeric_features = ['Price', 'Competitor_Price', 'Stock_Available', 'Marketing_Spend', 
                           'Weather_Condition', 'Economic_Indicator', 'Social_Media_Trend_Score',
                           'Market_Sentiment_Score', 'Competitor_Activity_Score', 'Month', 'Weekday']
        
        categorical_features = ['Holiday_Seasonal_Indicator']
        
        # Verify all features exist in the dataframe
        missing_features = [col for col in numeric_features + categorical_features if col not in product_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing columns in dataset: {missing_features}"}), 500
        
        # Create the preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        X = product_df[numeric_features + categorical_features]
        y = product_df['Units_Sold']
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        model.fit(X, y)
        
        # Generate dates for forecast period
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Create a dataframe for forecast periods
        forecast_records = []
        
        # Use the last record as a template
        last_record = product_df.iloc[-1].copy()
        
        for forecast_date in forecast_dates:
            forecast_record = last_record.copy()
            forecast_record['Date'] = forecast_date
            forecast_record['Year'] = forecast_date.year
            forecast_record['Month'] = forecast_date.month
            forecast_record['Day'] = 1
            forecast_record['Weekday'] = forecast_date.weekday()
            
            # Update with any provided external factors
            if 'competitor_activity' in external_factors:
                forecast_record['Competitor_Activity_Score'] = external_factors['competitor_activity']
            if 'weather_condition' in external_factors:
                forecast_record['Weather_Condition'] = external_factors['weather_condition']
            if 'market_sentiment' in external_factors:
                forecast_record['Market_Sentiment_Score'] = external_factors['market_sentiment']
            if 'holiday_indicator' in external_factors:
                forecast_record['Holiday_Seasonal_Indicator'] = external_factors['holiday_indicator']
            
            forecast_records.append(forecast_record)
        
        forecast_df = pd.DataFrame(forecast_records)
        
        # Make predictions
        X_forecast = forecast_df[numeric_features + categorical_features]
        forecast_df['Predicted_Units'] = model.predict(X_forecast)
        
        # Calculate feature importance
        feature_importance = model.named_steps['regressor'].feature_importances_
        
        # Getting the feature names after one-hot encoding
        preprocessor_features = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(['Holiday_Seasonal_Indicator']))
        
        # Create a dataframe of feature importances
        importance_df = pd.DataFrame({
            'Feature': preprocessor_features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Create visualization
        # plt.figure(figsize=(12, 6))
        
        # # Plot historical data (last 12 months or all if less) for comparison
        # historical = product_df.tail(min(12, len(product_df)))
        
        # plt.plot(historical['Date'], historical['Units_Sold'], 
        #          marker='o', label=f'Historical ({product_name})')
        
        # plt.plot(forecast_df['Date'], forecast_df['Predicted_Units'], 
        #          marker='x', linestyle='--', label=f'Predicted ({product_name})')
        
        # plt.title(f'Sales Forecast for {product_name}')
        # plt.xlabel('Month')
        # plt.ylabel('Units Sold')
        # plt.legend()
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        
        # forecast_img = get_plot_as_base64(plt.gcf())
        # plt.close()
        
        # Format dates for nicer output
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
        
        response = {
            'product_name': product_name,
            'forecast_period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'number_of_months': len(forecast_dates)
            },
            'forecast': forecast_df[['Date', 'Predicted_Units']].to_dict(orient='records'),
            'factor_impact': importance_df.head(5).to_dict(orient='records'), 
            # 'visualization': forecast_img
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)