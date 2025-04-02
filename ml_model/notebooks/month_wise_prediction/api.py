import flask
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)


# Load the saved model and preprocessing artifacts
model = joblib.load('demand_prediction_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')
scaler = joblib.load('scaler.joblib')
date_product_mapping = joblib.load('date_product_mapping.joblib')
feature_names = joblib.load('feature_names.joblib')

@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    try:
        data = request.json
        product_name = data['product_name']
        future_date_str = data['date']
        
        # Convert string date to datetime
        future_date = pd.to_datetime(future_date_str)
        
        # Get the encoded product name
        try:
            encoded_product = label_encoders['Product Name'].transform([product_name])[0]
        except:
            return jsonify({'error': f"Product '{product_name}' not found in training data"})
        
        # Extract date features
        year = future_date.year
        month = future_date.month
        day = future_date.day
        day_of_week = future_date.dayofweek
        quarter = (month - 1) // 3 + 1
        
        # Create input features
        input_features = np.array([[encoded_product, 0, 0, year, month, day, day_of_week, quarter]])
        
        # Scale the input features
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        predicted_demand = model.predict(input_scaled)[0]
        
        return jsonify({
            'product': product_name,
            'date': future_date_str,
            'predicted_demand': round(float(predicted_demand), 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/sales_report', methods=['POST'])
def sales_report():
    try:
        data = request.json
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])
        
        # Filter data for the specified date range
        date_range_data = date_product_mapping[
            (date_product_mapping['Order Date'] >= start_date) & 
            (date_product_mapping['Order Date'] <= end_date)
        ]
        
        # Group by product and sum quantities
        product_demand = date_range_data.groupby('Product Name')['Quantity'].sum().reset_index()
        
        # Sort by demand
        product_demand = product_demand.sort_values(by='Quantity', ascending=False)
        
        # Decode product names
        product_names = []
        for code in product_demand['Product Name']:
            try:
                name = label_encoders['Product Name'].inverse_transform([int(code)])[0]
                product_names.append(name)
            except:
                product_names.append(f"Unknown-{code}")
        
        product_demand['Product'] = product_names
        
        # Format the response
        results = []
        for _, row in product_demand.iterrows():
            results.append({
                'product': row['Product'],
                'total_quantity': int(row['Quantity'])
            })
        
        return jsonify({
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'products': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/monthly_trends', methods=['POST'])
def monthly_trends():
    try:
        data = request.json
        start_month = pd.to_datetime(data['start_month'])
        end_month = pd.to_datetime(data['end_month'])
        
        # Filter data
        date_range_data = date_product_mapping[
            (date_product_mapping['Order Date'] >= start_month) & 
            (date_product_mapping['Order Date'] <= end_month)
        ]
        
        # Add month-year column for grouping
        date_range_data['Month-Year'] = date_range_data['Order Date'].dt.strftime('%Y-%m')
        
        # Group by month and sum quantities
        monthly_demand = date_range_data.groupby('Month-Year')['Quantity'].sum().reset_index()
        
        # Calculate month-over-month growth
        monthly_demand['Previous'] = monthly_demand['Quantity'].shift(1)
        monthly_demand['Growth'] = (monthly_demand['Quantity'] - monthly_demand['Previous']) / monthly_demand['Previous'] * 100
        
        # Format the response
        results = []
        for _, row in monthly_demand.iterrows():
            growth = row['Growth'] if not pd.isna(row['Growth']) else None
            
            results.append({
                'month': row['Month-Year'],
                'total_quantity': int(row['Quantity']),
                'growth_percentage': round(float(growth), 2) if growth is not None else None
            })
        
        # Identify months with highest and lowest sales
        if len(results) > 0:
            highest_month = max(results, key=lambda x: x['total_quantity'])
            lowest_month = min(results, key=lambda x: x['total_quantity'])
            
            return jsonify({
                'start_month': start_month.strftime('%Y-%m'),
                'end_month': end_month.strftime('%Y-%m'),
                'monthly_data': results,
                'highest_sales_month': highest_month['month'],
                'highest_sales_quantity': highest_month['total_quantity'],
                'lowest_sales_month': lowest_month['month'],
                'lowest_sales_quantity': lowest_month['total_quantity']
            })
        else:
            return jsonify({
                'message': 'No data found for the specified date range'
            })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/product_monthly_trends', methods=['POST'])
def product_monthly_trends():
    try:
        data = request.json
        product_name = data['product_name']
        start_month = pd.to_datetime(data['start_month'])
        end_month = pd.to_datetime(data['end_month'])
        
        # Get the encoded product name
        try:
            encoded_product = label_encoders['Product Name'].transform([product_name])[0]
        except:
            return jsonify({'error': f"Product '{product_name}' not found in training data"})
        
        # Filter data
        product_data = date_product_mapping[date_product_mapping['Product Name'] == encoded_product]
        date_range_data = product_data[
            (product_data['Order Date'] >= start_month) & 
            (product_data['Order Date'] <= end_month)
        ]
        
        # Add month-year column for grouping
        date_range_data['Month-Year'] = date_range_data['Order Date'].dt.strftime('%Y-%m')
        
        # Group by month and sum quantities
        monthly_demand = date_range_data.groupby('Month-Year')['Quantity'].sum().reset_index()
        
        # Format the response
        results = []
        for _, row in monthly_demand.iterrows():
            results.append({
                'month': row['Month-Year'],
                'quantity': int(row['Quantity'])
            })
        
        return jsonify({
            'product': product_name,
            'start_month': start_month.strftime('%Y-%m'),
            'end_month': end_month.strftime('%Y-%m'),
            'monthly_data': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


