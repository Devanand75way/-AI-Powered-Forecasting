# app.py - Flask API for Product Demand Prediction
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
MODEL_PATH = 'product_demand_model.pkl'
PREDICTION_DATA_PATH = 'prediction_results.csv'

def load_model():
    """Load the trained prediction model"""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def load_predictions():
    """Load the pre-generated prediction data"""
    if not os.path.exists(PREDICTION_DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(PREDICTION_DATA_PATH)

def generate_predictions_for_product(product_name, num_months=6):
    """Generate predictions for a specific product for the next num_months"""
    model = load_model()
    predictions_df = load_predictions()
    
    # Check if we have pre-generated predictions for this product
    product_predictions = predictions_df[predictions_df['ProductName'] == product_name]
    
    if not product_predictions.empty:
        # Return pre-generated predictions
        return format_prediction_response(product_predictions)
    
    # If product not found in pre-generated predictions, generate new predictions
    if model is None:
        return None
    
    # Get current date to start predictions from
    current_date = datetime.now()
    future_dates = []
    
    # Generate future dates
    for i in range(1, num_months + 1):
        future_month = current_date.month + i
        future_year = current_date.year
        
        if future_month > 12:
            future_month = future_month % 12
            future_year += 1
            if future_month == 0:
                future_month = 12
                future_year -= 1
                
        future_dates.append((future_year, future_month))
    
    # Create features for prediction
    # Note: This is a simplified approach - in production, you'd use the same
    # feature extraction logic as in the training pipeline
    prediction_data = []
    
    # Using average demand of 10 as a placeholder (this would come from historical data)
    avg_demand = 10
    
    for year, month in future_dates:
        prediction_data.append({
            'Year': year,
            'Month': month,
            'AvgDemand': avg_demand,
            'ProductName': product_name
        })
    
    prediction_df = pd.DataFrame(prediction_data)
    
    try:
        # Make predictions
        predictions = model.predict(prediction_df)
        
        # Add predictions to the dataframe
        prediction_df['PredictedDemand'] = np.round(predictions).astype(int)
        prediction_df['MonthName'] = prediction_df['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%b'))
        
        return format_prediction_response(prediction_df)
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def format_prediction_response(prediction_df):
    """Format the prediction dataframe into a JSON response"""
    months_data = {}
    
    for _, row in prediction_df.iterrows():
        month_key = f"{row['MonthName']}-{row['Year']}"
        months_data[month_key] = int(row['PredictedDemand'])
    
    # Calculate insights
    demands = list(months_data.values())
    avg_demand = sum(demands) / len(demands) if demands else 0
    
    # Calculate trend
    trend = "Stable"
    if len(demands) >= 2:
        first_half = demands[:len(demands)//2]
        second_half = demands[len(demands)//2:]
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            trend = "Strongly Increasing"
        elif second_avg > first_avg:
            trend = "Slightly Increasing"
        elif second_avg < first_avg * 0.9:
            trend = "Strongly Decreasing"
        elif second_avg < first_avg:
            trend = "Slightly Decreasing"
    
    # Determine peak month
    peak_month = max(months_data.items(), key=lambda x: x[1])[0] if months_data else "N/A"
    
    # Determine low month
    low_month = min(months_data.items(), key=lambda x: x[1])[0] if months_data else "N/A"
    
    # Calculate growth percentage
    if len(demands) >= 2:
        growth_pct = ((demands[-1] - demands[0]) / demands[0] * 100) if demands[0] > 0 else 0
    else:
        growth_pct = 0
        
    return {
        "product": prediction_df['ProductName'].iloc[0],
        "months": months_data,
        "insights": {
            "average_demand": round(avg_demand, 1),
            "trend": trend,
            "peak_month": peak_month,
            "peak_demand": months_data.get(peak_month, 0),
            "low_month": low_month,
            "low_demand": months_data.get(low_month, 0),
            "total_predicted_demand": sum(demands),
            "growth_percentage": round(growth_pct, 1),
            "recommendation": get_recommendation(trend, growth_pct)
        }
    }

def get_recommendation(trend, growth_pct):
    """Generate a business recommendation based on the trend and growth"""
    if trend in ["Strongly Increasing", "Slightly Increasing"] and growth_pct > 10:
        return "Consider increasing inventory and production capacity to meet growing demand."
    elif trend == "Stable":
        return "Maintain current inventory levels with slight adjustments for seasonal variations."
    elif trend in ["Slightly Decreasing"] and growth_pct > -10:
        return "Monitor inventory closely and consider modest reductions in ordering."
    elif trend in ["Strongly Decreasing"] and growth_pct < -10:
        return "Reduce inventory levels and consider promotional activities to boost sales."
    else:
        return "Review historical performance and adjust inventory based on seasonal patterns."

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get list of all products available for prediction"""
    predictions_df = load_predictions()
    
    if predictions_df.empty:
        return jsonify({
            "error": "No prediction data available",
            "products": []
        }), 404
    
    products = predictions_df['ProductName'].unique().tolist()
    return jsonify({"products": products})

@app.route('/api/demand/<product_name>', methods=['GET'])
def get_product_demand(product_name):
    """Get demand prediction for a specific product"""
    if not product_name:
        return jsonify({"error": "Product name is required"}), 400
    
    # URL decode the product name if needed
    product_name = request.path.split('/')[-1]
    
    predictions = generate_predictions_for_product(product_name)
    
    if predictions is None:
        return jsonify({
            "error": "Unable to generate predictions for this product",
            "product": product_name
        }), 404
    
    return jsonify(predictions)

@app.route('/api/demand', methods=['GET'])
def get_all_predictions():
    """Get predictions for all products"""
    predictions_df = load_predictions()
    
    if predictions_df.empty:
        return jsonify({
            "error": "No prediction data available",
            "predictions": []
        }), 404
    
    # Group predictions by product
    all_predictions = []
    for product in predictions_df['ProductName'].unique():
        product_data = predictions_df[predictions_df['ProductName'] == product]
        predictions = format_prediction_response(product_data)
        all_predictions.append(predictions)
    
    return jsonify({"predictions": all_predictions})

if __name__ == '__main__':
    app.run(debug=True, port=5000)