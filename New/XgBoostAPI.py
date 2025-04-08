from flask import Flask, request, jsonify
import pandas as pd
from xgboost import XGBRegressor
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/XgBoost_trained_model.pkl")

# Define the same feature columns used during training
features = [
    'Product Name', 'RAM', 'Memory', 'Price (USD)', 'Competitor Price',
    'Stock Available', 'Marketing Spend', 'Holiday/Season Indicator',
    'Weather Condition', 'Economic Indicator', 'Social Media Trend Score',
    'Market Sentiment Score', 'Competitor Activity Score', 'Discount (%)',
    'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfWeek', 'IsWeekend',
    'Discount_Price_Impact', 'Marketing_Stock_Interaction',
    'Effective_Price', 'Price_Ratio'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Convert to proper data types
        for col in ['Product Name', 'RAM', 'Memory']:
            if input_df[col].dtype == 'object':
                input_df[col] = input_df[col].astype('category').cat.codes

        # Add derived features
        input_df['Discount_Price_Impact'] = input_df['Discount (%)'] * input_df['Price (USD)'] / 100
        input_df['Marketing_Stock_Interaction'] = input_df['Marketing Spend'] * input_df['Stock Available']
        input_df['Effective_Price'] = input_df['Price (USD)'] - input_df['Discount_Price_Impact']
        input_df['Price_Ratio'] = input_df['Price (USD)'] / input_df['Competitor Price']

        # Ensure all features are included
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        return jsonify({'predicted_sales': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
