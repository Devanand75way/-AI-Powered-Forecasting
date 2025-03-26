# sales-forecast-project/backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

from models.sales_forecast_model import SalesForecastModel
from models.preprocessing import preprocess_data

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict_sales():
    try:
        # Load data and preprocess
        data = request.json['data']
        df = pd.DataFrame(data)
        preprocessed_df, features = preprocess_data(df)
        
        # Load pre-trained model
        model = joblib.load('ml_model/saved_models/sales_forecast_model.pkl')
        
        # Make predictions
        predictions = model.predict(preprocessed_df[features])
        
        return jsonify({
            'predictions': predictions.tolist(),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # Load training data
        data = request.json['data']
        df = pd.DataFrame(data)
        
        # Preprocess data
        preprocessed_df, features = preprocess_data(df)
        
        # Initialize and train model
        sales_model = SalesForecastModel(preprocessed_df, features, 'Sales')
        sales_model.split_data()
        sales_model.train_random_forest()
        
        # Evaluate model
        performance = sales_model.evaluate_model()
        
        # Save model
        joblib.dump(sales_model.model, 'ml_model/saved_models/sales_forecast_model.pkl')
        
        return jsonify({
            'performance': performance,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True)