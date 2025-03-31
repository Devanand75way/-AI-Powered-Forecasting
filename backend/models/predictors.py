import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MarketInsightPredictor:
    def __init__(self):
        # Default market insights
        self.current_market_insights = {}

    def update_market_insights(self, insights):
        """
        Update market insights dynamically
        """
        self.current_market_insights = insights

    def get_market_insights(self, category):
        """
        Retrieve market insights for a specific category
        """
        # If specific category insights exist in current insights, use those
        if category in self.current_market_insights:
            return self.current_market_insights[category]
        
        # Fallback to default insights if no dynamic insights found
        default_insights = {
            'Electronics': {
                'growth_rate': 0.15,
                'market_sentiment': 'Positive',
                'key_trends': ['Sustainable Technology', 'AI Integration'],
                'demand_multiplier': 1.2
            },
            'Furniture': {
                'growth_rate': 0.10,
                'market_sentiment': 'Stable',
                'key_trends': ['Smart Home', 'Minimalist Design'],
                'demand_multiplier': 1.1
            },
            'Office Supplies': {
                'growth_rate': 0.08,
                'market_sentiment': 'Moderate',
                'key_trends': ['Remote Work', 'Ergonomic Solutions'],
                'demand_multiplier': 1.05
            },
            'Technology': {
                'growth_rate': 0.12,
                'market_sentiment': 'Positive',
                'key_trends': ['Cloud Computing', 'Cybersecurity'],
                'demand_multiplier': 1.15
            }
        }
        
        return default_insights.get(category, {
            'growth_rate': 0.05,
            'market_sentiment': 'Neutral',
            'key_trends': ['General Market Trends'],
            'demand_multiplier': 1.0
        })
    

class AdvancedDemandPredictor:
    def __init__(self):
        # Load pre-trained model and preprocessing artifacts
        self.model = joblib.load('ml_model/saved_models/demand_prediction_model.joblib')
        self.label_encoders = joblib.load('ml_model/saved_models/label_encoders.joblib')
        self.scaler = joblib.load('ml_model/saved_models/scaler.joblib')
        self.market_insight_predictor = MarketInsightPredictor()
    
    def encode_categorical_feature(self, column, value):
        """
        Encode categorical feature with fallback mechanism
        """
        try:
            # Try to use existing encoder
            return self.label_encoders[column].transform([str(value)])[0]
        except ValueError:
            # If label is unseen, add the new label
            full_classes = list(self.label_encoders[column].classes_)
            full_classes.append(str(value))
            self.label_encoders[column].classes_ = np.array(full_classes)
            
            # Return the new encoded value
            return self.label_encoders[column].transform([str(value)])[0]
    
    def prepare_input_data(self, input_data):
        """
        Prepare input data for prediction
        """
        # Columns to encode
        categorical_columns = [
            'Segment', 'Country', 'City', 'State', 'Region', 
            'Category', 'Sub-Category', 'Product Name'
        ]
        
        # Create a copy of input data
        processed_data = input_data.copy()
        
        # Encode categorical variables
        for col in categorical_columns:
            processed_data[col] = self.encode_categorical_feature(col, processed_data[col])
        
        # Prepare features
        features = [
            'Segment', 'Country', 'City', 'State', 'Region', 
            'Category', 'Sub-Category', 'Product Name', 
            'Sales', 'Discount'
        ]
        
        # Convert to DataFrame to ensure correct format
        input_df = pd.DataFrame([processed_data])
        
        return input_df[features]
    
    def predict_future_demand(self, input_data, market_insights=None):
        """
        Predict demand with comprehensive market insights
        """
        # Update market insights if provided
        if market_insights:
            self.market_insight_predictor.update_market_insights(market_insights)
        
        # Prepare input data
        prepared_features = self.prepare_input_data(input_data)
        
        # Scale the features
        scaled_input = self.scaler.transform(prepared_features)
        
        # Predict base demand
        base_demand = self.model.predict(scaled_input)[0]
        
        # Get market insights for the category
        market_insights_data = self.market_insight_predictor.get_market_insights(
            input_data['Category']
        )
        
        # Ensure type conversion and handling for growth rate and demand multiplier
        try:
            growth_rate = float(market_insights_data.get('growth_rate', 0.05))
            demand_multiplier = float(market_insights_data.get('demand_multiplier', 1.0))
        except (ValueError, TypeError):
            # Fallback to default values if conversion fails
            growth_rate = 0.05
            demand_multiplier = 1.0
        
        # Calculate future demand
        # Consider market growth rate, sentiment, and demand multiplier
        future_demand_multiplier = (1 + growth_rate) * demand_multiplier
        
        # Project demand for 6 months
        projected_demand = base_demand * (future_demand_multiplier ** 1)
        
        # Recommend stock levels with buffer
        recommended_stock = int(projected_demand * 1.2)  # 20% buffer
        
        # Determine demand classification
        if projected_demand > base_demand * 1.5:
            demand_classification = "High Potential"
        elif projected_demand > base_demand * 1.2:
            demand_classification = "Moderate Growth"
        else:
            demand_classification = "Stable"
        
        return {
            'base_demand': float(base_demand),
            'projected_demand': float(projected_demand),
            'demand_classification': demand_classification,
            'recommended_stock': recommended_stock,
            'market_insights': {
                'growth_rate': growth_rate,
                'market_sentiment': market_insights_data.get('market_sentiment', 'Neutral'),
                'key_trends': market_insights_data.get('key_trends', ['General Market Trends'])
            },
            'prediction_period': {
                'months': 6,
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'end_date': (datetime.now() + timedelta(days=6*30)).strftime('%Y-%m-%d')
            }
        }