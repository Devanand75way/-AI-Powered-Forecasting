import joblib
import numpy as np
import pandas as pd
import requests
import re
from datetime import datetime, timedelta
from collections import defaultdict

class TrendAnalyzer:
    """
    Analyze trending hashtags and keywords to adjust demand predictions
    """
    def __init__(self):
        self.trending_cache = {}
        self.cache_expiry = datetime.now()
        self.trending_topics = []
        self.api_key = None  # Set your API key for trend analysis service
    
    def set_api_key(self, api_key):
        """Set API key for trend analysis service"""
        self.api_key = api_key
    
    def fetch_trending_topics(self, category=None):
        """
        Fetch trending hashtags and topics from social media or trend services
        Refreshes cache if expired (every 6 hours)
        """
        current_time = datetime.now()
        
        # Check if cache is expired
        if current_time > self.cache_expiry or category not in self.trending_cache:
            try:
                # This would be replaced with actual API call to trend service
                # For example: Twitter API, Google Trends, etc.
                if self.api_key:
                    # Example API call structure (replace with actual implementation)
                    # response = requests.get(
                    #     f"https://api.trendservice.com/trends",
                    #     params={"category": category, "api_key": self.api_key}
                    # )
                    # if response.status_code == 200:
                    #     self.trending_topics = response.json().get("trends", [])
                    
                    # Simulated response for demonstration
                    self.trending_topics = self._get_simulated_trends(category)
                else:
                    # Fallback to simulated trends if no API key
                    self.trending_topics = self._get_simulated_trends(category)
                
                # Update cache with category-specific trends
                self.trending_cache[category] = self.trending_topics
                
                # Set cache to expire in 6 hours
                self.cache_expiry = current_time + timedelta(hours=6)
            except Exception as e:
                print(f"Error fetching trends: {e}")
                # Use cached data if available, otherwise empty list
                self.trending_topics = self.trending_cache.get(category, [])
        else:
            # Use cached trends
            self.trending_topics = self.trending_cache.get(category, [])
        
        return self.trending_topics
    
    def _get_simulated_trends(self, category):
        """Generate simulated trends based on category"""
        trends_by_category = {
            'Electronics': [
                {'tag': '#SmartHome', 'volume': 8500, 'growth': 0.25},
                {'tag': '#AI', 'volume': 12000, 'growth': 0.35},
                {'tag': '#SustainableTech', 'volume': 7500, 'growth': 0.22},
                {'tag': '#5G', 'volume': 6800, 'growth': 0.18}
            ],
            'Furniture': [
                {'tag': '#MinimalistDesign', 'volume': 5600, 'growth': 0.15},
                {'tag': '#HomeOffice', 'volume': 9200, 'growth': 0.28},
                {'tag': '#SustainableHome', 'volume': 4800, 'growth': 0.12},
                {'tag': '#SmartFurniture', 'volume': 3500, 'growth': 0.10}
            ],
            'Office Supplies': [
                {'tag': '#RemoteWork', 'volume': 7800, 'growth': 0.20},
                {'tag': '#Ergonomics', 'volume': 4500, 'growth': 0.15},
                {'tag': '#ProductivityHacks', 'volume': 6300, 'growth': 0.18},
                {'tag': '#HomeWorkspace', 'volume': 5100, 'growth': 0.16}
            ],
            'Technology': [
                {'tag': '#Cybersecurity', 'volume': 9500, 'growth': 0.30},
                {'tag': '#CloudComputing', 'volume': 8800, 'growth': 0.25},
                {'tag': '#MachineLearning', 'volume': 10200, 'growth': 0.32},
                {'tag': '#DataPrivacy', 'volume': 7200, 'growth': 0.22}
            ]
        }
        
        # Return category-specific trends or general trends
        return trends_by_category.get(category, [
            {'tag': '#Innovation', 'volume': 7000, 'growth': 0.20},
            {'tag': '#Sustainability', 'volume': 6500, 'growth': 0.18},
            {'tag': '#DigitalTransformation', 'volume': 8000, 'growth': 0.25}
        ])
    
    def calculate_trend_relevance(self, product_name, product_category, product_description=""):
        """
        Calculate how relevant current trends are to a specific product
        Returns a score and trend adjustment factor
        """
        # Fetch latest trends for the category
        trends = self.fetch_trending_topics(product_category)
        
        if not trends:
            return 0.0, 1.0  # No trends available, return neutral impact
        
        # Combine product information for matching
        product_text = f"{product_name} {product_category} {product_description}".lower()
        
        # Calculate relevance scores for each trend
        relevance_scores = []
        for trend in trends:
            tag = trend['tag'].lower().replace('#', '')
            
            # Simple keyword matching (could be enhanced with NLP)
            if tag in product_text:
                # Direct match gets full relevance
                score = 1.0 * trend['volume'] / 10000  # Normalize by volume
            else:
                # Check for partial matches
                words = re.findall(r'\w+', tag)
                partial_matches = sum(1 for word in words if word in product_text)
                score = (partial_matches / len(words)) * trend['volume'] / 10000 if words else 0
            
            # Weight the score by trend growth
            weighted_score = score * (1 + trend['growth'])
            relevance_scores.append(weighted_score)
        
        # Calculate overall trend relevance (average of top 2 scores if available)
        relevance_scores.sort(reverse=True)
        top_scores = relevance_scores[:2] if len(relevance_scores) >= 2 else relevance_scores
        average_relevance = sum(top_scores) / len(top_scores) if top_scores else 0
        
        # Calculate trend adjustment factor (range: 0.8 to 1.5)
        # Low relevance: slight decrease, high relevance: significant increase
        trend_adjustment = 0.8 + (average_relevance * 0.7)
        trend_adjustment = min(1.5, max(0.8, trend_adjustment))  # Clamp to reasonable range
        
        return average_relevance, trend_adjustment


class MarketInsightPredictor:
    def __init__(self):
        # Default market insights
        self.current_market_insights = {}
        self.trend_analyzer = TrendAnalyzer()

    def set_trend_api_key(self, api_key):
        """Set API key for trend analysis"""
        self.trend_analyzer.set_api_key(api_key)

    def update_market_insights(self, insights):
        """
        Update market insights dynamically
        """
        self.current_market_insights = insights

    def get_market_insights(self, category, product_name="", product_description=""):
        """
        Retrieve market insights for a specific category
        Now incorporates trend analysis
        """
        # If specific category insights exist in current insights, use those
        if category in self.current_market_insights:
            base_insights = self.current_market_insights[category]
        else:
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
            
            base_insights = default_insights.get(category, {
                'growth_rate': 0.05,
                'market_sentiment': 'Neutral',
                'key_trends': ['General Market Trends'],
                'demand_multiplier': 1.0
            })
        
        # Apply trend analysis to adjust insights
        if product_name:
            relevance, trend_adjustment = self.trend_analyzer.calculate_trend_relevance(
                product_name, category, product_description
            )
            
            # If trend relevance is significant, adjust demand multiplier
            if relevance > 0.1:  # Threshold for meaningful trend relevance
                base_insights['demand_multiplier'] = base_insights.get('demand_multiplier', 1.0) * trend_adjustment
                base_insights['trend_relevance'] = relevance
                base_insights['trend_adjustment'] = trend_adjustment
                
                # Add trending hashtags to key trends
                trending_tags = [trend['tag'] for trend in self.trend_analyzer.trending_topics[:3]]
                if trending_tags:
                    base_insights['key_trends'] = trending_tags + base_insights.get('key_trends', [])[:2]
        
        return base_insights


class HistoricalDataAnalyzer:
    """
    Analyze historical sales data to dynamically calculate unit multipliers
    """
    def __init__(self):
        self.category_multipliers = {}
        self.product_multipliers = {}
        self.default_multiplier = 100  # Conservative default
    
    def load_historical_data(self, historical_data_path):
        """
        Load historical sales data from file
        """
        try:
            self.historical_data = pd.read_csv(historical_data_path , encoding="Windows-1252")
            return True
        except Exception as e:
            print(f"Error loading historical data: {e}")
            self.historical_data = pd.DataFrame()
            return False
    
    def calculate_category_multipliers(self):
        """
        Calculate average unit multipliers by category from historical data
        """
        if self.historical_data.empty:
            return
        
        # Group by category and calculate average units per model score
        if 'Category' in self.historical_data.columns and 'ActualUnits' in self.historical_data.columns and 'ModelScore' in self.historical_data.columns:
            category_data = self.historical_data.groupby('Category').apply(
                lambda x: np.median(x['ActualUnits'] / x['ModelScore']) if (x['ModelScore'] > 0).all() else self.default_multiplier
            )
            
            self.category_multipliers = category_data.to_dict()
    
    def calculate_product_multipliers(self):
        """
        Calculate product-specific multipliers from historical data
        """
        if self.historical_data.empty:
            return
        
        # Group by product and calculate average units per model score
        if 'ProductName' in self.historical_data.columns and 'ActualUnits' in self.historical_data.columns and 'ModelScore' in self.historical_data.columns:
            product_data = self.historical_data.groupby('ProductName').apply(
                lambda x: np.median(x['ActualUnits'] / x['ModelScore']) if (x['ModelScore'] > 0).all() else None
            )
            
            # Filter out None values (products without enough data)
            self.product_multipliers = {k: v for k, v in product_data.to_dict().items() if v is not None}
    
    def get_unit_multiplier(self, product_name, category):
        """
        Get appropriate unit multiplier for a product
        Prioritizes product-specific multiplier, then category, then default
        """
        # Try product-specific multiplier first
        if product_name in self.product_multipliers:
            return self.product_multipliers[product_name]
        
        # Try category multiplier next
        if category in self.category_multipliers:
            return self.category_multipliers[category]
        
        # Fall back to default multiplier
        return self.default_multiplier
    
    def add_data_point(self, product_name, category, model_score, actual_units):
        """
        Add new data point to improve future multiplier calculations
        """
        new_data = pd.DataFrame({
            'ProductName': [product_name],
            'Category': [category],
            'ModelScore': [model_score],
            'ActualUnits': [actual_units],
            'Date': [datetime.now().strftime('%Y-%m-%d')]
        })
        
        if self.historical_data.empty:
            self.historical_data = new_data
        else:
            self.historical_data = pd.concat([self.historical_data, new_data], ignore_index=True)
        
        # Recalculate multipliers with new data
        self.calculate_product_multipliers()
        self.calculate_category_multipliers()


class AdvancedDemandPredictor:
    def __init__(self):
        # Load pre-trained model and preprocessing artifacts
        self.model = joblib.load('ml_model/saved_models/demand_prediction_model.joblib')
        self.label_encoders = joblib.load('ml_model/saved_models/label_encoders.joblib')
        self.scaler = joblib.load('ml_model/saved_models/scaler.joblib')
        self.market_insight_predictor = MarketInsightPredictor()
        
        # Initialize historical data analyzer for dynamic unit multipliers
        self.historical_analyzer = HistoricalDataAnalyzer()
        
        # Try to load historical data if available
        try:
            self.historical_analyzer.load_historical_data('ml_model/data/dataset.csv')
            self.historical_analyzer.calculate_category_multipliers()
            self.historical_analyzer.calculate_product_multipliers()
        except Exception as e:
            print(f"Warning: Could not initialize historical data analysis: {e}")
    
    def set_trend_api_key(self, api_key):
        """Set API key for trend analysis"""
        self.market_insight_predictor.set_trend_api_key(api_key)
    
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
    
    def predict_future_demand(self, input_data, market_insights=None, product_description=""):
        """
        Predict demand with comprehensive market insights and convert to actual units
        Now with dynamic unit multiplier calculation and trend analysis
        """
        # Update market insights if provided
        if market_insights:
            self.market_insight_predictor.update_market_insights(market_insights)
        
        # Prepare input data
        prepared_features = self.prepare_input_data(input_data)
        
        # Scale the features
        scaled_input = self.scaler.transform(prepared_features)
        
        # Predict base demand from model
        base_demand_raw = self.model.predict(scaled_input)[0]
        
        # Get dynamic unit multiplier based on product and category
        product_name = input_data.get('Product Name', '')
        category = input_data.get('Category', '')
        unit_multiplier = self.historical_analyzer.get_unit_multiplier(product_name, category)
        
        # Convert to actual units
        base_demand_units = int(base_demand_raw * unit_multiplier)
        
        # Get market insights for the category, including trend analysis
        market_insights_data = self.market_insight_predictor.get_market_insights(
            category, product_name, product_description
        )
        
        # Ensure type conversion and handling for growth rate and demand multiplier
        try:
            growth_rate = float(market_insights_data.get('growth_rate', 0.05))
            demand_multiplier = float(market_insights_data.get('demand_multiplier', 1.0))
        except (ValueError, TypeError):
            # Fallback to default values if conversion fails
            growth_rate = 0.05
            demand_multiplier = 1.0
        
        # Calculate future demand with trend-adjusted multiplier
        future_demand_multiplier = (1 + growth_rate) * demand_multiplier
        
        # Get trend information if available
        trend_relevance = market_insights_data.get('trend_relevance', 0)
        trend_adjustment = market_insights_data.get('trend_adjustment', 1.0)
        
        # Project demand for 6 months
        projected_demand_units = int(base_demand_units * future_demand_multiplier)
        
        # Recommend stock levels with buffer
        recommended_stock = int(projected_demand_units * 1.2)  # 20% buffer
        
        # Determine demand classification
        if projected_demand_units > base_demand_units * 1.3:
            demand_classification = "High Potential"
        elif projected_demand_units > base_demand_units * 1.1:
            demand_classification = "Moderate Growth"
        else:
            demand_classification = "Stable"
        
        # Format numerical values for user-friendly display
        formatted_base_demand = f"{base_demand_units:,} units"
        formatted_projected_demand = f"{projected_demand_units:,} units"
        formatted_recommended_stock = f"{recommended_stock:,} units"
        
        # Build comprehensive result
        result = {
            'base_demand': formatted_base_demand,
            'projected_demand': formatted_projected_demand,
            'recommended_stock': formatted_recommended_stock,
            'demand_classification': demand_classification,
            'raw_values': {
                'base_demand_raw': float(base_demand_raw),
                'projected_demand_raw': float(base_demand_raw * future_demand_multiplier),
                'unit_multiplier': unit_multiplier
            },
            'market_insights': {
                'growth_rate': f"{growth_rate:.1%}",  # Format as percentage
                'market_sentiment': market_insights_data.get('market_sentiment', 'Neutral'),
                'key_trends': market_insights_data.get('key_trends', ['General Market Trends']),
                'demand_multiplier': f"{demand_multiplier:.2f}"
            },
            'prediction_period': {
                'months': 6,
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'end_date': (datetime.now() + timedelta(days=6*30)).strftime('%Y-%m-%d')
            }
        }
        
        # Add trend analysis data if available
        if trend_relevance > 0:
            result['trend_analysis'] = {
                'relevance_score': f"{trend_relevance:.2f}",
                'adjustment_factor': f"{trend_adjustment:.2f}",
                'trending_hashtags': [trend['tag'] for trend in self.market_insight_predictor.trend_analyzer.trending_topics[:3]]
            }
        
        return result
    
    def save_actual_demand(self, product_name, category, model_score, actual_units):
        """
        Save actual demand data to improve future predictions
        """
        self.historical_analyzer.add_data_point(product_name, category, model_score, actual_units)
        
        # Optional: save to file
        try:
            self.historical_analyzer.historical_data.to_csv('data/historical_sales.csv', index=False)
        except Exception as e:
            print(f"Warning: Could not save historical data: {e}")