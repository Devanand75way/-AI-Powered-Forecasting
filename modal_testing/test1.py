import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import requests
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("sales_prediction.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

class EnhancedSalesPredictionModel:
    def __init__(self):
        # Initialize data storage and model components
        self.external_data = {}
        self.model = None
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
    
    def fetch_weather_data(self, location):
        """
        Fetch weather data from an external API with robust error handling
        """
        try:
            api_key = '456da9c22bc74effa18104242252603'
            url = f'http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=yes'
            
            # Use timeout to prevent hanging
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            weather_data = response.json()
            
            # Validate response structure
            if 'current' not in weather_data:
                self.logger.warning("Unexpected weather data structure")
                raise ValueError("Invalid weather data response")
            
            # Extract weather features with safe get method
            current = weather_data['current']
            self.external_data['temperature'] = current.get('temp_c', 0)
            self.external_data['humidity'] = current.get('humidity', 0)
            self.external_data['precipitation'] = current.get('precip_mm', 0)
            
            self.logger.info(f"Successfully fetched weather data for {location}")
        
        except requests.RequestException as e:
            self.logger.error(f"Network error fetching weather data: {e}")
            # Fallback to default values
            self.external_data.update({
                'temperature': 0,
                'humidity': 0,
                'precipitation': 0
            })
        except ValueError as e:
            self.logger.error(f"Data validation error: {e}")
            self.external_data.update({
                'temperature': 0,
                'humidity': 0,
                'precipitation': 0
            })
        except Exception as e:
            self.logger.error(f"Unexpected error in weather data fetch: {e}")
            self.external_data.update({
                'temperature': 0,
                'humidity': 0,
                'precipitation': 0
            })
    
    def analyze_social_media_trends(self, product_keywords):
        """
        Analyze social media trends with robust error handling
        """
        try:
            from serpapi import GoogleSearch
            
            params = {
                "engine": "google_trends",
                "q": product_keywords,
                "data_type": "TIMESERIES",
                "api_key": "YOUR_SERPAPI_KEY"  # Replace with your actual key
            }

            search = GoogleSearch(params)
            response = search.get_dict()
            
            # Safely extract trend data
            timeline_data = response.get('interest_over_time', {}).get('timeline_data', [])
            
            self.external_data['sentiment_score'] = (
                timeline_data[0].get('value', 0) if timeline_data else 0
            )
            self.external_data['social_engagement'] = len(timeline_data)
            
            self.logger.info(f"Successfully analyzed trends for {product_keywords}")
        
        except ImportError:
            self.logger.error("SerpAPI library not installed. Please install with: pip install google-search-results")
            self.external_data.update({
                'sentiment_score': 0,
                'social_engagement': 0
            })
        except Exception as e:
            self.logger.error(f"Error analyzing social media trends: {e}")
            self.external_data.update({
                'sentiment_score': 0,
                'social_engagement': 0
            })
    
    def prepare_enhanced_dataset(self, historical_sales_data):
        """
        Prepare dataset with comprehensive error handling and feature engineering
        """
        try:
            # Create a copy to avoid modifying the original dataframe
            data = historical_sales_data.copy()
            
            # Validate essential columns
            required_columns = ['Sales', 'Order Date', 'Ship Date']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Drop irrelevant columns
            columns_to_drop = [
                'Row ID', 'Order ID', 'Customer ID', 'Customer Name', 
                'Product ID', 'Product Name', 'Country', 'City', 'State'
            ]
            data.drop(columns=[col for col in columns_to_drop if col in data.columns], 
                      inplace=True)

            # Convert date columns to datetime with error handling
            date_columns = ['Order Date', 'Ship Date']
            for col in date_columns:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                except Exception as e:
                    self.logger.error(f"Error converting {col} to datetime: {e}")
            
            # Drop rows with invalid dates
            data.dropna(subset=date_columns, inplace=True)

            # Extract date features
            for prefix in ['Order', 'Ship']:
                date_col = f'{prefix} Date'
                data[f'{prefix}_Year'] = data[date_col].dt.year
                data[f'{prefix}_Month'] = data[date_col].dt.month
                data[f'{prefix}_Day'] = data[date_col].dt.day
                data[f'{prefix}_DayOfWeek'] = data[date_col].dt.dayofweek

            # Drop original date columns
            data.drop(columns=date_columns, inplace=True)

            # Handle Postal Code
            data['Postal Code'] = data['Postal Code'].fillna(0).astype(int)

            # One-Hot Encode categorical variables
            categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
            categorical_cols = [col for col in categorical_cols if col in data.columns]
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

            # Merge with external data
            external_df = pd.DataFrame([self.external_data])
            combined_data = pd.concat([data, external_df], axis=1)

            # Handle missing values
            combined_data.fillna(0, inplace=True)

            self.logger.info("Successfully prepared enhanced dataset")
            return combined_data
        
        except Exception as e:
            self.logger.error(f"Error in dataset preparation: {e}")
            raise
    
    def train_enhanced_model(self, combined_data):
        """
        Train machine learning model with comprehensive error handling
        """
        try:
            # Validate 'Sales' column
            if 'Sales' not in combined_data.columns:
                raise ValueError("'Sales' column is missing from the dataset")
            
            # Separate features and target
            X = combined_data.drop('Sales', axis=1)
            y = combined_data['Sales']
            
            # Store column names before scaling
            self.X_train_columns = list(X.columns)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest Regressor
            self.model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1  # Use all available cores
            )
            self.model.fit(self.X_train, self.y_train)
            
            # Evaluate model
            train_score = self.model.score(self.X_train, self.y_train)
            test_score = self.model.score(self.X_test, self.y_test)
            
            self.logger.info(f"Model Training Score: {train_score}")
            self.logger.info(f"Model Testing Score: {test_score}")
            
            return train_score, test_score
        
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    def predict_sales(self, new_data):
        """
        Predict sales with robust error handling and flexible input types
        """
        try:
            # Validate model is trained
            if self.model is None:
                raise RuntimeError("Model has not been trained. Call train_enhanced_model first.")
            
            # Log input data type and details for debugging
            self.logger.info(f"Input data type: {type(new_data)}")
            
            # Ensure we have training column information
            if not hasattr(self, 'X_train') or not hasattr(self, 'X_train_columns'):
                raise RuntimeError("Training data column information is missing")
            
            # Convert input to DataFrame with flexible handling
            if isinstance(new_data, dict):
                new_data_df = pd.DataFrame([new_data])
            elif isinstance(new_data, (list, np.ndarray)):
                if len(new_data) == 0:
                    raise ValueError("Empty input data")
                
                # If it's a list of dictionaries
                if isinstance(new_data[0], dict):
                    new_data_df = pd.DataFrame(new_data)
                else:
                    # If it's a NumPy array or list of values
                    column_names = self.X_train_columns
                    
                    # Convert NumPy array to DataFrame
                    if isinstance(new_data, np.ndarray):
                        new_data_df = pd.DataFrame(new_data, columns=column_names)
                    else:
                        # List of values
                        new_data_df = pd.DataFrame([new_data], columns=column_names)
            
            elif isinstance(new_data, pd.DataFrame):
                new_data_df = new_data.copy()
            else:
                raise TypeError(f"Unsupported input type: {type(new_data)}")
            
            # Log DataFrame details
            self.logger.info(f"Input DataFrame columns: {new_data_df.columns}")
            self.logger.info(f"Input DataFrame shape: {new_data_df.shape}")
            
            # Ensure columns match training data
            expected_columns = self.X_train_columns
            
            # Add missing columns with default values
            for col in expected_columns:
                if col not in new_data_df.columns:
                    new_data_df[col] = 0
            
            # Reorder and select only expected columns
            new_data_df = new_data_df[expected_columns]
            
            # Scale input data
            input_scaled = self.scaler.transform(new_data_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)
            
            self.logger.info(f"Sales prediction: {prediction[0]}")
            return prediction[0]
        
        except Exception as e:
            self.logger.error(f"Error in sales prediction: {e}")
            raise

def main():
    # Initialize the enhanced sales prediction model
    sales_predictor = EnhancedSalesPredictionModel()
    
    try:
        # Fetch external data
        sales_predictor.fetch_weather_data('India')
        sales_predictor.analyze_social_media_trends('')
        
        # Load historical sales data
        historical_data = pd.read_csv('./dataset.csv', encoding='Windows-1252')
        
        # Prepare enhanced dataset
        enhanced_dataset = sales_predictor.prepare_enhanced_dataset(historical_data)
        
        # Train the enhanced model
        sales_predictor.train_enhanced_model(enhanced_dataset)
        
        # Example prediction
        new_data = {
            "temperature": 5,
            "humidity": 60,
            "precipitation": 5,
            "sentiment_score": 0.45,
            "social_engagement": 600
        }

        # Make prediction
        sales_prediction = sales_predictor.predict_sales(new_data)
        print("Predicted Sales:", sales_prediction)
    
    except Exception as e:
        logging.error(f"Critical error in main execution: {e}")

if __name__ == "__main__":
    main()