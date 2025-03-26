import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class SalesDataProcessor:
    def __init__(self, file_path):
        """
        Initialize the data processor with the sales dataset
        """
        self.raw_data = pd.read_csv(file_path, encoding='Windows-1252')
        self.preprocessed_data = None
        self.features = None
        self.target = None
        
    def preprocess_data(self):
        """
        Comprehensive data preprocessing steps
        """
        # Create a copy of the original dataframe
        df = self.raw_data.copy()
        
        # 1. Date Preprocessing
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        # Extract additional date features
        df['Order_Year'] = df['Order Date'].dt.year
        df['Order_Month'] = df['Order Date'].dt.month
        df['Order_Day'] = df['Order Date'].dt.day
        df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek
        
        # 2. Categorical Encoding
        categorical_columns = [
            'Ship Mode', 'Category', 'Sub-Category'
        ]
        
        # Use Label Encoding for categorical variables
        le = LabelEncoder()
        for col in categorical_columns:
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
        
        # 3. Feature Engineering
        # Calculate profit margin
        df['Profit_Margin'] = df['Profit'] / df['Sales'] * 100
        
        # Calculate sales per quantity
        df['Sales_per_Quantity'] = df['Sales'] / df['Quantity']
        
        # 4. Feature Selection
        selected_features = [
            'Quantity', 
            'Discount', 
            'Profit_Margin', 
            'Sales_per_Quantity',
            'Order_Year', 
            'Order_Month', 
            'Order_Day', 
            'Order_DayOfWeek',
            'Ship Mode_Encoded', 
            'Category_Encoded', 
            'Sub-Category_Encoded'
        ]
        
        # Store preprocessed data and features
        self.preprocessed_data = df
        self.features = selected_features
        self.target = 'Sales'
        
        return self.preprocessed_data

    def prepare_training_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for model training
        """
        if self.preprocessed_data is None:
            raise ValueError("Please preprocess data first using preprocess_data() method")
        
        # Select features and target
        X = self.preprocessed_data[self.features]
        y = self.preprocessed_data[self.target]
        
        # 5. Scale Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 6. Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test, scaler

    def predict_monthly_demand(self, product_name):
          """
          Predict monthly demand for a specific product
          """
          # Convert 'Order Date' to datetime format
          self.raw_data['Order Date'] = pd.to_datetime(self.raw_data['Order Date'])

          # Filter by Product Name and Group by Month
          monthly_demand = self.raw_data[
               self.raw_data['Product Name'] == product_name
          ].groupby(self.raw_data['Order Date'].dt.month)['Sales'].sum().reset_index()
          
          print(monthly_demand)

          # Rename columns
          monthly_demand.columns = ['Month', 'Total Sales']

          # Find months with highest and lowest demand
          highest_demand_month = monthly_demand.loc[monthly_demand['Total Sales'].idxmax(), 'Month']
          lowest_demand_month = monthly_demand.loc[monthly_demand['Total Sales'].idxmin(), 'Month']

          # Convert month numbers to month names
          month_names = {
               1: 'January', 2: 'February', 3: 'March', 4: 'April', 
               5: 'May', 6: 'June', 7: 'July', 8: 'August', 
               9: 'September', 10: 'October', 11: 'November', 12: 'December'
          }

          return {
               'product_name': product_name,
               'highest_demand_month': month_names[highest_demand_month],
               'lowest_demand_month': month_names[lowest_demand_month],
               'monthly_demand_details': monthly_demand.to_dict(orient='records')
          }

class SalesForecastModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize model with training and testing data
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.performance_metrics = {}
    
    def train_model(self, n_estimators=100, random_state=42):
        """
        Train Random Forest Regressor
        """
        # 7. Model Training
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.model.fit(self.X_train, self.y_train)
        
        return self
    
    def evaluate_model(self):
        """
        Evaluate model performance
        """
        # 8. Model Evaluation
        y_pred = self.model.predict(self.X_test)
        
        self.performance_metrics = {
            'Mean Absolute Error': mean_absolute_error(self.y_test, y_pred),
            'Mean Squared Error': mean_squared_error(self.y_test, y_pred),
            'Root Mean Squared Error': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'R-squared Score': r2_score(self.y_test, y_pred)
        }
        
        return self.performance_metrics
    
    def save_model(self, filepath='./sales_forecast_model.pkl'):
        """
        Save trained model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath='./sales_forecast_model.pkl'):
        """
        Load pre-trained model
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self

def main():
    # 1. Data Loading and Preprocessing
    processor = SalesDataProcessor('../data/dataset.csv')  
    preprocessed_data = processor.preprocess_data()
    
    # 2. Prepare Training Data
    X_train, X_test, y_train, y_test, scaler = processor.prepare_training_data()
    
    # 3. Model Training
    forecast_model = SalesForecastModel(X_train, X_test, y_train, y_test)
    forecast_model.train_model()
    
    # 4. Model Evaluation
    performance_metrics = forecast_model.evaluate_model()
    print("\nModel Performance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")
    
    # 5. Save Model
    model_path = forecast_model.save_model()
    
    # 6. Example of predicting monthly demand for a product
    example_product = processor.raw_data['Product Name'].unique()[1]
    monthly_demand = processor.predict_monthly_demand(example_product)
    print("\nMonthly Demand for Product:", monthly_demand)

if __name__ == "__main__":
    main()