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
        self.label_encoders = {}
        self.scaler = None
        
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
        df['Days_to_Ship'] = (df['Ship Date'] - df['Order Date']).dt.days
        
        # 2. Categorical Encoding
        categorical_columns = [
            'Ship Mode', 'Category', 'Sub-Category', 'Product Name'
        ]
        
        # Use Label Encoding for categorical variables
        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # 3. Feature Engineering
        # Calculate profit margin
        df['Profit_Margin'] = df['Profit'] / df['Sales'] * 100
        
        # Calculate sales per quantity
        df['Sales_per_Quantity'] = df['Sales'] / df['Quantity']
        
        # Interaction features
        df['Discount_Quantity_Interaction'] = df['Discount'] * df['Quantity']
        
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
            'Product Name_Encoded',
            'Discount_Quantity_Interaction',
            'Days_to_Ship'
        ]
        
        # Store preprocessed data and features
        self.preprocessed_data = df
        self.features = selected_features
        self.target = 'Sales'
        
        # Scale features
        X = self.preprocessed_data[self.features]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
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
        
        # 6. Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def predict_monthly_demand(self, product_name):
        """
        Predict monthly demand for a specific product
        """
        # Ensure 'Order Date' is in datetime format before extracting the month
        self.raw_data['Order Date'] = pd.to_datetime(self.raw_data['Order Date'], errors='coerce')

        # Group by Month and aggregate sales for the specific product
        monthly_demand = self.raw_data[
            self.raw_data['Product Name'] == product_name
        ].groupby(self.raw_data['Order Date'].dt.month)['Sales'].agg(['sum', 'mean', 'count']).reset_index()
        
        # Rename columns
        monthly_demand.columns = ['Month', 'Total Sales', 'Average Sales', 'Transaction Count']
        
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
    def __init__(self, processor):
        """
        Initialize model with data processor
        """
        self.processor = processor
        self.model = None
        self.performance_metrics = {}
    
    def train_model(self, n_estimators=100, random_state=42):
        """
        Train Random Forest Regressor
        """
        # Prepare training data
        X_train, X_test, y_train, y_test = self.processor.prepare_training_data()
        
        # 7. Model Training
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        
        return self
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        # 8. Model Evaluation
        y_pred = self.model.predict(X_test)
        
        self.performance_metrics = {
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'Mean Squared Error': mean_squared_error(y_test, y_pred),
            'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R-squared Score': r2_score(y_test, y_pred)
        }
        
        return self.performance_metrics
    
    def predict_sales(self, input_data):
        """
        Predict sales for given input data
        
        :param input_data: Dictionary of input features
        :return: Predicted sales
        """
        # Prepare input dataframe
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, le in self.processor.label_encoders.items():
            if col in input_data:
                input_df[f'{col}_Encoded'] = le.transform(input_df[col])
        
        # Calculate engineered features
        input_df['Profit_Margin'] = input_df.get('Profit', 0) / input_df.get('Sales', 1) * 100
        input_df['Sales_per_Quantity'] = input_df.get('Sales', 1) / input_df.get('Quantity', 1)
        input_df['Discount_Quantity_Interaction'] = input_df['Discount'] * input_df['Quantity']
        
        # Select and scale features
        X_input = input_df[self.processor.features]
        X_scaled = self.processor.scaler.transform(X_input)
        
        # Predict
        predicted_sales = self.model.predict(X_scaled)
        
        return predicted_sales[0]
    
    def save_model(self, filepath='./sales_forecast_model.pkl'):
        """
        Save trained model and processor
        """
        # Save model
        joblib.dump(self.model, filepath)
        
        # Save processor metadata
        joblib.dump({
            'label_encoders': self.processor.label_encoders,
            'scaler': self.processor.scaler,
            'features': self.processor.features
        }, filepath.replace('.pkl', '_processor.pkl'))
        
        print(f"Model and processor saved to {filepath}")
        return filepath
    
    def load_model(self, model_path, processor_path=None):
        """
        Load pre-trained model and processor
        """
        # Load model
        self.model = joblib.load(model_path)
        
        # If processor path not provided, generate from model path
        if processor_path is None:
            processor_path = model_path.replace('.pkl', '_processor.pkl')
        
        # Load processor metadata
        processor_data = joblib.load(processor_path)
        
        # Recreate processor attributes
        self.processor.label_encoders = processor_data['label_encoders']
        self.processor.scaler = processor_data['scaler']
        self.processor.features = processor_data['features']
        
        print(f"Model loaded from {model_path}")
        return self

def main():
    # 1. Data Loading and Preprocessing
    processor = SalesDataProcessor('../data/dataset.csv')  
    preprocessed_data = processor.preprocess_data()
    
    # 2. Model Training
    forecast_model = SalesForecastModel(processor)
    forecast_model.train_model()
    
    # 3. Print Performance Metrics
    print("\nModel Performance Metrics:")
    for metric, value in forecast_model.performance_metrics.items():
        print(f"{metric}: {value}")
    
    # 4. Save Model
    model_path = forecast_model.save_model()
    
    # 5. Example Prediction
    # Get a sample product
    example_product = processor.raw_data['Product Name'].unique()[22]
    example_product_category = processor.raw_data[
        processor.raw_data['Product Name'] == example_product
    ]['Category'].iloc[0]
    
    # Prediction input example
    prediction_input = {
        'Product Name': example_product,
        'Category': example_product_category,
        'Ship Mode': 'Standard Class',
        'Quantity': 5,
        'Discount': 0.1,
        'Order_Year': 2025,
        'Order_Month': 1,
        'Order_Day': 10,
        'Order_DayOfWeek': 5,
        'Days_to_Ship': 3,
        'Sales': 300,
        'Profit': 100
    }
    
    # Predict sales
    predicted_sales = forecast_model.predict_sales(prediction_input)
    print(f"\nPredicted Sales for {example_product}: ${predicted_sales:.2f}")
    
    # 6. Monthly Demand Prediction
    monthly_demand = processor.predict_monthly_demand(example_product)
    print("\nMonthly Demand Details:")
    print(monthly_demand)

if __name__ == "__main__":
    main()