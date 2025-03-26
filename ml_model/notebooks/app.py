import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import chardet

def detect_encoding(file_path):
    """
    Detect the file encoding
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def read_csv_with_encoding(file_path):
    """
    Read CSV file with detected or specified encoding
    """
    # First, try to detect encoding
    try:
        encoding = detect_encoding(file_path)
        print(f"Detected Encoding: {encoding}")
    except Exception as e:
        print(f"Encoding detection failed: {e}")
        encoding = 'utf-8'  # Fallback to utf-8

    # List of encodings to try
    encodings_to_try = [
        encoding,  # Detected encoding
        'utf-8',
        'latin-1',
        'iso-8859-1',
        'cp1252'
    ]

    for enc in encodings_to_try:
        try:
            # Try reading with the current encoding
            df = pd.read_csv(file_path, encoding=enc, errors='replace')
            print(f"Successfully read file with {enc} encoding")
            return df
        except Exception as e:
            print(f"Failed to read with {enc} encoding: {e}")
    
    # If all attempts fail
    raise ValueError("Could not read the CSV file with any of the attempted encodings")

class SalesDataProcessor:
    def __init__(self, file_path):
        """
        Initialize the data processor with the sales dataset
        """
        self.raw_data = pd.read_csv(file_path , encoding='Windows-1252')
        self.preprocessed_data = None
        self.features = None
        self.target = None

        # self.raw_data = read_csv_with_encoding(file_path)
        # self.preprocessed_data = None
        # self.features = None
        # self.target = None
        
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
            'Ship Mode', 'Segment', 'Country', 
            'Category', 'Sub-Category', 'Region'
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
            'Segment_Encoded', 
            'Category_Encoded', 
            'Sub-Category_Encoded',
            'Region_Encoded'
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

# Main Execution
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
    
    # 6. Optional: Load and Verify Model
    loaded_model = SalesForecastModel(X_train, X_test, y_train, y_test)
    loaded_model.load_model(model_path)

if __name__ == "__main__":
    main()