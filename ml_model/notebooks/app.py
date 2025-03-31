import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load and preprocess the dataset
def preprocess_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path , encoding="Windows-1252")
    
    # Feature extraction and engineering
    # Select relevant features for demand prediction
    features = [
        'Segment', 'Country', 'City', 'State', 'Region', 
        'Category', 'Sub-Category', 'Product Name', 
        'Sales', 'Quantity', 'Discount'
    ]
    
    # Create a working dataframe
    working_df = df[features].copy()
    
    # Encode categorical variables
    categorical_columns = [
        'Segment', 'Country', 'City', 'State', 'Region', 
        'Category', 'Sub-Category', 'Product Name'
    ]
    
    # Label Encoding for categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        working_df[col] = le.fit_transform(working_df[col].astype(str))
        label_encoders[col] = le
    
    # Split features and target
    # We'll use Quantity as our target variable for demand prediction
    X = working_df.drop('Quantity', axis=1)
    y = working_df['Quantity']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'scaler': scaler
    }

# Train the model
def train_demand_prediction_model(preprocessed_data):
    # Use Random Forest Regressor for demand prediction
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    
    # Train the model
    model.fit(
        preprocessed_data['X_train'], 
        preprocessed_data['y_train']
    )
    
    # Evaluate the model
    train_score = model.score(
        preprocessed_data['X_train'], 
        preprocessed_data['y_train']
    )
    test_score = model.score(
        preprocessed_data['X_test'], 
        preprocessed_data['y_test']
    )
    
    print(f"Train R² Score: {train_score}")
    print(f"Test R² Score: {test_score}")
    
    return model

# Main execution
def main():
    # Preprocess the data
    preprocessed_data = preprocess_data('ml_model/data/dataset.csv')
    
    # Train the model
    demand_model = train_demand_prediction_model(preprocessed_data)
    
    # Save the model and preprocessing artifacts
    joblib.dump(demand_model, 'demand_prediction_model.joblib')
    joblib.dump(preprocessed_data['label_encoders'], 'label_encoders.joblib')
    joblib.dump(preprocessed_data['scaler'], 'scaler.joblib')

if __name__ == '__main__':
    main()