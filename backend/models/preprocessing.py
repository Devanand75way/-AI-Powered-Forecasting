# sales-forecast-project/backend/models/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Comprehensive data preprocessing function
    """
    # Date Preprocessing
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    # Extract date features
    df['Order_Year'] = df['Order Date'].dt.year
    df['Order_Month'] = df['Order Date'].dt.month
    df['Order_Day'] = df['Order Date'].dt.day
    df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek
    
    # Categorical Encoding
    cat_columns = ['Ship Mode', 'Segment', 'Category', 'Sub-Category']
    le = LabelEncoder()
    
    for col in cat_columns:
        df[f'{col}_Encoded'] = le.fit_transform(df[col])
    
    # Feature Engineering
    df['Profit_Margin'] = df['Profit'] / df['Sales'] * 100
    df['Sales_per_Quantity'] = df['Sales'] / df['Quantity']
    
    # Select Features
    features = [
        'Sales_per_Quantity', 'Quantity', 'Discount', 
        'Profit_Margin', 'Order_Year', 'Order_Month', 
        'Order_Day', 'Order_DayOfWeek',
        'Ship Mode_Encoded', 'Segment_Encoded', 
        'Category_Encoded', 'Sub-Category_Encoded'
    ]
    
    return df, features