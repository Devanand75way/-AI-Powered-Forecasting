# sales-forecast-project/backend/models/sales_forecast_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)
import xgboost as xgb

class SalesForecastModel:
    def __init__(self, preprocessed_data, features, target):
        self.data = preprocessed_data
        self.features = features
        self.target = target
        self.model = None
        self.performance_metrics = {}

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data[self.features]
        y = self.data[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_random_forest(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        
        self.performance_metrics = {
            'mean_absolute_error': mean_absolute_error(self.y_test, y_pred),
            'mean_squared_error': mean_squared_error(self.y_test, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'r2_score': r2_score(self.y_test, y_pred)
        }
        
        return self.performance_metrics

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model has not been trained. Train the model first.")
        
        return self.model.predict(input_data)