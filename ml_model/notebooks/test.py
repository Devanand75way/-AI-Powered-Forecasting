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
        """
        Split data into training and testing sets
        """
        X = self.data[self.features]
        y = self.data[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_random_forest(self, n_estimators=100, random_state=42):
        """
        Train Random Forest Regressor
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        return self

    def train_xgboost(self, learning_rate=0.1, n_estimators=100):
        """
        Train XGBoost Regressor
        """
        self.model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        y_pred = self.model.predict(self.X_test)
        
        self.performance_metrics = {
            'mean_absolute_error': mean_absolute_error(self.y_test, y_pred),
            'mean_squared_error': mean_squared_error(self.y_test, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'r2_score': r2_score(self.y_test, y_pred),
            'cross_val_scores': cross_val_score(
                self.model, 
                self.X_train, 
                self.y_train, 
                cv=5, 
                scoring='neg_mean_squared_error'
            ).mean()
        }
        
        return self.performance_metrics

    def feature_importance(self):
        """
        Extract and rank feature importances
        """
        importances = self.model.feature_importances_
        feature_importances = sorted(
            zip(self.features, importances), 
            key=lambda x: x[1], 
            reverse=True
        )
        return feature_importances

    def save_model(self, filepath='ml_model/saved_models/sales_forecast_model.pkl'):
        """
        Save trained model
        """
        joblib.dump(self.model, filepath)
        return filepath

    def load_model(self, filepath='ml_model/saved_models/sales_forecast_model.pkl'):
        """
        Load pre-trained model
        """
        self.model = joblib.load(filepath)
        return self

    def predict(self, input_data):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Train the model first.")
        
        return self.model.predict(input_data)

def generate_prediction_report(model):
    """
    Generate a comprehensive prediction report
    """
    report = {
        'performance_metrics': model.performance_metrics,
        'feature_importances': model.feature_importance(),
        'model_type': type(model.model).__name__
    }
    return report

# Example Usage
# Assuming preprocessed_df, selected_features, sales_target are from previous preprocessing
# sales_model = SalesForecastModel(preprocessed_df, selected_features, sales_target)
# sales_model.split_data()
# sales_model.train_random_forest()
# model_performance = sales_model.evaluate_model()
# prediction_report = generate_prediction_report(sales_model)