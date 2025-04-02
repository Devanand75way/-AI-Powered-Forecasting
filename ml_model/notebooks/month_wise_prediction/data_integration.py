import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataIntegrationModule:
    """
    Module for integrating external data with sales data for improved demand prediction.
    This handles merging, feature engineering, and preprocessing for the combined dataset.
    """
    
    def __init__(self, sales_data_path="ml_model/data/dataset.csv", external_data_path="ml_model/data/external_factors.csv"):
        """
        Initialize the data integration module
        
        Parameters:
        sales_data_path (str): Path to the sales data CSV file
        external_data_path (str): Path to external data CSV file (optional)
        """
        self.sales_data_path = sales_data_path
        self.external_data_path = external_data_path
        self.sales_df = None
        self.external_df = None
        self.integrated_df = None
    
    def load_data(self):
        """Load sales and external data"""
        # Load sales data
        self.sales_df = pd.read_csv(self.sales_data_path, encoding="Windows-1252")
        
        # Convert date columns to datetime
        if 'Order Date' in self.sales_df.columns:
            self.sales_df['Order Date'] = pd.to_datetime(self.sales_df['Order Date'])
        
        # Load external data if provided
        if self.external_data_path and os.path.exists(self.external_data_path):
            self.external_df = pd.read_csv(self.external_data_path)
            
            # Convert date columns to datetime
            if 'Date' in self.external_df.columns:
                self.external_df['Date'] = pd.to_datetime(self.external_df['Date'])
        
        return self.sales_df, self.external_df
    
    def engineer_date_features(self, df, date_column='Order Date'):
        """
        Engineer date-related features from a date column
        
        Parameters:
        df (DataFrame): Input DataFrame
        date_column (str): Name of the date column
        
        Returns:
        DataFrame: DataFrame with additional date features
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Extract basic date components
        result['Year'] = result[date_column].dt.year
        result['Month'] = result[date_column].dt.month
        result['Day'] = result[date_column].dt.day
        result['DayOfWeek'] = result[date_column].dt.dayofweek
        result['Quarter'] = result[date_column].dt.quarter
        
        # Extract week number and day of year
        result['WeekOfYear'] = result[date_column].dt.isocalendar().week
        result['DayOfYear'] = result[date_column].dt.dayofyear
        
        # Is weekend/weekday
        result['IsWeekend'] = result['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Is month start/end
        result['IsMonthStart'] = result[date_column].dt.is_month_start.astype(int)
        result['IsMonthEnd'] = result[date_column].dt.is_month_end.astype(int)
        
        # Is quarter start/end
        result['IsQuarterStart'] = result[date_column].dt.is_quarter_start.astype(int)
        result['IsQuarterEnd'] = result[date_column].dt.is_quarter_end.astype(int)
        
        # Season in Northern Hemisphere
        # Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov
        result['Season'] = result['Month'].apply(
            lambda month: 0 if month in [12, 1, 2] else  # Winter
                         1 if month in [3, 4, 5] else    # Spring
                         2 if month in [6, 7, 8] else    # Summer
                         3                               # Fall
        )
        
        return result
    
    def merge_sales_with_external(self):
        """
        Merge sales data with external data based on date
        
        Returns:
        DataFrame: Integrated dataset with sales and external data
        """
        if self.sales_df is None or self.external_df is None:
            self.load_data()
        
        # Ensure date columns are prepared for merging
        self.sales_df['Date_Key'] = self.sales_df['Order Date'].dt.strftime('%Y-%m-%d')
        self.external_df['Date_Key'] = self.external_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Merge the datasets
        self.integrated_df = pd.merge(
            self.sales_df, 
            self.external_df.drop('Date', axis=1), 
            on='Date_Key', 
            how='left'
        )
        
        # Handle missing values from the merge
        # For demonstration, we'll fill with zeros but you might want more sophisticated imputation
        for col in self.external_df.columns:
            if col not in ['Date', 'Date_Key'] and col in self.integrated_df.columns:
                self.integrated_df[col] = self.integrated_df[col].fillna(0)
        
        return self.integrated_df
    
    def aggregate_sales_by_date(self, group_by_product=True):
        """
        Aggregate sales data by date, optionally grouping by product as well
        
        Parameters:
        group_by_product (bool): Whether to group by product as well as date
        
        Returns:
        DataFrame: Aggregated sales data
        """
        if self.sales_df is None:
            self.load_data()
        
        # Create a copy for aggregation
        agg_df = self.sales_df.copy()
        
        # Define groupby columns
        if group_by_product:
            groupby_cols = ['Order Date', 'Product Name']
        else:
            groupby_cols = ['Order Date']
        
        # Aggregate sales metrics
        aggregated = agg_df.groupby(groupby_cols).agg({
            'Quantity': 'sum',
            'Sales': 'sum',
            'Discount': 'mean'  # Average discount for the day/product
        }).reset_index()
        
        return aggregated
    
    def calculate_lagged_features(self, df, group_cols=['Product Name'], target_col='Quantity', lags=[1, 7, 14, 28]):
        """
        Calculate lagged features for time series forecasting
        
        Parameters:
        df (DataFrame): Input DataFrame with time series data
        group_cols (list): Columns to group by (e.g., product)
        target_col (str): Target column to create lags for
        lags (list): List of lag periods to create
        
        Returns:
        DataFrame: DataFrame with additional lag features
        """
        # Make a copy to avoid modifying the original
        result = df.copy().sort_values(['Order Date'] + group_cols)
        
        # Create lag features
        for lag in lags:
            for col in group_cols:
                result[f'{target_col}_lag_{lag}'] = result.groupby(col)[target_col].shift(lag)
        
        # Create rolling window features
        for window in [7, 14, 30]:
            for col in group_cols:
                # Rolling mean
                result[f'{target_col}_roll_mean_{window}'] = result.groupby(col)[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation (volatility)
                result[f'{target_col}_roll_std_{window}'] = result.groupby(col)[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        
        # Fill NA values introduced by lagging
        for col in result.columns:
            if '_lag_' in col or '_roll_' in col:
                result[col] = result[col].fillna(0)
        
        return result
    
    def calculate_growth_rates(self, df, value_cols=['Quantity', 'Sales'], period_type='MoM'):
     """
     Calculate growth rates for specified value columns
     
     Parameters:
     df (DataFrame): Input DataFrame
     value_cols (list): Value columns to calculate growth rates for
     period_type (str): Period type - 'MoM' (Month-over-Month), 'YoY' (Year-over-Year)
     
     Returns:
     DataFrame: DataFrame with growth rate columns added
     """
     # Make a copy
     result = df.copy()
     
     # Check if date columns exist, if not, try to create them from Order Date
     if 'Order Date' in result.columns:
          if 'Year' not in result.columns:
               result['Year'] = result['Order Date'].dt.year
          if 'Month' not in result.columns:
               result['Month'] = result['Order Date'].dt.month
     else:
          raise ValueError("DataFrame must have either 'Year'/'Month' columns or 'Order Date' column")
     
     # Determine shift value based on period type
     if period_type == 'MoM':
          shift_value = 1
          groupby_cols = ['Year', 'Month']
     elif period_type == 'YoY':
          shift_value = 12
          groupby_cols = ['Month']
     else:
          raise ValueError("period_type must be 'MoM' or 'YoY'")
     
     # Calculate growth rates
     for col in value_cols:
          # Group by and calculate previous period values
          grouped = result.groupby(groupby_cols)[col].sum().reset_index()
          grouped[f'prev_{col}'] = grouped[col].shift(shift_value)
          
          # Calculate growth rate
          grouped[f'{col}_growth'] = (
               (grouped[col] - grouped[f'prev_{col}']) / grouped[f'prev_{col}'] * 100
          ).fillna(0)
          
          # Drop the temporary column
          grouped = grouped.drop(f'prev_{col}', axis=1)
     
     return grouped
    
    def generate_date_range_features(self, start_date, end_date):
        """
        Generate features for a specific date range (useful for predictions)
        
        Parameters:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
        Returns:
        DataFrame: DataFrame with date features for the specified range
        """
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create DataFrame
        date_df = pd.DataFrame({'Order Date': date_range})
        
        # Add date features
        date_df = self.engineer_date_features(date_df)
        
        # If external data is available, merge it
        if self.external_df is not None:
            date_df['Date_Key'] = date_df['Order Date'].dt.strftime('%Y-%m-%d')
            self.external_df['Date_Key'] = self.external_df['Date'].dt.strftime('%Y-%m-%d')
            
            date_df = pd.merge(
                date_df,
                self.external_df.drop('Date', axis=1),
                on='Date_Key',
                how='left'
            )
            
            # Fill missing values
            for col in self.external_df.columns:
                if col not in ['Date', 'Date_Key'] and col in date_df.columns:
                    date_df[col] = date_df[col].fillna(0)
        
        return date_df
    
    def prepare_features_for_model(self, start_date=None, end_date=None, include_product=True):
        """
        Prepare integrated features for modeling
        
        Parameters:
        start_date (str): Start date in YYYY-MM-DD format (optional)
        end_date (str): End date in YYYY-MM-DD format (optional)
        include_product (bool): Whether to include product-level features
        
        Returns:
        DataFrame: DataFrame with features ready for modeling
        """
        # Load data if not already loaded
        if self.sales_df is None:
            self.load_data()
        
        # If start_date and end_date are provided, generate date range features
        if start_date and end_date:
            base_df = self.generate_date_range_features(start_date, end_date)
        else:
            # Otherwise, use the integrated data if available
            if self.integrated_df is None:
                # If not available, try to merge sales with external data
                if self.external_df is not None:
                    self.merge_sales_with_external()
                    base_df = self.integrated_df.copy()
                else:
                    # If no external data, just use sales data
                    base_df = self.sales_df.copy()
            else:
                base_df = self.integrated_df.copy()
        
        # Add date features if not already present
        if 'Year' not in base_df.columns:
            base_df = self.engineer_date_features(base_df)
        
        # Determine whether to aggregate by date only or by date and product
        if include_product:
            # Ensure we have product level aggregation
            if 'Product Name' in base_df.columns:
                # Aggregate data by date and product
                agg_df = self.aggregate_sales_by_date(group_by_product=True)
                
                # Calculate lagged features for product level
                feature_df = self.calculate_lagged_features(
                    agg_df,
                    group_cols=['Product Name'],
                    target_col='Quantity',
                    lags=[1, 7, 14, 28]
                )
            else:
                raise ValueError("Product data not available but include_product is True")
        else:
            # Aggregate data by date only
            agg_df = self.aggregate_sales_by_date(group_by_product=False)
            
            # Calculate lagged features for overall sales
            feature_df = self.calculate_lagged_features(
                agg_df,
                group_cols=[],  # No group cols for date-only aggregation
                target_col='Quantity',
                lags=[1, 7, 14, 28]
            )
        
        # Remove rows with NaN values introduced by lagging
        feature_df = feature_df.dropna()
        
        # Add growth rate features
        month_growth = self.calculate_growth_rates(
            feature_df,
            value_cols=['Quantity', 'Sales'],
            period_type='MoM'
        )
        
        # Merge back with year-month identifier if needed
        # This depends on how you want to use the growth rates
        
        # Select relevant columns for modeling
        # This depends on the specific modeling needs
        feature_cols = [col for col in feature_df.columns 
                       if col not in ['Order Date', 'Date_Key'] and
                          not col.endswith('_growth')]  # Exclude growth rates if they're in a separate DF
        
        model_features = feature_df[feature_cols].copy()
        
        # Handle categorical features if needed
        if include_product and 'Product Name' in model_features.columns:
            # One-hot encode product names
            model_features = pd.get_dummies(model_features, columns=['Product Name'], prefix='prod')
        
        # Scale numerical features if needed
        # This would typically be done with sklearn's StandardScaler or MinMaxScaler
        
        return model_features
