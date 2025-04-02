# Import the class
# Assuming the class is in a file called data_integration.py
from data_integration import DataIntegrationModule

# Create an instance of the class
di_module = DataIntegrationModule(
    sales_data_path="ml_model/data/dataset.csv",
    external_data_path="ml_model/data/external_factors.csv"  # Optional
)

# Step 1: Load data
sales_df, external_df = di_module.load_data()
print(f"Sales data shape: {sales_df.shape}")
if external_df is not None:
    print(f"External data shape: {external_df.shape}")

# Step 2: Engineer date features (if needed separately)
sales_with_date_features = di_module.engineer_date_features(sales_df)
print(f"Date features added: {list(set(sales_with_date_features.columns) - set(sales_df.columns))}")

# Step 3: Merge sales with external data
integrated_data = di_module.merge_sales_with_external()
print(f"Integrated data shape: {integrated_data.shape}")

# Step 4: Aggregate sales by date and product
aggregated_sales = di_module.aggregate_sales_by_date(group_by_product=True)
aggregated_sales = di_module.engineer_date_features(aggregated_sales) 
print(f"Aggregated sales shape: {aggregated_sales.shape}")

# Step 5: Calculate lagged features
sales_with_lags = di_module.calculate_lagged_features(
    aggregated_sales,
    group_cols=['Product Name'],
    target_col='Quantity',
    lags=[1, 7, 14, 28]
)
print(f"Lagged features added: {list(set(sales_with_lags.columns) - set(aggregated_sales.columns))}")

# Step 6: Calculate growth rates
growth_rates = di_module.calculate_growth_rates(
    aggregated_sales,
    value_cols=['Quantity', 'Sales'],
    period_type='MoM'  # Month-over-Month
)
print(f"Growth rate columns: {[col for col in growth_rates.columns if 'growth' in col]}")

# Step 7: Generate features for a specific date range
future_dates = di_module.generate_date_range_features(
    start_date="2023-04-01",
    end_date="2023-04-30"
)
print(f"Generated future date features with shape: {future_dates.shape}")

# Step 8: Prepare features for modeling
model_features = di_module.prepare_features_for_model(
    # Optional date range for prediction
    start_date="2023-01-01",
    end_date="2023-03-31",
    include_product=False
)
print(f"Model features shape: {model_features.shape}")
print(f"Model feature columns: {model_features.columns.tolist()[:5]}...")  # First 5 columns

# Example of how to use the prepared features for prediction
# (This assumes you have a trained model)
# prediction = trained_model.predict(model_features)