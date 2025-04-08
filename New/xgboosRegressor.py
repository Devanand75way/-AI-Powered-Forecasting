import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

# Load data
df = pd.read_csv("New/processed_smartphone_sales_with_discount.csv", parse_dates=['Date'])

# Date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Encode categorical variables
for col in ['Product Name', 'RAM', 'Memory']:
    df[col] = df[col].astype('category').cat.codes

# Feature Engineering
df['Discount_Price_Impact'] = df['Discount (%)'] * df['Price (USD)'] / 100
df['Marketing_Stock_Interaction'] = df['Marketing Spend'] * df['Stock Available']
df['Effective_Price'] = df['Price (USD)'] - df['Discount_Price_Impact']
df['Price_Ratio'] = df['Price (USD)'] / df['Competitor Price']

# Final features
features = [
    'Product Name', 'RAM', 'Memory', 'Price (USD)', 'Competitor Price',
    'Stock Available', 'Marketing Spend', 'Holiday/Season Indicator',
    'Weather Condition', 'Economic Indicator', 'Social Media Trend Score',
    'Market Sentiment Score', 'Competitor Activity Score', 'Discount (%)',
    'Year', 'Month', 'WeekOfYear', 'DayOfWeek', 'IsWeekend',
    'Discount_Price_Impact', 'Marketing_Stock_Interaction',
    'Effective_Price', 'Price_Ratio'
]

X = df[features]
y = df['Adjusted Units Sold']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# ---------- Predict for next 6 months ----------
future_dates = pd.date_range(start="2024-12-01", periods=6, freq='MS')
future_df = pd.DataFrame({'Date': future_dates})
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week
future_df['DayOfWeek'] = 0  # assuming 1st of month is Monday
future_df['IsWeekend'] = 0

# Fill other features using average values
for col in ['Product Name', 'RAM', 'Memory', 'Price (USD)', 'Competitor Price',
            'Stock Available', 'Marketing Spend', 'Holiday/Season Indicator',
            'Weather Condition', 'Economic Indicator', 'Social Media Trend Score',
            'Market Sentiment Score', 'Competitor Activity Score', 'Discount (%)']:
    future_df[col] = df[col].mean()

# Feature Engineering
future_df['Discount_Price_Impact'] = future_df['Discount (%)'] * future_df['Price (USD)'] / 100
future_df['Marketing_Stock_Interaction'] = future_df['Marketing Spend'] * future_df['Stock Available']
future_df['Effective_Price'] = future_df['Price (USD)'] - future_df['Discount_Price_Impact']
future_df['Price_Ratio'] = future_df['Price (USD)'] / future_df['Competitor Price']

# Predict
X_future = future_df[features]
future_df['Predicted Sales'] = model.predict(X_future)

# ---------- Suggest improvements if sales are low ----------
avg_sales = future_df['Predicted Sales'].mean()
suggestions = []

for _, row in future_df.iterrows():
    if row['Predicted Sales'] < avg_sales:
        new_discount = row['Discount (%)'] + 10
        new_marketing = row['Marketing Spend'] + 1000

        row_copy = row.copy()
        row_copy['Discount (%)'] = new_discount
        row_copy['Marketing Spend'] = new_marketing
        row_copy['Discount_Price_Impact'] = new_discount * row['Price (USD)'] / 100
        row_copy['Marketing_Stock_Interaction'] = new_marketing * row['Stock Available']
        row_copy['Effective_Price'] = row['Price (USD)'] - row_copy['Discount_Price_Impact']
        row_copy['Price_Ratio'] = row['Price (USD)'] / row['Competitor Price']

        # Predict again with updated factors
        improved_sales = model.predict(pd.DataFrame([row_copy[features]]))[0]

        suggestion = {
            'Suggested Discount (%)': round(new_discount, 2),
            'Suggested Marketing Spend': round(new_marketing),
            'New Predicted Sales': round(improved_sales, 2)
        }
    else:
        suggestion = {}

    suggestions.append(suggestion)

future_df['Suggestions'] = suggestions

# Final output
output = future_df[['Year', 'Month', 'Predicted Sales', 'Suggestions']]
print(output)
