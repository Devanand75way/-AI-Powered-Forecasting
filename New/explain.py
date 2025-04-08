import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


# Load your dataset
df = pd.read_csv("New/processed_smartphone_sales_with_discount.csv", parse_dates=['Date'])

# Time-based Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Interaction Features
df['Marketing_Stock_Interaction'] = df['Marketing Spend'] * df['Stock Available']
df['Discount_Price_Impact'] = df['Discount (%)'] * df['Price (USD)']

# Convert object columns to category dtype
for col in ['Product Name', 'RAM', 'Memory']:
    df[col] = df[col].astype('category')


# Drop unneeded columns
X = df.drop(columns=['Date', 'Units Sold'],axis=1)
y = df['Units Sold']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model
# Now use XGBoost with categorical enabled
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    enable_categorical=True,
    random_state=42
)

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
joblib.dump(model, 'models/trained_model.pkl')

# Plot
plt.figure(figsize=(14,6))
plt.plot(y_test.values, label='Actual', alpha=0.6)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title("Actual vs Predicted Sales")
plt.xlabel("Sample Index")
plt.ylabel("Units Sold")
plt.legend()
plt.show()
