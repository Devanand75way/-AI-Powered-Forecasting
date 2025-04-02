from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load dataset
df = pd.read_csv("random_test/datasets/dataset_copy.csv", encoding="Windows-1252", parse_dates=["Order Date"])
df.set_index("Order Date", inplace=True)

# Function to get monthly sales for a product
def get_monthly_sales(product_name):
    product_df = df[df["Product Name"] == product_name]

    if product_df.empty:
        return pd.Series(dtype=float)

    monthly_sales = product_df["Sales"].resample('M').sum().fillna(0)

    print("Data Range:", monthly_sales.index.min(), "to", monthly_sales.index.max())
    
    # Handle zero values for better forecasting
    monthly_sales = monthly_sales.replace(0, np.nan).interpolate().fillna(0)

    return monthly_sales

@app.route('/predict', methods=['GET'])
def predict_demand():
    product_name = request.args.get('product_name')
    months = int(request.args.get('months', 1))  # Default to 1 month ahead
    
    if not product_name:
        return jsonify({"error": "Please provide a product name"}), 400

    monthly_sales = get_monthly_sales(product_name)

    if monthly_sales.empty:
        return jsonify({"error": "No sales data found for this product"}), 404

    # Auto-select best ARIMA parameters
    best_model = auto_arima(monthly_sales, seasonal=True, m=min(12, len(monthly_sales)//2), stepwise=True, trace=False)
    order = best_model.order
    seasonal_order = best_model.seasonal_order

    # Train SARIMA Model
    model = SARIMAX(monthly_sales, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    future_index = pd.date_range(start=monthly_sales.index[-1], periods=months + 1, freq='M')[1:]
    future_forecast = pd.Series(model_fit.forecast(steps=months), index=future_index)
    print("future_forecast", future_forecast)

    return jsonify({
    "product": product_name,
    "forecast": {str(date): value for date, value in future_forecast.items()}
    })


if __name__ == '__main__':
    app.run(debug=True)



#  In this Project i need to Show case the Current month sales, based on the current sale report AI will help us to predict the next Month Sales.

# Using this Ai modal the Customer not only view their past sale , instead it will check their Future Demand products as well which month of the their sales going to be High. These are are my some key ideas that is i wanted to Add and made this AI modal , Instead of this Points if you have any another or Better approach than apply it Create the Steps and also create the overall modal