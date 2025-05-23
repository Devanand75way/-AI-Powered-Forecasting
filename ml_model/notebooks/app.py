import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import base64
from io import BytesIO
import matplotlib
from textblob import TextBlob
from serpapi import GoogleSearch
import os
import json
from flask_cors import CORS

app = Flask(__name__) 
CORS(app)

sales_data  = pd.read_csv('ml_model/data/smartphone_sales_updated.csv')
API_KEY = "4465ed83b7aca9208cbab70e152493e8fdddd8646ac4ffab2b690e575229402e"


# Load the actual dataset
def load_dataset():

    dataset_path = 'ml_model/data/smartphone_sales_updated.csv'

    # Check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

# Helper function to get product name from one-hot encoded columns
def get_product_name(row):
    product_columns = [col for col in row.index if col.startswith('Product Name_')]
    for col in product_columns:
        if row[col]:
            return col.replace('Product Name_', '')
    return None

# 1. Past Sales Analysis (By Date Interval)
@app.route('/sales-analysis-by-date', methods=['POST'])
def sales_analysis_by_date():
    data = request.get_json()
    print("sales_analysis_by_date", data)

    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        # Load data
        df = load_dataset()
        
        # Filter by date range
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        
        if filtered_df.empty:
            return jsonify({"error": "No data found for the selected date range"}), 400
        
        # Get product name for each row
        filtered_df['Product'] = filtered_df.apply(get_product_name, axis=1)
        
        # Calculate actual sales (using Units Sold directly)
        filtered_df['Actual Sales'] = filtered_df['Units Sold']
        
        # Total sales per product
        product_sales = filtered_df.groupby('Product')['Actual Sales'].sum().reset_index()
        product_sales = product_sales.sort_values('Actual Sales', ascending=False)
        
        # Monthly sales trends
        filtered_df['Month'] = filtered_df['Date'].dt.strftime('%Y-%m')
        monthly_sales = filtered_df.groupby('Month')['Actual Sales'].sum().reset_index()
        
        # Best & worst-performing months
        best_month = monthly_sales.loc[monthly_sales['Actual Sales'].idxmax()]['Month']
        worst_month = monthly_sales.loc[monthly_sales['Actual Sales'].idxmin()]['Month']
        
        
        # Prepare response
        response = {
            "total_sales": filtered_df['Actual Sales'].sum(),
            "product_sales": product_sales.to_dict('records'),
            "monthly_trends": monthly_sales.to_dict('records'),
            "best_month": best_month,
            "worst_month": worst_month,
            # "graph": graph_image
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. Past Sales Report (By Product Name)
@app.route('/sales-analysis-by-product', methods=['POST'])
def sales_analysis_by_product():
    data = request.get_json()
    product_name = data.get('product_name')
    
    try:
        # Load data
        df = load_dataset()
        
        # Filter for the specific product
        product_col = f'Product Name_{product_name}'
        if product_col not in df.columns:
            return jsonify({"error": f"Product not found: {product_name}"}), 400
        
        filtered_df = df[df[product_col] == True].copy()
        
        if filtered_df.empty:
            return jsonify({"error": f"No data found for product {product_name}"}), 400
        
        # Calculate actual sales
        filtered_df['Actual Sales'] = filtered_df['Units Sold']
        
        # Monthly sales trends for this product
        filtered_df['Month'] = filtered_df['Date'].dt.strftime('%Y-%m')
        monthly_sales = filtered_df.groupby('Month')['Actual Sales'].sum().reset_index()
        
        # Best-selling months for this product
        best_months = monthly_sales.sort_values('Actual Sales', ascending=False).head(3)
        

        
        response = {
            "product_name": product_name,
            "total_sales": filtered_df['Actual Sales'].sum(),
            "monthly_trends": monthly_sales.to_dict('records'),
            "best_selling_months": best_months.to_dict('records'),
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. Future Sales Forecasting
@app.route('/sales-forecast', methods=['POST'])
def sales_forecast():
    data = request.get_json()
    product_name = data.get('product_name')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    forecast_months = data.get('forecast_months', 3)
    external_factors = data.get('external_factors', {})
    ram = data.get('ram')  
    memory = data.get('memory')
    print(data)
    try:
        # Load historical data
        df = load_dataset()
        
        # Identify the correct product column
        product_col = f'Product Name_{product_name}'
        if product_col not in df.columns:
            return jsonify({"error": f"Product not found: {product_name}"}), 400
        
        # Filter historical data for the specific product
        filtered_df = df[df[product_col] == True].copy()
        
        # Debugging: Check how many rows match the product
        print(f"Rows for {product_name}: {len(filtered_df)}")
        
        # Check if there's any data for this product
        if filtered_df.empty:
            return jsonify({"error": f"No historical data found for product {product_name}"}), 400
        
        # Additional filtering for RAM if provided
        if ram:
            ram_col = f'RAM_{ram}'
            # Debugging: Check if the RAM column exists
            print(f"RAM column {ram_col} exists: {ram_col in filtered_df.columns}")
            if ram_col in filtered_df.columns:
                # Debugging: Count rows before RAM filter
                print(f"Rows before RAM filter: {len(filtered_df)}")
                # Check if any rows have this RAM
                if not filtered_df[ram_col].any():
                    return jsonify({"error": f"No data for {product_name} with RAM {ram}"}), 400
                filtered_df = filtered_df[filtered_df[ram_col] == True]
                # Debugging: Count rows after RAM filter
                print(f"Rows after RAM filter: {len(filtered_df)}")
            else:
                return jsonify({"error": f"RAM option not found: {ram}"}), 400
        
        # Additional filtering for Memory if provided
        if memory:
            memory_col = f'Memory_{memory}'
            # Debugging: Check if the Memory column exists
            print(f"Memory column {memory_col} exists: {memory_col in filtered_df.columns}")
            if memory_col in filtered_df.columns:
                # Debugging: Count rows before Memory filter
                print(f"Rows before Memory filter: {len(filtered_df)}")
                # Check if any rows have this Memory
                if not filtered_df[memory_col].any():
                    return jsonify({"error": f"No data for {product_name} with Memory {memory}"}), 400
                filtered_df = filtered_df[filtered_df[memory_col] == True]
                # Debugging: Count rows after Memory filter
                print(f"Rows after Memory filter: {len(filtered_df)}")
            else:
                return jsonify({"error": f"Memory option not found: {memory}"}), 400
        
        if filtered_df.empty:
            # Provide more detailed error message
            if ram and memory:
                error_msg = f"No historical data found for product {product_name} with {ram} RAM and {memory} Memory configuration"
            elif ram:
                error_msg = f"No historical data found for product {product_name} with {ram} RAM"
            elif memory:
                error_msg = f"No historical data found for product {product_name} with {memory} Memory"
            else:
                error_msg = f"No historical data found for product {product_name}"
            
            return jsonify({"error": error_msg}), 400

        # Get today's date to separate past and future data
        today = pd.Timestamp.now().normalize()
        
        # Get the last 12 months of data for training
        twelve_months_ago = today - pd.DateOffset(months=12)
        training_df = filtered_df[(filtered_df['Date'] >= twelve_months_ago) & (filtered_df['Date'] <= today)].copy()

        if training_df.empty:
            return jsonify({"error": "Insufficient historical data for the last 12 months"}), 400
        
        # Prepare data for training
        training_df['Month'] = training_df['Date'].dt.month
        training_df['Year'] = training_df['Date'].dt.year
        training_df['Day'] = training_df['Date'].dt.day
        
        # Extract relevant features
        feature_columns = [
            'Month', 'Year', 'Day', 'Price (USD)', 'Competitor Price', 
            'Stock Available', 'Marketing Spend', 'Holiday/Season Indicator',
            'Weather Condition', 'Economic Indicator', 'Social Media Trend Score',
            'Market Sentiment Score', 'Competitor Activity Score'
        ]
        
        # Check if all features exist
        missing_features = [col for col in feature_columns if col not in training_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing required features in dataset: {missing_features}"}), 400
        
        X = training_df[feature_columns]
        y = training_df['Units Sold']
        
        # Train a model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        
        # Generate forecast dates - starting from today (or end_date if provided and it's in the future)
        forecast_start = max(today, pd.to_datetime(end_date)) if end_date else today
        forecast_dates = pd.date_range(start=forecast_start, periods=forecast_months, freq='ME')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({'Date': forecast_dates})
        forecast_df['Month'] = forecast_df['Date'].dt.month
        forecast_df['Year'] = forecast_df['Date'].dt.year
        forecast_df['Day'] = forecast_df['Date'].dt.day
        
        # Fill in features for forecasting
        # Use the average values from last 12 months as base values
        for col in feature_columns:
            if col not in ['Month', 'Year', 'Day']:
                forecast_df[col] = training_df[col].mean()
        
        # Apply external factors if provided
        if 'competitor_activity' in external_factors:
            forecast_df['Competitor Activity Score'] = external_factors['competitor_activity']
        
        if 'weather_condition' in external_factors:
            forecast_df['Weather Condition'] = external_factors['weather_condition']
        
        if 'market_sentiment' in external_factors:
            forecast_df['Market Sentiment Score'] = external_factors['market_sentiment']
        
        if 'holiday_indicator' in external_factors:
            forecast_df['Holiday/Season Indicator'] = external_factors['holiday_indicator']
        
        # Make predictions
        X_forecast = forecast_df[feature_columns]
        forecast_df['Predicted Sales'] = model.predict(X_forecast)
        
        historical_monthly = training_df.groupby([training_df['Date'].dt.year.rename('Year'),training_df['Date'].dt.month.rename('Month')])[['Units Sold']].mean().reset_index()
        historical_monthly['Date'] = pd.to_datetime(historical_monthly['Year'].astype(str) + '-' + historical_monthly['Month'].astype(str) + '-01')
        historical_monthly['Actual Sales'] = historical_monthly['Units Sold'] 
        

        # Customize title to include RAM and Memory
        title = f'Sales Forecast for {product_name}'
        if ram:
            title += f' ({ram}'
            if memory:
                title += f', {memory})'
            else:
                title += ')'
        elif memory:
            title += f' ({memory})'
        

        impact = {}
        feature_importances = dict(zip(feature_columns, model.feature_importances_))
        
        for factor, value in external_factors.items():
            if factor == 'competitor_activity' and 'Competitor Activity Score' in feature_importances:
                baseline = training_df['Competitor Activity Score'].mean()
                impact[factor] = (value - baseline) * feature_importances['Competitor Activity Score']
            elif factor == 'weather_condition' and 'Weather Condition' in feature_importances:
                baseline = training_df['Weather Condition'].mean()
                impact[factor] = (value - baseline) * feature_importances['Weather Condition']
            elif factor == 'market_sentiment' and 'Market Sentiment Score' in feature_importances:
                baseline = training_df['Market Sentiment Score'].mean()
                impact[factor] = (value - baseline) * feature_importances['Market Sentiment Score']
            elif factor == 'holiday_indicator' and 'Holiday/Season Indicator' in feature_importances:
                baseline = training_df['Holiday/Season Indicator'].mean()
                impact[factor] = (value - baseline) * feature_importances['Holiday/Season Indicator']
        
        # Normalize impact values to make them more interpretable
        max_impact = max([abs(v) for v in impact.values()]) if impact else 1
        for factor in impact:
            impact[factor] = impact[factor] / max_impact
        
        # Add past 12 months performance metrics
        past_performance = {
            "avg_monthly_sales": training_df['Units Sold'].mean(),
            "total_sales_last_12_months": training_df['Units Sold'].sum(),
            "best_month": historical_monthly.loc[historical_monthly['Actual Sales'].idxmax()]['Date'].strftime('%Y-%m'),
            "worst_month": historical_monthly.loc[historical_monthly['Actual Sales'].idxmin()]['Date'].strftime('%Y-%m'),
            "sales_trend": "increasing" if historical_monthly['Actual Sales'].iloc[-1] > historical_monthly['Actual Sales'].iloc[0] else "decreasing"
        }
        
        # Get competitor products (same RAM and Memory but different product)
        competitor_products = []
        if ram and memory:
            ram_col = f'RAM_{ram}'
            memory_col = f'Memory_{memory}'
            
            # Get products with the same RAM and Memory configuration
            competitor_df = df[(df[ram_col] == True) & (df[memory_col] == True) & (df[product_col] == False)]
            
            # Get unique competitor products
            for i, row in competitor_df.iterrows():
                for col in [c for c in competitor_df.columns if c.startswith('Product Name_') and c != product_col]:
                    if row[col]:
                        comp_name = col.replace('Product Name_', '')
                        if comp_name not in competitor_products:
                            competitor_products.append(comp_name)
                            break
        
        # Prepare response
        response = {
            "product_name": product_name,
            "ram": ram,
            "memory": memory,
            "forecast_months": forecast_months,
            "forecast_data": forecast_df[['Date', 'Predicted Sales']].to_dict('records'),
            "past_performance": past_performance,
            "external_factors_impact": impact,
            "competitor_products": competitor_products[:5] if competitor_products else [],
            # "graph": graph_image
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# get the Product reviews from the API 
@app.route('/get_product_reviews', methods=['GET'])
def get_product_reviews():
    product_query = request.args.get("q")
    
    if not product_query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    # Step 1: Search by product name using google_shopping
    shopping_params = {
        "engine": "google_shopping",
        "q": product_query,
        "gl": "us",
        "hl": "en",
        "api_key": API_KEY
    }

    shopping_search = GoogleSearch(shopping_params)
    shopping_results = shopping_search.get_dict()

    product_id = ""
    results = shopping_results.get("shopping_results", [])
    if not results:
        return jsonify({"error": "No shopping results found"}), 404

    product_id = results[0].get("product_id")
    if not product_id:
        return jsonify({"error": "Product ID not found"}), 404

    # Step 2: Use product_id to get reviews
    product_params = {
        "engine": "google_product",
        "product_id": product_id,
        "reviews": "1",
        "gl": "us",
        "hl": "en",
        "api_key": API_KEY
    }

    product_search = GoogleSearch(product_params)
    product_results = product_search.get_dict()

    reviews_data = product_results.get("reviews_results", {})
    reviews = reviews_data.get("reviews", [])

    review_list = [{"content": review.get("content")} for review in reviews]

    return jsonify({
        "query": product_query,
        "product_id": product_id,
        "reviews": review_list
    })


# Add a route to calculate the Market sentiment using Product reviews
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    # Get feedback from the request
    feedback = request.json.get('feedback', [])
    
    if not feedback or not isinstance(feedback, list):
        return jsonify({'error': 'Invalid input. Please provide a list of feedback texts.'}), 400
    
    # Calculate sentiment scores
    sentiment_scores = [TextBlob(text).sentiment.polarity for text in feedback]
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
    # Return the sentiment score
    return jsonify({'average_sentiment': round(average_sentiment, 2)})


# Add a route to calculate the competitor scores
def get_own_product_info(product_name):
    # Load your product data from the JSON file

    with open("ml_model/data/products.json") as f:
        data = json.load(f)
    for product in data["products"]:
        if product["name"].lower() == product_name.lower():
            return product
    return None

def simplify_query(name):
    """Reduce noise in product names for Google Trends."""
    keywords = name.split()
    filtered = [word for word in keywords if word.lower() not in ['for', 't-mobile', 'unlocked', '128gb', '256gb']]
    return " ".join(filtered[:4])  # Keep first 3-4 main words

def get_google_trend_score(product_name):
    from serpapi import GoogleSearch
    simplified_query = simplify_query(product_name)
    print("Simplified query", simplified_query)
    params = {
        "engine": "google_trends",
        "q": simplified_query,
        "data_type": "TIMESERIES",
        "api_key": API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    interest_over_time = results['interest_over_time']
    timeline_data = interest_over_time['timeline_data']
    print(timeline_data)
    for values in timeline_data[:1]:
        extracted_value = values['values'][0]['value']
        return extracted_value


def get_competitor_products(query):
    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": "us",
        "hl": "en",
        "api_key": API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("shopping_results", [])

@app.route('/get_competitor_data', methods=['GET'])
def get_competitor_data():
    product_name = request.args.get("q")
    if not product_name:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    own_product = get_own_product_info(product_name)
    if not own_product:
        return jsonify({"error": "Product not found in your database"}), 404

    competitor_candidates = get_competitor_products(product_name)
    competitors = []

    for result in competitor_candidates[:4]:  # limit to top 4 competitors
        competitor_name = result.get("title")
        raw_price = result.get("price")
        source = result.get("source", "Unknown")

        # Clean and extract price
        if isinstance(raw_price, dict):
            competitor_price = raw_price.get("extracted_value", 0)
        elif isinstance(raw_price, str):
            competitor_price = float(''.join(c for c in raw_price if c.isdigit() or c == '.'))
        else:
            competitor_price = 0

        if competitor_name and competitor_price:
            trend_score = get_google_trend_score(competitor_name)

            # Try matching competitor name with your own product list
            matched_product = get_own_product_info(competitor_name)
            our_price = matched_product["price"] if matched_product else "Not available"

            competitors.append({
                "name": competitor_name,
                "price": competitor_price,
                "ourPrice": our_price,
                "source": source,
                "googleTrends": trend_score
            })

    return jsonify({
        "data": {
            "competitors": competitors
        }
    })


if __name__ == '__main__':
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.run(debug=True)