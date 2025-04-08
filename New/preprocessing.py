import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import joblib
import os
from flask import Flask, request, jsonify, render_template
import io
import base64

import matplotlib
matplotlib.use('Agg')

# Create Flask app
app = Flask(__name__)

# Make sure models directory exists
os.makedirs('models', exist_ok=True)

# Data preprocessing function with improved data type handling
def preprocess_data(df):
    """Preprocess the sales data for modeling with robust data type handling"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    print("Original column data types:")
    print(df.dtypes)
    
    # Convert date to datetime
    try:
        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    except KeyError:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except KeyError:
            print("No date column found. Creating a dummy date column.")
            df['date'] = pd.to_datetime('today')
    
    # Extract date features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    
    # Handle categorical columns properly
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['Date', 'date']]
    
    print(f"Categorical columns to encode: {categorical_cols}")
    
    # Create a new DataFrame to build our features properly
    processed_df = pd.DataFrame()
    
    # Process target variable first
    target_col = 'Units Sold'
    if target_col not in df.columns:
        print(f"ERROR: Required column '{target_col}' not found in dataset")
        raise ValueError(f"Required column '{target_col}' not found in dataset")
    
    # Convert target to numeric and handle NaN values
    processed_df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Process numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    for col in numeric_cols:
        processed_df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle categorical columns with one-hot encoding
    if categorical_cols:
        # Use pandas get_dummies with a prefix to avoid collisions
        dummies = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, drop_first=False)
        processed_df = pd.concat([processed_df, dummies], axis=1)
    
    # Add date-derived features
    processed_df['day'] = df['date'].dt.day
    processed_df['month'] = df['date'].dt.month
    processed_df['year'] = df['date'].dt.year
    processed_df['day_of_week'] = df['date'].dt.dayofweek
    processed_df['quarter'] = df['date'].dt.quarter
    
    # Fill NaN values
    processed_df = processed_df.fillna(0)
    
    # Double-check all columns are numeric
    for col in processed_df.columns:
        if not np.issubdtype(processed_df[col].dtype, np.number):
            print(f"WARNING: Column {col} is not numeric: {processed_df[col].dtype}")
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
    
    # Drop rows with NaN in target variable
    processed_df = processed_df.dropna(subset=[target_col])
    
    # Print final data types
    print("Processed column data types:")
    print(processed_df.dtypes)
    
    # Prepare features and target
    X = processed_df.drop([target_col], axis=1)
    y = processed_df[target_col]
    
    print(f"Final X data types after preprocessing:")
    print(X.dtypes.value_counts())
    
    # Explicitly convert X to float64 to ensure compatibility with SHAP
    X = X.astype(np.float64)
    
    print("Everything is good - data preprocessed successfully")
    return X, y


# Train the Random Forest model with safety checks
def train_model(X, y):
    """Train the Random Forest sales prediction model with safety checks"""
    print("Starting model training...")
    print(f"Input shape — X: {X.shape}, y: {y.shape}")
    
    # Handle object-type columns safely
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(np.float64)
                print(f" Converted column '{col}' to float64.")
            except ValueError:
                print(f"Cannot convert column '{col}' to float64. Consider encoding.")
                raise
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Final dtype check
    if X_train.dtypes.apply(lambda dt: dt == 'object').any():
        print("Object-type columns still exist. Forcing full DataFrame conversion to float.")
        try:
            X_train = X_train.astype(np.float64)
            X_test = X_test.astype(np.float64)
        except Exception as e:
            print(f"Conversion failed: {str(e)}")
            raise

    # Train the model
    print("🌲 Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    print(f"Evaluation — MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    
    # SHAP explainer
    print("Creating SHAP explainer...")
    explainer = None
    X_sample = X_train.iloc[:1000].astype(np.float64)

    try:
        explainer = shap.Explainer(model, X_sample)
        print("SHAP explainer (auto) created.")
    except Exception as e:
        print(f"shap.Explainer failed: {str(e)}")
        print("Trying shap.TreeExplainer...")
        try:
            explainer = shap.TreeExplainer(model)
            print("SHAP TreeExplainer created.")
        except Exception as e:
            print(f"SHAP fallback failed: {str(e)}")
    
    return model, explainer, X_test, metrics

# Generate insights based on feature importance (fallback when SHAP is unavailable)
def generate_insights_from_importance(feature_importance, prediction, input_data):
    """Generate business insights based on feature importance when SHAP is unavailable"""
    insights = []
    recommendations = []
    
    # Get top factors
    top_features = feature_importance.head(6)['Feature'].tolist()
    
    insights.append(f"Predicted Units Sold: {prediction:.2f}")
    
    # Check for price-related features
    price_features = [f for f in top_features if 'Price' in f and 'Competitor' not in f]
    competitor_price_features = [f for f in top_features if 'Competitor Price' in f]
    
    if price_features and competitor_price_features:
        price_feature = price_features[0]
        competitor_price_feature = competitor_price_features[0]
        
        price_value = input_data[price_feature].values[0] if price_feature in input_data.columns else 0
        competitor_price = input_data[competitor_price_feature].values[0] if competitor_price_feature in input_data.columns else 0
        
        if price_value > competitor_price:
            recommendations.append(f"Consider reducing price from ${price_value:.2f} to be more competitive with ${competitor_price:.2f}")
        else:
            recommendations.append(f"Price positioning is good (${price_value:.2f} vs competitor ${competitor_price:.2f})")
    
    # Check for discount features
    discount_features = [f for f in top_features if 'Discount' in f]
    if discount_features:
        discount_feature = discount_features[0]
        current_discount = input_data[discount_feature].values[0] if discount_feature in input_data.columns else 0
        
        if current_discount < 15:
            recommendations.append(f"Consider increasing discount from {current_discount:.1f}% to improve sales")
        elif current_discount > 30:
            recommendations.append(f"Current discount of {current_discount:.1f}% may be too high, consider value-based promotions instead")
    
    # Check for marketing features
    marketing_features = [f for f in top_features if 'Marketing' in f]
    if marketing_features:
        marketing_feature = marketing_features[0]
        marketing_spend = input_data[marketing_feature].values[0] if marketing_feature in input_data.columns else 0
        
        recommendations.append(f"Marketing is influential - optimize current spend of ${marketing_spend:.2f}")
    
    # Check for social media features
    social_features = [f for f in top_features if 'Social Media' in f]
    if social_features:
        recommendations.append("Social media presence is important - focus on increasing engagement")
    
    # Check for stock features
    stock_features = [f for f in top_features if 'Stock' in f]
    if stock_features:
        stock_feature = stock_features[0]
        stock_value = input_data[stock_feature].values[0] if stock_feature in input_data.columns else 0
        
        if stock_value < 100:
            recommendations.append(f"Low stock level ({stock_value:.0f} units) may be limiting sales potential")
    
    # If we don't have enough recommendations, add a general one
    if len(recommendations) < 3:
        recommendations.append("Review seasonal marketing strategies to improve sales during this period")
    
    return insights, recommendations

# Generate SHAP summary plot safely
# Modify the generate_shap_plot function to avoid multiprocessing issues
def generate_shap_plot(explainer, X_test):
    """Generate SHAP summary plot with error handling"""
    if explainer is None:
        return None
    
    try:
        # Set matplotlib to use Agg backend to avoid GUI issues
        import matplotlib
        matplotlib.use('Agg')
        
        plt.figure(figsize=(10, 6))
        # Limit sample size and ensure proper types
        X_sample = X_test.astype(np.float64).iloc[:50]  # Reduced sample size
        shap_values = explainer(X_sample)
        
        # Use threading lock if needed
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close('all')
        
        return plot_data
    except Exception as e:
        print(f"Error generating SHAP plot: {str(e)}")
        plt.close('all')
        return None

# Explain prediction with SHAP values or feature importance
def explain_prediction(model, explainer, input_data):
    """Generate prediction and explanation with fallback to feature importance"""
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Try to use SHAP if available
    if explainer is not None:
        try:
            # Use Agg backend
            import matplotlib
            matplotlib.use('Agg')
            
            # Calculate SHAP values
            shap_values = explainer(input_data.astype(np.float64))
            
            # Get SHAP values for this prediction
            shap_df = pd.DataFrame({
                'Feature': input_data.columns,
                'SHAP Value': shap_values.values[0, :]
            }).sort_values('SHAP Value', key=abs, ascending=False)
            
            # Generate insights and recommendations
            insights, recommendations = generate_insights(shap_df, prediction, input_data)
            
            # Generate feature impact plot
            plt.figure(figsize=(10, 6))
            shap.bar_plot(shap_values[0], feature_names=input_data.columns.tolist(), show=False)
            plt.tight_layout()
            
            # Convert plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            impact_plot = base64.b64encode(buffer.getvalue()).decode()
            plt.close('all')  # Close all figures, not just current one
            
            return prediction, shap_df, feature_importance, insights, recommendations, impact_plot
        except Exception as e:
            print(f"Error using SHAP explainer: {str(e)}")
            print("Falling back to feature importance...")
            plt.close('all')  # Ensure plot resources are freed
    
    # Fallback to feature importance
    insights, recommendations = generate_insights_from_importance(feature_importance, prediction, input_data)
    
    # Generate simple feature importance plot with Agg backend
    import matplotlib
    matplotlib.use('Agg')
    plt.figure(figsize=(10, 6))
    feature_importance.head(15).plot(kind='barh', x='Feature', y='Importance', legend=False)
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    impact_plot = base64.b64encode(buffer.getvalue()).decode()
    plt.close('all')  # Close all figures
    
    # Create a placeholder SHAP df with feature importance values
    shap_df = pd.DataFrame({
        'Feature': feature_importance['Feature'],
        'SHAP Value': feature_importance['Importance']  # Using importance as placeholder
    }).head(10)
    
    return prediction, shap_df, feature_importance, insights, recommendations, impact_plot


# Flask routes
@app.route('/train', methods=['POST'])
def train_endpoint():
    """Endpoint to train the model with local dataset"""
    try:
        # Read the CSV file from a fixed local path
        filepath = "New/processed_smartphone_sales_with_discount.csv"  # Change this to your actual filename
        print(f"Reading CSV file from: {filepath}")
        
        # Read the dataset
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        print(f"Sample columns: {df.columns[:5].tolist()}...")
        
        # Print sample data for debugging
        print("Sample data:")
        print(df.head(2))
        
        # Preprocess data
        try:
            X, y = preprocess_data(df)
            print(f"Data preprocessed successfully: X shape={X.shape}, y shape={y.shape}")
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error during data preprocessing: {str(e)}'}), 500
        
        # Train model
        try:
            model, explainer, X_test, metrics = train_model(X, y)
            print("Model trained successfully")
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error during model training: {str(e)}'}), 500
        
        # Save model and explainer
        try:
            joblib.dump(model, 'models/sales_forecast_model.pkl')
            if explainer is not None:
                joblib.dump(explainer, 'models/sales_forecast_explainer.pkl')
                print("Model and explainer saved successfully")
            else:
                print("Model saved successfully (explainer not available)")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return jsonify({'error': f'Error saving model: {str(e)}'}), 500
        
        # Generate SHAP summary plot
        try:
            plot_data = generate_shap_plot(explainer, X_test) if explainer is not None else None
            if plot_data:
                print("Plot generated successfully")
            else:
                print("No plot generated (explainer not available)")
        except Exception as e:
            print(f"Error generating plot: {str(e)}")
            plot_data = None
        
        # Return success
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': {
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'r2': float(metrics['r2'])
            },
            'features': X.columns.tolist(),
            'shap_plot': plot_data
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Endpoint for making predictions with explanations"""
    try:
        # Get input data
        data = request.json
        
        # Convert to dataframe
        input_df = pd.DataFrame([data])
        
        # Process date if present
        if 'date' in input_df.columns or 'Date' in input_df.columns:
            date_col = 'date' if 'date' in input_df.columns else 'Date'
            input_df['date'] = pd.to_datetime(input_df[date_col], errors='coerce')
            
            # Extract date features
            input_df['day'] = input_df['date'].dt.day
            input_df['month'] = input_df['date'].dt.month
            input_df['year'] = input_df['date'].dt.year
            input_df['day_of_week'] = input_df['date'].dt.dayofweek
            input_df['quarter'] = input_df['date'].dt.quarter
            
            # Remove date column
            input_df = input_df.drop(['date'], axis=1)
            if date_col in input_df.columns:
                input_df = input_df.drop([date_col], axis=1)
        
        # Convert all columns to float64 for consistency
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # Load model and explainer
        try:
            model = joblib.load('models/sales_forecast_model.pkl')
            try:
                explainer = joblib.load('models/sales_forecast_explainer.pkl')
            except FileNotFoundError:
                print("Explainer not found, will use feature importance instead")
                explainer = None
        except FileNotFoundError:
            return jsonify({'error': 'Model not found. Please train the model first.'}), 404
        
        # Ensure input data has all required columns
        missing_cols = set(model.feature_names_in_) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        
        # Remove any extra columns
        extra_cols = set(input_df.columns) - set(model.feature_names_in_)
        if extra_cols:
            input_df = input_df.drop(columns=extra_cols)
            
        # Reorder columns to match model
        input_df = input_df[model.feature_names_in_]
        
        # Ensure all data is float64
        input_df = input_df.astype(np.float64)
        
        # Get prediction and explanation
        prediction, shap_df, feature_importance, insights, recommendations, impact_plot = explain_prediction(model, explainer, input_df)
        
        # Return results
        return jsonify({
            'prediction': {
                'units_sold': float(prediction),
            },
            'explanations': {
                'top_factors': shap_df.head(10).to_dict(orient='records'),
                'feature_importance': feature_importance.head(10).to_dict(orient='records'),
                'insights': insights,
                'recommendations': recommendations,
                # 'impact_plot': impact_plot
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Generate insights based on SHAP values
def generate_insights(shap_df, prediction, input_data):
    """Generate business insights based on SHAP values"""
    insights = []
    recommendations = []
    
    # Get top positive and negative factors
    positive_factors = shap_df[shap_df['SHAP Value'] > 0].sort_values('SHAP Value', ascending=False).head(3)
    negative_factors = shap_df[shap_df['SHAP Value'] < 0].sort_values('SHAP Value', ascending=True).head(3)
    
    insights.append(f"Predicted Units Sold: {prediction:.2f}")
    
    # Process price features
    price_features = [f for f in shap_df['Feature'] if 'Price' in f and 'Competitor' not in f]
    competitor_price_features = [f for f in shap_df['Feature'] if 'Competitor Price' in f]
    
    if price_features and competitor_price_features:
        price_feature = price_features[0]
        competitor_price_feature = competitor_price_features[0]
        
        price_value = input_data[price_feature].values[0] if price_feature in input_data.columns else 0
        competitor_price = input_data[competitor_price_feature].values[0] if competitor_price_feature in input_data.columns else 0
        
        price_impact = shap_df[shap_df['Feature'] == price_feature]['SHAP Value'].values[0] if price_feature in shap_df['Feature'].values else 0
        
        if price_impact < 0 and price_value > competitor_price:
            # Current price is higher than competitor and negatively impacting sales
            optimal_price = competitor_price * 0.97  # Slightly less than competitor
            recommendations.append(f"Consider reducing price from ${price_value:.2f} to ${optimal_price:.2f} (slightly below competitor's ${competitor_price:.2f})")
        elif price_impact < 0 and price_value <= competitor_price:
            # Price is already competitive but still negatively impacting sales
            recommendations.append(f"Current price (${price_value:.2f}) seems competitive compared to competitors (${competitor_price:.2f}). Consider bundle deals instead of price reductions.")
    
    # Process discount features
    discount_features = [f for f in shap_df['Feature'] if 'Discount' in f]
    if discount_features:
        discount_feature = discount_features[0]
        discount_impact = shap_df[shap_df['Feature'] == discount_feature]['SHAP Value'].values[0] if discount_feature in shap_df['Feature'].values else 0
        current_discount = input_data[discount_feature].values[0] if discount_feature in input_data.columns else 0
        
        if discount_impact > 0:
            # Discount is positively impacting sales
            optimal_discount = min(current_discount + 5, 35)  # Increase discount but cap at 35%
            recommendations.append(f"Increase discount from {current_discount:.1f}% to {optimal_discount:.1f}% to boost sales further")
        elif discount_impact < 0 and current_discount > 10:
            # Discount is negatively impacting sales (might indicate quality perception issues)
            recommendations.append("Current discount strategy isn't working effectively. Consider premium positioning instead")
    
    # Process marketing features
    marketing_features = [f for f in shap_df['Feature'] if 'Marketing' in f]
    if marketing_features:
        marketing_feature = marketing_features[0]
        marketing_impact = shap_df[shap_df['Feature'] == marketing_feature]['SHAP Value'].values[0] if marketing_feature in shap_df['Feature'].values else 0
        marketing_spend = input_data[marketing_feature].values[0] if marketing_feature in input_data.columns else 0
        
        if marketing_impact > 0:
            # Marketing is working
            recommendations.append(f"Current marketing strategy is effective. Consider increasing budget by 15% from ${marketing_spend:.2f}")
        elif marketing_impact < 0:
            # Marketing isn't effective
            recommendations.append("Current marketing approach isn't driving sales. Review marketing channels")
    
    # Process social media features
    social_features = [f for f in shap_df['Feature'] if 'Social Media' in f]
    if social_features:
        social_feature = social_features[0]
        social_impact = shap_df[shap_df['Feature'] == social_feature]['SHAP Value'].values[0] if social_feature in shap_df['Feature'].values else 0
        social_score = input_data[social_feature].values[0] if social_feature in input_data.columns else 0
        
        if social_impact > 0:
            recommendations.append("Social media engagement is positively influencing sales. Increase activity")
        elif social_impact < 0 and social_score > 0:
            recommendations.append("Current social media approach may be creating negative sentiment. Review strategy")
    
    # Process stock features
    stock_features = [f for f in shap_df['Feature'] if 'Stock' in f]
    if stock_features:
        stock_feature = stock_features[0]
        stock_impact = shap_df[shap_df['Feature'] == stock_feature]['SHAP Value'].values[0] if stock_feature in shap_df['Feature'].values else 0
        stock_value = input_data[stock_feature].values[0] if stock_feature in input_data.columns else 0
        
        if stock_impact > 0 and stock_value < 100:
            recommendations.append(f"Low stock level ({stock_value:.0f} units) may be limiting sales. Increase inventory")
    
    # If we don't have enough recommendations, add a general one
    if len(recommendations) < 3:
        recommendations.append("Optimize seasonal marketing strategies to improve sales during this period")
    
    return insights, recommendations


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)