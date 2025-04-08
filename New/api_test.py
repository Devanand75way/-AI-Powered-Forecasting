from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/trained_model.pkl")
expected_columns = [
    'Product Name', 'RAM', 'Memory', 'Price (USD)', 'Competitor Price',
    'Stock Available', 'Marketing Spend', 'Holiday/Season Indicator',
    'Weather Condition', 'Economic Indicator', 'Social Media Trend Score',
    'Market Sentiment Score', 'Competitor Activity Score', 'Discount (%)',
    'Adjusted Units Sold', 'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfWeek',
    'IsWeekend', 'Marketing_Stock_Interaction', 'Discount_Price_Impact'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Fill in missing columns with default values (or handle appropriately)
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # or a default value you prefer

        # Reorder columns to match training
        input_df = input_df[expected_columns]

        prediction = model.predict(input_df)
        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
