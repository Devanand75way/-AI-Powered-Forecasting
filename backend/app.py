from flask import Flask, request, jsonify
import traceback
from flask_cors import CORS
from models.predictors import AdvancedDemandPredictor

# Flask API Setup
app = Flask(__name__)
demand_predictor = AdvancedDemandPredictor()

@app.route('/update_market_insights', methods=['POST'])
def update_market_insights():
    """
    Endpoint to update market insights dynamically
    """
    try:
        market_insights = request.json
        
        # Validate input
        if not isinstance(market_insights, dict):
            return jsonify({'error': 'Invalid market insights format'}), 400
        
        # Update market insights in the predictor
        demand_predictor.market_insight_predictor.update_market_insights(market_insights)
        
        return jsonify({
            'status': 'Market insights updated successfully',
            'received_insights': market_insights
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'details': 'An error occurred while updating market insights'
        }), 500

@app.route('/predict_future_demand', methods=['POST'])
def predict_future_demand():
    try:
        # Get input data from request
        input_data = request.json.get('demandData', {})
        market_insights = request.json.get('marketInsights', {})
        
        # Validate input data
        required_fields = [
            'Segment', 'Country', 'City', 'State', 'Region', 
            'Category', 'Sub-Category', 'Product Name', 
            'Sales', 'Discount'
        ]
        
        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Predict future demand
        prediction = demand_predictor.predict_future_demand(
            input_data, 
            market_insights
        )
        
        return jsonify(prediction)
    
    except Exception as e:
        # Log the full traceback for debugging
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'details': 'An error occurred during prediction',
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.run(debug=True, port=5000)