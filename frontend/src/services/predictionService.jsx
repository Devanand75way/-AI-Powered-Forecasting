// sales-forecast-project/frontend/src/services/predictionService.js
import axios from 'axios';

const API_BASE_URL =  'http://localhost:5000/api';

export const fetchSalesPrediction = async (params = {}) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/predict`, params);
        return response.data;
    } catch (error) {
        console.error('Prediction fetch error:', error);
        throw error;
    }
};

export const trainNewModel = async (trainingData) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/train`, trainingData);
        return response.data;
    } catch (error) {
        console.error('Model training error:', error);
        throw error;
    }
};