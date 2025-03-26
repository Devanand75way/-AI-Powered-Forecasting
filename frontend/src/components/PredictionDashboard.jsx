import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Card, CardContent, CardHeader, Typography, Grid } from '@mui/material';
import { fetchSalesPrediction } from '../services/predictionService';

const PredictionDashboard = () => {
    const [predictionData, setPredictionData] = useState(null);
    const [performanceMetrics, setPerformanceMetrics] = useState(null);

    useEffect(() => {
        const loadPredictions = async () => {
            try {
                const result = await fetchSalesPrediction();
                setPredictionData(result.predictions);
                setPerformanceMetrics(result.performance_metrics);
            } catch (error) {
                console.error("Failed to fetch predictions", error);
            }
        };

        loadPredictions();
    }, []);

    return (
        <Grid container spacing={4} padding={4}>
            {/* Sales Forecast Chart */}
            <Grid item xs={12} md={6}>
                <Card>
                    <CardHeader title="Sales Forecast" />
                    <CardContent>
                        <LineChart width={500} height={300} data={predictionData}>
                            <XAxis dataKey="month" />
                            <YAxis />
                            <CartesianGrid strokeDasharray="3 3" />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="sales" stroke="#3f51b5" />
                        </LineChart>
                    </CardContent>
                </Card>
            </Grid>

            {/* Model Performance Metrics */}
            <Grid item xs={12} md={6}>
                <Card>
                    <CardHeader title="Model Performance" />
                    <CardContent>
                        {performanceMetrics && (
                            <div>
                                <Typography variant="body1">
                                    <strong>R2 Score:</strong> {performanceMetrics.r2_score.toFixed(4)}
                                </Typography>
                                <Typography variant="body1">
                                    <strong>MAE:</strong> {performanceMetrics.mean_absolute_error.toFixed(4)}
                                </Typography>
                                <Typography variant="body1">
                                    <strong>RMSE:</strong> {performanceMetrics.root_mean_squared_error.toFixed(4)}
                                </Typography>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </Grid>
        </Grid>
    );
};

export default PredictionDashboard;
