import React, { useState, useEffect } from 'react';
import { 
  ThemeProvider, 
  createTheme 
} from '@mui/material/styles';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Box,
  Chip,
  Alert
} from '@mui/material';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function MarketTrendsPage() {
  const navigate = useNavigate();
  const [marketInsight, setMarketInsight] = useState({
    category: '',
    growth_rate: 0.0,
    market_sentiment: '',
    key_trends: [],
    demand_multiplier: 1.0
  });

  const [newTrend, setNewTrend] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);

  const CATEGORIES = ['Furniture', 'Office Supplies', 'Technology'];
  const MARKET_SENTIMENTS = ['Positive', 'Stable', 'Moderate', 'Negative'];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setMarketInsight(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const addTrend = () => {
    if (newTrend && !marketInsight.key_trends.includes(newTrend)) {
      setMarketInsight(prev => ({
        ...prev,
        key_trends: [...prev.key_trends, newTrend]
      }));
      setNewTrend('');
    }
  };

  const removeTrend = (trendToRemove) => {
    setMarketInsight(prev => ({
      ...prev,
      key_trends: prev.key_trends.filter(trend => trend !== trendToRemove)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // First, update market insights in the backend
      await axios.post(
        'http://localhost:5000/update_market_insights', 
        { [marketInsight.category]: marketInsight }
      );
      
      // Redirect to Demand Prediction page with market insights
      navigate('/demand-prediction', { 
        state: { 
          marketInsights: { [marketInsight.category]: marketInsight },
          prefillCategory: marketInsight.category
        } 
      });
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
    }
  };

  return (
    <Container maxWidth="lg">
      <Box my={4}>
        <Typography variant="h4" gutterBottom>
          Market Trends Management
        </Typography>

        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Category Selection */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Product Category</InputLabel>
                <Select
                  name="category"
                  value={marketInsight.category}
                  onChange={handleInputChange}
                >
                  {CATEGORIES.map(category => (
                    <MenuItem key={category} value={category}>
                      {category}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Growth Rate */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="growth_rate"
                label="Growth Rate (%)"
                type="number"
                inputProps={{ step: 0.01, min: 0, max: 1 }}
                value={marketInsight.growth_rate}
                onChange={handleInputChange}
                helperText="Enter growth rate between 0-1"
                required
              />
            </Grid>

            {/* Market Sentiment */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth required>
                <InputLabel>Market Sentiment</InputLabel>
                <Select
                  name="market_sentiment"
                  value={marketInsight.market_sentiment}
                  onChange={handleInputChange}
                >
                  {MARKET_SENTIMENTS.map(sentiment => (
                    <MenuItem key={sentiment} value={sentiment}>
                      {sentiment}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Demand Multiplier */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="demand_multiplier"
                label="Demand Multiplier"
                type="number"
                inputProps={{ step: 0.1, min: 0.5, max: 2 }}
                value={marketInsight.demand_multiplier}
                onChange={handleInputChange}
                helperText="Adjust demand prediction (0.5-2)"
                required
              />
            </Grid>

            {/* Key Trends Management */}
            <Grid item xs={12}>
              <Box display="flex" alignItems="center" mb={2}>
                <TextField
                  fullWidth
                  label="Add Market Trend"
                  value={newTrend}
                  onChange={(e) => setNewTrend(e.target.value)}
                  sx={{ mr: 2 }}
                />
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={addTrend}
                >
                  Add Trend
                </Button>
              </Box>
              
              {/* Displayed Trends */}
              <Box>
                {marketInsight.key_trends.map((trend) => (
                  <Chip
                    key={trend}
                    label={trend}
                    onDelete={() => removeTrend(trend)}
                    color="primary"
                    variant="outlined"
                    sx={{ mr: 1, mb: 1 }}
                  />
                ))}
              </Box>
            </Grid>

            {/* Submit Button */}
            <Grid item xs={12}>
              <Button 
                type="submit" 
                variant="contained" 
                color="secondary" 
                fullWidth
              >
                Update Market Insights & Predict Demand
              </Button>
            </Grid>
          </Grid>
        </form>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Box>
    </Container>
  );
}

export default MarketTrendsPage;