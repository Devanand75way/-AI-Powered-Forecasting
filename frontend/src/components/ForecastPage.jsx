import React, { useState } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  TextField, 
  Button, 
  Grid,
  MenuItem,
  Slider,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Chip,
  Stack,
  Divider,
  Card,
  CardContent,
  Autocomplete,
  Alert,
  Snackbar
} from '@mui/material';
import { motion } from 'framer-motion';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import axios from 'axios';

const ForecastPage = () => {
  const [productName, setProductName] = useState('');
  const [forecastMonths, setForecastMonths] = useState(8);
  const [ram, setRam] = useState('6GB');
  const [memory, setMemory] = useState('256GB');
  const [competitorActivity, setCompetitorActivity] = useState(0.5);
  const [weatherCondition, setWeatherCondition] = useState(0);
  const [marketSentiment, setMarketSentiment] = useState(0.0);
  const [holidayIndicator, setHolidayIndicator] = useState(1);
  const [feedback, setFeedback] = useState('');
  const [feedbackList, setFeedbackList] = useState([]);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [productDetails, setProductDetails] = useState(null);

  const productOptions = [
    'iPhone 14','iPhone 15', 'Samsung Galaxy S22', 'Google Pixel 8', 'OnePlus 12','OnePlus 11' ,'Xiaomi 14'
  ];
  
  const ramOptions = ['6GB', '8GB', '16GB'];
  const memoryOptions = ['256GB', '512GB'];

  const handleAddFeedback = () => {
    if (feedback.trim()) {
      setFeedbackList([...feedbackList, feedback]);
      setFeedback('');
      
      // Recalculate market sentiment based on feedback
      // This is a simple simulation - in a real app, this would call an API
      const newSentiment = Math.min(Math.random() * 0.2 + marketSentiment, 1);
      setMarketSentiment(newSentiment);
    }
  };

  const handleRemoveFeedback = (index) => {
    const newList = [...feedbackList];
    newList.splice(index, 1);
    setFeedbackList(newList);
  };

  const handleCalculateCompetitorActivity = () => {
    // Simulate competitor activity calculation
    const newValue = Math.min(Math.random() * 0.3 + competitorActivity, 1);
    setCompetitorActivity(parseFloat(newValue.toFixed(2)));
  };

  const calculateMarketSentiment = async () => {
    try {
      const response = await axios.post('http://localhost:5000/analyze_sentiment', {
        feedback: feedbackList
      });
      console.log(response);
      setMarketSentiment(response.data.average_sentiment);
    } catch (error) {
      console.log(error);
      setError("Error calculating market sentiment. Using current value.");
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return months[date.getMonth()];
  };

  const handleForecast = async () => {
    if (!productName) return;
    
    setLoading(true);
    setError(null);
    setForecastData(null);
    setProductDetails(null);
    
    // Create the forecast request body
    const forecastRequest = {
      product_name: productName,
      forecast_months: forecastMonths,
      ram: ram,
      memory: memory,
      external_factors: {
        competitor_activity: competitorActivity,
        weather_condition: weatherCondition,
        market_sentiment: marketSentiment,
        holiday_indicator: holidayIndicator
      }
    };
    
    console.log("Forecast request:", forecastRequest);
    
    try {
      const response = await axios.post("http://127.0.0.1:5000/sales-forecast", forecastRequest);
      console.log("Forecast response:", response.data);
      
      // Process the API response data
      if (response.data.forecast_data) {
        // Transform API response to chart data format
        const chartData = response.data.forecast_data.map(item => ({
          month: formatDate(item.Date),
          sales: Math.round(item["Predicted Sales"] * 1000), // Scale for better visualization
          forecast: true
        }));
        
        setForecastData(chartData);
        setProductDetails(response.data);
      }
      
      setLoading(false);
    } catch (error) {
      console.error("Forecast error:", error);
      
      // Handle specific error messages from API
      if (error.response && error.response.data && error.response.data.error) {
        setError(error.response.data.error);
      } else {
        setError("Error generating forecast. Please try again later.");
      }
      
      setForecastData(null);
      setLoading(false);
    }
  };

  const handleCloseError = () => {
    setError(null);
  };

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          Future Sales Forecast Prediction
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Autocomplete
              options={productOptions}
              value={productName}
              onChange={(event, newValue) => {
                setProductName(newValue);
              }}
              renderInput={(params) => <TextField {...params} label="Product Name" fullWidth />}
            />
          </Grid>
          
          <Grid item xs={12} md={3}>
            <TextField
              select
              label="RAM"
              value={ram}
              onChange={(e) => setRam(e.target.value)}
              fullWidth
            >
              {ramOptions.map((option) => (
                <MenuItem key={option} value={option}>{option}</MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              select
              label="Memory"
              value={memory}
              onChange={(e) => setMemory(e.target.value)}
              fullWidth
            >
              {memoryOptions.map((option) => (
                <MenuItem key={option} value={option}>{option}</MenuItem>
              ))}
            </TextField>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <TextField
              label="Forecast Months"
              type="number"
              value={forecastMonths}
              onChange={(e) => setForecastMonths(parseInt(e.target.value))}
              InputProps={{ inputProps: { min: 1, max: 24 } }}
              fullWidth
            />
          </Grid>
          
          <Grid item xs={12}>
            <Divider sx={{ my: 2 }}>
              <Chip label="External Factors" />
            </Divider>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Competitor Activity Calculator
                </Typography>
                <Typography gutterBottom>Current Value: {competitorActivity}</Typography>
                <Slider
                  value={competitorActivity}
                  min={0}
                  max={1}
                  step={0.1}
                  onChange={(e, newValue) => setCompetitorActivity(newValue)}
                  valueLabelDisplay="auto"
                  marks
                  sx={{ mb: 2 }}
                />
                <Button variant="outlined" onClick={handleCalculateCompetitorActivity}>
                  Calculate Competition
                </Button>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Weather Condition
                </Typography>
                <Slider
                  value={weatherCondition}
                  min={-1}
                  max={1}
                  step={0.1}
                  onChange={(e, newValue) => setWeatherCondition(newValue)}
                  valueLabelDisplay="auto"
                  marks={[
                    { value: -1, label: 'Bad' },
                    { value: 0, label: 'Neutral' },
                    { value: 1, label: 'Good' },
                  ]}
                />
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Market Sentiment Calculator
                </Typography>
                <Typography gutterBottom>Current Sentiment: {marketSentiment.toFixed(2)}</Typography>
                
                <TextField
                  label="Product Feedback"
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  fullWidth
                  multiline
                  rows={2}
                  sx={{ mb: 2 }}
                />
                
                <Button 
                  variant="outlined" 
                  onClick={handleAddFeedback}
                  disabled={!feedback.trim()}
                  sx={{ mr: 2 , mb: 2 }}
                >
                  Add Feedback
                </Button>
                <Button 
                  variant="outlined" 
                  onClick={calculateMarketSentiment}
                  disabled={feedbackList.length === 0}
                  sx={{ mb: 2 }}
                >
                 Calculate
                </Button>
                
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {feedbackList.map((item, index) => (
                    <Chip 
                      key={index} 
                      label={item} 
                      onDelete={() => handleRemoveFeedback(index)}
                      sx={{ mb: 1 }}
                    />
                  ))}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Holiday Indicator
                </Typography>
                <FormControl>
                  <RadioGroup
                    value={holidayIndicator}
                    onChange={(e) => setHolidayIndicator(parseInt(e.target.value))}
                  >
                    <FormControlLabel value={1} control={<Radio />} label="Holiday Period" />
                    <FormControlLabel value={0} control={<Radio />} label="Non-Holiday Period" />
                  </RadioGroup>
                </FormControl>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleForecast}
              disabled={loading || !productName}
              fullWidth
              size="large"
              sx={{ py: 1.5 }}
            >
              {loading ? 'Calculating Forecast...' : 'Generate Sales Forecast'}
            </Button>
          </Grid>
        </Grid>
        
        {error && (
          <Alert 
            severity="error" 
            sx={{ mt: 3 }}
            onClose={handleCloseError}
          >
            {error}
          </Alert>
        )}
        
        {forecastData && productDetails && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            style={{ height: 400, marginTop: 30 }}
          >
            <Typography variant="h6" gutterBottom textAlign="center">
              Sales Forecast for {productDetails.product_name} ({productDetails.ram || ram})
            </Typography>
            
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={forecastData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="sales" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={{ r: 6 }}
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                 1 Year Past Sales Data:
              </Typography>
              <Typography variant="body2">
                • Product: {productDetails.product_name} ({productDetails.ram || ram})
              </Typography>
              {productDetails.past_performance && (
                <>
                  <Typography variant="body2">
                    • Sales Trend: {productDetails.past_performance.sales_trend}
                  </Typography>
                  <Typography variant="body2">
                    • Best Month: {productDetails.past_performance.best_month}
                  </Typography>
                  <Typography variant="body2">
                    • Worst Month: {productDetails.past_performance.worst_month}
                  </Typography>
                </>
              )}
              {productDetails.external_factors_impact && (
                <>
                  <Typography variant="subtitle1" sx={{ mt: 1 }}>
                    External Factors Impact:
                  </Typography>
                  <Typography variant="body2">
                    • Competitor Activity: {productDetails.external_factors_impact.competitor_activity.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    • Market Sentiment: {productDetails.external_factors_impact.market_sentiment.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    • Weather Impact: {productDetails.external_factors_impact.weather_condition.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    • Holiday Effect: {productDetails.external_factors_impact.holiday_indicator.toFixed(2)}
                  </Typography>
                </>
              )}
            </Box>
          </motion.div>
        )}
      </Paper>
      
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={handleCloseError}
      >
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>

      </Snackbar>
    </motion.div>
  );
};

export default ForecastPage;