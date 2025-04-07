import React, { useState, useEffect } from 'react';
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
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress
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
  
  // New state variables for dynamic competitor activity
  const [competitors, setCompetitors] = useState([]);
  const [loadingCompetitors, setLoadingCompetitors] = useState(false);
  
  // New state variables for market sentiment
  const [productFeedback, setProductFeedback] = useState([]);
  const [loadingFeedback, setLoadingFeedback] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState('');

  const productOptions = [
    'Apple iPhone 14 Plus','Apple iPhone 15 Plus', 'Samsung Galaxy S22', 'Google Pixel 8', 'OnePlus 12 5G','OnePlus 11 5G' ,'Xiaomi 14'
  ];
  
  const ramOptions = ['6GB', '8GB', '16GB'];
  const memoryOptions = ['128GB','256GB', '512GB'];

  // Load competitors when product changes
  useEffect(() => {
    if (productName) {
      fetchCompetitors(productName);
    }
  }, [productName]);

  const fetchCompetitors = async (product) => {
    setLoadingCompetitors(true);
    try {
      // In a real app, this would be an API call to get competitor data
      // For this example, we'll simulate the API response
      const response = await axios.get(`http://localhost:5000/get_competitor_data?q=${product}`);
      console.log(response.data.data);
      setCompetitors(response.data.data.competitors);
      
      // Calculate an overall competitor activity score
      const calculatedScore = calculateCompetitorScore(response.data.data.competitors);
      setCompetitorActivity(calculatedScore);
      
    } catch (error) {
      console.error('Error fetching competitors:', error);
      setError('Failed to load competitor data');
    } finally {
      setLoadingCompetitors(false);
    }
  };

  // Function to calculate competitor score based on available data
  const calculateCompetitorScore = (competitorList) => {
    console.log('Calculating', competitorList);
    if (!competitorList || competitorList.length === 0) return 0.0;
  
    // Define weights
    const weights = {
      priceDifference: 0.3,
      googleTrends: 0.3,
    };
  
    const competitorScores = competitorList.map(competitor => {
      let totalScore = 0;
      let totalWeight = 0;
  
      // Safe parse of ourPrice
      const ourPrice = parseFloat(competitor.ourPrice);
      const compPrice = parseFloat(competitor.price);
  
      //  Price difference score (only if both prices are valid numbers)
      if (!isNaN(ourPrice) && !isNaN(compPrice) && ourPrice > 0) {
        const priceDiff = (ourPrice - compPrice) / ourPrice;
        const priceScore = Math.min(Math.max(priceDiff + 0.5, 0), 1); // normalized
        totalScore += priceScore * weights.priceDifference;
        totalWeight += weights.priceDifference;
      }
  
      //  Google Trends score (safe parse)
      const trendsScore = parseFloat(competitor.googleTrends);
      if (!isNaN(trendsScore)) {
        totalScore += (trendsScore / 100) * weights.googleTrends;
        totalWeight += weights.googleTrends;
      }
  
      // //  Ad Spend score (optional)
      // if (!isNaN(parseFloat(competitor.adSpend))) {
      //   totalScore += parseFloat(competitor.adSpend) * weights.adSpend;
      //   totalWeight += weights.adSpend;
      // }  
  
      // //  Social Media score (optional)
      // if (!isNaN(parseFloat(competitor.socialMediaScore))) {
      //   totalScore += parseFloat(competitor.socialMediaScore) * weights.socialMediaScore;
      //   totalWeight += weights.socialMediaScore;
      // }
  
      // Fallback score
      return totalWeight > 0 ? totalScore / totalWeight : 0.5;
    });
  
    const avgScore = competitorScores.reduce((sum, score) => sum + score, 0) / competitorScores.length;
    return parseFloat(avgScore.toFixed(2));
  };
  

  // Fetch product feedback for market sentiment
  const handleFetchProductFeedback = async () => {
    if (!selectedProduct) return;
    
    setLoadingFeedback(true);
    try {
      // Real API call to get product feedback
      const response = await axios.get(`http://localhost:5000/get_product_reviews?q=${selectedProduct}`);
      
      // Process the reviews from the response
      const reviewData = response.data.reviews.map(review => ({
        text: review.content,
        // We'll calculate sentiment on frontend for now
        // In a real app, the API might provide sentiment scores
        sentiment: Math.random() * 0.6 + 0.2 // Dummy sentiment between 0.2 and 0.8
      }));
      
      setProductFeedback(reviewData);
      
      // Calculate average sentiment
      const avgSentiment = reviewData.reduce((sum, item) => sum + item.sentiment, 0) / 
                          reviewData.length;
      
      setMarketSentiment(parseFloat(avgSentiment.toFixed(2)));
      
      // Add fetched feedback to feedbackList for display
      setFeedbackList(reviewData.map(item => item.text));
      
    } catch (error) {
      console.error('Error fetching product feedback:', error);
      setError('Failed to load product feedback');
    } finally {
      setLoadingFeedback(false);
    }
  };

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
          
          {/* Updated Competitor Activity Calculator */}
          <Grid item xs={12} md={8}>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Competitor Activity Calculator
                </Typography>
                <Typography gutterBottom>
                  Current Value: {competitorActivity} {loadingCompetitors && <CircularProgress size={16} sx={{ ml: 1 }} />}
                </Typography>
                <Slider
                  value={competitorActivity}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(e, newValue) => setCompetitorActivity(newValue)}
                  valueLabelDisplay="auto"
                  marks={[
                    { value: 0, label: 'Low' },
                    { value: 0.5, label: 'Medium' },
                    { value: 1, label: 'High' }
                  ]}
                  sx={{ mb: 2 }}
                />
                
                {competitors && (
                  <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Competitor</TableCell>
                          <TableCell align="right">Price</TableCell>
                          <TableCell align="right">Our Price</TableCell>
                          <TableCell align="right">Google Trends</TableCell>
                          <TableCell align="right">Source</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {competitors.map((competitor, index) => (
                          <TableRow key={index}>
                            <TableCell component="th" scope="row">
                              {competitor.name}
                            </TableCell>
                            <TableCell align="right">${competitor.price}</TableCell>
                            <TableCell align="right">${competitor.ourPrice}</TableCell>
                            <TableCell align="center">{competitor.googleTrends}%</TableCell>
                            <TableCell align="right">{competitor.source}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
                
                <Button 
                  variant="outlined" 
                  onClick={() => fetchCompetitors(productName)}
                  disabled={!productName || loadingCompetitors}
                >
                  {loadingCompetitors ? 'Loading...' : 'Analyze Competition'}
                </Button>
              </CardContent>
            </Card>
          </Grid>
          
          {/* <Grid item xs={12} md={6}>
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
          </Grid> */}
         
          
          {/* Updated Market Sentiment Calculator */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Market Sentiment Calculator
                </Typography>
                <Typography gutterBottom>
                  Current Sentiment: {marketSentiment.toFixed(2)} {loadingFeedback && <CircularProgress size={16} sx={{ ml: 1 }} />}
                </Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} md={8}>
                    <Autocomplete
                      options={productOptions}
                      value={selectedProduct}
                      onChange={(event, newValue) => {
                        setSelectedProduct(newValue);
                      }}
                      renderInput={(params) => <TextField {...params} label="Select Product for Feedback" fullWidth />}
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Button 
                      variant="outlined" 
                      onClick={handleFetchProductFeedback}
                      disabled={!selectedProduct || loadingFeedback}
                      fullWidth
                      sx={{ height: '100%' }}
                    >
                      Fetch Feedback
                    </Button>
                  </Grid>
                </Grid>
                
                <Divider sx={{ my: 2 }}>
                  <Chip label="Add Custom Feedback" size="small" />
                </Divider>
                
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
                  sx={{ mr: 2, mb: 2 }}
                >
                  Add Feedback
                </Button>
                <Button 
                  variant="outlined" 
                  onClick={calculateMarketSentiment}
                  disabled={feedbackList.length === 0}
                  sx={{ mb: 2 }}
                >
                 Calculate Sentiment
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