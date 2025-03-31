import React, { useEffect, useState } from 'react';
import { 
  ThemeProvider, 
  createTheme 
} from '@mui/material/styles';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
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
  Paper,
  Alert,
  Tooltip,
  IconButton,
  Chip
} from '@mui/material';
import { 
  InfoOutlined as InfoIcon, 
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon
} from '@mui/icons-material';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import MarketTrendsPage from './MarkeTrends';

// Theme Configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
  },
});

// Predefined Options for Select Inputs
const COUNTRIES = ['United States', 'Canada', 'United Kingdom'];
const CATEGORIES = ['Furniture', 'Technology', 'Office Supplies'];
const SUBCATEGORY = ['Bookcases','Chairs' , 'Tables',  'Accessories', 'Phones', 'Machines','Storage','Appliances', 'Art']
const REGION = ['South' ,'West' ,'Central', 'East']

function HomePage() {
  return (
    <Container maxWidth="lg">
      <Box my={4} textAlign="center">
        <Typography variant="h3" gutterBottom>
          AI Powered Demand Forecasting
        </Typography>
        <Typography variant="h6" color="textSecondary" paragraph>
          Leverage advanced machine learning to predict product demand and optimize your inventory strategy.
        </Typography>
        
        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h5">Predictive Analytics</Typography>
                <Typography variant="body2">
                  Use our AI model to forecast product demand based on historical data and market trends.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h5">Market Insights</Typography>
                <Typography variant="body2">
                  Get detailed market insights including growth rates, sentiment, and key trends.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h5">Stock Optimization</Typography>
                <Typography variant="body2">
                  Receive recommended stock levels to minimize overstock and stockouts.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Box mt={4}>
          <Button 
            variant="contained" 
            color="primary" 
            component={Link} 
            to="/demand-prediction"
            size="large"
          >
            Test AI Demand Prediction
          </Button>
        </Box>
      </Box>
    </Container>
  );
}

function DemandPredictionPage() {
  const location = useLocation();
  
  const [formData, setFormData] = useState({
    Segment: "Consumer",
    Country: 'United States',
    City: '',
    State: '',
    Region: 'East',
    Category: 'Furniture',
    'Sub-Category': '',
    'Product Name': '',
    Sales: 50.00,
    Discount: 0.1,
    PredictionPeriodMonths: 6
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);
  const [marketInsights, setMarketInsights] = useState({});
  console.log(marketInsights)
  // Handle location state for prefilling and market insights
  useEffect(() => {
    if (location.state) {
      // Prefill category if passed from Market Trends
      if (location.state.prefillCategory) {
        setFormData(prev => ({
          ...prev,
          Category: location.state.prefillCategory
        }));
      }

      // Store market insights if passed
      if (location.state.marketInsights) {
        setMarketInsights(location.state.marketInsights);
      }
    }
  }, [location.state]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        'http://localhost:5000/predict_future_demand', 
        {
          demandData: formData,
          marketInsights: marketInsights
        }
      );
      setPredictionResult(response.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
      setPredictionResult(null);
    }
  };
  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          AI Demand Prediction
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Country</InputLabel>
                <Select
                  name="Country"
                  value={formData.Country}
                  onChange={handleInputChange}
                >
                  {COUNTRIES.map(country => (
                    <MenuItem key={country} value={country}>
                      {country}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="City"
                label="City"
                value={formData.City}
                onChange={handleInputChange}
                helperText="Enter the city for more precise prediction"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="State"
                label="State"
                value={formData.State}
                onChange={handleInputChange}
                helperText="Enter the state for regional insights"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Region</InputLabel>
                <Select
                  name="Region"
                  value={formData.Region}
                  onChange={handleInputChange}
                >
                  {REGION.map(region => (
                    <MenuItem key={region} value={region}>
                      {region}
                    </MenuItem>
                  ))}
                </Select> 
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Category</InputLabel>
                <Select
                  name="Category"
                  value={formData.Category}
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

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Sub Category</InputLabel>
                <Select
                  name="Sub-Category"
                  value={formData["Sub-Category"]}
                  onChange={handleInputChange}
                >
                  {SUBCATEGORY.map(category => (
                    <MenuItem key={category} value={category}>
                      {category}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="Product Name"
                label="Product Name"
                value={formData['Product Name']}
                onChange={handleInputChange}
                helperText="Specific product name for detailed prediction"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="Sales"
                label="Sales Amount"
                type="number"
                value={formData.Sales}
                onChange={handleInputChange}
                helperText="Current sales amount"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                name="Discount"
                label="Discount Rate"
                type="number"
                inputProps={{ step: 0.01, min: 0, max: 1 }}
                value={formData.Discount}
                onChange={handleInputChange}
                helperText="Discount rate (0-1)"
              />
            </Grid>

            <Grid item xs={12}>
              <Button 
                type="submit" 
                variant="contained" 
                color="primary" 
                fullWidth
              >
                Predict Demand
              </Button>
            </Grid>
          </Grid>
        </form>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

{predictionResult && (
          <Box mt={4}>
            <Grid container spacing={3}>
              {/* Main Prediction Insights */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Demand Forecast Summary
                      <Tooltip title="AI-powered demand prediction based on historical data and market trends">
                        <IconButton size="small" sx={{ ml: 1 }}>
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle1">
                          Base Demand
                          <Tooltip title="Current baseline demand before market adjustments">
                            <IconButton size="small" sx={{ ml: 1 }}>
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Typography>
                        <Typography variant="h6" color="primary">
                          {predictionResult.base_demand.toFixed(2)}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle1">
                          Projected Demand
                          <Tooltip title="Estimated future demand considering market factors">
                            <IconButton size="small" sx={{ ml: 1 }}>
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Typography>
                        <Typography variant="h6" color="secondary">
                          {predictionResult.projected_demand.toFixed(2)}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="subtitle1">
                          Demand Classification
                          {predictionResult.demand_classification}
                        </Typography>
                        <Typography variant="body1">
                          {predictionResult.demand_classification}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="subtitle1">
                          Recommended Stock
                          <Tooltip title="Optimal inventory level to meet projected demand">
                            <IconButton size="small" sx={{ ml: 1 }}>
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Typography>
                        <Typography variant="h6">
                          {predictionResult.recommended_stock} units
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Market Insights */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Market Insights
                      <Tooltip title="Comprehensive market analysis and trends">
                        <IconButton size="small" sx={{ ml: 1 }}>
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle1">Growth Rate</Typography>
                        <Typography variant="h6" color="primary">
                          {(predictionResult.market_insights.growth_rate * 100).toFixed(2)}%
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle1">Market Sentiment</Typography>
                        <Typography variant="h6" color="secondary">
                          {predictionResult.market_insights.market_sentiment}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="subtitle1">Key Market Trends</Typography>
                        {predictionResult.market_insights.key_trends.map((trend, index) => (
                          <Chip 
                            key={index} 
                            label={trend} 
                            variant="outlined" 
                            color="primary" 
                            sx={{ mr: 1, mb: 1 }}
                          />
                        ))}
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="subtitle1">Prediction Period</Typography>
                        <Typography variant="body1">
                          {predictionResult.prediction_period.start_date} to {predictionResult.prediction_period.end_date}
                          <br />
                          Duration: {predictionResult.prediction_period.months} months
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}
      </Paper>
    </Container>
  );
}



function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              AI Demand Forecasting
            </Typography>
            <Button color="inherit" component={Link} to="/">Home</Button>
            <Button color="inherit" component={Link} to="/demand-prediction">Predict Demand</Button>
            <Button color="inherit" component={Link} to="/market-trends">Market Trends</Button>
          </Toolbar>
        </AppBar>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/demand-prediction" element={<DemandPredictionPage />} />
          <Route path="/market-trends" element={<MarketTrendsPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;