import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  TextField, 
  Button, 
  Grid,
  MenuItem,
  Autocomplete,
  CircularProgress,
  Tabs,
  Tab
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
  ResponsiveContainer,
  BarChart,
  Bar,
  ReferenceLine
} from 'recharts';
import axios from 'axios';

const ProductVisualizationPage = () => {
  const [product, setProduct] = useState('');
  const [productData, setProductData] = useState(null);
  const [chartType, setChartType] = useState('line');
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [productOptions] = useState([
    'iPhone 14', 'iPhone 15', 'Samsung Galaxy S22', 'Google Pixel 8', 'OnePlus 12', 'Xiaomi 14','Google Pixel 8'
  ]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const formatMonthlyData = (data) => {
    if (!data) return [];
    // Convert API format to chart format
    return data.map(item => ({
      month: item.Month.substring(5), // Taking only MM part from YYYY-MM
      sales: parseFloat(item["Actual Sales"].toFixed(2))
    }));
  };

  // const mockResponse = [
  // {
  //   "best_selling_months": [
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" }
  //   ],
  //   "monthly_trends": [
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales":0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" },
  //     { "Actual Sales": 0.0, "Month": "YYYY-MM" }
  //   ],
  //   "product_name": product,
  //   "total_sales": 0.0
  // }];

  const handleSearch = async() => {
    if (!product) return;
    setLoading(true);
    try {
      const response  = await axios.post("http://127.0.0.1:5000/sales-analysis-by-product", {
        product_name: product
      })
      setProductData(response.data);
      setLoading(false);
      setActiveTab(0);
    } catch (error) {
      console.error("Sales analysis by product error:", error);
      setProductData(null);
      setLoading(false);
      setActiveTab(0);
      alert("Failed to fetch product data. Please try again.");
    }
    
  };

  // Calculate average sales for reference line
  const getAverageSales = () => {
    if (!productData?.monthly_trends) return 0;
    const sum = productData.monthly_trends.reduce((acc, curr) => acc + curr["Actual Sales"], 0);
    return sum / productData.monthly_trends.length;
  };

  // Group data by year for better visualization
  const getYearlyGroupedData = () => {
    if (!productData?.monthly_trends) return [];
    
    const groupedByYear = {};
    
    productData.monthly_trends.forEach(item => {
      const year = item.Month.substring(0, 4);
      const month = item.Month.substring(5);
      
      if (!groupedByYear[year]) {
        groupedByYear[year] = [];
      }
      
      groupedByYear[year].push({
        month,
        sales: item["Actual Sales"]
      });
    });
    
    return groupedByYear;
  };

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          Past Product Sales Visualization
        </Typography>
        
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={5}>
            <Autocomplete
              options={productOptions}
              value={product}
              onChange={(event, newValue) => {
                setProduct(newValue);
              }}
              renderInput={(params) => <TextField {...params} label="Select Product" fullWidth />}
            />
          </Grid>
          
          <Grid item xs={12} md={3}>
            <TextField
              select
              label="Chart Type"
              value={chartType}
              onChange={(e) => setChartType(e.target.value)}
              fullWidth
            >
              <MenuItem value="line">Line Chart</MenuItem>
              <MenuItem value="bar">Bar Chart</MenuItem>
            </TextField>
          </Grid>
          
          <Grid item xs={12} md={4} sx={{ display: 'flex', alignItems: 'center' }}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleSearch}
              disabled={loading || !product}
              fullWidth
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
            >
              {loading ? 'Loading...' : 'Visualize Sales'}
            </Button>
          </Grid>
        </Grid>
        
        {productData && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
              <Tabs value={activeTab} onChange={handleTabChange}>
                <Tab label="Monthly Trends" />
                <Tab label="Yearly Comparison" />
              </Tabs>
            </Box>
            
            {activeTab === 0 && (
              <Box>
                <Typography variant="h6" gutterBottom textAlign="center">
                  Monthly Sales Trends for {productData.product_name}
                </Typography>
                <Typography variant="subtitle2" gutterBottom textAlign="center" color="text.secondary">
                  Total Sales: {productData.total_sales.toFixed(2)} units
                </Typography>
                <Box sx={{ height: 400, mt: 3 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    {chartType === 'line' ? (
                      <LineChart
                        data={formatMonthlyData(productData.monthly_trends)}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip formatter={(value) => [`${value} units`, 'Sales']} />
                        <Legend />
                        <ReferenceLine y={getAverageSales()} stroke="green" strokeDasharray="3 3" label="Average" />
                        <Line type="monotone" dataKey="sales" stroke="#8884d8" activeDot={{ r: 8 }} />
                      </LineChart>
                    ) : (
                      <BarChart
                        data={formatMonthlyData(productData.monthly_trends)}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip formatter={(value) => [`${value} units`, 'Sales']} />
                        <Legend />
                        <ReferenceLine y={getAverageSales()} stroke="green" strokeDasharray="3 3" label="Average" />
                        <Bar dataKey="sales" fill="#8884d8" />
                      </BarChart>
                    )}
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}
            
            {activeTab === 1 && (
              <Box>
                <Typography variant="h6" gutterBottom textAlign="center">
                  Yearly Sales Comparison for {productData.product_name}
                </Typography>
                <Box sx={{ height: 500, mt: 3 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" type="category" allowDuplicatedCategory={false} />
                      <YAxis />
                      <Tooltip formatter={(value) => [`${value} units`, 'Sales']} />
                      <Legend />
                      
                      {Object.entries(getYearlyGroupedData()).map(([year, data], index) => (
                        <Line 
                          key={year}
                          data={data} 
                          type="monotone" 
                          dataKey="sales" 
                          stroke={index === 0 ? '#8884d8' : index === 1 ? '#82ca9d' : '#ff7300'} 
                          name={`Sales ${year}`}
                          activeDot={{ r: 8 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}
          </motion.div>
        )}
      </Paper>
    </motion.div>
  );
};

export default ProductVisualizationPage;