import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

const ProductDemandDashboard = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [products, setProducts] = useState([]);
  const [viewMode, setViewMode] = useState('bar');
  const [activeTab, setActiveTab] = useState('individual');
  
  // API base URL - change this to your actual API endpoint
  const API_BASE_URL = 'http://localhost:5000/api';
  
  useEffect(() => {
    // Fetch list of available products
    const fetchProducts = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/products`);
        if (!response.ok) {
          throw new Error('Failed to fetch products');
        }
        const data = await response.json();
        setProducts(data.products || []);
        
        // Select first product by default if available
        if (data.products && data.products.length > 0) {
          setSelectedProduct(data.products[0]);
          fetchProductData(data.products[0]);
        } else {
          setLoading(false);
        }
      } catch (error) {
        console.error('Error fetching products:', error);
        setError('Failed to load products. Please try again later.');
        setLoading(false);
        
        // Use sample data for demo purposes
        useSampleData();
      }
    };
    
    const fetchProductData = async (productName) => {
      try {
        const response = await fetch(`${API_BASE_URL}/demand/${encodeURIComponent(productName)}`);
        if (!response.ok) {
          throw new Error('Failed to fetch product data');
        }
        const productData = await response.json();
        
        // Add to existing data or replace if already exists
        setData(prevData => {
          const filteredData = prevData.filter(item => item.product !== productName);
          return [...filteredData, productData];
        });
        
        setLoading(false);
      } catch (error) {
        console.error(`Error fetching data for ${productName}:`, error);
        setError(`Failed to load data for ${productName}. Please try again later.`);
        setLoading(false);
      }
    };
    
    const fetchAllData = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/demand`);
        if (!response.ok) {
          throw new Error('Failed to fetch all products data');
        }
        const allData = await response.json();
        setData(allData.predictions || []);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching all product data:', error);
        setError('Failed to load product data. Please try again later.');
        setLoading(false);
        
        // Use sample data for demo purposes
        useSampleData();
      }
    };
    
    const useSampleData = () => {
      // Sample data for demonstration
      const sampleData = [
        {
          "product": "Bush Somerset Collection Bookcase",
          "months": {
            "Apr-2024": 12,
            "May-2024": 15,
            "Jun-2024": 21,
            "Jul-2024": 18,
            "Aug-2024": 25,
            "Sep-2024": 22
          },
          "insights": {
            "average_demand": 18.8,
            "trend": "Increasing",
            "peak_month": "Aug-2024",
            "peak_demand": 25,
            "low_month": "Apr-2024",
            "low_demand": 12,
            "total_predicted_demand": 113,
            "growth_percentage": 83.3,
            "recommendation": "Consider increasing inventory and production capacity to meet growing demand."
          }
        },
        {
          "product": "Hon Deluxe Fabric Upholstered Stacking Chairs",
          "months": {
            "Apr-2024": 33,
            "May-2024": 36,
            "Jun-2024": 42,
            "Jul-2024": 38,
            "Aug-2024": 45,
            "Sep-2024": 48
          },
          "insights": {
            "average_demand": 40.3,
            "trend": "Increasing",
            "peak_month": "Sep-2024",
            "peak_demand": 48,
            "low_month": "Apr-2024",
            "low_demand": 33,
            "total_predicted_demand": 242,
            "growth_percentage": 45.5,
            "recommendation": "Consider increasing inventory and production capacity to meet growing demand."
          }
        },
        {
          "product": "Self-Adhesive Address Labels",
          "months": {
            "Apr-2024": 55,
            "May-2024": 48,
            "Jun-2024": 52,
            "Jul-2024": 63,
            "Aug-2024": 59,
            "Sep-2024": 67
          },
          "insights": {
            "average_demand": 57.3,
            "trend": "Slightly Increasing",
            "peak_month": "Sep-2024",
            "peak_demand": 67,
            "low_month": "May-2024",
            "low_demand": 48,
            "total_predicted_demand": 344,
            "growth_percentage": 21.8,
            "recommendation": "Maintain current inventory levels with modest increases to meet growing demand."
          }
        }
      ];
      
      setData(sampleData);
      setProducts(sampleData.map(item => item.product));
      setSelectedProduct(sampleData[0].product);
    };
    
    // Start fetching data
    fetchProducts();
    fetchAllData();
  }, []);
  
  // Watch for product selection changes
  useEffect(() => {
    if (selectedProduct && !data.some(item => item.product === selectedProduct)) {
      const fetchSelectedProductData = async () => {
        try {
          const response = await fetch(`${API_BASE_URL}/demand/${encodeURIComponent(selectedProduct)}`);
          if (!response.ok) {
            throw new Error('Failed to fetch product data');
          }
          const productData = await response.json();
          
          setData(prevData => {
            const filteredData = prevData.filter(item => item.product !== selectedProduct);
            return [...filteredData, productData];
          });
        } catch (error) {
          console.error(`Error fetching data for ${selectedProduct}:`, error);
          setError(`Failed to load data for ${selectedProduct}. Please try again later.`);
        }
      };
      
      fetchSelectedProductData();
    }
  }, [selectedProduct]);
  
  // Transform data for charts
  const getChartData = () => {
    if (!selectedProduct || loading) return [];
    
    const productData = data.find(item => item.product === selectedProduct);
    if (!productData) return [];
    
    return Object.entries(productData.months).map(([month, demand]) => ({
      month,
      demand
    }));
  };
  
  // Get all products' data for comparison chart
  const getComparisonData = () => {
    if (loading) return [];
    
    // Create an array of all months from all products
    const allMonths = new Set();
    data.forEach(product => {
      Object.keys(product.months).forEach(month => allMonths.add(month));
    });
    
    // Convert to sorted array
    const sortedMonths = Array.from(allMonths).sort();
    
    // Create data for chart
    return sortedMonths.map(month => {
      const monthData = { month };
      data.forEach(product => {
        if (product.months && product.months[month] !== undefined) {
          monthData[product.product] = product.months[month];
        }
      });
      return monthData;
    });
  };
  
  // Get insights for selected product
  const getInsights = () => {
    if (!selectedProduct || loading) return null;
    
    const productData = data.find(item => item.product === selectedProduct);
    if (!productData || !productData.insights) return null;
    
    return productData.insights;
  };
  
  // Prepare data for trend distribution pie chart
  const getTrendDistributionData = () => {
    if (loading || data.length === 0) return [];
    
    const trendCounts = {};
    data.forEach(product => {
      if (product.insights && product.insights.trend) {
        const trend = product.insights.trend;
        trendCounts[trend] = (trendCounts[trend] || 0) + 1;
      }
    });
    
    return Object.entries(trendCounts).map(([trend, count]) => ({
      name: trend,
      value: count
    }));
  };
  
  // Prepare data for product comparison radar chart
  const getRadarData = () => {
    if (loading || !selectedProduct) return [];
    
    const productData = data.find(item => item.product === selectedProduct);
    if (!productData || !productData.months) return [];
    
    return Object.entries(productData.months).map(([month, demand]) => ({
      month,
      [selectedProduct]: demand
    }));
  };
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Loading product demand data...</div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-red-500">{error}</div>
      </div>
    );
  }

  const insights = getInsights();

  return (
    <div className="p-4 max-w-6xl mx-auto">
      <div className="bg-white shadow-lg rounded-lg p-6 mb-6">
        <h1 className="text-2xl font-bold mb-2">Product Demand Prediction Dashboard</h1>
        <p className="text-gray-600 mb-6">Analyze and visualize product demand predictions for the next 6 months</p>
        
        {/* Product Selection Controls */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 space-y-4 md:space-y-0">
          <div className="w-full md:w-1/2">
            <label className="block text-sm font-medium mb-2">Select Product:</label>
            <select 
              className="w-full p-2 border rounded-md"
              value={selectedProduct || ''}
              onChange={(e) => setSelectedProduct(e.target.value)}
            >
              {products.map((product, index) => (
                <option key={index} value={product}>{product}</option>
              ))}
            </select>
          </div>
          
          <div className="flex space-x-2">
            <button 
              onClick={() => setActiveTab('individual')}
              className={`px-4 py-2 rounded-md ${activeTab === 'individual' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            >
              Individual Analysis
            </button>
            <button 
              onClick={() => setActiveTab('comparison')}
              className={`px-4 py-2 rounded-md ${activeTab === 'comparison' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            >
              Comparison
            </button>
          </div>
        </div>
        
        {/* Visualization Controls */}
        {activeTab === 'individual' && (
          <div className="flex justify-end mb-4">
            <div className="flex space-x-2">
              <button 
                onClick={() => setViewMode('bar')}
                className={`px-4 py-2 rounded-md ${viewMode === 'bar' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
              >
                Bar Chart
              </button>
              <button 
                onClick={() => setViewMode('line')}
                className={`px-4 py-2 rounded-md ${viewMode === 'line' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
              >
                Line Chart
              </button>
              <button 
                onClick={() => setViewMode('radar')}
                className={`px-4 py-2 rounded-md ${viewMode === 'radar' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
              >
                Radar Chart
              </button>
            </div>
          </div>
        )}
        
        {/* Individual Product Analysis Tab */}
        {activeTab === 'individual' && (
          <div>
            {/* Individual Product Chart */}
            <div className="mb-8">
              <h2 className="text-xl font-semibold mb-4">Predicted Demand: {selectedProduct}</h2>
              <div className="h-64 md:h-80">
                <ResponsiveContainer width="100%" height="100%">
                  {viewMode === 'bar' && (
                    <BarChart data={getChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="demand" fill="#3B82F6" name="Predicted Quantity" />
                    </BarChart>
                  )}
                  {viewMode === 'line' && (
                    <LineChart data={getChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="demand" stroke="#3B82F6" name="Predicted Quantity" strokeWidth={2} />
                    </LineChart>
                  )}
                  {viewMode === 'radar' && (
                    <RadarChart data={getRadarData()} outerRadius={90}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="month" />
                      <PolarRadiusAxis />
                      <Radar name={selectedProduct} dataKey={selectedProduct} stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                      <Legend />
                      <Tooltip />
                    </RadarChart>
                  )}
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Product Insights */}
            {insights && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-4">Demand Insights</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Average Monthly Demand:</span>
                      <span className="font-medium">{insights.average_demand} units</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Predicted Demand:</span>
                      <span className="font-medium">{insights.total_predicted_demand} units</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Peak Demand Month:</span>
                      <span className="font-medium">{insights.peak_month} ({insights.peak_demand} units)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Lowest Demand Month:</span>
                      <span className="font-medium">{insights.low_month} ({insights.low_demand} units)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Demand Growth:</span>
                      <span className={`font-medium ${insights.growth_percentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {insights.growth_percentage >= 0 ? '+' : ''}{insights.growth_percentage}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Trend Analysis:</span>
                      <span className={`font-medium ${
                        insights.trend.includes('Increasing') ? 'text-green-600' : 
                        insights.trend.includes('Decreasing') ? 'text-red-600' : 'text-blue-600'
                      }`}>
                        {insights.trend}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-2">Business Recommendation</h3>
                  <p className="text-gray-700">{insights.recommendation}</p>
                  
                  <div className="mt-4">
                    <h4 className="font-medium mb-2">Actions to Consider:</h4>
                    <ul className="list-disc list-inside space-y-1 text-gray-700">
                      {insights.trend.includes('Increasing') && (
                        <>
                          <li>Increase inventory levels to meet growing demand</li>
                          <li>Optimize supply chain for higher volume</li>
                          <li>Consider expanding production capacity</li>
                        </>
                      )}
                      {insights.trend === 'Stable' && (
                        <>
                          <li>Maintain current inventory levels</li>
                          <li>Focus on operational efficiency</li>
                          <li>Monitor for seasonal variations</li>
                        </>
                      )}
                      {insights.trend.includes('Decreasing') && (
                        <>
                          <li>Reduce inventory to prevent overstocking</li>
                          <li>Implement promotional strategies</li>
                          <li>Review product positioning</li>
                        </>
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Product Comparison Tab */}
        {activeTab === 'comparison' && (
          <div>
            {/* Product Comparison Chart */}
            <div className="mb-8">
              <h2 className="text-xl font-semibold mb-4">Product Demand Comparison</h2>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getComparisonData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {products.map((product, index) => (
                      <Bar 
                        key={index} 
                        dataKey={product} 
                        fill={COLORS[index % COLORS.length]} 
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Trend Distribution */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div>
                <h3 className="text-lg font-semibold mb-4">Product Trend Distribution</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={getTrendDistributionData()}
                        cx="50%"
                        cy="50%"
                        labelLine={true}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({name, percent}) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {getTrendDistributionData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => [`${value} products`, 'Count']} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-4">Overall Insights</h3>
                <div className="space-y-4">
                  <p className="text-gray-700">
                    <strong>Top Demand Product:</strong> {
                      data.reduce((max, product) => {
                        const totalDemand = Object.values(product.months || {}).reduce((sum, val) => sum + val, 0);
                        return totalDemand > max.demand ? {name: product.product, demand: totalDemand} : max;
                      }, {name: 'None', demand: 0}).name
                    }
                  </p>
                  
                  <p className="text-gray-700">
                    <strong>Best Growth Product:</strong> {
                      data.reduce((max, product) => {
                        const growth = product.insights?.growth_percentage || 0;
                        return growth > max.growth ? {name: product.product, growth} : max;
                      }, {name: 'None', growth: -Infinity}).name
                    }
                  </p>
                  
                  <p className="text-gray-700">
                    <strong>Highest Average Demand:</strong> {
                      data.reduce((max, product) => {
                        const avg = product.insights?.average_demand || 0;
                        return avg > max.avg ? {name: product.product, avg} : max;
                      }, {name: 'None', avg: 0}).name
                    }
                  </p>
                  
                  