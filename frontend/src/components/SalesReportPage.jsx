import React, { useState, useEffect } from "react";
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  MenuItem,
  Autocomplete,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  Divider,
  Chip,
} from "@mui/material";
import { motion } from "framer-motion";
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
} from "recharts";
import axios from "axios";

const ProductVisualizationPage = () => {
  const [product, setProduct] = useState("");
  const [chartType, setChartType] = useState("line");
  const [loading, setLoading] = useState(false);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const sales = [
    {
      best_month: "YYYY-MM",
      monthly_trends: [
        {
          "Actual Sales": 0.0,
          Month: "YYYY-MM",
        },
        {
          "Actual Sales": 0.0,
          Month: "YYYY-MM",
        },
        {
          "Actual Sales": 0.0,
          Month: "YYYY-MM",
        },
        {
          "Actual Sales": 0.0,
          Month: "YYYY-MM",
        },
      ],
      product_sales: [
        {
          "Actual Sales": 0.0,
          Product: "Samsung Galaxy S22",
        },
        {
          "Actual Sales": 0.0,
          Product: "Oppo Find X6",
        },
        {
          "Actual Sales": 0.0,
          Product: "iPhone 14",
        },
        {
          "Actual Sales": 0.0,
          Product: "Samsung Galaxy S23",
        },
        {
          "Actual Sales": 0.0,
          Product: "OnePlus 10 Pro",
        },
        {
          "Actual Sales": 0.0,
          Product: "Xiaomi Mi 12",
        },
        {
          "Actual Sales": 0.0,
          Product: "Xiaomi Mi 13",
        },
        {
          "Actual Sales": 0.0,
          Product: "OnePlus 11",
        },
        {
          "Actual Sales": 0.0,
          Product: "Google Pixel 8",
        },
        {
          "Actual Sales": 0.0,
          Product: "Oppo Find X5",
        },
        {
          "Actual Sales": 0.0,
          Product: "iPhone 15",
        },
      ],
      total_sales: 0.0,
      worst_month: "yyyy-mm",
    },
  ];
  const [salesData, setSalesData] = useState(sales[0]);

  const handleSearch = async () => {
    if (!startDate || !endDate) return;
    setLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/sales-analysis-by-date",
        {
          start_date: startDate,
          end_date: endDate,
        }
      );
      if(response.status === 400){
        console.error("No response from server");
        setLoading(false);
        return;
      }
      setSalesData(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Sales report fetch error:", error);
      setSalesData(null);
      setLoading(false);
    }
  };

  // Format numbers to 2 decimal places
  const formatNumber = (num) => {
    return Number(num).toFixed(2);
  };

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Past Product Sales Visualization
        </Typography>
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
            <TextField
              label="Start Date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              fullWidth
              InputLabelProps={{
                shrink: true,
              }}
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              label="End Date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              fullWidth
              InputLabelProps={{
                shrink: true,
              }}
            />
          </Grid>
          <Grid
            item
            xs={12}
            md={4}
            sx={{ display: "flex", alignItems: "center" }}
          >
            <Button
              variant="contained"
              color="primary"
              onClick={handleSearch}
              disabled={loading}
              fullWidth
            >
              {loading ? "Searching..." : "Search Reports"}
            </Button>
          </Grid>
        </Grid>
{/*  */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
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
        </Grid>


        {/* Sales Overview Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  Total Sales
                </Typography>
                <Typography variant="h4">
                  {formatNumber(salesData.total_sales)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="success.main">
                  Best Month
                </Typography>
                <Typography variant="h4">{salesData.best_month}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="error.main">
                  Worst Month
                </Typography>
                <Typography variant="h4">{salesData.worst_month}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="info.main">
                  Total Products
                </Typography>
                <Typography variant="h4">
                  {salesData.product_sales.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Monthly Trends Chart */}
        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          Monthly Sales Trends
        </Typography>
        <Paper sx={{ p: 2, mb: 4 }} elevation={2}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            style={{ height: 300 }}
          >
            <ResponsiveContainer width="100%" height="100%">
              {chartType === "line" ? (
                <LineChart
                  data={salesData.monthly_trends}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="Month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="Actual Sales"
                    stroke="#8884d8"
                    activeDot={{ r: 8 }}
                  />
                </LineChart>
              ) : (
                <BarChart
                  data={salesData.monthly_trends}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="Month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Actual Sales" fill="#8884d8" />
                </BarChart>
              )}
            </ResponsiveContainer>
          </motion.div>
        </Paper>

        {/* Product Sales Chart */}
        <Typography variant="h6" gutterBottom>
          Product Sales Comparison
        </Typography>
        <Paper sx={{ p: 2, mb: 4 }} elevation={2}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            style={{ height: 400 }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={salesData.product_sales}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="Product" type="category" width={150} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Actual Sales" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        </Paper>

        {/* Data Tables */}
        <Divider sx={{ my: 4 }}>
          <Chip label="Detailed Data" />
        </Divider>

        <Grid container spacing={4}>
          {/* Monthly Trends Table */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Monthly Sales Data
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Month</TableCell>
                    <TableCell align="right">Actual Sales</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {salesData.monthly_trends.map((row, index) => (
                    <TableRow
                      key={index}
                      sx={{
                        backgroundColor:
                          row.Month === salesData.best_month
                            ? "rgba(76, 175, 80, 0.1)"
                            : row.Month === salesData.worst_month
                            ? "rgba(244, 67, 54, 0.1)"
                            : "inherit",
                      }}
                    >
                      <TableCell>{row.Month}</TableCell>
                      <TableCell align="right">
                        {formatNumber(row["Actual Sales"])}
                      </TableCell>
                    </TableRow>
                  ))}
                  <TableRow
                    sx={{
                      fontWeight: "bold",
                      backgroundColor: "rgba(0, 0, 0, 0.05)",
                    }}
                  >
                    <TableCell>
                      <strong>Total</strong>
                    </TableCell>
                    <TableCell align="right">
                      <strong>{formatNumber(salesData.total_sales)}</strong>
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>

          {/* Product Sales Table */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Product Sales Data
            </Typography>
            <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Product</TableCell>
                    <TableCell align="right">Actual Sales</TableCell>
                    <TableCell align="right">% of Total</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {salesData.product_sales.map((row, index) => (
                    <TableRow key={index}>
                      <TableCell>{row.Product}</TableCell>
                      <TableCell align="right">
                        {formatNumber(row["Actual Sales"])}
                      </TableCell>
                      <TableCell align="right">
                        {formatNumber(
                          (row["Actual Sales"] / salesData.total_sales) * 100
                        )}
                        %
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>
      </Paper>
    </motion.div>
  );
};

export default ProductVisualizationPage;
