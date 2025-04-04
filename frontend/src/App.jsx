// App.jsx
import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { 
  CssBaseline, 
  Box, 
  Container, 
  AppBar, 
  Toolbar, 
  Typography, 
  Button,
  ThemeProvider,
  createTheme
} from '@mui/material';
import { motion } from 'framer-motion';
import SalesReportPage from './components/SalesReportPage';
import ProductVisualizationPage from './components/ProductVisualizationPage';
import ForecastPage from './components/ForecastPage';
import ForecastWorking from "./components/ForcastWorking"

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Mobile Phone Sales Forecast
              </Typography>
              <Button color="inherit" component={Link} to="/">Sales Reports</Button>
              <Button color="inherit" component={Link} to="/visualization">Product Visualization</Button>
              <Button color="inherit" component={Link} to="/forecast">Sales Forecast</Button>
              <Button color="inherit" component={Link} to="/working">Working</Button>
            </Toolbar>
          </AppBar>
        </Box>
        
        <Container maxWidth="lg" sx={{ mt: 4 }}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Routes>
              <Route path="/" element={<SalesReportPage />} />
              <Route path="/visualization" element={<ProductVisualizationPage />} />
              <Route path="/forecast" element={<ForecastPage />} />
              <Route path="/working" element={<ForecastWorking />} />
            </Routes>
          </motion.div>
        </Container>
      </Router>
    </ThemeProvider>
  );
};

export default App;
