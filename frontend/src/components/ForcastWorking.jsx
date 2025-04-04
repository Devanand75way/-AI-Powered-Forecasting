import React, { useState } from "react";
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Slider,
  Card,
  CardContent,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  CircularProgress,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import InfoIcon from "@mui/icons-material/Info";
import SentimentSatisfiedAltIcon from "@mui/icons-material/SentimentSatisfiedAlt";
import SentimentVeryDissatisfiedIcon from "@mui/icons-material/SentimentVeryDissatisfied";
import SentimentNeutralIcon from "@mui/icons-material/SentimentNeutral";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

// TabPanel component for handling tab content
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const CompetitorSentimentAnalysisPage = () => {
  const [tabValue, setTabValue] = useState(0);
  const [competitorInputs, setCompetitorInputs] = useState({
    salesScore: 0.7,
    priceScore: 0.5,
    discountScore: 0.8,
    trendsScore: 0.4,
    adSpendScore: 0.9,
    socialMediaScore: 0.6,
  });

  const [sentimentInputs, setSentimentInputs] = useState({
    positiveReviews: 75,
    neutralReviews: 15,
    negativeReviews: 10,
    sampleText:
      "I really love this product! The features are amazing and it has great battery life. The price is a bit high though.",
  });

  const [calculatingCompetitor, setCalculatingCompetitor] = useState(false);
  const [calculatingSentiment, setCalculatingSentiment] = useState(false);
  const [competitorResult, setCompetitorResult] = useState(null);
  const [sentimentResult, setSentimentResult] = useState(null);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleCompetitorChange = (prop) => (event, newValue) => {
    setCompetitorInputs({
      ...competitorInputs,
      [prop]:
        newValue !== undefined ? newValue : parseFloat(event.target.value),
    });
  };

  const handleSentimentChange = (prop) => (event) => {
    setSentimentInputs({ ...sentimentInputs, [prop]: event.target.value });
  };

  const calculateCompetitorScore = () => {
    setCalculatingCompetitor(true);

    // Simulate API call or calculation time
    setTimeout(() => {
      // Weights for each factor
      const weights = {
        salesScore: 0.25,
        priceScore: 0.15,
        discountScore: 0.2,
        trendsScore: 0.15,
        adSpendScore: 0.15,
        socialMediaScore: 0.1,
      };

      // Calculate weighted score
      let totalScore = 0;
      const detailedCalculation = [];

      Object.keys(weights).forEach((key) => {
        const factorScore = competitorInputs[key] * weights[key];
        totalScore += factorScore;
        detailedCalculation.push({
          factor: key,
          inputValue: competitorInputs[key],
          weight: weights[key],
          score: factorScore,
        });
      });

      // Normalize to a 0-100 scale for easier understanding
      const normalizedScore = Math.round(totalScore * 100);

      // Determine impact level
      let impactLevel;
      if (normalizedScore >= 80) impactLevel = "Very High";
      else if (normalizedScore >= 60) impactLevel = "High";
      else if (normalizedScore >= 40) impactLevel = "Medium";
      else if (normalizedScore >= 20) impactLevel = "Low";
      else impactLevel = "Very Low";

      setCompetitorResult({
        totalScore,
        normalizedScore,
        impactLevel,
        detailedCalculation,
        radarData: Object.keys(competitorInputs).map((key) => ({
          factor: key.replace("Score", ""),
          value: competitorInputs[key],
          fullMark: 1,
        })),
      });

      setCalculatingCompetitor(false);
    }, 1000);
  };

  const calculateSentimentScore = () => {
    setCalculatingSentiment(true);

    // Simulate API call or TextBlob analysis
    setTimeout(() => {
      // For demonstration, we'll calculate sentiment based on the distribution
      // In reality, you would use TextBlob or another NLP tool to analyze the text

      const { positiveReviews, neutralReviews, negativeReviews, sampleText } =
        sentimentInputs;

      // Calculate overall sentiment score (-1 to 1 scale)
      const totalReviews = positiveReviews + neutralReviews + negativeReviews;
      const sentimentScore =
        (positiveReviews * 1 + neutralReviews * 0 + negativeReviews * -1) /
        totalReviews;

      // Simulate TextBlob analysis of sample text
      // In a real implementation, you would use the TextBlob library in Python
      // Here we're just simulating a result
      let textSentiment;
      if (
        sampleText.toLowerCase().includes("love") ||
        sampleText.toLowerCase().includes("great")
      ) {
        textSentiment = {
          polarity: 0.7,
          subjectivity: 0.8,
          classification: "Positive",
        };
      } else if (
        sampleText.toLowerCase().includes("hate") ||
        sampleText.toLowerCase().includes("terrible")
      ) {
        textSentiment = {
          polarity: -0.7,
          subjectivity: 0.8,
          classification: "Negative",
        };
      } else {
        textSentiment = {
          polarity: 0.1,
          subjectivity: 0.5,
          classification: "Neutral",
        };
      }

      // Determine overall sentiment category
      let sentimentCategory;
      if (sentimentScore > 0.3) sentimentCategory = "Positive";
      else if (sentimentScore < -0.3) sentimentCategory = "Negative";
      else sentimentCategory = "Neutral";

      // Prepare chart data
      const pieChartData = [
        { name: "Positive", value: positiveReviews },
        { name: "Neutral", value: neutralReviews },
        { name: "Negative", value: negativeReviews },
      ];

      setSentimentResult({
        sentimentScore,
        sentimentCategory,
        textSentiment,
        pieChartData,
        normalizedScore: Math.round((sentimentScore + 1) * 50), // Convert -1 to 1 scale to 0-100
      });

      setCalculatingSentiment(false);
    }, 1000);
  };

  const COLORS = ["#0088FE", "#FFBB28", "#FF8042"];

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
        Competitive Analysis Dashboard
      </Typography>

      <Paper sx={{ width: "100%", mb: 4 }}>
        <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            aria-label="analysis tabs"
          >
            <Tab label="Competitor Activity Score" />
            <Tab label="Market Sentiment Score" />
          </Tabs>
        </Box>

        {/* Competitor Activity Score Tab */}
        <TabPanel value={tabValue} index={0}>
          <Typography variant="h5" gutterBottom>
            Competitor Activity Score Calculator
          </Typography>

          <Typography variant="body1" paragraph>
            The Competitor Activity Score helps quantify how much impact a
            competitor is having in the market. It combines multiple factors
            with different weights to create a single score that represents the
            competitor's market presence and threat level.
          </Typography>

          <Accordion defaultExpanded>
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls="panel1a-content"
              id="panel1a-header"
            >
              <Typography variant="h6">How It's Calculated</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography paragraph>
                The Competitor Activity Score is calculated using a weighted
                average of six key factors:
              </Typography>

              <TableContainer
                component={Paper}
                variant="outlined"
                sx={{ mb: 3 }}
              >
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Factor</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell align="right">Weight</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>S = Sales Score</TableCell>
                      <TableCell>
                        Normalized sales volume (higher sales = higher impact)
                      </TableCell>
                      <TableCell align="right">25%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>P = Price Difference Score</TableCell>
                      <TableCell>
                        Lower competitor prices create higher impact
                      </TableCell>
                      <TableCell align="right">15%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>D = Discount Score</TableCell>
                      <TableCell>
                        Higher discounts from competitors create higher impact
                      </TableCell>
                      <TableCell align="right">20%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>T = Google Trends Score</TableCell>
                      <TableCell>
                        Search volume impact (higher searches = higher impact)
                      </TableCell>
                      <TableCell align="right">15%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>A = Ad Spend Score</TableCell>
                      <TableCell>Competitor advertising intensity</TableCell>
                      <TableCell align="right">15%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>M = Social Media Score</TableCell>
                      <TableCell>
                        Social discussion volume (Twitter, Reddit, etc.)
                      </TableCell>
                      <TableCell align="right">10%</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>

              <Typography paragraph>
                The formula used is:{" "}
                <strong>
                  Competitor Activity Score = (S × 0.25) + (P × 0.15) + (D ×
                  0.20) + (T × 0.15) + (A × 0.15) + (M × 0.10)
                </strong>
              </Typography>

              <Typography>
                Each factor is normalized to a 0-1 scale, where 1 means maximum
                possible impact and 0 means no impact. The final score is
                between 0 and 1, where higher scores indicate stronger
                competitor activity and market threat.
              </Typography>
            </AccordionDetails>
          </Accordion>

          <Box sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Calculate Competitor Activity Score
            </Typography>

            <Card variant="outlined" sx={{ mb: 4 }}>
              <CardContent>
                <Grid container spacing={3}>
                  {[
                    {
                      key: "salesScore",
                      label: "Sales Score",
                      description:
                        "Higher value means competitor has more sales volume",
                    },
                    {
                      key: "priceScore",
                      label: "Price Difference Score",
                      description:
                        "Higher value means competitor has lower prices",
                    },
                    {
                      key: "discountScore",
                      label: "Discount Score",
                      description:
                        "Higher value means competitor offers larger discounts",
                    },
                    {
                      key: "trendsScore",
                      label: "Google Trends Score",
                      description:
                        "Higher value means more search interest for competitor",
                    },
                    {
                      key: "adSpendScore",
                      label: "Ad Spend Score",
                      description:
                        "Higher value means competitor has higher advertising spend",
                    },
                    {
                      key: "socialMediaScore",
                      label: "Social Media Score",
                      description:
                        "Higher value means more social media discussion",
                    },
                  ].map((item) => (
                    <Grid item xs={12} md={6} key={item.key}>
                      <Box
                        sx={{ display: "flex", alignItems: "center", mb: 1 }}
                      >
                        <Typography id={`${item.key}-slider`}>
                          {item.label}
                        </Typography>
                        <Tooltip title={item.description}>
                          <InfoIcon
                            fontSize="small"
                            sx={{ ml: 1, color: "text.secondary" }}
                          />
                        </Tooltip>
                      </Box>
                      <Box sx={{ display: "flex", alignItems: "center" }}>
                        <Slider
                          value={competitorInputs[item.key]}
                          onChange={handleCompetitorChange(item.key)}
                          aria-labelledby={`${item.key}-slider`}
                          step={0.1}
                          marks
                          min={0}
                          max={1}
                          valueLabelDisplay="auto"
                          sx={{ mr: 2, flex: 1 }}
                        />
                        <TextField
                          value={competitorInputs[item.key]}
                          onChange={handleCompetitorChange(item.key)}
                          inputProps={{
                            step: 0.1,
                            min: 0,
                            max: 1,
                            type: "number",
                          }}
                          sx={{ width: "80px" }}
                          size="small"
                        />
                      </Box>
                    </Grid>
                  ))}
                </Grid>

                <Box sx={{ mt: 4, display: "flex", justifyContent: "center" }}>
                  <Button
                    variant="contained"
                    color="primary"
                    size="large"
                    onClick={calculateCompetitorScore}
                    disabled={calculatingCompetitor}
                    startIcon={
                      calculatingCompetitor && (
                        <CircularProgress size={20} color="inherit" />
                      )
                    }
                  >
                    {calculatingCompetitor
                      ? "Calculating..."
                      : "Calculate Score"}
                  </Button>
                </Box>
              </CardContent>
            </Card>

            {competitorResult && (
              <Card variant="outlined" sx={{ mt: 4 }}>
                <CardContent>
                  <Typography variant="h6" align="center" gutterBottom>
                    Competitor Activity Score Results
                  </Typography>

                  <Box
                    sx={{ display: "flex", justifyContent: "center", mb: 3 }}
                  >
                    <Box sx={{ position: "relative", display: "inline-flex" }}>
                      <CircularProgress
                        variant="determinate"
                        value={competitorResult.normalizedScore}
                        size={120}
                        thickness={5}
                        color={
                          competitorResult.normalizedScore >= 80
                            ? "error"
                            : competitorResult.normalizedScore >= 60
                            ? "warning"
                            : competitorResult.normalizedScore >= 40
                            ? "primary"
                            : "success"
                        }
                      />
                      <Box
                        sx={{
                          top: 0,
                          left: 0,
                          bottom: 0,
                          right: 0,
                          position: "absolute",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <Typography
                          variant="h5"
                          component="div"
                          color="text.secondary"
                        >
                          {`${competitorResult.normalizedScore}%`}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>

                  <Typography
                    variant="h6"
                    align="center"
                    color="text.secondary"
                    gutterBottom
                  >
                    Impact Level: {competitorResult.impactLevel}
                  </Typography>

                  <Divider sx={{ my: 3 }} />

                  <Grid container spacing={4}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        Detailed Calculation
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Factor</TableCell>
                              <TableCell align="right">Value</TableCell>
                              <TableCell align="right">Weight</TableCell>
                              <TableCell align="right">
                                Weighted Score
                              </TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {competitorResult.detailedCalculation.map((row) => (
                              <TableRow key={row.factor}>
                                <TableCell component="th" scope="row">
                                  {row.factor.replace("Score", "")}
                                </TableCell>
                                <TableCell align="right">
                                  {row.inputValue.toFixed(2)}
                                </TableCell>
                                <TableCell align="right">
                                  {(row.weight * 100).toFixed(0)}%
                                </TableCell>
                                <TableCell align="right">
                                  {row.score.toFixed(4)}
                                </TableCell>
                              </TableRow>
                            ))}
                            <TableRow>
                              <TableCell colSpan={3}>
                                <strong>Total Score</strong>
                              </TableCell>
                              <TableCell align="right">
                                <strong>
                                  {competitorResult.totalScore.toFixed(4)}
                                </strong>
                              </TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        Factor Visualization
                      </Typography>
                      <ResponsiveContainer width="100%" height={300}>
                        <RadarChart
                          outerRadius={90}
                          data={competitorResult.radarData}
                        >
                          <PolarGrid />
                          <PolarAngleAxis dataKey="factor" />
                          <PolarRadiusAxis angle={30} domain={[0, 1]} />
                          <Radar
                            name="Competitor"
                            dataKey="value"
                            stroke="#8884d8"
                            fill="#8884d8"
                            fillOpacity={0.6}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </Grid>
                  </Grid>

                  <Box sx={{ mt: 3 }}>
                    <Typography variant="body1">
                      <strong>Interpretation:</strong> This competitor is having
                      a {competitorResult.impactLevel.toLowerCase()} impact in
                      the market.
                      {competitorResult.normalizedScore >= 60
                        ? " You should closely monitor their activities and develop a strategic response."
                        : " Continue to monitor their activities as part of your regular competitive analysis."}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            )}
          </Box>
        </TabPanel>

        {/* Market Sentiment Score Tab */}
        <TabPanel value={tabValue} index={1}>
          <Typography variant="h5" gutterBottom>
            Market Sentiment Score Analysis
          </Typography>

          <Typography variant="body1" paragraph>
            Market Sentiment Score helps understand how consumers perceive a
            product or brand – whether their opinions are positive, negative, or
            neutral. This analysis uses natural language processing techniques
            to quantify sentiment in text data.
          </Typography>

          <Accordion defaultExpanded>
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls="panel2a-content"
              id="panel2a-header"
            >
              <Typography variant="h6">How It's Calculated</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography paragraph>
                The Market Sentiment Score is calculated using TextBlob, a
                Python library for processing textual data that provides
                sentiment analysis capabilities:
              </Typography>

              <Box
                sx={{
                  bgcolor: "background.paper",
                  p: 2,
                  borderRadius: 1,
                  mb: 3,
                }}
              >
               
              </Box>

              <Typography paragraph>
                <strong>
                  TextBlob's sentiment analysis provides two key metrics:
                </strong>
              </Typography>

              <Box component="ul" sx={{ pl: 4 }}>
                <Box component="li" sx={{ mb: 1 }}>
                  <Typography>
                    <strong>Polarity:</strong> A float value between -1 and 1,
                    where -1 indicates very negative sentiment, 0 indicates
                    neutral sentiment, and 1 indicates very positive sentiment.
                  </Typography>
                </Box>
                <Box component="li">
                  <Typography>
                    <strong>Subjectivity:</strong> A float value between 0 and
                    1, where 0 indicates very objective content and 1 indicates
                    very subjective content (personal opinion).
                  </Typography>
                </Box>
              </Box>

              <Typography paragraph sx={{ mt: 2 }}>
                For market sentiment analysis at scale, we typically:
              </Typography>

              <Box component="ol" sx={{ pl: 4 }}>
                <Box component="li" sx={{ mb: 1 }}>
                  <Typography>
                    Collect customer reviews, social media mentions, and other
                    text data
                  </Typography>
                </Box>
                <Box component="li" sx={{ mb: 1 }}>
                  <Typography>
                    Process each text through TextBlob's sentiment analyzer
                  </Typography>
                </Box>
                <Box component="li" sx={{ mb: 1 }}>
                  <Typography>
                    Aggregate the results to get an overall sentiment score
                  </Typography>
                </Box>
                <Box component="li">
                  <Typography>
                    Track changes over time to identify trends in market
                    perception
                  </Typography>
                </Box>
              </Box>
            </AccordionDetails>
          </Accordion>

          <Box sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Calculate Market Sentiment Score
            </Typography>

            <Card variant="outlined" sx={{ mb: 4 }}>
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Typography gutterBottom>Positive Reviews</Typography>
                    <TextField
                      fullWidth
                      type="number"
                      InputProps={{ inputProps: { min: 0, max: 100 } }}
                      value={sentimentInputs.positiveReviews}
                      onChange={handleSentimentChange("positiveReviews")}
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Typography gutterBottom>Neutral Reviews</Typography>
                    <TextField
                      fullWidth
                      type="number"
                      InputProps={{ inputProps: { min: 0, max: 100 } }}
                      value={sentimentInputs.neutralReviews}
                      onChange={handleSentimentChange("neutralReviews")}
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Typography gutterBottom>Negative Reviews</Typography>
                    <TextField
                      fullWidth
                      type="number"
                      InputProps={{ inputProps: { min: 0, max: 100 } }}
                      value={sentimentInputs.negativeReviews}
                      onChange={handleSentimentChange("negativeReviews")}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Typography gutterBottom>
                      Sample Text for Analysis
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={4}
                      value={sentimentInputs.sampleText}
                      onChange={handleSentimentChange("sampleText")}
                      placeholder="Enter a product review or social media post to analyze its sentiment..."
                    />
                  </Grid>
                </Grid>

                <Box sx={{ mt: 4, display: "flex", justifyContent: "center" }}>
                  <Button
                    variant="contained"
                    color="primary"
                    size="large"
                    onClick={calculateSentimentScore}
                    disabled={calculatingSentiment}
                    startIcon={
                      calculatingSentiment && (
                        <CircularProgress size={20} color="inherit" />
                      )
                    }
                  >
                    {calculatingSentiment
                      ? "Analyzing..."
                      : "Analyze Sentiment"}
                  </Button>
                </Box>
              </CardContent>
            </Card>

            {sentimentResult && (
              <Card variant="outlined" sx={{ mt: 4 }}>
                <CardContent>
                  <Typography variant="h6" align="center" gutterBottom>
                    Market Sentiment Analysis Results
                  </Typography>

                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "center",
                      mb: 3,
                      alignItems: "center",
                    }}
                  >
                    {sentimentResult.sentimentCategory === "Positive" ? (
                      <SentimentSatisfiedAltIcon
                        color="success"
                        sx={{ fontSize: 60, mr: 2 }}
                      />
                    ) : sentimentResult.sentimentCategory === "Negative" ? (
                      <SentimentVeryDissatisfiedIcon
                        color="error"
                        sx={{ fontSize: 60, mr: 2 }}
                      />
                    ) : (
                      <SentimentNeutralIcon
                        color="action"
                        sx={{ fontSize: 60, mr: 2 }}
                      />
                    )}

                    <Typography variant="h5">
                      Overall Sentiment: {sentimentResult.sentimentCategory}
                    </Typography>
                  </Box>

                  <Box
                    sx={{ display: "flex", justifyContent: "center", mb: 3 }}
                  >
                    <Box sx={{ width: "100%", maxWidth: 500 }}>
                      <Box
                        sx={{ display: "flex", alignItems: "center", mb: 1 }}
                      >
                        <Box sx={{ minWidth: 100 }}>
                          <Typography variant="body2" color="text.secondary">
                            Very Negative
                          </Typography>
                        </Box>
                        <Box sx={{ width: "100%", mx: 2 }}>
                          <Box
                            sx={{
                              width: "100%",
                              bgcolor: "grey.300",
                              height: 10,
                              borderRadius: 5,
                            }}
                          >
                            <Box
                              sx={{
                                width: `${sentimentResult.normalizedScore}%`,
                                bgcolor:
                                  sentimentResult.sentimentCategory ===
                                  "Positive"
                                    ? "success.main"
                                    : sentimentResult.sentimentCategory ===
                                      "Negative"
                                    ? "error.main"
                                    : "warning.main",
                                height: 10,
                                borderRadius: 5,
                              }}
                            />
                          </Box>
                        </Box>
                        <Box sx={{ minWidth: 100, textAlign: "right" }}>
                          <Typography variant="body2" color="text.secondary">
                            Very Positive
                          </Typography>
                        </Box>
                      </Box>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        align="center"
                      >
                        Sentiment Score:{" "}
                        {sentimentResult.sentimentScore.toFixed(2)} (on a scale
                        from -1 to 1)
                      </Typography>
                    </Box>
                  </Box>

                  <Divider sx={{ my: 3 }} />

                  <Grid container spacing={4}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        TextBlob Analysis of Sample Text
                      </Typography>

                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableBody>
                            <TableRow>
                              <TableCell component="th" scope="row">
                                Polarity
                              </TableCell>
                              <TableCell align="right">
                                {sentimentResult.textSentiment.polarity.toFixed(
                                  2
                                )}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell component="th" scope="row">
                                Subjectivity
                              </TableCell>
                              <TableCell align="right">
                                {sentimentResult.textSentiment.subjectivity.toFixed(
                                  2
                                )}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell component="th" scope="row">
                                Classification
                              </TableCell>
                              <TableCell align="right">
                                {sentimentResult.textSentiment.classification}
                              </TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </TableContainer>

                      <Box
                        sx={{
                          mt: 3,
                          p: 2,
                          bgcolor: "background.paper",
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="body2" fontStyle="italic">
                          "{sentimentInputs.sampleText}"
                        </Typography>
                      </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        Sentiment Distribution
                      </Typography>

                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={sentimentResult.pieChartData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) =>
                              `${name} (${(percent * 100).toFixed(0)}%)`
                            }
                            outerRadius={80}
                            fill="#8884d8"
                          >
                            {sentimentResult.pieChartData.map(
                              (entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={COLORS[index % COLORS.length]}
                                />
                              )
                            )}
                          </Pie>
                          <RechartsTooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </Grid>
                  </Grid>
                  <Box
                    sx={{ mt: 4, display: "flex", justifyContent: "center" }}
                  >
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      onClick={handleExportToCSV}
                    >
                      Export to CSV
                    </Button>
                  </Box>
                  <Box
                    sx={{ mt: 4, display: "flex", justifyContent: "center" }}
                  >
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      onClick={handleShare}
                    >
                      Share on Social Media
                    </Button>
                  </Box>
                  <Box
                    sx={{ mt: 4, display: "flex", justifyContent: "center" }}
                  >
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      onClick={handleReset}
                    >
                      Reset
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            )}
          </Box>
        </TabPanel>
      </Paper>
    </Box>
  );
};
export default CompetitorSentimentAnalysisPage;
