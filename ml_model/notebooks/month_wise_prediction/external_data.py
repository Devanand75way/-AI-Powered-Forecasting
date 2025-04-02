import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os

class ExternalDataHandler:
    """
    Class to handle external data sources that may impact sales predictions.
    External factors could include:
    - Weather data
    - Economic indicators
    - Holiday/seasonal events
    - Competitor pricing
    - Social media sentiment
    """
    
    def __init__(self, config_file=None):
        """Initialize with optional configuration file"""
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        
        # Default data paths
        self.data_dir = self.config.get('data_dir', 'ml_model/data/')
        self.external_data_file = os.path.join(self.data_dir, 'external_factors.csv')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_weather_data(self, locations, start_date, end_date):
        """
        Fetch weather data for given locations and date range
        
        Parameters:
        locations (list): List of city names or coordinates
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
        Returns:
        DataFrame: Weather data for the locations and date range
        """
        # This would use a weather API in production
        # For demonstration, we'll generate sample data
        
        date_range = pd.date_range(start=start_date, end=end_date)
        weather_data = []
        
        for location in locations:
            for date in date_range:
                # Generate sample weather data
                weather_data.append({
                    'Date': date,
                    'Location': location,
                    'Temperature': np.random.normal(22, 8),  # Mean 22°C with std 8
                    'Precipitation': max(0, np.random.normal(2, 5)),  # Mean 2mm with std 5
                    'Humidity': np.random.uniform(30, 90),
                    'Wind_Speed': max(0, np.random.normal(15, 10))  # Mean 15 km/h with std 10
                })
        
        return pd.DataFrame(weather_data)
    
    def get_holiday_data(self, countries, start_date, end_date):
        """
        Get holiday data for given countries and date range
        
        Parameters:
        countries (list): List of country codes
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
        Returns:
        DataFrame: Holiday data for the countries and date range
        """
        # This would use a holiday API in production
        # For demonstration, we'll use a sample holiday list
        
        sample_holidays = {
                    'IN': [
                         {'date': '2024-01-26', 'name': 'Republic Day', 'importance': 'Major'},
                         {'date': '2024-03-08', 'name': 'Maha Shivratri', 'importance': 'Medium'},
                         {'date': '2024-03-25', 'name': 'Holi', 'importance': 'Major'},
                         {'date': '2024-04-10', 'name': 'Eid-ul-Fitr', 'importance': 'Major'},
                         {'date': '2024-04-17', 'name': 'Ram Navami', 'importance': 'Medium'},
                         {'date': '2024-04-21', 'name': 'Mahavir Jayanti', 'importance': 'Medium'},
                         {'date': '2024-05-23', 'name': 'Buddha Purnima', 'importance': 'Medium'},
                         {'date': '2024-06-17', 'name': 'Eid-ul-Adha (Bakrid)', 'importance': 'Major'},
                         {'date': '2024-08-15', 'name': 'Independence Day', 'importance': 'Major'},
                         {'date': '2024-09-07', 'name': 'Ganesh Chaturthi', 'importance': 'Medium'},
                         {'date': '2024-10-02', 'name': 'Gandhi Jayanti', 'importance': 'Major'},
                         {'date': '2024-10-12', 'name': 'Dussehra', 'importance': 'Major'},
                         {'date': '2024-10-31', 'name': 'Diwali', 'importance': 'Major'},
                         {'date': '2024-11-15', 'name': 'Guru Nanak Jayanti', 'importance': 'Medium'},
                         {'date': '2024-12-25', 'name': 'Christmas', 'importance': 'Major'},
                    ]
          }

        
        holiday_data = []
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        for country in countries:
            if country in sample_holidays:
                for holiday in sample_holidays[country]:
                    holiday_date = pd.to_datetime(holiday['date'])
                    if start <= holiday_date <= end:
                        holiday_data.append({
                            'Date': holiday_date,
                            'Country': country,
                            'Holiday': holiday['name'],
                            'Importance': holiday['importance']
                        })
        
        if not holiday_data:
            # Create empty DataFrame with correct columns
            return pd.DataFrame(columns=['Date', 'Country', 'Holiday', 'Importance'])
        
        return pd.DataFrame(holiday_data)
    
    def get_economic_indicators(self, indicators, start_date, end_date):
        """
        Get economic indicators for given date range
        
        Parameters:
        indicators (list): List of economic indicators to fetch
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
        Returns:
        DataFrame: Economic indicator data
        """
        # This would use an economic data API in production
        # For demonstration, we'll generate sample data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly frequency
        economic_data = []
        
        indicator_params = {
            'CPI': {'mean': 2.5, 'std': 0.5},
            'Unemployment': {'mean': 4.5, 'std': 0.8},
            'Consumer_Confidence': {'mean': 100, 'std': 10},
            'Retail_Sales_Growth': {'mean': 3.2, 'std': 1.5},
            'Disposable_Income': {'mean': 0.5, 'std': 0.3}
        }
        
        for date in date_range:
            data_point = {'Date': date}
            
            for indicator in indicators:
                if indicator in indicator_params:
                    params = indicator_params[indicator]
                    data_point[indicator] = np.random.normal(params['mean'], params['std'])
            
            economic_data.append(data_point)
        
        return pd.DataFrame(economic_data)
    
    def get_competitor_pricing(self, products, start_date, end_date):
        """
        Get competitor pricing data for given products and date range
        
        Parameters:
        products (list): List of product names
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
        Returns:
        DataFrame: Competitor pricing data
        """
        # This would use a web scraping service or competitor API in production
        # For demonstration, we'll generate sample data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly frequency
        pricing_data = []
        
        for product in products:
            base_price = np.random.uniform(50, 500)  # Random base price
            
            for date in date_range:
                # Generate competitor pricing with some variance
                pricing_data.append({
                    'Date': date,
                    'Product': product,
                    'Competitor_A_Price': max(0, base_price * np.random.normal(1, 0.1)),
                    'Competitor_B_Price': max(0, base_price * np.random.normal(0.95, 0.15)),
                    'Competitor_C_Price': max(0, base_price * np.random.normal(1.05, 0.2)),
                    'Market_Average_Price': max(0, base_price * np.random.normal(1, 0.05))
                })
        
        return pd.DataFrame(pricing_data)
    
    def get_social_media_sentiment(self, products, start_date, end_date):
        """
        Get social media sentiment data for given products and date range
        
        Parameters:
        products (list): List of product names
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
        Returns:
        DataFrame: Social media sentiment data
        """
        # This would use a social media API or sentiment analysis service in production
        # For demonstration, we'll generate sample data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')  # Daily frequency
        sentiment_data = []
        
        for product in products:
            base_sentiment = np.random.uniform(0.5, 0.8)  # Random base sentiment (0-1)
            
            for date in date_range:
                # Generate sentiment metrics with some variance
                sentiment_data.append({
                    'Date': date,
                    'Product': product,
                    'Sentiment_Score': min(1, max(0, base_sentiment * np.random.normal(1, 0.2))),
                    'Mentions_Count': int(np.random.poisson(100)),  # Poisson distribution for counts
                    'Positive_Mentions': int(np.random.poisson(70)),
                    'Negative_Mentions': int(np.random.poisson(30)),
                    'Viral_Score': min(100, max(0, np.random.normal(50, 20)))
                })
        
        return pd.DataFrame(sentiment_data)
    
    def get_marketing_data(self, products, start_date, end_date):
        """
        Get marketing campaign data for given products and date range
        
        Parameters:
        products (list): List of product names
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
        Returns:
        DataFrame: Marketing campaign data
        """
        # This would use marketing analytics APIs in production
        # For demonstration, we'll generate sample data
        
        # Generate some campaign dates
        campaign_starts = pd.date_range(start=start_date, end=end_date, freq='30D')
        marketing_data = []
        
        for product in products:
            for campaign_start in campaign_starts:
                campaign_end = campaign_start + timedelta(days=14)  # 2-week campaigns
                
                if campaign_end <= pd.to_datetime(end_date):
                    # Generate campaign metrics
                    campaign_budget = np.random.uniform(5000, 50000)
                    marketing_data.append({
                        'Start_Date': campaign_start,
                        'End_Date': campaign_end,
                        'Product': product,
                        'Campaign_Name': f"Campaign_{campaign_start.strftime('%Y%m%d')}",
                        'Budget': campaign_budget,
                        'Impressions': int(campaign_budget * np.random.uniform(10, 20)),
                        'Clicks': int(campaign_budget * np.random.uniform(0.5, 1.5)),
                        'Conversions': int(campaign_budget * np.random.uniform(0.01, 0.05)),
                        'ROI': np.random.normal(2.5, 0.8)  # Return on Investment
                    })
        
        return pd.DataFrame(marketing_data)
    
    def compile_external_data(self, start_date, end_date, products=None, locations=None, countries=None):
        """
        Compile all external data sources into a single DataFrame
        
        Parameters:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        products (list): List of product names
        locations (list): List of location names
        countries (list): List of country codes
        
        Returns:
        DataFrame: Combined external data
        """
        # Default values
        products = products or ['Product A', 'Product B', 'Product C']
        locations = locations or ['New York', 'Los Angeles', 'Chicago']
        countries = countries or ['US', 'CA']
        indicators = ['CPI', 'Unemployment', 'Consumer_Confidence', 'Retail_Sales_Growth']
        
        # Get data from different sources
        weather_df = self.get_weather_data(locations, start_date, end_date)
        holiday_df = self.get_holiday_data(countries, start_date, end_date)
        economic_df = self.get_economic_indicators(indicators, start_date, end_date)
        competitor_df = self.get_competitor_pricing(products, start_date, end_date)
        sentiment_df = self.get_social_media_sentiment(products, start_date, end_date)
        marketing_df = self.get_marketing_data(products, start_date, end_date)
        
        # Create date range for the master dataset
        date_range = pd.date_range(start=start_date, end=end_date)
        master_df = pd.DataFrame({'Date': date_range})
        
        # Process and aggregate weather data
        if not weather_df.empty:
            weather_agg = weather_df.groupby('Date').agg({
                'Temperature': 'mean',
                'Precipitation': 'mean',
                'Humidity': 'mean',
                'Wind_Speed': 'mean'
            }).reset_index()
            master_df = pd.merge(master_df, weather_agg, on='Date', how='left')
        
        # Process holiday data
        if not holiday_df.empty:
            # Create binary columns for holidays
            for date in date_range:
                holiday_on_date = holiday_df[holiday_df['Date'] == date]
                if not holiday_on_date.empty:
                    master_df.loc[master_df['Date'] == date, 'Is_Holiday'] = 1
                    master_df.loc[master_df['Date'] == date, 'Holiday_Importance'] = \
                        3 if 'Major' in holiday_on_date['Importance'].values else 2
                else:
                    master_df.loc[master_df['Date'] == date, 'Is_Holiday'] = 0
                    master_df.loc[master_df['Date'] == date, 'Holiday_Importance'] = 0
        
        # Process economic data
        if not economic_df.empty:
            # Forward fill economic indicators (they're typically monthly)
            economic_df = economic_df.set_index('Date')
            economic_daily = economic_df.reindex(date_range).ffill()
            economic_daily = economic_daily.reset_index().rename(columns={'index': 'Date'})
            master_df = pd.merge(master_df, economic_daily, on='Date', how='left')
        
        # Process competitor pricing data
        if not competitor_df.empty:
            # Average across all products for simplicity
            pricing_agg = competitor_df.groupby('Date').agg({
                'Market_Average_Price': 'mean',
                'Competitor_A_Price': 'mean',
                'Competitor_B_Price': 'mean',
                'Competitor_C_Price': 'mean'
            }).reset_index()
            
            # Forward fill pricing data (it's typically weekly)
            pricing_agg = pricing_agg.set_index('Date')
            pricing_daily = pricing_agg.reindex(date_range).ffill()
            pricing_daily = pricing_daily.reset_index().rename(columns={'index': 'Date'})
            
            master_df = pd.merge(master_df, pricing_daily, on='Date', how='left')
        
        # Process sentiment data
        if not sentiment_df.empty:
            # Average across all products
            sentiment_agg = sentiment_df.groupby('Date').agg({
                'Sentiment_Score': 'mean',
                'Mentions_Count': 'sum',
                'Positive_Mentions': 'sum',
                'Negative_Mentions': 'sum',
                'Viral_Score': 'mean'
            }).reset_index()
            
            master_df = pd.merge(master_df, sentiment_agg, on='Date', how='left')
        
        # Process marketing data
        if not marketing_df.empty:
            # Create a campaign activity indicator for each date
            for date in date_range:
                active_campaigns = ((marketing_df['Start_Date'] <= date) & 
                                    (marketing_df['End_Date'] >= date)).sum()
                if active_campaigns > 0:
                    # Get active campaign budgets for this date
                    active_budgets = marketing_df.loc[
                        (marketing_df['Start_Date'] <= date) & 
                        (marketing_df['End_Date'] >= date), 'Budget'
                    ].sum()
                    
                    master_df.loc[master_df['Date'] == date, 'Active_Campaigns'] = active_campaigns
                    master_df.loc[master_df['Date'] == date, 'Marketing_Budget'] = active_budgets
                else:
                    master_df.loc[master_df['Date'] == date, 'Active_Campaigns'] = 0
                    master_df.loc[master_df['Date'] == date, 'Marketing_Budget'] = 0
        
        # Fill NA values
        master_df = master_df.fillna(0)
        
        return master_df
    
    def save_external_data(self, data, file_path=None):
        """Save compiled external data to a CSV file"""
        file_path = file_path or self.external_data_file
        data.to_csv(file_path, index=False)
        print(f"External data saved to {file_path}")
        return file_path
    
    def generate_external_data(self, start_date, end_date, products=None, save=True):
        """Generate and optionally save external data"""
        external_data = self.compile_external_data(
            start_date, end_date, products=products
        )
        
        if save:
            self.save_external_data(external_data)
        
        return external_data

# Example usage
if __name__ == "__main__":
    handler = ExternalDataHandler()
    
    # Generate one year of external data
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    external_data = handler.generate_external_data(
        start_date, 
        end_date,
        products=['Office Chair', 'Desk', 'Filing Cabinet', 'Bookshelf', 'Desk Lamp'],
        save=True
    )
    
    print(f"Generated external data with {len(external_data)} rows and {len(external_data.columns)} columns")
    print(f"Columns: {external_data.columns.tolist()}")
    print(external_data.head())