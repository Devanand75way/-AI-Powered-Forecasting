import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from collections import Counter
import ast

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

class HashtagTrendAnalyzer:
    def __init__(self, data_path, top_n=10, trending_window_days=7, 
                 base_multiplier=1.0, engagement_scaling_factor=1000, 
                 similarity_threshold=0.6):
        """
        Initialize the HashtagTrendAnalyzer with configuration parameters.
        
        Args:
            data_path (str): Path to the dataset containing social media posts
            top_n (int): Number of top trending hashtags to track
            trending_window_days (int): Time window for trending analysis in days
            base_multiplier (float): Base demand multiplier value
            engagement_scaling_factor (float): Scaling factor for engagement metrics
            similarity_threshold (float): Threshold for determining product name matches
        """
        self.data_path = data_path
        self.top_n = top_n
        self.trending_window_days = trending_window_days
        self.base_multiplier = base_multiplier
        self.engagement_scaling_factor = engagement_scaling_factor
        self.similarity_threshold = similarity_threshold
        self.trending_hashtags = []
        self.hashtag_engagement = {}
        self.stop_words = set(stopwords.words('english'))
        
        # Load and preprocess the data
        self.load_data()
        self.preprocess_data()
        self.analyze_trending_hashtags()
    
    def load_data(self):
        """Load the dataset and extract relevant columns."""
        try:
            # Load the data
            self.df = pd.read_csv(self.data_path, sep='\t')
            
            # Extract only the columns we need
            relevant_columns = [
                'hashtags', 'views', 'likes', 'reposts', 'replies', 
                'date_posted', 'name', 'description'
            ]
            
            available_columns = [col for col in relevant_columns if col in self.df.columns]
            self.df = self.df[available_columns]
            
            print(f"Data loaded successfully with {len(self.df)} records")
            
            # Print column info for debugging
            print(f"Available columns: {list(self.df.columns)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame(columns=relevant_columns)
    
    def preprocess_data(self):
        """Preprocess the loaded data for analysis."""
        try:
            # Convert date_posted to datetime
            if 'date_posted' in self.df.columns:
                self.df['date_posted'] = pd.to_datetime(self.df['date_posted'])
                print("Sample dates:", self.df['date_posted'].head())
            
            # Extract hashtags from the hashtags column
            if 'hashtags' in self.df.columns:
                self.df['hashtags_list'] = self.df['hashtags'].apply(
                    lambda x: self._extract_hashtags(x)
                )
                # Print sample hashtags for debugging
                print("Sample hashtags:")
                for idx, hashtags in enumerate(self.df['hashtags_list'].head(5)):
                    print(f"  Row {idx}: {hashtags}")
            
            # Create an engagement score
            engagement_cols = ['views', 'likes', 'reposts', 'replies']
            available_engagement_cols = [col for col in engagement_cols if col in self.df.columns]
            
            if available_engagement_cols:
                self.df['engagement'] = 0
                if 'views' in available_engagement_cols:
                    self.df['engagement'] += self.df['views'].fillna(0) * 0.01
                if 'likes' in available_engagement_cols:
                    self.df['engagement'] += self.df['likes'].fillna(0) * 0.5
                if 'reposts' in available_engagement_cols:
                    self.df['engagement'] += self.df['reposts'].fillna(0) * 2.0
                if 'replies' in available_engagement_cols:
                    self.df['engagement'] += self.df['replies'].fillna(0) * 1.5
                
                print("Engagement score statistics:")
                print(self.df['engagement'].describe())
            
            print("Data preprocessing completed")
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_hashtags(self, hashtags_str):
        """Extract hashtags from various string formats."""
        if pd.isna(hashtags_str) or hashtags_str == '':
            return []
            
        try:
            # Handle string representation of a list
            if isinstance(hashtags_str, str):
                if hashtags_str.startswith('[') and hashtags_str.endswith(']'):
                    try:
                        # Parse as a list using ast.literal_eval for safety
                        hashtag_list = ast.literal_eval(hashtags_str)
                        if isinstance(hashtag_list, list):
                            return [tag.lower().strip('#') for tag in hashtag_list if isinstance(tag, str)]
                    except (ValueError, SyntaxError):
                        # If parsing fails, try other methods
                        pass
                    
                # Try removing brackets and splitting
                cleaned = hashtags_str.strip('[]').replace('"', '').replace("'", '')
                return [tag.lower().strip('#').strip() for tag in cleaned.split(',') if tag.strip()]
            
            # If it's already a list
            elif isinstance(hashtags_str, list):
                return [tag.lower().strip('#') for tag in hashtags_str if isinstance(tag, str)]
                
        except Exception as e:
            print(f"Error extracting hashtags from {hashtags_str}: {e}")
            
        return []
    
    def analyze_trending_hashtags(self):
        """Analyze and identify trending hashtags within the specified time window."""
        try:
            # Check if required columns exist
            if 'hashtags_list' not in self.df.columns or 'engagement' not in self.df.columns:
                print("Required columns 'hashtags_list' or 'engagement' missing")
                self.trending_hashtags = []
                return
            
            # Skip time filtering if date_posted is not available
            if 'date_posted' in self.df.columns:
                # Get cutoff date (using days instead of hours to avoid timezone issues)
                current_time = datetime.now(pytz.UTC)
                cutoff_time = current_time - timedelta(days=self.trending_window_days)
                
                # Filter for recent data, handling timezone-aware datetimes properly
                self.df['is_recent'] = self.df['date_posted'] >= cutoff_time
                recent_data = self.df[self.df['is_recent']]
                
                if recent_data.empty:
                    print(f"No data found within the last {self.trending_window_days} days")
                    print(f"Data range: {self.df['date_posted'].min()} to {self.df['date_posted'].max()}")
                    # Fall back to using all data
                    recent_data = self.df
                else:
                    print(f"Using {len(recent_data)} records from the last {self.trending_window_days} days")
            else:
                # If no date column, use all data
                recent_data = self.df
                print("No date_posted column, using all available data")
            
            # Extract hashtags and their engagement
            hashtag_engagement = {}
            total_hashtags = 0
            
            for _, row in recent_data.iterrows():
                for hashtag in row['hashtags_list']:
                    if hashtag:  # Skip empty hashtags
                        total_hashtags += 1
                        if hashtag in hashtag_engagement:
                            hashtag_engagement[hashtag] += row['engagement']
                        else:
                            hashtag_engagement[hashtag] = row['engagement']
            
            print(f"Total unique hashtags found: {len(hashtag_engagement)}")
            print(f"Total hashtag occurrences: {total_hashtags}")
            
            if not hashtag_engagement:
                print("No hashtags found in the dataset")
                self.trending_hashtags = []
                return
            
            # Sort hashtags by engagement
            sorted_hashtags = sorted(hashtag_engagement.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            # Print top hashtags for debugging
            print("\nTop hashtags by engagement:")
            for i, (tag, score) in enumerate(sorted_hashtags[:min(10, len(sorted_hashtags))]):
                print(f"  {i+1}. #{tag}: {score:.2f}")
            
            self.trending_hashtags = [tag for tag, _ in sorted_hashtags[:self.top_n]]
            self.hashtag_engagement = hashtag_engagement
            
            print(f"\nStored top {len(self.trending_hashtags)} trending hashtags")
            
        except Exception as e:
            print(f"Error analyzing trending hashtags: {e}")
            import traceback
            traceback.print_exc()
    
    def normalize_text(self, text):
        """Normalize text for comparison (lowercase, remove special chars, etc.)."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove stopwords
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def check_product_trending(self, product_name):
        """
        Check if a product is trending based on hashtag analysis.
        
        Args:
            product_name (str): The product name to check
            
        Returns:
            dict: Result containing trending status and demand multiplier
        """
        normalized_product = self.normalize_text(product_name)
        
        if not normalized_product:
            return {
                "trending": False,
                "multiplier": self.base_multiplier * 0.8,
                "message": "Invalid product name provided"
            }
        
        # Check for exact matches in trending hashtags
        exact_match = False
        for hashtag in self.trending_hashtags:
            if normalized_product == hashtag or normalized_product in hashtag:
                exact_match = True
                break
        
        # If no exact match, use fuzzy matching
        best_match = None
        best_score = 0
        
        if not exact_match:
            for hashtag in self.trending_hashtags:
                # Calculate similarity score using fuzzy matching
                similarity_score = fuzz.token_set_ratio(normalized_product, hashtag) / 100.0
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = hashtag
        
        # Determine if the product is trending based on matching
        is_trending = exact_match or (best_match and best_score >= self.similarity_threshold)
        
        # Calculate the demand multiplier
        multiplier = self.base_multiplier
        trend_message = ""
        
        if is_trending:
            # Get the engagement score for the hashtag
            if exact_match:
                hashtag_to_use = normalized_product
                match_type = "exact match"
            else:
                hashtag_to_use = best_match
                match_type = f"similar to '{best_match}' (similarity: {best_score:.2f})"
            
            # Get engagement score and calculate multiplier
            engagement = self.hashtag_engagement.get(hashtag_to_use, 0)
            multiplier = self.base_multiplier * (1 + (engagement / self.engagement_scaling_factor))
            
            trend_message = f"Product is trending ({match_type}) with engagement score of {engagement:.2f}"
        else:
            # Reduce demand for non-trending products
            multiplier = self.base_multiplier * 0.8
            trend_message = "Product is not trending in the current dataset"
        
        return {
            "trending": is_trending,
            "multiplier": multiplier,
            "message": trend_message,
            "best_match": best_match if not exact_match and is_trending else None,
            "similarity_score": best_score if not exact_match and is_trending else None
        }
    
    def get_trending_hashtags(self):
        """Return the current list of trending hashtags."""
        return self.trending_hashtags
    
    def get_hashtag_stats(self):
        """Return detailed statistics about hashtags in the dataset."""
        if not self.hashtag_engagement:
            return {"message": "No hashtags found in the dataset"}
        
        # Get all hashtags sorted by engagement
        sorted_hashtags = sorted(self.hashtag_engagement.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Count total occurrences
        total_hashtags = sum(1 for row in self.df['hashtags_list'] 
                            for _ in row if 'hashtags_list' in self.df.columns)
        
        return {
            "total_unique_hashtags": len(self.hashtag_engagement),
            "total_hashtag_occurrences": total_hashtags,
            "top_hashtags": [{"hashtag": tag, "engagement": score} 
                            for tag, score in sorted_hashtags[:20]]
        }
    
    def explore_dataset(self):
        """Explore the dataset and return useful statistics."""
        stats = {}
        
        # Basic dataset info
        stats["total_records"] = len(self.df)
        stats["columns"] = list(self.df.columns)
        
        # Date range if available
        if 'date_posted' in self.df.columns:
            stats["date_range"] = {
                "earliest": self.df['date_posted'].min().strftime('%Y-%m-%d'),
                "latest": self.df['date_posted'].max().strftime('%Y-%m-%d')
            }
        
        # Engagement stats if available
        if 'engagement' in self.df.columns:
            stats["engagement"] = {
                "mean": self.df['engagement'].mean(),
                "median": self.df['engagement'].median(),
                "max": self.df['engagement'].max(),
                "min": self.df['engagement'].min()
            }
        
        # Hashtag statistics
        if 'hashtags_list' in self.df.columns:
            # Count hashtags per post
            self.df['hashtag_count'] = self.df['hashtags_list'].apply(len)
            
            stats["hashtags"] = {
                "total_posts_with_hashtags": (self.df['hashtag_count'] > 0).sum(),
                "avg_hashtags_per_post": self.df['hashtag_count'].mean(),
                "max_hashtags_in_post": self.df['hashtag_count'].max()
            }
            
            # Most common hashtags
            all_hashtags = [tag for tags in self.df['hashtags_list'] for tag in tags]
            if all_hashtags:
                hashtag_counter = Counter(all_hashtags)
                stats["most_common_hashtags"] = hashtag_counter.most_common(10)
        
        return stats

class DemandPredictionAPI:
    def __init__(self, data_path, config=None):
        """
        Initialize the DemandPredictionAPI.
        
        Args:
            data_path (str): Path to the dataset
            config (dict, optional): Configuration parameters for the analyzer
        """
        default_config = {
            'top_n': 20,
            'trending_window_days': 30,
            'base_multiplier': 1.0,
            'engagement_scaling_factor': 1000,
            'similarity_threshold': 0.6
        }
        
        if config:
            default_config.update(config)
        
        self.analyzer = HashtagTrendAnalyzer(
            data_path=data_path,
            top_n=default_config['top_n'],
            trending_window_days=default_config['trending_window_days'],
            base_multiplier=default_config['base_multiplier'],
            engagement_scaling_factor=default_config['engagement_scaling_factor'],
            similarity_threshold=default_config['similarity_threshold']
        )
    
    def predict_demand(self, product_name):
        """
        Predict demand for a product based on hashtag trending analysis.
        
        Args:
            product_name (str): Name of the product to check
            
        Returns:
            dict: Result containing trending status and demand multiplier
        """
        return self.analyzer.check_product_trending(product_name)
    
    def get_trending_products(self):
        """Return the current list of trending products/hashtags."""
        return self.analyzer.get_trending_hashtags()
    
    def get_hashtag_statistics(self):
        """Get detailed statistics about hashtags in the dataset."""
        return self.analyzer.get_hashtag_stats()
    
    def dataset_exploration(self):
        """Explore the dataset and return useful information."""
        return self.analyzer.explore_dataset()


# Example usage
if __name__ == "__main__":
    # Initialize the API with the data path
    api = DemandPredictionAPI("ml_model/data/twitter_post.tsv", {
        'top_n': 50,
        'trending_window_days': 60,  # Look at last 30 days of data
        'base_multiplier': 1.0,
        'engagement_scaling_factor': 500,
        'similarity_threshold': 0.7
    })
    
    # Explore the dataset
    print("\n===== DATASET EXPLORATION =====")
    dataset_info = api.dataset_exploration()
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    
    # Get detailed hashtag statistics
    print("\n===== HASHTAG STATISTICS =====")
    hashtag_stats = api.get_hashtag_statistics()
    if "top_hashtags" in hashtag_stats:
        print(f"Total unique hashtags: {hashtag_stats['total_unique_hashtags']}")
        print(f"Total hashtag occurrences: {hashtag_stats['total_hashtag_occurrences']}")
        print("\nTop hashtags by engagement:")
        for i, item in enumerate(hashtag_stats['top_hashtags']):
            print(f"{i+1}. #{item['hashtag']}: {item['engagement']:.2f}")
    
    # Get trending hashtags
    trending_hashtags = api.get_trending_products()
    print("\n===== CURRENT TRENDING HASHTAGS =====")
    if trending_hashtags:
        for i, tag in enumerate(trending_hashtags):
            print(f"{i+1}. #{tag}")
    else:
        print("No trending hashtags identified")
    
    # Check if a product is trending
    product_to_check = "tennis"
    print(f"\n===== CHECKING PRODUCT: {product_to_check} =====")
    result = api.predict_demand(product_to_check)
    print(f"Trending: {result['trending']}")
    print(f"Demand Multiplier: {result['multiplier']:.2f}")
    print(f"Details: {result['message']}")