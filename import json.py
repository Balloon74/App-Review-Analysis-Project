import json
import sqlite3
import requests
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

class Platform(Enum):
    APP_STORE = "app_store"
    PLAY_STORE = "play_store"

@dataclass
class Review:
    id: str
    app_id: str
    platform: Platform
    content: str
    rating: int
    date: datetime
    language: str
    sentiment: SentimentType
    confidence: float
    raw_data: Dict

class ReviewAnalyzer:
    def __init__(self, grok_api_key: str = None):
        """
        Initialize the Review Analyzer
        
        Args:
            grok_api_key: API key for Grok sentiment analysis (to be implemented)
        """
        self.grok_api_key = grok_api_key
        self.db_path = "review_analysis.db"
        self.init_database()
        
        # Supported languages
        self.supported_languages = ['zh', 'en', 'es', 'hi', 'ar']
        
        # Manual sentiment keywords for fallback
        self.positive_keywords = {
            'en': ['excellent', 'amazing', 'great', 'love', 'perfect', 'awesome', 'fantastic', 'wonderful', 'good', 'like', 'best', 'brilliant'],
            'es': ['excelente', 'increíble', 'genial', 'amor', 'perfecto', 'fantástico', 'maravilloso', 'bueno', 'mejor', 'brillante'],
            'zh': ['很好', '优秀', '完美', '喜欢', '棒', '太好了', '很棒', '不错'],
            'hi': ['बेहतरीन', 'अच्छा', 'शानदार', 'प्यार', 'परफेक्ट', 'बहुत बढ़िया'],
            'ar': ['ممتاز', 'رائع', 'جيد', 'أحب', 'مثالي', 'رائع جداً']
        }
        
        self.negative_keywords = {
            'en': ['terrible', 'awful', 'hate', 'worst', 'bad', 'horrible', 'useless', 'trash', 'sucks', 'broken', 'crash', 'bug'],
            'es': ['terrible', 'horrible', 'odio', 'peor', 'malo', 'inútil', 'basura', 'roto', 'error'],
            'zh': ['糟糕', '讨厌', '最差', '坏', '垃圾', '崩溃', '错误', '不好'],
            'hi': ['बुरा', 'घिनौना', 'बेकार', 'खराब', 'क्रैश', 'बग'],
            'ar': ['سيء', 'فظيع', 'أكره', 'الأسوأ', 'عديم الفائدة', 'معطل', 'خطأ']
        }
        
        # Rate limiting
        self.last_api_call = 0
        self.min_api_interval = 1  # Minimum seconds between API calls
        
    def init_database(self):
        """Initialize SQLite database for storing reviews and analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                app_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                content TEXT NOT NULL,
                rating INTEGER,
                date TIMESTAMP,
                language TEXT,
                sentiment TEXT,
                confidence REAL,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_id TEXT NOT NULL,
                platform TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_reviews INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                unknown_count INTEGER,
                avg_confidence REAL,
                trend_direction TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns
        This is a basic implementation - could be enhanced with proper language detection library
        """
        # Arabic detection
        if re.search(r'[\u0600-\u06FF]', text):
            return 'ar'
        
        # Chinese detection
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        
        # Hindi detection
        if re.search(r'[\u0900-\u097F]', text):
            return 'hi'
        
        # Spanish vs English (basic approach)
        spanish_indicators = ['ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü', '¿', '¡']
        if any(char in text.lower() for char in spanish_indicators):
            return 'es'
        
        # Default to English
        return 'en'
        
    async def analyze_sentiment_grok(self, text: str, language: str) -> Tuple[SentimentType, float]:
        """
        Analyze sentiment using Grok API
        TODO: Implement actual Grok API integration
        """
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_api_call < self.min_api_interval:
                time.sleep(self.min_api_interval - (current_time - self.last_api_call))
            
            # TODO: Replace with actual Grok API call
            # This is a placeholder for the actual implementation
            """
            headers = {
                'Authorization': f'Bearer {self.grok_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'text': text,
                'language': language,
                'task': 'sentiment_analysis'
            }
            
            response = requests.post(
                'https://api.grok.com/v1/sentiment',  # Placeholder URL
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                sentiment = SentimentType(result['sentiment'])
                confidence = result['confidence']
                self.last_api_call = time.time()
                return sentiment, confidence
            else:
                raise Exception(f"Grok API error: {response.status_code}")
            """
            
            # Placeholder return - remove when implementing actual API
            raise Exception("Grok API not implemented yet")
            
        except Exception as e:
            logger.error(f"Grok API failed: {e}")
            # Fall back to manual analysis
            return self.analyze_sentiment_manual(text, language)
    
    def analyze_sentiment_manual(self, text: str, language: str) -> Tuple[SentimentType, float]:
        """
        Fallback manual sentiment analysis using keywords
        """
        text_lower = text.lower()
        
        # Get keywords for the language (fallback to English if not supported)
        pos_keywords = self.positive_keywords.get(language, self.positive_keywords['en'])
        neg_keywords = self.negative_keywords.get(language, self.negative_keywords['en'])
        
        pos_count = sum(1 for word in pos_keywords if word in text_lower)
        neg_count = sum(1 for word in neg_keywords if word in text_lower)
        
        # Calculate sentiment based on keyword counts and rating patterns
        total_keywords = pos_count + neg_count
        
        if total_keywords == 0:
            return SentimentType.NEUTRAL, 0.5
        
        pos_ratio = pos_count / total_keywords
        
        if pos_ratio > 0.6:
            return SentimentType.POSITIVE, min(pos_ratio, 0.8)
        elif pos_ratio < 0.4:
            return SentimentType.NEGATIVE, min(1 - pos_ratio, 0.8)
        else:
            return SentimentType.NEUTRAL, 0.5
    
    def create_review_hash(self, app_id: str, content: str, platform: Platform) -> str:
        """Create unique hash for review to avoid duplicates"""
        unique_string = f"{app_id}_{platform.value}_{content[:100]}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def fetch_app_store_reviews(self, app_id: str, limit: int = 200) -> List[Dict]:
        """
        Fetch reviews from App Store
        TODO: Implement actual App Store API or scraping logic
        """
        # Placeholder - replace with actual implementation
        logger.info(f"Fetching App Store reviews for {app_id}")
        
        # This would typically involve:
        # 1. iTunes Search API for basic app info
        # 2. App Store scraping or unofficial APIs for reviews
        # 3. Handling pagination and rate limits
        
        return []  # Placeholder
    
    def fetch_play_store_reviews(self, app_id: str, limit: int = 200) -> List[Dict]:
        """
        Fetch reviews from Google Play Store
        TODO: Implement actual Play Store API or scraping logic
        """
        # Placeholder - replace with actual implementation
        logger.info(f"Fetching Play Store reviews for {app_id}")
        
        # This would typically involve:
        # 1. Google Play Developer API (if available)
        # 2. Play Store scraping
        # 3. Handling different languages and regions
        
        return []  # Placeholder
    
    def is_duplicate_review(self, review_id: str) -> bool:
        """Check if review already exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM reviews WHERE id = ?", (review_id,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def save_review(self, review: Review):
        """Save review to database"""
        if self.is_duplicate_review(review.id):
            logger.debug(f"Skipping duplicate review: {review.id}")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reviews (id, app_id, platform, content, rating, date, language, 
                               sentiment, confidence, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            review.id, review.app_id, review.platform.value, review.content,
            review.rating, review.date, review.language, review.sentiment.value,
            review.confidence, json.dumps(review.raw_data)
        ))
        
        conn.commit()
        conn.close()
        
    async def process_review(self, raw_review: Dict, app_id: str, platform: Platform) -> Review:
        """Process a single review"""
        content = raw_review.get('content', '')
        rating = raw_review.get('rating', 0)
        date = raw_review.get('date', datetime.now())
        
        # Create unique ID
        review_id = self.create_review_hash(app_id, content, platform)
        
        # Detect language
        language = self.detect_language(content)
        
        # Skip if language not supported
        if language not in self.supported_languages:
            sentiment, confidence = SentimentType.UNKNOWN, 0.0
        else:
            # Analyze sentiment
            if self.grok_api_key:
                sentiment, confidence = await self.analyze_sentiment_grok(content, language)
            else:
                sentiment, confidence = self.analyze_sentiment_manual(content, language)
        
        return Review(
            id=review_id,
            app_id=app_id,
            platform=platform,
            content=content,
            rating=rating,
            date=date,
            language=language,
            sentiment=sentiment,
            confidence=confidence,
            raw_data=raw_review
        )
    
    async def analyze_app_reviews(self, app_id: str, max_reviews_per_platform: int = 500) -> Dict:
        """
        Main function to analyze reviews for an app
        """
        logger.info(f"Starting analysis for app: {app_id}")
        
        results = {
            'app_id': app_id,
            'analysis_date': datetime.now(),
            'platforms': {},
            'combined_stats': {
                'total_reviews': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'unknown': 0
            }
        }
        
        # Fetch reviews from both platforms
        platforms_data = [
            (Platform.APP_STORE, self.fetch_app_store_reviews(app_id, max_reviews_per_platform)),
            (Platform.PLAY_STORE, self.fetch_play_store_reviews(app_id, max_reviews_per_platform))
        ]
        
        for platform, raw_reviews in platforms_data:
            if not raw_reviews:
                logger.warning(f"No reviews found for {app_id} on {platform.value}")
                continue
            
            platform_stats = {
                'total_reviews': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'unknown': 0,
                'avg_confidence': 0.0
            }
            
            total_confidence = 0
            processed_reviews = []
            
            # Process reviews in batches to manage API rate limits
            batch_size = 10
            for i in range(0, len(raw_reviews), batch_size):
                batch = raw_reviews[i:i + batch_size]
                
                # Process batch concurrently
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        self.process_review(raw_review, app_id, platform)
                        for raw_review in batch
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            review = await future
                            processed_reviews.append(review)
                            
                            # Update stats
                            platform_stats['total_reviews'] += 1
                            platform_stats[review.sentiment.value] += 1
                            total_confidence += review.confidence
                            
                            # Save to database
                            self.save_review(review)
                            
                        except Exception as e:
                            logger.error(f"Error processing review: {e}")
                
                # Rate limiting between batches
                time.sleep(1)
            
            # Calculate average confidence
            if platform_stats['total_reviews'] > 0:
                platform_stats['avg_confidence'] = total_confidence / platform_stats['total_reviews']
            
            results['platforms'][platform.value] = platform_stats
            
            # Update combined stats
            for key in ['total_reviews', 'positive', 'negative', 'neutral', 'unknown']:
                results['combined_stats'][key] += platform_stats[key]
        
        # Save analysis summary
        self.save_analysis_summary(results)
        
        logger.info(f"Analysis completed for {app_id}")
        return results
    
    def save_analysis_summary(self, results: Dict):
        """Save analysis summary to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        combined = results['combined_stats']
        
        cursor.execute('''
            INSERT INTO analysis_summary (app_id, analysis_date, total_reviews, 
                                        positive_count, negative_count, neutral_count, 
                                        unknown_count, avg_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            results['app_id'],
            results['analysis_date'],
            combined['total_reviews'],
            combined['positive'],
            combined['negative'],
            combined['neutral'],
            combined['unknown'],
            0.0  # Will calculate avg confidence across platforms later
        ))
        
        conn.commit()
        conn.close()
    
    def get_trend_analysis(self, app_id: str, days: int = 30) -> Dict:
        """Get trend analysis for an app over specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get historical data
        cursor.execute('''
            SELECT analysis_date, positive_count, negative_count, neutral_count, total_reviews
            FROM analysis_summary 
            WHERE app_id = ? AND analysis_date >= datetime('now', '-{} days')
            ORDER BY analysis_date
        '''.format(days), (app_id,))
        
        historical_data = cursor.fetchall()
        conn.close()
        
        if not historical_data:
            return {'trend': 'insufficient_data', 'data': []}
        
        # Calculate trend
        if len(historical_data) >= 2:
            recent_positive_ratio = historical_data[-1][1] / max(historical_data[-1][4], 1)
            older_positive_ratio = historical_data[0][1] / max(historical_data[0][4], 1)
            
            if recent_positive_ratio > older_positive_ratio + 0.1:
                trend = 'improving'
            elif recent_positive_ratio < older_positive_ratio - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'data': historical_data,
            'summary': {
                'total_analyses': len(historical_data),
                'date_range': f"{historical_data[0][0]} to {historical_data[-1][0]}" if historical_data else None
            }
        }

# Example usage and configuration
async def main():
    """Main function to run the review analyzer"""
    
    # Initialize analyzer
    # TODO: Add your Grok API key here
    analyzer = ReviewAnalyzer(grok_api_key=None)
    
    # List of apps to analyze (replace with actual app IDs)
    apps_to_analyze = [
        "com.example.app1",  # Android package name
        "123456789",         # iOS app ID
        # Add more apps as needed
    ]
    
    for app_id in apps_to_analyze:
        try:
            # Analyze reviews (targeting ~1000 reviews per week = ~500 per platform)
            results = await analyzer.analyze_app_reviews(app_id, max_reviews_per_platform=500)
            
            print(f"\nAnalysis Results for {app_id}:")
            print(f"Total Reviews: {results['combined_stats']['total_reviews']}")
            print(f"Positive: {results['combined_stats']['positive']}")
            print(f"Negative: {results['combined_stats']['negative']}")
            print(f"Neutral: {results['combined_stats']['neutral']}")
            
            # Get trend analysis
            trend_data = analyzer.get_trend_analysis(app_id)
            print(f"Trend: {trend_data['trend']}")
            
        except Exception as e:
            logger.error(f"Error analyzing {app_id}: {e}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())