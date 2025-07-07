import json
import sqlite3
import requests
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import statistics
import numpy as np
from collections import defaultdict, Counter
from difflib import SequenceMatcher

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

class TrendDirection(Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"

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
    is_filtered: bool = False
    quality_score: float = 0.0
    word_count: int = 0

@dataclass
class SentimentBatch:
    reviews: List[Review]
    sentiment_distribution: Dict[str, int]
    avg_confidence: float
    quality_score: float
    anomaly_score: float = 0.0

@dataclass
class TrendAnalysis:
    app_id: str
    time_period: str
    current_week: Dict[str, int]
    previous_week: Dict[str, int]
    trend_direction: TrendDirection
    confidence_change: float
    anomaly_detected: bool
    weekly_comparison: Dict[str, float]

class SmartReviewFilter:
    """Enhanced preprocessing layer for review filtering"""
    
    def __init__(self):
        self.spam_patterns = [
            r'^.{1,3}$',  # Very short reviews (1-3 characters)
            r'^(.)\1{4,}',  # Repeated characters
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
            r'(https?://[^\s]+)',  # URLs
            r'(\d{10,})',  # Phone numbers
        ]
        
        self.quality_keywords = {
            'high_quality': [
                'feature', 'functionality', 'interface', 'performance', 'experience', 
                'improvement', 'bug', 'issue', 'update', 'design', 'usability',
                'navigation', 'speed', 'loading', 'crash', 'error', 'problem',
                'solution', 'helpful', 'useful', 'recommend', 'disappointing',
                'excellent', 'terrible', 'amazing', 'awful', 'love', 'hate',
                'works', 'broken', 'fix', 'better', 'worse', 'easy', 'difficult'
            ],
            'low_quality': [
                'first', 'test', 'trying', 'downloaded', 'install', 'ok', 'nice',
                'cool', 'wow', 'meh', 'hm', 'hmm', 'yep', 'nope'
            ]
        }
        
        self.duplicate_threshold = 0.85
        self.review_cache = {}
    
    def calculate_quality_score(self, review: Review) -> float:
        """Calculate quality score for a review"""
        score = 0.0
        content = review.content.lower()
        
        # Word count factor (structured reviews have more words)
        word_count = len(content.split())
        if word_count < 3:
            score -= 0.5
        elif word_count >= 10:
            score += 0.3
        elif word_count >= 5:
            score += 0.1
        
        # Quality keywords
        high_quality_count = sum(1 for keyword in self.quality_keywords['high_quality'] if keyword in content)
        low_quality_count = sum(1 for keyword in self.quality_keywords['low_quality'] if keyword in content)
        
        score += (high_quality_count * 0.1) - (low_quality_count * 0.1)
        
        # Punctuation and structure
        if '.' in content or '!' in content or '?' in content:
            score += 0.1
        
        # Check for specific feedback
        feedback_patterns = [
            r'because', r'since', r'when', r'after', r'before', r'however',
            r'although', r'but', r'and', r'also', r'additionally'
        ]
        
        for pattern in feedback_patterns:
            if re.search(pattern, content):
                score += 0.05
        
        # Rating consistency (detailed reviews usually have consistent ratings)
        if review.rating in [1, 2] and any(word in content for word in ['good', 'great', 'excellent']):
            score -= 0.2  # Inconsistent rating
        elif review.rating in [4, 5] and any(word in content for word in ['bad', 'terrible', 'awful']):
            score -= 0.2  # Inconsistent rating
        
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def is_spam_or_junk(self, review: Review) -> bool:
        """Check if review is spam or junk"""
        content = review.content.strip()
        
        # Check spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, content):
                return True
        
        # Check for very short reviews without substance
        if len(content) < 10 and not any(word in content.lower() for word in self.quality_keywords['high_quality']):
            return True
        
        # Check for repeated content (exact duplicates)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.review_cache:
            return True
        
        self.review_cache[content_hash] = True
        return False
    
    def is_similar_to_existing(self, review: Review, existing_reviews: List[Review]) -> bool:
        """Check if review is too similar to existing ones"""
        if len(existing_reviews) == 0:
            return False
        
        # Check similarity with recent reviews
        recent_reviews = existing_reviews[-20:]  # Check last 20 reviews
        
        for existing in recent_reviews:
            similarity = SequenceMatcher(None, review.content.lower(), existing.content.lower()).ratio()
            if similarity > self.duplicate_threshold:
                return True
        
        return False
    
    def filter_reviews(self, reviews: List[Review]) -> List[Review]:
        """Filter reviews based on quality and spam detection"""
        filtered_reviews = []
        
        for review in reviews:
            # Skip spam/junk
            if self.is_spam_or_junk(review):
                review.is_filtered = True
                continue
            
            # Calculate quality score
            review.quality_score = self.calculate_quality_score(review)
            review.word_count = len(review.content.split())
            
            # Only keep high-quality reviews (score > 0.3)
            if review.quality_score > 0.3:
                # Check for duplicates
                if not self.is_similar_to_existing(review, filtered_reviews):
                    filtered_reviews.append(review)
                else:
                    review.is_filtered = True
            else:
                review.is_filtered = True
        
        logger.info(f"Filtered {len(reviews)} reviews down to {len(filtered_reviews)} high-quality reviews")
        return filtered_reviews

class SentimentDriftTracker:
    """Track sentiment changes and detect anomalies"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.anomaly_threshold = 0.20  # 20% change threshold
    
    def get_weekly_sentiment_data(self, app_id: str) -> Dict:
        """Get sentiment data for current and previous week"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current week data (last 7 days)
        cursor.execute("""
            SELECT sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM reviews 
            WHERE app_id = ? AND date >= datetime('now', '-7 days')
            GROUP BY sentiment
        """, (app_id,))
        
        current_week_data = cursor.fetchall()
        
        # Get previous week data (8-14 days ago)
        cursor.execute("""
            SELECT sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM reviews 
            WHERE app_id = ? AND date >= datetime('now', '-14 days') 
            AND date < datetime('now', '-7 days')
            GROUP BY sentiment
        """, (app_id,))
        
        previous_week_data = cursor.fetchall()
        
        conn.close()
        
        # Process data
        current_week = {row[0]: row[1] for row in current_week_data}
        previous_week = {row[0]: row[1] for row in previous_week_data}
        
        return {
            'current_week': current_week,
            'previous_week': previous_week,
            'current_confidence': {row[0]: row[2] for row in current_week_data},
            'previous_confidence': {row[0]: row[2] for row in previous_week_data}
        }
    
    def detect_sentiment_anomaly(self, app_id: str) -> TrendAnalysis:
        """Detect sentiment anomalies and trends"""
        data = self.get_weekly_sentiment_data(app_id)
        current_week = data['current_week']
        previous_week = data['previous_week']
        
        # Calculate totals
        current_total = sum(current_week.values()) or 1
        previous_total = sum(previous_week.values()) or 1
        
        # Calculate percentages
        current_percentages = {k: v/current_total for k, v in current_week.items()}
        previous_percentages = {k: v/previous_total for k, v in previous_week.items()}
        
        # Calculate changes
        sentiment_changes = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            current_pct = current_percentages.get(sentiment, 0)
            previous_pct = previous_percentages.get(sentiment, 0)
            sentiment_changes[sentiment] = current_pct - previous_pct
        
        # Determine trend direction
        positive_change = sentiment_changes.get('positive', 0)
        negative_change = sentiment_changes.get('negative', 0)
        
        anomaly_detected = False
        
        if abs(positive_change) > self.anomaly_threshold or abs(negative_change) > self.anomaly_threshold:
            anomaly_detected = True
        
        if positive_change > 0.1 and negative_change < -0.1:
            trend_direction = TrendDirection.IMPROVING
        elif positive_change < -0.1 and negative_change > 0.1:
            trend_direction = TrendDirection.DECLINING
        elif max(abs(positive_change), abs(negative_change)) > 0.15:
            trend_direction = TrendDirection.VOLATILE
        else:
            trend_direction = TrendDirection.STABLE
        
        # Calculate confidence change
        current_conf = statistics.mean(data['current_confidence'].values()) if data['current_confidence'] else 0
        previous_conf = statistics.mean(data['previous_confidence'].values()) if data['previous_confidence'] else 0
        confidence_change = current_conf - previous_conf
        
        return TrendAnalysis(
            app_id=app_id,
            time_period="weekly",
            current_week=current_week,
            previous_week=previous_week,
            trend_direction=trend_direction,
            confidence_change=confidence_change,
            anomaly_detected=anomaly_detected,
            weekly_comparison=sentiment_changes
        )

class PlatformWeightCalculator:
    """Calculate platform weights based on review volume"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_platform_weights(self, app_id: str) -> Dict[Platform, float]:
        """Calculate platform weights based on historical review volume"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT platform, COUNT(*) as count
            FROM reviews 
            WHERE app_id = ? AND date >= datetime('now', '-30 days')
            GROUP BY platform
        """, (app_id,))
        
        platform_counts = cursor.fetchall()
        conn.close()
        
        if not platform_counts:
            # Default equal weights
            return {Platform.APP_STORE: 0.5, Platform.PLAY_STORE: 0.5}
        
        # Calculate total reviews
        total_reviews = sum(count for _, count in platform_counts)
        
        # Calculate weights
        weights = {}
        for platform_str, count in platform_counts:
            platform = Platform(platform_str)
            weights[platform] = count / total_reviews
        
        # Ensure both platforms have weights
        if Platform.APP_STORE not in weights:
            weights[Platform.APP_STORE] = 0.0
        if Platform.PLAY_STORE not in weights:
            weights[Platform.PLAY_STORE] = 0.0
        
        return weights
    
    def calculate_review_distribution(self, app_id: str, total_target: int = 1000) -> Dict[Platform, int]:
        """Calculate how many reviews to fetch from each platform"""
        weights = self.get_platform_weights(app_id)
        
        distribution = {}
        for platform, weight in weights.items():
            distribution[platform] = int(total_target * weight)
        
        # Ensure at least some reviews from each platform if available
        for platform in distribution:
            if distribution[platform] == 0 and weights[platform] > 0:
                distribution[platform] = min(100, total_target // 4)
        
        return distribution

class ReviewAnalyzer:
    def __init__(self, grok_api_key: str = None):
        """
        Initialize the Enhanced Review Analyzer
        
        Args:
            grok_api_key: API key for Grok sentiment analysis
        """
        self.grok_api_key = grok_api_key
        self.db_path = "review_analysis.db"
        self.init_database()
        
        # Initialize new components
        self.smart_filter = SmartReviewFilter()
        self.drift_tracker = SentimentDriftTracker(self.db_path)
        self.platform_calculator = PlatformWeightCalculator(self.db_path)
        
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
        
        # Cache for processed reviews
        self.review_cache = {}
        
    def init_database(self):
        """Initialize SQLite database with enhanced tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced reviews table
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
                quality_score REAL DEFAULT 0.0,
                word_count INTEGER DEFAULT 0,
                is_filtered BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Enhanced analysis summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_id TEXT NOT NULL,
                platform TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_reviews INTEGER,
                filtered_reviews INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                unknown_count INTEGER,
                avg_confidence REAL,
                avg_quality_score REAL,
                trend_direction TEXT,
                anomaly_detected BOOLEAN DEFAULT FALSE,
                platform_weight REAL DEFAULT 1.0,
                manual_review_flagged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # New table for trend tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trend_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_id TEXT NOT NULL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                current_week_positive INTEGER,
                current_week_negative INTEGER,
                current_week_neutral INTEGER,
                previous_week_positive INTEGER,
                previous_week_negative INTEGER,
                previous_week_neutral INTEGER,
                trend_direction TEXT,
                confidence_change REAL,
                anomaly_detected BOOLEAN,
                sentiment_change_positive REAL,
                sentiment_change_negative REAL
            )
        ''')
        
        # Cache table for processed reviews
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_cache (
                id TEXT PRIMARY KEY,
                app_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sentiment TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns
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
    
    def is_cached_review(self, review_id: str) -> Optional[Tuple[SentimentType, float]]:
        """Check if review is already processed and cached"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT sentiment, confidence FROM review_cache 
            WHERE id = ? AND processed_at >= datetime('now', '-7 days')
        """, (review_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return SentimentType(result[0]), result[1]
        return None
    
    def cache_review_result(self, review: Review):
        """Cache review analysis result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        content_hash = hashlib.md5(review.content.encode()).hexdigest()
        
        cursor.execute("""
            INSERT OR REPLACE INTO review_cache 
            (id, app_id, platform, content_hash, sentiment, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (review.id, review.app_id, review.platform.value, content_hash, 
              review.sentiment.value, review.confidence))
        
        conn.commit()
        conn.close()
        
    async def analyze_sentiment_grok(self, text: str, language: str) -> Tuple[SentimentType, float]:
        """
        Analyze sentiment using Grok API
        """
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_api_call < self.min_api_interval:
                time.sleep(self.min_api_interval - (current_time - self.last_api_call))
            
            # TODO: Replace with actual Grok API call
            """
            headers = {
                'Authorization': f'Bearer {self.grok_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'text': text,
                'language': language,
                'task': 'sentiment_analysis',
                'return_confidence': True
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
        Enhanced fallback manual sentiment analysis
        """
        text_lower = text.lower()
        
        # Get keywords for the language
        pos_keywords = self.positive_keywords.get(language, self.positive_keywords['en'])
        neg_keywords = self.negative_keywords.get(language, self.negative_keywords['en'])
        
        pos_count = sum(1 for word in pos_keywords if word in text_lower)
        neg_count = sum(1 for word in neg_keywords if word in text_lower)
        
        # Enhanced sentiment calculation
        total_keywords = pos_count + neg_count
        
        if total_keywords == 0:
            return SentimentType.NEUTRAL, 0.5
        
        pos_ratio = pos_count / total_keywords
        
        # More nuanced confidence calculation
        word_count = len(text_lower.split())
        confidence_boost = min(word_count / 20, 0.2)  # Longer reviews get confidence boost
        
        if pos_ratio > 0.65:
            confidence = min(pos_ratio + confidence_boost, 0.9)
            return SentimentType.POSITIVE, confidence
        elif pos_ratio < 0.35:
            confidence = min((1 - pos_ratio) + confidence_boost, 0.9)
            return SentimentType.NEGATIVE, confidence
        else:
            return SentimentType.NEUTRAL, 0.5 + confidence_boost
    
    def create_review_hash(self, app_id: str, content: str, platform: Platform) -> str:
        """Create unique hash for review to avoid duplicates"""
        unique_string = f"{app_id}_{platform.value}_{content[:100]}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def fetch_app_store_reviews(self, app_id: str, limit: int = 200) -> List[Dict]:
        """
        Fetch reviews from App Store
        TODO: Implement actual App Store API or scraping logic
        """
        logger.info(f"Fetching {limit} App Store reviews for {app_id}")
        
        # This would typically involve:
        # 1. iTunes Search API for basic app info
        # 2. App Store scraping or unofficial APIs for reviews
        # 3. Handling pagination and rate limits
        # 4. Filtering for recent reviews (last week)
        
        return []  # Placeholder
    
    def fetch_play_store_reviews(self, app_id: str, limit: int = 200) -> List[Dict]:
        """
        Fetch reviews from Google Play Store
        TODO: Implement actual Play Store API or scraping logic
        """
        logger.info(f"Fetching {limit} Play Store reviews for {app_id}")
        
        # This would typically involve:
        # 1. Google Play Developer API (if available)
        # 2. Play Store scraping
        # 3. Handling different languages and regions
        # 4. Filtering for recent reviews (last week)
        
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
        """Save review to database with enhanced fields"""
        if self.is_duplicate_review(review.id):
            logger.debug(f"Skipping duplicate review: {review.id}")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reviews (id, app_id, platform, content, rating, date, language, 
                               sentiment, confidence, raw_data, quality_score, word_count, is_filtered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            review.id, review.app_id, review.platform.value, review.content,
            review.rating, review.date, review.language, review.sentiment.value,
            review.confidence, json.dumps(review.raw_data), review.quality_score,
            review.word_count, review.is_filtered
        ))
        
        conn.commit()
        conn.close()
        
        # Cache the result
        self.cache_review_result(review)
        
    async def process_review(self, raw_review: Dict, app_id: str, platform: Platform) -> Review:
        """Process a single review with enhanced analysis"""
        content = raw_review.get('content', '')
        rating = raw_review.get('rating', 0)
        date = raw_review.get('date', datetime.now())
        
        # Create unique ID
        review_id = self.create_review_hash(app_id, content, platform)
        
        # Check cache first
        cached_result = self.is_cached_review(review_id)
        if cached_result:
            sentiment, confidence = cached_result
            logger.debug(f"Using cached result for review {review_id}")
        else:
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
        
        # Create review object
        review = Review(
            id=review_id,
            app_id=app_id,
            platform=platform,
            content=content,
            rating=rating,
            date=date,
            language=self.detect_language(content),
            sentiment=sentiment,
            confidence=confidence,
            raw_data=raw_review,
            word_count=len(content.split())
        )
        
        return review
    
    def should_flag_for_manual_review(self, batch: SentimentBatch) -> bool:
        """Determine if a batch should be flagged for manual review"""
        # Flag if average confidence is below threshold
        if batch.avg_confidence < 0.65:
            return True
        
        # Flag if sentiment distribution is very unclear
        total_reviews = len(batch.reviews)
        if total_reviews > 0:
            pos_ratio = batch.sentiment_distribution.get('positive', 0) / total_reviews
            neg_ratio = batch.sentiment_distribution.get('negative', 0) / total_reviews
            
            # Flag if sentiment is very mixed (no clear majority)
            if 0.3 < pos_ratio < 0.7 and 0.3 < neg_ratio < 0.7:
                return True
        
        return False
    
async def analyze_app_reviews(self, app_id: str) -> Dict:
        """
        Complete enhanced review analysis with all new features
        """
        logger.info(f"Starting enhanced analysis for app: {app_id}")
        
        # Step 1: Calculate platform distribution based on historical data
        platform_distribution = self.platform_calculator.calculate_review_distribution(app_id)
        logger.info(f"Platform distribution: {platform_distribution}")
        
        # Step 2: Fetch reviews from both platforms
        all_reviews = []
        
        # Fetch from App Store
        if platform_distribution.get(Platform.APP_STORE, 0) > 0:
            app_store_reviews = self.fetch_app_store_reviews(
                app_id, 
                limit=platform_distribution[Platform.APP_STORE]
            )
            for raw_review in app_store_reviews:
                review = await self.process_review(raw_review, app_id, Platform.APP_STORE)
                all_reviews.append(review)
        
        # Fetch from Play Store
        if platform_distribution.get(Platform.PLAY_STORE, 0) > 0:
            play_store_reviews = self.fetch_play_store_reviews(
                app_id, 
                limit=platform_distribution[Platform.PLAY_STORE]
            )
            for raw_review in play_store_reviews:
                review = await self.process_review(raw_review, app_id, Platform.PLAY_STORE)
                all_reviews.append(review)
        
        logger.info(f"Fetched {len(all_reviews)} total reviews")
        
        # Step 3: Apply smart filtering to keep only high-quality reviews
        filtered_reviews = self.smart_filter.filter_reviews(all_reviews)
        logger.info(f"After filtering: {len(filtered_reviews)} high-quality reviews")
        
        # Step 4: Analyze sentiment drift and trends
        trend_analysis = self.drift_tracker.detect_sentiment_anomaly(app_id)
        
        # Step 5: Process reviews in batches and save to database
        batch_size = 50
        processed_batches = []
        
        for i in range(0, len(filtered_reviews), batch_size):
            batch_reviews = filtered_reviews[i:i + batch_size]
            
            # Create sentiment batch
            sentiment_distribution = Counter()
            confidence_scores = []
            quality_scores = []
            
            for review in batch_reviews:
                sentiment_distribution[review.sentiment.value] += 1
                confidence_scores.append(review.confidence)
                quality_scores.append(review.quality_score)
                
                # Save review to database
                self.save_review(review)
            
            # Calculate batch metrics
            batch = SentimentBatch(
                reviews=batch_reviews,
                sentiment_distribution=dict(sentiment_distribution),
                avg_confidence=statistics.mean(confidence_scores) if confidence_scores else 0.0,
                quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
                anomaly_score=1.0 if trend_analysis.anomaly_detected else 0.0
            )
            
            processed_batches.append(batch)
        
        # Step 6: Calculate platform weights for final analysis
        platform_weights = self.platform_calculator.get_platform_weights(app_id)
        
        # Step 7: Generate comprehensive analysis summary
        analysis_summary = self.generate_analysis_summary(
            app_id, 
            processed_batches, 
            trend_analysis, 
            platform_weights
        )
        
        # Step 8: Save analysis summary to database
        self.save_analysis_summary(analysis_summary)
        
        # Step 9: Save trend analysis
        self.save_trend_analysis(trend_analysis)
        
        logger.info(f"Analysis complete for app: {app_id}")
        return analysis_summary
    
    def generate_analysis_summary(self, app_id: str, batches: List[SentimentBatch], 
                                 trend_analysis: TrendAnalysis, 
                                 platform_weights: Dict[Platform, float]) -> Dict:
        """Generate comprehensive analysis summary"""
        
        # Aggregate all reviews from batches
        all_reviews = []
        for batch in batches:
            all_reviews.extend(batch.reviews)
        
        if not all_reviews:
            return {
                'app_id': app_id,
                'status': 'no_reviews',
                'message': 'No reviews found for analysis'
            }
        
        # Calculate weighted sentiment distribution
        platform_sentiment = {Platform.APP_STORE: Counter(), Platform.PLAY_STORE: Counter()}
        
        for review in all_reviews:
            platform_sentiment[review.platform][review.sentiment.value] += 1
        
        # Apply platform weights
        weighted_sentiment = Counter()
        for platform, weight in platform_weights.items():
            for sentiment, count in platform_sentiment[platform].items():
                weighted_sentiment[sentiment] += count * weight
        
        # Calculate final percentages
        total_weighted = sum(weighted_sentiment.values())
        sentiment_percentages = {
            sentiment: (count / total_weighted) * 100 if total_weighted > 0 else 0
            for sentiment, count in weighted_sentiment.items()
        }
        
        # Calculate confidence metrics
        confidence_scores = [review.confidence for review in all_reviews]
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        # Calculate quality metrics
        quality_scores = [review.quality_score for review in all_reviews]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # Check if manual review is needed
        manual_review_needed = any(self.should_flag_for_manual_review(batch) for batch in batches)
        
        # Determine overall app sentiment
        if sentiment_percentages.get('positive', 0) > 60:
            overall_sentiment = 'positive'
        elif sentiment_percentages.get('negative', 0) > 40:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'app_id': app_id,
            'analysis_date': datetime.now().isoformat(),
            'total_reviews_analyzed': len(all_reviews),
            'total_reviews_filtered': len([r for r in all_reviews if r.is_filtered]),
            'sentiment_distribution': sentiment_percentages,
            'overall_sentiment': overall_sentiment,
            'confidence_metrics': {
                'average_confidence': avg_confidence,
                'high_confidence_reviews': len([r for r in all_reviews if r.confidence > 0.8]),
                'low_confidence_reviews': len([r for r in all_reviews if r.confidence < 0.5])
            },
            'quality_metrics': {
                'average_quality_score': avg_quality,
                'high_quality_reviews': len([r for r in all_reviews if r.quality_score > 0.7]),
                'low_quality_reviews': len([r for r in all_reviews if r.quality_score < 0.3])
            },
            'platform_analysis': {
                'app_store_weight': platform_weights.get(Platform.APP_STORE, 0),
                'play_store_weight': platform_weights.get(Platform.PLAY_STORE, 0),
                'app_store_reviews': len([r for r in all_reviews if r.platform == Platform.APP_STORE]),
                'play_store_reviews': len([r for r in all_reviews if r.platform == Platform.PLAY_STORE])
            },
            'trend_analysis': {
                'trend_direction': trend_analysis.trend_direction.value,
                'anomaly_detected': trend_analysis.anomaly_detected,
                'confidence_change': trend_analysis.confidence_change,
                'weekly_sentiment_change': trend_analysis.weekly_comparison
            },
            'flags': {
                'manual_review_needed': manual_review_needed,
                'anomaly_detected': trend_analysis.anomaly_detected,
                'low_confidence_batch': avg_confidence < 0.65
            },
            'language_distribution': self.get_language_distribution(all_reviews),
            'processing_stats': {
                'batches_processed': len(batches),
                'cache_hits': len([r for r in all_reviews if self.is_cached_review(r.id)]),
                'api_calls_saved': len([r for r in all_reviews if self.is_cached_review(r.id)])
            }
        }
    
    def get_language_distribution(self, reviews: List[Review]) -> Dict[str, int]:
        """Get language distribution of reviews"""
        language_count = Counter()
        for review in reviews:
            language_count[review.language] += 1
        return dict(language_count)
    
    def save_analysis_summary(self, summary: Dict):
        """Save analysis summary to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_summary (
                app_id, platform, total_reviews, filtered_reviews,
                positive_count, negative_count, neutral_count, unknown_count,
                avg_confidence, avg_quality_score, trend_direction, anomaly_detected,
                platform_weight, manual_review_flagged
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            summary['app_id'],
            'combined',
            summary['total_reviews_analyzed'],
            summary['total_reviews_filtered'],
            int(summary['sentiment_distribution'].get('positive', 0) * summary['total_reviews_analyzed'] / 100),
            int(summary['sentiment_distribution'].get('negative', 0) * summary['total_reviews_analyzed'] / 100),
            int(summary['sentiment_distribution'].get('neutral', 0) * summary['total_reviews_analyzed'] / 100),
            int(summary['sentiment_distribution'].get('unknown', 0) * summary['total_reviews_analyzed'] / 100),
            summary['confidence_metrics']['average_confidence'],
            summary['quality_metrics']['average_quality_score'],
            summary['trend_analysis']['trend_direction'],
            summary['trend_analysis']['anomaly_detected'],
            1.0,  # Combined analysis weight
            summary['flags']['manual_review_needed']
        ))
        
        conn.commit()
        conn.close()
    
    def save_trend_analysis(self, trend_analysis: TrendAnalysis):
        """Save trend analysis to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trend_analysis (
                app_id, current_week_positive, current_week_negative, current_week_neutral,
                previous_week_positive, previous_week_negative, previous_week_neutral,
                trend_direction, confidence_change, anomaly_detected,
                sentiment_change_positive, sentiment_change_negative
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trend_analysis.app_id,
            trend_analysis.current_week.get('positive', 0),
            trend_analysis.current_week.get('negative', 0),
            trend_analysis.current_week.get('neutral', 0),
            trend_analysis.previous_week.get('positive', 0),
            trend_analysis.previous_week.get('negative', 0),
            trend_analysis.previous_week.get('neutral', 0),
            trend_analysis.trend_direction.value,
            trend_analysis.confidence_change,
            trend_analysis.anomaly_detected,
            trend_analysis.weekly_comparison.get('positive', 0),
            trend_analysis.weekly_comparison.get('negative', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_app_analysis_history(self, app_id: str, days: int = 30) -> List[Dict]:
        """Get historical analysis data for an app"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analysis_summary 
            WHERE app_id = ? AND analysis_date >= datetime('now', '-{} days')
            ORDER BY analysis_date DESC
        '''.format(days), (app_id,))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        conn.close()
        return results
    
    def get_trend_analysis_history(self, app_id: str, days: int = 30) -> List[Dict]:
        """Get trend analysis history for an app"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trend_analysis 
            WHERE app_id = ? AND analysis_date >= datetime('now', '-{} days')
            ORDER BY analysis_date DESC
        '''.format(days), (app_id,))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        conn.close()
        return results
    
    def get_apps_requiring_manual_review(self) -> List[str]:
        """Get list of apps that require manual review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT app_id FROM analysis_summary 
            WHERE manual_review_flagged = 1 
            AND analysis_date >= datetime('now', '-7 days')
        ''')
        
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_anomaly_apps(self) -> List[str]:
        """Get list of apps with detected anomalies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT app_id FROM trend_analysis 
            WHERE anomaly_detected = 1 
            AND analysis_date >= datetime('now', '-7 days')
        ''')
        
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean old reviews
        cursor.execute('''
            DELETE FROM reviews 
            WHERE created_at < datetime('now', '-{} days')
        '''.format(days))
        
        # Clean old cache
        cursor.execute('''
            DELETE FROM review_cache 
            WHERE processed_at < datetime('now', '-{} days')
        '''.format(days))
        
        # Clean old analysis summaries
        cursor.execute('''
            DELETE FROM analysis_summary 
            WHERE analysis_date < datetime('now', '-{} days')
        '''.format(days))
        
        # Clean old trend analysis
        cursor.execute('''
            DELETE FROM trend_analysis 
            WHERE analysis_date < datetime('now', '-{} days')
        '''.format(days))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up data older than {days} days")

# Usage example and main execution
async def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = ReviewAnalyzer(grok_api_key="your_grok_api_key_here")
    
    # Example app IDs
    app_ids = ["com.example.app1", "com.example.app2"]
    
    # Run analysis for each app
    for app_id in app_ids:
        try:
            logger.info(f"Starting analysis for {app_id}")
            
            # Run comprehensive analysis
            result = await analyzer.analyze_app_reviews(app_id)
            
            # Print summary
            print(f"\n--- Analysis Results for {app_id} ---")
            print(f"Overall Sentiment: {result['overall_sentiment']}")
            print(f"Total Reviews: {result['total_reviews_analyzed']}")
            print(f"Confidence: {result['confidence_metrics']['average_confidence']:.2f}")
            print(f"Quality Score: {result['quality_metrics']['average_quality_score']:.2f}")
            print(f"Trend: {result['trend_analysis']['trend_direction']}")
            
            if result['flags']['manual_review_needed']:
                print("⚠️  Manual Review Recommended")
            
            if result['flags']['anomaly_detected']:
                print("🚨 Anomaly Detected")
            
            # Wait between apps to respect rate limits
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error analyzing {app_id}: {e}")
    
    # Clean up old data
    analyzer.cleanup_old_data(days=90)
    
    # Get apps requiring attention
    manual_review_apps = analyzer.get_apps_requiring_manual_review()
    anomaly_apps = analyzer.get_anomaly_apps()
    
    if manual_review_apps:
        print(f"\n📋 Apps requiring manual review: {manual_review_apps}")
    
    if anomaly_apps:
        print(f"\n🚨 Apps with anomalies: {anomaly_apps}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
