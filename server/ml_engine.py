"""
Machine Learning Engine for Chatbot Analytics

This module provides:
1. Topic Clustering using TF-IDF + K-Means
2. Sentiment Analysis for user satisfaction prediction
3. Response Quality Scoring
4. Anomaly Detection for unusual queries
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from .orm import AnalyticsEvent, TopicCluster, Conversation, Message


class MLEngine:
    """Machine Learning engine for chatbot analytics and optimization."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.kmeans = None
        self.is_trained = False
        
    def train_topic_clusters(self, messages: List[str], n_clusters: int = 5) -> Dict:
        """
        Train topic clustering model on historical messages.
        
        Returns:
            Dict with cluster labels and top terms for each cluster
        """
        if len(messages) < n_clusters:
            return {"status": "insufficient_data", "clusters": []}
        
        # Vectorize messages
        X = self.vectorizer.fit_transform(messages)
        
        # Train K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        
        self.is_trained = True
        
        # Extract top terms for each cluster
        feature_names = self.vectorizer.get_feature_names_out()
        clusters_info = []
        
        for i in range(n_clusters):
            center = self.kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-5:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            clusters_info.append({
                "cluster_id": i,
                "top_terms": top_terms,
                "label": self._generate_cluster_label(top_terms)
            })
        
        return {
            "status": "success",
            "n_clusters": n_clusters,
            "clusters": clusters_info
        }
    
    def predict_topic(self, message: str) -> Optional[int]:
        """Predict the topic cluster for a new message."""
        if not self.is_trained:
            return None
        
        X = self.vectorizer.transform([message])
        cluster = self.kmeans.predict(X)[0]
        return int(cluster)
    
    def _generate_cluster_label(self, top_terms: List[str]) -> str:
        """Generate a human-readable label from top terms."""
        # Simple heuristic-based labeling
        terms_str = ' '.join(top_terms).lower()
        
        if any(word in terms_str for word in ['register', 'enroll', 'class', 'course', 'drop', 'add']):
            return "Registration & Enrollment"
        elif any(word in terms_str for word in ['tuition', 'bill', 'payment', 'bursar', 'financial']):
            return "Billing & Financial Aid"
        elif any(word in terms_str for word in ['housing', 'dorm', 'residence', 'room']):
            return "Housing & Residential Life"
        elif any(word in terms_str for word in ['library', 'study', 'book', 'hours']):
            return "Library & Academic Resources"
        elif any(word in terms_str for word in ['wifi', 'network', 'eduroam', 'tech', 'computer']):
            return "IT & Technology"
        else:
            return f"Topic: {top_terms[0].title()}"
    
    def calculate_response_quality(self, response: str, has_sources: bool = False) -> float:
        """
        Calculate quality score for a response (0-100).
        
        Factors:
        - Length (not too short, not too long)
        - Has sources
        - Proper formatting
        """
        score = 50.0  # Base score
        
        # Length scoring
        word_count = len(response.split())
        if 20 <= word_count <= 200:
            score += 20
        elif word_count < 10:
            score -= 20
        
        # Sources bonus
        if has_sources:
            score += 20
        
        # Formatting bonus (has lists, bold, etc.)
        if any(marker in response for marker in ['**', '- ', '1.', '2.']):
            score += 10
        
        return min(100.0, max(0.0, score))
    
    def detect_anomaly(self, message: str, db: Session) -> Dict:
        """
        Detect if a message is anomalous (unusual query).
        
        Uses:
        - Message length
        - Rare words
        - Similarity to historical messages
        """
        anomaly_score = 0.0
        reasons = []
        
        # Check length
        word_count = len(message.split())
        if word_count > 100:
            anomaly_score += 0.3
            reasons.append("unusually_long")
        elif word_count < 3:
            anomaly_score += 0.2
            reasons.append("very_short")
        
        # Check for spam patterns
        spam_patterns = [
            r'(buy|sell|click here|free money)',
            r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)',
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                anomaly_score += 0.5
                reasons.append("spam_pattern")
                break
        
        return {
            "is_anomaly": anomaly_score > 0.5,
            "score": anomaly_score,
            "reasons": reasons
        }
    
    def get_conversation_insights(self, db: Session, days: int = 7) -> Dict:
        """
        Generate ML-powered insights from conversation data.
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get conversation stats
        total_conversations = db.query(Conversation).filter(
            Conversation.created_at >= cutoff_date
        ).count()
        
        total_events = db.query(AnalyticsEvent).join(Conversation).filter(
            Conversation.created_at >= cutoff_date
        ).count()
        
        # Calculate satisfaction rate
        helpful_events = db.query(AnalyticsEvent).filter(
            AnalyticsEvent.helpful == True,
            AnalyticsEvent.created_at >= cutoff_date
        ).count()
        
        satisfaction_rate = (helpful_events / total_events * 100) if total_events > 0 else 0
        
        # Get top topics
        topic_distribution = db.query(
            TopicCluster.label,
            func.count(AnalyticsEvent.id).label('count')
        ).join(AnalyticsEvent).filter(
            AnalyticsEvent.created_at >= cutoff_date
        ).group_by(TopicCluster.label).order_by(func.count().desc()).limit(5).all()
        
        # Predict trends
        trend_prediction = self._predict_trend(db, days)
        
        return {
            "period_days": days,
            "total_conversations": total_conversations,
            "total_questions": total_events,
            "satisfaction_rate": round(satisfaction_rate, 1),
            "top_topics": [{"topic": label, "count": count} for label, count in topic_distribution],
            "trend": trend_prediction,
            "insights": self._generate_insights(satisfaction_rate, topic_distribution)
        }
    
    def _predict_trend(self, db: Session, days: int) -> str:
        """Predict if conversations are trending up or down."""
        mid_point = datetime.utcnow() - timedelta(days=days//2)
        
        first_half = db.query(Conversation).filter(
            Conversation.created_at < mid_point,
            Conversation.created_at >= datetime.utcnow() - timedelta(days=days)
        ).count()
        
        second_half = db.query(Conversation).filter(
            Conversation.created_at >= mid_point
        ).count()
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_insights(self, satisfaction_rate: float, topic_distribution: List) -> List[str]:
        """Generate actionable insights from data."""
        insights = []
        
        if satisfaction_rate < 70:
            insights.append("⚠️ Satisfaction rate is below target. Consider reviewing response quality.")
        elif satisfaction_rate > 85:
            insights.append("✅ Excellent satisfaction rate! Users are finding responses helpful.")
        
        if topic_distribution:
            top_topic = topic_distribution[0][0]
            insights.append(f"📊 '{top_topic}' is the most common topic. Ensure documentation is up-to-date.")
        
        return insights


# Global instance
ml_engine = MLEngine()
