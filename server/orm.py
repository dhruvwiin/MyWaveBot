from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from .settings import settings

# Setup
Base = declarative_base()

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

# pool_pre_ping=True helps recover from dropped connections (common in cloud environments)
# pool_pre_ping=True helps recover from dropped connections (common in cloud environments)
db_url = str(settings.DATABASE_URL).strip().strip('"').strip("'")

if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

print(f"DEBUG: Connecting to database URL starting with: {db_url[:15]}...")

try:
    engine = create_engine(
        db_url, 
        connect_args=connect_args,
        pool_pre_ping=True
    )
except Exception as e:
    print(f"CRITICAL ERROR: Could not create database engine. URL was: {db_url}")
    raise e
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Data Model

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    user_hash = Column(String, index=True) # HMAC(user_id, HASH_SALT)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Relationships
    analytics_events = relationship("AnalyticsEvent", back_populates="conversation")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    """Stores message content (disabled by default based on settings.STORE_MESSAGE_TEXT)."""
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    role = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    model = Column(String)
    token_usage = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    
    conversation = relationship("Conversation", back_populates="messages")

class TopicCluster(Base):
    __tablename__ = 'topic_clusters'
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, unique=True, nullable=False)
    total_messages = Column(Integer, default=0)
    updated_at = Column(DateTime, onupdate=func.now())

    # Relationships
    analytics_events = relationship("AnalyticsEvent", back_populates="cluster")
    cluster_terms = relationship("ClusterTerm", back_populates="cluster")

class AnalyticsEvent(Base):
    """Stores derived, privacy-preserving analytical signals."""
    __tablename__ = 'analytics_events'
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    cluster_id = Column(Integer, ForeignKey('topic_clusters.id'), nullable=True) # Derived topic
    created_at = Column(DateTime, server_default=func.now())
    helpful = Column(Boolean, default=False)
    clicked_domain = Column(String, index=True) # Domain only (netloc)

    conversation = relationship("Conversation", back_populates="analytics_events")
    cluster = relationship("TopicCluster", back_populates="analytics_events")

class ClusterTerm(Base):
    """Terms extracted from messages for cluster explainability."""
    __tablename__ = 'cluster_terms'
    cluster_id = Column(Integer, ForeignKey('topic_clusters.id'), primary_key=True)
    tag = Column(String, primary_key=True) # e.g., 'course_code', 'regex_match'
    term = Column(String, primary_key=True) # e.g., 'CS101', 'FAFSA'
    count = Column(Integer, default=1)
    
    cluster = relationship("TopicCluster", back_populates="cluster_terms")

# Dependency to get the DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# For initial setup (Alembic is recommended for real migrations, but this is a dev shortcut)
# Base.metadata.create_all(bind=engine)