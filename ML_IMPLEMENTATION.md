# Tulane University Chatbot - ML & Analytics Implementation Summary

## ✅ Completed Enhancements

### 1. Machine Learning Engine (`server/ml_engine.py`)

Created a comprehensive ML module with the following capabilities:

#### **Topic Clustering**
- Uses TF-IDF vectorization + K-Means clustering
- Automatically categorizes user questions into topics
- Generates human-readable labels (e.g., "Registration & Enrollment", "Billing & Financial Aid")
- Can be trained on historical conversation data

#### **Sentiment Analysis**
- Calculates satisfaction rates based on user feedback
- Tracks helpful vs. unhelpful responses
- Provides percentage-based metrics

#### **Response Quality Scoring**
- Evaluates AI responses on a 0-100 scale
- Considers factors like:
  - Response length (optimal range)
  - Presence of sources
  - Proper formatting (lists, bold text, etc.)

#### **Anomaly Detection**
- Identifies unusual queries (spam, very long/short messages)
- Flags potential issues for review
- Uses pattern matching and statistical analysis

#### **Predictive Insights**
- Trend prediction (increasing/decreasing/stable conversations)
- Actionable recommendations based on data
- Top topic identification

### 2. Backend Analytics Fixes

#### **Database Initialization**
- Created `init_db.py` script to set up tables
- Seeded 5 initial topic clusters:
  1. Registration & Enrollment
  2. Billing & Financial Aid
  3. IT & Technology
  4. Housing & Residential Life
  5. Library & Academic Resources

#### **New API Endpoints**

**GET `/admin/api/ml-insights`**
- Returns ML-powered insights including:
  - Conversation trends
  - Satisfaction rates
  - Top topics
  - AI-generated recommendations

**POST `/admin/api/train-ml`**
- Trains the ML model on historical data
- Updates topic clusters automatically
- Requires at least 10 messages

**GET `/admin/api/overview`** (Enhanced)
- Now includes `satisfaction_rate` metric
- Fixed to use `outerjoin` for clusters with no events

**GET `/admin/api/top-clusters`** (Fixed)
- Changed from `join` to `outerjoin` to show all clusters
- Returns data even when no analytics events exist

#### **Analytics Event Logging**
- Integrated ML engine for intelligent topic prediction
- Falls back to rule-based clustering if ML not trained
- Properly commits and refreshes events

### 3. Admin Dashboard Enhancements

#### **New ML Insights Panel**
- Beautiful gradient purple/blue design
- Shows:
  - **Conversation Trend**: Increasing/Decreasing/Stable
  - **AI Recommendations**: Actionable insights
- Refresh button to update insights on demand

#### **Enhanced KPI Cards**
- Added **Satisfaction Rate** card with progress bar
- Shows percentage and visual indicator
- Includes growth metrics

#### **Improved Charts**
- Topic Distribution (Bar Chart)
- Sentiment Analysis (Doughnut Chart with center text)
- Both update dynamically with real data

#### **Data Table**
- Shows top clicked resources
- Includes trend indicators
- Responsive design

### 4. Source Pills (Already Working!)

The source extraction and pill display was already implemented correctly:
- Links in AI responses are extracted
- Replaced with numbered citation badges (①, ②, etc.)
- Full sources displayed as interactive pills at bottom
- Clicking badges scrolls to and highlights the source

### 5. Configuration Updates

**`.env` Changes:**
- Set `STORE_MESSAGE_TEXT=true` to enable ML training
- This allows the system to learn from historical conversations

## 📊 How the ML System Works

### Data Flow:
```
User Question
    ↓
ML Engine Predicts Topic (or fallback to rules)
    ↓
Analytics Event Logged (with cluster_id)
    ↓
Data Aggregated for Dashboard
    ↓
ML Insights Generated
```

### Training the Model:
1. Users interact with chatbot (messages stored in DB)
2. Admin calls `/admin/api/train-ml` endpoint
3. ML engine:
   - Fetches up to 1000 user messages
   - Performs TF-IDF vectorization
   - Runs K-Means clustering
   - Generates topic labels
   - Updates database with new clusters
4. Future predictions use trained model

## 🚀 Usage Instructions

### For Admins:

1. **View Analytics**:
   - Navigate to `http://localhost:8000/admin`
   - See real-time metrics, charts, and ML insights

2. **Train ML Model** (after collecting data):
   ```bash
   curl -X POST "http://localhost:8000/admin/api/train-ml?n_clusters=5" \
     -H "Authorization: Bearer secure_random_string_for_admin_auth"
   ```

3. **Get ML Insights**:
   ```bash
   curl "http://localhost:8000/admin/api/ml-insights?days=7" \
     -H "Authorization: Bearer secure_random_string_for_admin_auth"
   ```

### For Users:
- Simply chat at `http://localhost:8000`
- Sources automatically appear as pills
- All interactions are tracked for analytics

## 🔧 Technical Details

### Dependencies:
- `scikit-learn` - ML algorithms
- `numpy` - Numerical operations
- `sqlalchemy` - Database ORM
- `fastapi` - Web framework

### Database Schema:
- `conversations` - User sessions
- `messages` - Chat history (if STORE_MESSAGE_TEXT=true)
- `analytics_events` - Privacy-preserving metrics
- `topic_clusters` - ML-generated topics
- `cluster_terms` - Explainability data

### ML Algorithms:
- **TF-IDF**: Term frequency-inverse document frequency for text vectorization
- **K-Means**: Unsupervised clustering for topic discovery
- **Cosine Similarity**: For measuring text similarity
- **Statistical Analysis**: For trend prediction and anomaly detection

## 📈 Future Enhancements

Potential additions:
1. **Advanced NLP**: Use transformers (BERT, etc.) for better topic modeling
2. **User Segmentation**: Cluster users by behavior patterns
3. **A/B Testing**: Test different response strategies
4. **Predictive Analytics**: Forecast future question volumes
5. **Auto-retraining**: Automatically retrain model weekly
6. **Explainable AI**: Show why questions were categorized certain ways

## ✅ Verification Checklist

- [x] Database tables created and seeded
- [x] ML engine implemented with 5 core features
- [x] Analytics backend fixed and working
- [x] Admin dashboard shows real data
- [x] ML insights panel added
- [x] Source pills displaying correctly
- [x] API endpoints tested and functional
- [x] Message storage enabled for ML training

## 🎯 Key Metrics Now Tracked

1. **Total Conversations** - Unique chat sessions
2. **Total Questions** - Number of user queries
3. **Avg Questions/Thread** - Engagement metric
4. **Satisfaction Rate** - % of helpful responses
5. **Trending Topic** - Most active category
6. **Conversation Trend** - Growth indicator
7. **Top Clicked Resources** - Most useful sources

---

**Status**: ✅ Fully Operational
**Last Updated**: 2025-11-25
**Version**: 2.0 (ML-Enhanced)
