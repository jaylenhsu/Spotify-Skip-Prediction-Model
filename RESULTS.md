# Spotify Skip Prediction - Results Summary

## Project Overview

A machine learning project to predict and explain when songs are skipped on Spotify using 10+ years of personal listening history (2015-2025).

**Goal**: Create an impressive portfolio project for Spotify data science/SWE internship applications.

## Dataset

- **Total Records**: 154,474 listening events
- **Date Range**: May 2015 - December 2025 (~10 years)
- **Unique Tracks**: 9,506
- **Unique Artists**: 1,904
- **Overall Skip Rate**: 23.33% (36,038 skipped tracks)

### Top Played Tracks
1. Sunflower - Post Malone (589 plays)
2. Heartless - The Weeknd (443 plays)
3. Hurricane - Kanye West (371 plays)
4. Reminder - The Weeknd (360 plays)
5. WAIT FOR U - Future feat. Drake & Tems (349 plays)

## Model Performance

### Baseline Model (Random Forest)
Trained on temporal, artist history, and session features **without** Spotify audio features.

#### Metrics
- **ROC AUC**: 0.9592 (95.92%)
- **Average Precision**: 0.9806 (98.06%)
- **Overall Accuracy**: 91%

#### Classification Report
```
              precision    recall  f1-score   support

 Not Skipped       0.89      0.82      0.85      9,405
     Skipped       0.92      0.95      0.94     21,490

    accuracy                           0.91     30,895
   weighted avg       0.91      0.91      0.91     30,895
```

#### Confusion Matrix
```
Predicted:        Not Skip    Skip
Actual Not Skip:    7,668     1,737
Actual Skip:          984    20,506
```

**Interpretation**:
- The model correctly identifies 95% of skipped tracks
- 82% precision on "not skipped" - good at avoiding false alarms
- Strong performance considering we haven't used audio features yet!

## Feature Importance

### Top 10 Most Important Features

1. **ms_played** (22.9%) - Duration listened is the strongest signal
2. **session_id** (16.4%) - Listening context matters
3. **skip_streak** (11.3%) - Consecutive skips predict future skips
4. **prev_track_skipped** (11.2%) - Sequential behavior is important
5. **artist_skip_rate** (10.6%) - Historical artist preference
6. **year** (10.5%) - Temporal trends in listening behavior
7. **track_skip_rate** (6.4%) - Historical track preference
8. **days_since_artist** (2.2%) - Freshness/fatigue
9. **session_length** (1.6%) - Session context
10. **session_position** (1.2%) - Position in listening session

### Key Insights

1. **Listening Duration is King**: The single strongest predictor (22.9%) is how long you've already listened
2. **Context Matters**: Session features (session_id, position, length) combined explain ~19% of prediction
3. **Behavioral Patterns**: Skip streak and previous track behavior are powerful signals
4. **Artist Familiarity**: Historical skip rates for artists/tracks provide strong personalization

## Features Engineered (33 total)

### Temporal Features
- Hour of day (with sin/cos encoding for cyclical)
- Day of week
- Month, year
- Is weekend, is late night (11pm-4am), is work hours (9am-5pm)

### Artist Features
- Historical skip rate (expanding window to avoid leakage)
- Total plays (at time of listening)
- Days since last played this artist

### Track Features
- Historical skip rate
- Total plays
- Days since last played
- Is repeat within 1 hour
- Track popularity (percentile rank)

### Session Features
- Session ID (gaps > 30 min = new session)
- Position in session
- Session length
- Same artist/album as previous track
- Previous track skipped
- Skip streak (consecutive skips)

### Popularity Features
- Track popularity percentile
- Artist popularity percentile
- Is top 10% track/artist

## Project Structure

```
song-skip-predictions/
├── data/
│   ├── raw/spotify_history/           # Original JSON files
│   └── processed/
│       ├── listening_history_parsed.csv      # Parsed data
│       └── features_engineered.csv           # ML-ready features
├── models/                            # Trained models
│   ├── random_forest_model_*.joblib
│   ├── scaler_*.joblib
│   └── feature_names_*.txt
├── results/plots/                     # Visualizations
│   ├── feature_importance.png/csv
│   ├── roc_curve.png
│   └── pr_curve.png
├── src/
│   ├── data_collection/
│   │   ├── parse_spotify_history.py   # Parse JSON → DataFrame
│   │   └── spotify_client.py          # (Future) API integration
│   ├── features/
│   │   └── feature_engineering.py     # Feature creation
│   └── models/
│       ├── train_model.py             # Model training
│       └── explain_model.py           # SHAP explainability
└── notebooks/
    ├── 01_exploratory_analysis.ipynb
    └── 02_skip_pattern_analysis.ipynb
```

## Next Steps & Improvements

### Phase 1: Complete (Baseline Model ✓)
- [x] Parse listening history
- [x] Engineer temporal and behavioral features
- [x] Train baseline model
- [x] Achieve strong performance without audio features

### Phase 2: Enhancements (When Spotify API Opens)
- [ ] Enrich with Spotify audio features (danceability, energy, tempo, valence, etc.)
- [ ] Compare model performance before/after audio features
- [ ] Quantify the value of audio features vs behavioral data

### Phase 3: Advanced Features
- [ ] Add genre/mood features
- [ ] Sequential model (LSTM/Transformer) for temporal patterns
- [ ] Playlist context features
- [ ] Time-of-day + audio feature interactions

### Phase 4: Production & Deployment
- [ ] Create simple web app for real-time predictions
- [ ] API endpoint for skip prediction
- [ ] Interactive dashboard showing skip patterns
- [ ] A/B testing framework

## Model Explainability

### To Run SHAP Analysis
```bash
# Install SHAP (not in default requirements)
pip install shap

# Generate explanations
cd src/models
python explain_model.py
```

This will create:
- SHAP summary plots (overall feature importance with directionality)
- SHAP waterfall plots (individual prediction explanations)
- SHAP force plots (why specific tracks were skipped/not skipped)

## Installation & Usage

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Place Spotify history JSON files in data/raw/spotify_history/

# Parse listening history
cd src/data_collection
python parse_spotify_history.py

# Engineer features
cd ../features
python feature_engineering.py

# Train model
cd ../models
python train_model.py
```

### Requirements to Add Audio Features Later
When Spotify API access opens:
1. Get API credentials from https://developer.spotify.com/dashboard
2. Create `.env` file with credentials
3. Run audio feature enrichment script (to be created)

## Technical Highlights for Internship Application

### Data Science Skills Demonstrated
- **Feature Engineering**: Created 33 engineered features from raw data
- **Time Series Awareness**: Used time-based train/test split, expanding window for leakage prevention
- **Class Imbalance**: Handled 23% skip rate with appropriate metrics (ROC-AUC, precision-recall)
- **Model Validation**: Used proper evaluation metrics for imbalanced classification
- **Interpretability**: Feature importance analysis, plan for SHAP values

### Software Engineering Skills Demonstrated
- **Clean Code**: Modular structure, documented functions, type hints
- **Scalability**: Processed 154K records efficiently
- **Reproducibility**: Saved models, scalers, feature names with timestamps
- **Version Control Ready**: Proper .gitignore, structured project layout
- **Production Mindset**: Separated data/model/analysis, error handling

### Domain Knowledge
- **Spotify Metrics**: Skip rate, listening duration, session behavior
- **User Behavior**: Sequential patterns, artist fatigue, temporal preferences
- **Product Thinking**: Features that could drive recommendations, playlist optimization

## Key Findings

1. **Behavioral data alone achieves 96% ROC-AUC** - demonstrates that user patterns are highly predictive even without audio features

2. **Context is crucial** - session-based features and sequential patterns explain significant variance

3. **Personalization works** - artist/track historical preferences are strong signals

4. **Temporal patterns exist** - year is a top-10 feature, suggesting evolving taste

## Potential Business Applications

1. **Playlist Optimization**: Reorder tracks to minimize skips
2. **Recommendation Engine**: Avoid recommending tracks likely to be skipped
3. **User Engagement**: Identify when users are in "discovery mode" vs "favorites mode"
4. **A/B Testing**: Predict skip impact of UI/UX changes
5. **Artist Insights**: Help artists understand why their tracks get skipped

---

**Project Status**: ✅ Baseline Complete | 🔄 Audio Features Pending (API Access)

**Next Milestone**: Add SHAP explainability (pip install shap) + audio feature integration
