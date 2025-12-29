"""
Feature engineering for skip prediction model.

This module creates features from listening history data without requiring
Spotify audio features (those will be added later when API access is available).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from datetime import datetime


class SkipFeatureEngineeer:
    """Engineer features for skip prediction from listening history."""

    def __init__(self):
        """Initialize feature engineer."""
        self.artist_skip_rates = {}
        self.track_skip_rates = {}
        self.album_skip_rates = {}

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.

        Features created:
        - Hour of day (already have)
        - Day of week (already have)
        - Is weekend (already have)
        - Time period (already have)
        - Month
        - Is late night (11pm-4am)
        - Is work hours (9am-5pm)
        - Sin/cos encoding of cyclical features

        Args:
            df: DataFrame with played_at_dt column

        Returns:
            DataFrame with temporal features added
        """
        # Already have: hour_of_day, day_of_week, is_weekend, time_period

        # Add month
        if 'month' not in df.columns:
            df['month'] = df['played_at_dt'].dt.month

        # Late night listening (11pm - 4am)
        df['is_late_night'] = df['hour_of_day'].isin(range(23, 24)) | df['hour_of_day'].isin(range(0, 5))

        # Work hours (9am - 5pm on weekdays)
        df['is_work_hours'] = (~df['is_weekend']) & df['hour_of_day'].between(9, 17)

        # Cyclical encoding for hour and day of week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def create_artist_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create artist-based features.

        Features created:
        - Artist historical skip rate (excluding current track)
        - Artist total plays
        - Artist days since last played
        - Artist play count in last 7/30 days

        Args:
            df: DataFrame sorted by played_at_dt

        Returns:
            DataFrame with artist features added
        """
        # Calculate artist skip rate (using expanding window to avoid leakage)
        df['artist_skip_rate'] = df.groupby('artist_name')['is_skip'].transform(
            lambda x: x.expanding().mean().shift(1)
        )

        # Artist total plays (at time of listening)
        df['artist_total_plays'] = df.groupby('artist_name').cumcount()

        # Days since last played this artist
        df['artist_last_played'] = df.groupby('artist_name')['played_at_dt'].shift(1)
        df['days_since_artist'] = (df['played_at_dt'] - df['artist_last_played']).dt.total_seconds() / 86400

        # Fill NaN values
        df['artist_skip_rate'] = df['artist_skip_rate'].fillna(0.5)  # Use overall mean as default
        df['days_since_artist'] = df['days_since_artist'].fillna(999)  # Large value for first plays

        return df

    def create_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create track-based features.

        Features created:
        - Track historical skip rate
        - Track total plays
        - Days since last played
        - Track play count in last 7/30 days
        - Is repeat (played in last hour)

        Args:
            df: DataFrame sorted by played_at_dt

        Returns:
            DataFrame with track features added
        """
        # Track skip rate (expanding window to avoid leakage)
        df['track_skip_rate'] = df.groupby('track_name')['is_skip'].transform(
            lambda x: x.expanding().mean().shift(1)
        )

        # Track total plays
        df['track_total_plays'] = df.groupby('track_name').cumcount()

        # Days since last played
        df['track_last_played'] = df.groupby('track_name')['played_at_dt'].shift(1)
        df['days_since_track'] = (df['played_at_dt'] - df['track_last_played']).dt.total_seconds() / 86400

        # Is this a repeat within an hour?
        df['is_repeat_1h'] = (df['days_since_track'] * 24 < 1).astype(int)

        # Fill NaN values
        df['track_skip_rate'] = df['track_skip_rate'].fillna(0.5)
        df['days_since_track'] = df['days_since_track'].fillna(999)

        return df

    def create_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create listening session features.

        A session is defined as continuous listening with gaps < 30 minutes.

        Features created:
        - Session ID
        - Position in session
        - Session length (tracks)
        - Same artist as previous track
        - Same album as previous track
        - Previous track skipped
        - Skip streak (consecutive skips)

        Args:
            df: DataFrame sorted by played_at_dt

        Returns:
            DataFrame with session features added
        """
        # Define session breaks (gaps > 30 minutes)
        df['time_diff'] = df['played_at_dt'].diff().dt.total_seconds() / 60
        df['session_break'] = (df['time_diff'] > 30) | (df['time_diff'].isna())
        df['session_id'] = df['session_break'].cumsum()

        # Position in session
        df['session_position'] = df.groupby('session_id').cumcount()

        # Session length (total tracks in session)
        df['session_length'] = df.groupby('session_id')['session_id'].transform('count')

        # Same artist/album as previous track
        df['prev_artist'] = df['artist_name'].shift(1)
        df['prev_album'] = df['album_name'].shift(1)
        df['same_artist_as_prev'] = (df['artist_name'] == df['prev_artist']).astype(int)
        df['same_album_as_prev'] = (df['album_name'] == df['prev_album']).astype(int)

        # Previous track skipped
        df['prev_track_skipped'] = df['is_skip'].shift(1).fillna(0).astype(int)

        # Skip streak (consecutive skips before this track)
        df['skip_streak'] = self._calculate_skip_streak(df)

        # Clean up
        df = df.drop(['time_diff', 'session_break', 'prev_artist', 'prev_album'], axis=1)

        return df

    def create_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create popularity-based features.

        Features created:
        - Overall track popularity (percentile rank by play count)
        - Artist popularity (percentile rank)
        - Is in top 10% most played tracks
        - Is in top 10% most played artists

        Args:
            df: DataFrame

        Returns:
            DataFrame with popularity features added
        """
        # Calculate total plays for each track/artist in the dataset
        track_play_counts = df.groupby('track_name').size()
        artist_play_counts = df.groupby('artist_name').size()

        # Map to percentiles
        df['track_popularity_pct'] = df['track_name'].map(
            lambda x: track_play_counts[x] / len(df)
        )
        df['artist_popularity_pct'] = df['artist_name'].map(
            lambda x: artist_play_counts[x] / len(df)
        )

        # Top 10% flags
        track_top10_threshold = track_play_counts.quantile(0.9)
        artist_top10_threshold = artist_play_counts.quantile(0.9)

        df['is_top_track'] = df['track_name'].map(
            lambda x: int(track_play_counts[x] >= track_top10_threshold)
        )
        df['is_top_artist'] = df['artist_name'].map(
            lambda x: int(artist_play_counts[x] >= artist_top10_threshold)
        )

        return df

    def _count_plays_in_window(self, df: pd.DataFrame, groupby_col: str, days: int) -> pd.Series:
        """Count plays in a rolling time window."""
        result = []
        for idx, row in df.iterrows():
            cutoff_date = row['played_at_dt'] - pd.Timedelta(days=days)
            mask = (df['played_at_dt'] < row['played_at_dt']) & \
                   (df['played_at_dt'] >= cutoff_date) & \
                   (df[groupby_col] == row[groupby_col])
            result.append(mask.sum())
        return pd.Series(result, index=df.index)

    def _calculate_skip_streak(self, df: pd.DataFrame) -> pd.Series:
        """Calculate consecutive skip streak before each track."""
        streak = []
        current_streak = 0

        for skip in df['is_skip']:
            streak.append(current_streak)
            if skip:
                current_streak += 1
            else:
                current_streak = 0

        # Shift because we want streak BEFORE current track
        return pd.Series([0] + streak[:-1], index=df.index)

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features.

        Args:
            df: Raw DataFrame with basic fields

        Returns:
            DataFrame with all engineered features
        """
        print("Engineering features...")
        print(f"Starting with {len(df)} records")

        # Ensure sorted by time
        df = df.sort_values('played_at_dt').reset_index(drop=True)

        # Temporal features
        print("  - Temporal features...")
        df = self.create_temporal_features(df)

        # Artist features
        print("  - Artist features...")
        df = self.create_artist_features(df)

        # Track features
        print("  - Track features...")
        df = self.create_track_features(df)

        # Session features
        print("  - Session features...")
        df = self.create_session_features(df)

        # Popularity features
        print("  - Popularity features...")
        df = self.create_popularity_features(df)

        print(f"✓ Feature engineering complete. {len(df.columns)} columns.")

        return df


def main():
    """Main feature engineering pipeline."""
    # Load parsed data
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    input_file = data_dir / 'listening_history_parsed.csv'

    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    df['played_at_dt'] = pd.to_datetime(df['played_at_dt'])

    print(f"Loaded {len(df):,} records")

    # Initialize feature engineer
    engineer = SkipFeatureEngineeer()

    # Engineer features
    df = engineer.engineer_all_features(df)

    # Save engineered features
    output_file = data_dir / 'features_engineered.csv'
    df.to_csv(output_file, index=False)

    print(f"\n✓ Saved engineered features to {output_file}")

    # Print feature summary
    print("\n" + "="*50)
    print("FEATURE SUMMARY")
    print("="*50)

    feature_cols = [col for col in df.columns if col not in [
        'track_name', 'artist_name', 'album_name', 'track_uri', 'track_id',
        'played_at', 'played_at_dt', 'reason_start', 'reason_end',
        'track_last_played', 'artist_last_played'
    ]]

    print(f"Total features: {len(feature_cols)}")
    print(f"Target variable: is_skip")
    print(f"\nFeature list:")
    for col in sorted(feature_cols):
        if col != 'is_skip':
            print(f"  - {col}")

    # Check for missing values
    print("\nMissing values:")
    missing = df[feature_cols].isnull().sum()
    if missing.sum() == 0:
        print("  None!")
    else:
        print(missing[missing > 0])

    # Sample
    print("\nSample of engineered data:")
    sample_cols = ['track_name', 'artist_name', 'is_skip', 'artist_skip_rate',
                   'track_total_plays', 'session_position', 'hour_of_day']
    print(df[sample_cols].head(10))


if __name__ == '__main__':
    main()
