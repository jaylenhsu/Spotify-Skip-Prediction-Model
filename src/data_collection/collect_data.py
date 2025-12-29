"""
Script to collect listening history data from Spotify.
Run this periodically to build up a dataset.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from spotify_client import SpotifyDataCollector


def save_raw_data(data: List[Dict], filename: str):
    """Save raw API response data as JSON."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath = data_dir / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw data to {filepath}")


def collect_listening_history(collector: SpotifyDataCollector, limit: int = 50) -> pd.DataFrame:
    """
    Collect recent listening history with audio features.

    Args:
        collector: SpotifyDataCollector instance
        limit: Number of recent tracks to fetch

    Returns:
        DataFrame with listening history and features
    """
    print(f"Fetching {limit} recently played tracks...")
    recently_played = collector.get_recently_played(limit=limit)

    # Save raw data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_raw_data(recently_played, f'recently_played_{timestamp}.json')

    # Extract track IDs
    track_ids = [item['track']['id'] for item in recently_played]

    print("Fetching audio features...")
    audio_features = collector.get_audio_features(track_ids)

    # Build dataset
    data = []
    for i, item in enumerate(tqdm(recently_played, desc="Processing tracks")):
        track = item['track']
        features = audio_features[i] if i < len(audio_features) else {}

        # Detect skip (compare with next track's played_at time)
        next_played_at = recently_played[i-1]['played_at'] if i > 0 else None
        skip_info = collector.detect_skip(
            item['played_at'],
            track['duration_ms'],
            next_played_at
        )

        record = {
            # Track identifiers
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_name': track['artists'][0]['name'],
            'album_name': track['album']['name'],

            # Temporal features
            'played_at': item['played_at'],
            'played_timestamp': datetime.fromisoformat(item['played_at'].replace('Z', '+00:00')).timestamp(),

            # Track metadata
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'explicit': track['explicit'],

            # Audio features
            'danceability': features.get('danceability'),
            'energy': features.get('energy'),
            'key': features.get('key'),
            'loudness': features.get('loudness'),
            'mode': features.get('mode'),
            'speechiness': features.get('speechiness'),
            'acousticness': features.get('acousticness'),
            'instrumentalness': features.get('instrumentalness'),
            'liveness': features.get('liveness'),
            'valence': features.get('valence'),
            'tempo': features.get('tempo'),
            'time_signature': features.get('time_signature'),

            # Skip detection
            'is_skip': skip_info['is_skip'],
            'listen_duration_ms': skip_info['listen_duration_ms'],
            'listen_percentage': skip_info['listen_percentage'],
        }
        data.append(record)

    df = pd.DataFrame(data)
    return df


def main():
    """Main data collection routine."""
    print("Initializing Spotify client...")
    collector = SpotifyDataCollector()

    # Get user profile
    profile = collector.get_user_profile()
    print(f"Authenticated as: {profile['display_name']}")

    # Collect data
    df = collect_listening_history(collector, limit=50)

    # Save processed data
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = data_dir / f'listening_history_{timestamp}.csv'
    df.to_csv(output_file, index=False)

    print(f"\nCollected {len(df)} tracks")
    print(f"Saved to {output_file}")

    # Print summary statistics
    skip_count = df['is_skip'].sum()
    total_with_skip_info = df['is_skip'].notna().sum()
    if total_with_skip_info > 0:
        print(f"\nSkip rate: {skip_count}/{total_with_skip_info} ({skip_count/total_with_skip_info*100:.1f}%)")

    print("\nSample data:")
    print(df[['track_name', 'artist_name', 'is_skip', 'listen_percentage']].head())


if __name__ == '__main__':
    main()
