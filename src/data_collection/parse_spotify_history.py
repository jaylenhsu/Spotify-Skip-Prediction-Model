"""
Parse Spotify extended streaming history from privacy download.

The extended streaming history JSON files contain:
- ts (timestamp)
- ms_played (milliseconds played)
- master_metadata_track_name
- master_metadata_album_artist_name
- master_metadata_album_album_name
- spotify_track_uri
- reason_start, reason_end (why playback started/ended)
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union
from tqdm import tqdm


class SpotifyHistoryParser:
    """Parse Spotify extended streaming history JSON files."""

    def __init__(self, history_dir: Union[str, Path]):
        """
        Initialize parser with directory containing history files.

        Args:
            history_dir: Directory containing StreamingHistory*.json files
        """
        self.history_dir = Path(history_dir)

    def load_all_history_files(self) -> List[Dict]:
        """
        Load all streaming history JSON files from directory.

        Returns:
            List of all streaming records
        """
        all_records = []

        # Look for extended streaming history files
        # Multiple possible formats:
        # - endsong_0.json, endsong_1.json, etc. (newest format)
        # - StreamingHistory0.json, StreamingHistory1.json (older format)
        # - Streaming_History_Audio_*.json (extended history format)
        history_files = sorted(self.history_dir.glob('endsong_*.json'))
        if not history_files:
            history_files = sorted(self.history_dir.glob('StreamingHistory*.json'))
        if not history_files:
            history_files = sorted(self.history_dir.glob('Streaming_History_Audio_*.json'))

        if not history_files:
            raise FileNotFoundError(
                f"No streaming history files found in {self.history_dir}\n"
                f"Expected files like 'endsong_*.json', 'StreamingHistory*.json', or 'Streaming_History_Audio_*.json'"
            )

        print(f"Found {len(history_files)} history file(s)")

        for file_path in tqdm(history_files, desc="Loading files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_records.extend(data)
                print(f"  {file_path.name}: {len(data)} records")

        print(f"\nTotal records loaded: {len(all_records)}")
        return all_records

    def parse_to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """
        Parse streaming records into a pandas DataFrame.

        Args:
            records: List of streaming history records

        Returns:
            DataFrame with parsed and cleaned data
        """
        df = pd.DataFrame(records)

        # Handle both old and new format field names
        # New format: ts, master_metadata_track_name, etc.
        # Old format: endTime, trackName, artistName, etc.

        column_mapping = {
            # New format -> standard names
            'ts': 'played_at',
            'master_metadata_track_name': 'track_name',
            'master_metadata_album_artist_name': 'artist_name',
            'master_metadata_album_album_name': 'album_name',
            'ms_played': 'ms_played',
            'spotify_track_uri': 'track_uri',
            'reason_start': 'reason_start',
            'reason_end': 'reason_end',
            'shuffle': 'shuffle',
            'skipped': 'skipped',

            # Old format -> standard names
            'endTime': 'played_at',
            'trackName': 'track_name',
            'artistName': 'artist_name',
            'msPlayed': 'ms_played',
        }

        # Rename columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        print(f"\nColumns available: {df.columns.tolist()}")

        return df

    def detect_skips(self, df: pd.DataFrame, skip_threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect skipped tracks based on listen duration.

        Args:
            df: DataFrame with streaming history
            skip_threshold: Minimum fraction of track that must be played (default 0.5 = 50%)

        Returns:
            DataFrame with skip detection added
        """
        # Note: We don't have track duration in the streaming history
        # We'll need to fetch this from Spotify API or use a heuristic

        # If 'skipped' field exists (newer extended history), use it
        if 'skipped' in df.columns:
            df['is_skip'] = df['skipped']
            print("Using 'skipped' field from streaming history")
        else:
            # Heuristic: if played < 30 seconds, likely a skip
            # This is a placeholder - we'll refine when we get track durations
            df['is_skip'] = df['ms_played'] < 30000
            print(f"Using heuristic: tracks played < 30s marked as skipped")

        return df

    def extract_track_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Spotify track ID from URI.

        Args:
            df: DataFrame with track_uri column

        Returns:
            DataFrame with track_id column added
        """
        if 'track_uri' in df.columns:
            # URI format: spotify:track:TRACK_ID
            df['track_id'] = df['track_uri'].str.extract(r'spotify:track:(.+)')[0]
        else:
            df['track_id'] = None

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features from timestamp.

        Args:
            df: DataFrame with played_at column

        Returns:
            DataFrame with temporal features added
        """
        # Convert to datetime
        df['played_at_dt'] = pd.to_datetime(df['played_at'])

        # Extract temporal features
        df['hour_of_day'] = df['played_at_dt'].dt.hour
        df['day_of_week'] = df['played_at_dt'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['played_at_dt'].dt.month
        df['year'] = df['played_at_dt'].dt.year

        # Create time periods
        df['time_period'] = pd.cut(
            df['hour_of_day'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )

        # Weekend flag
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and filter streaming data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)

        # Remove null track names (these are usually podcast episodes or errors)
        df = df.dropna(subset=['track_name'])

        # Remove very short listens (< 1 second, likely errors)
        df = df[df['ms_played'] >= 1000]

        # Sort by timestamp
        df = df.sort_values('played_at_dt')

        # Reset index
        df = df.reset_index(drop=True)

        print(f"\nCleaning: {initial_count} -> {len(df)} records ({initial_count - len(df)} removed)")

        return df


def main():
    """Main parsing routine."""
    # Update this path to where your JSON files are located
    history_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'spotify_history'

    print(f"Looking for streaming history in: {history_dir}")
    print("Make sure your JSON files from Spotify are in this directory!\n")

    parser = SpotifyHistoryParser(history_dir)

    # Load all history files
    records = parser.load_all_history_files()

    # Parse to DataFrame
    df = parser.parse_to_dataframe(records)

    # Extract track IDs
    df = parser.extract_track_id(df)

    # Add temporal features
    df = parser.add_temporal_features(df)

    # Detect skips
    df = parser.detect_skips(df)

    # Clean data
    df = parser.clean_data(df)

    # Save processed data
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'listening_history_parsed.csv'
    df.to_csv(output_file, index=False)

    print(f"\n✓ Saved parsed data to {output_file}")

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total tracks: {len(df):,}")
    print(f"Date range: {df['played_at_dt'].min()} to {df['played_at_dt'].max()}")
    print(f"Unique tracks: {df['track_id'].nunique():,}")
    print(f"Unique artists: {df['artist_name'].nunique():,}")

    if 'is_skip' in df.columns:
        skip_rate = df['is_skip'].sum() / len(df) * 100
        print(f"\nSkip rate: {skip_rate:.2f}% ({df['is_skip'].sum():,} skipped)")

    print(f"\nAverage listen time: {df['ms_played'].mean() / 1000:.1f} seconds")
    print(f"Median listen time: {df['ms_played'].median() / 1000:.1f} seconds")

    print("\nTop 10 most played tracks:")
    top_tracks = df.groupby(['track_name', 'artist_name']).size().sort_values(ascending=False).head(10)
    for (track, artist), count in top_tracks.items():
        print(f"  {track} - {artist}: {count} plays")

    print("\nSample of processed data:")
    print(df[['track_name', 'artist_name', 'ms_played', 'is_skip', 'hour_of_day', 'day_of_week']].head(10))


if __name__ == '__main__':
    main()
