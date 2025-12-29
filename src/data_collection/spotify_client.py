"""
Spotify API client for data collection.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv


class SpotifyDataCollector:
    """Handles authentication and data collection from Spotify API."""

    def __init__(self):
        """Initialize Spotify client with OAuth authentication."""
        load_dotenv()

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv('SPOTIFY_CLIENT_ID'),
            client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
            redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI'),
            scope='user-read-recently-played user-read-playback-state user-top-read'
        ))

    def get_recently_played(self, limit: int = 50) -> List[Dict]:
        """
        Fetch recently played tracks.

        Args:
            limit: Number of tracks to fetch (max 50 per request)

        Returns:
            List of recently played track dictionaries
        """
        results = self.sp.current_user_recently_played(limit=limit)
        return results['items']

    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """
        Get audio features for multiple tracks.

        Args:
            track_ids: List of Spotify track IDs

        Returns:
            List of audio feature dictionaries
        """
        # API accepts up to 100 tracks at once
        features = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            features.extend(self.sp.audio_features(batch))
        return features

    def get_track_details(self, track_id: str) -> Dict:
        """
        Get detailed information about a track.

        Args:
            track_id: Spotify track ID

        Returns:
            Track details dictionary
        """
        return self.sp.track(track_id)

    def get_user_profile(self) -> Dict:
        """
        Get current user's profile information.

        Returns:
            User profile dictionary
        """
        return self.sp.current_user()

    def detect_skip(self, played_at: str, duration_ms: int,
                    next_played_at: Optional[str] = None) -> Dict:
        """
        Infer if a track was skipped based on play duration.

        Args:
            played_at: ISO timestamp when track started
            duration_ms: Track duration in milliseconds
            next_played_at: ISO timestamp when next track started (if available)

        Returns:
            Dictionary with skip detection results
        """
        if next_played_at is None:
            # Can't determine skip without next track info
            return {'is_skip': None, 'listen_duration_ms': None, 'listen_percentage': None}

        played_time = datetime.fromisoformat(played_at.replace('Z', '+00:00'))
        next_time = datetime.fromisoformat(next_played_at.replace('Z', '+00:00'))

        listen_duration_ms = (next_time - played_time).total_seconds() * 1000
        listen_percentage = (listen_duration_ms / duration_ms) * 100 if duration_ms > 0 else 0

        # Consider it a skip if less than 80% of the track was played
        # and less than 30 seconds from the end
        is_skip = (listen_percentage < 80) and (duration_ms - listen_duration_ms > 30000)

        return {
            'is_skip': is_skip,
            'listen_duration_ms': listen_duration_ms,
            'listen_percentage': listen_percentage
        }
