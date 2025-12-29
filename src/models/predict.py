"""
Use the trained model to make predictions on new listening data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, List


class SkipPredictionModel:
    """Load and use the trained skip prediction model."""

    def __init__(self):
        """Load the most recent trained model."""
        models_dir = Path(__file__).parent.parent.parent / 'models'
        model_files = sorted(models_dir.glob('random_forest_model_*.joblib'))

        if not model_files:
            raise FileNotFoundError("No trained model found. Run train_model.py first.")

        model_path = model_files[-1]
        # Extract timestamp: random_forest_model_20251229_171013.joblib -> 20251229_171013
        timestamp = '_'.join(model_path.stem.split('_')[-2:])

        # Load model components
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(models_dir / f'scaler_{timestamp}.joblib')

        with open(models_dir / f'feature_names_{timestamp}.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f]

        print(f"Loaded model from {model_path}")
        print(f"Features required: {len(self.feature_names)}")

    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Predict skip probability for tracks.

        Args:
            features: DataFrame with engineered features

        Returns:
            Dictionary with predictions and probabilities
        """
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Select and order features
        X = features[self.feature_names].fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return {
            'will_skip': predictions.astype(bool),
            'skip_probability': probabilities,
            'confidence': np.abs(probabilities - 0.5) * 2  # 0 = uncertain, 1 = very confident
        }

    def predict_single(self, features: Dict) -> Dict:
        """
        Predict for a single track.

        Args:
            features: Dictionary of feature values

        Returns:
            Prediction results
        """
        df = pd.DataFrame([features])
        result = self.predict(df)

        return {
            'will_skip': bool(result['will_skip'][0]),
            'skip_probability': float(result['skip_probability'][0]),
            'confidence': float(result['confidence'][0])
        }

    def get_top_factors(self, features: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        Get top factors influencing predictions.

        Args:
            features: DataFrame with features
            top_n: Number of top factors to return

        Returns:
            List of dictionaries with feature name and importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n).to_dict('records')


def example_usage():
    """Demonstrate model usage."""
    # Load model
    predictor = SkipPredictionModel()

    # Example: Predict on test data
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    df = pd.read_csv(data_dir / 'features_engineered.csv')

    # Get last 10 tracks
    recent_tracks = df.tail(10).copy()

    # Make predictions
    predictions = predictor.predict(recent_tracks)

    # Display results
    print("\n" + "="*70)
    print("SKIP PREDICTIONS FOR RECENT TRACKS")
    print("="*70)

    for i, (_, row) in enumerate(recent_tracks.iterrows()):
        will_skip = predictions['will_skip'][i]
        prob = predictions['skip_probability'][i]
        conf = predictions['confidence'][i]

        print(f"\n{i+1}. {row['track_name']} - {row['artist_name']}")
        print(f"   Prediction: {'WILL SKIP' if will_skip else 'WILL NOT SKIP'}")
        print(f"   Skip Probability: {prob:.1%}")
        print(f"   Confidence: {conf:.1%}")
        print(f"   Actual: {'SKIPPED' if row['is_skip'] else 'NOT SKIPPED'}")

    # Show top factors
    print("\n" + "="*70)
    print("TOP 10 FACTORS INFLUENCING SKIP BEHAVIOR")
    print("="*70)

    top_factors = predictor.get_top_factors(recent_tracks, top_n=10)
    for i, factor in enumerate(top_factors, 1):
        print(f"{i}. {factor['feature']}: {factor['importance']:.4f}")


if __name__ == '__main__':
    example_usage()
