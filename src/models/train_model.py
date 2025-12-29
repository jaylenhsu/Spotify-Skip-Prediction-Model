"""
Train skip prediction model using engineered features.

This script trains a baseline model using temporal, artist history,
and session features (without audio features from Spotify API).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import XGBoost, but don't fail if it's not available
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    print("Warning: XGBoost not available. Using Random Forest instead.")
    print("To use XGBoost, install libomp: brew install libomp")


class SkipPredictor:
    """Train and evaluate skip prediction models."""

    def __init__(self, model_type='xgboost'):
        """
        Initialize predictor.

        Args:
            model_type: 'xgboost' or 'random_forest'
        """
        # Use Random Forest if XGBoost is not available
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            print("XGBoost not available, falling back to Random Forest")
            model_type = 'random_forest'

        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Prepare data for training.

        Args:
            df: DataFrame with engineered features
            test_size: Fraction of data to use for testing

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Define feature columns (exclude metadata and target)
        exclude_cols = [
            'is_skip',  # Target
            'skipped',  # Duplicate of target
            'track_name', 'artist_name', 'album_name',  # Metadata
            'track_uri', 'track_id', 'played_at', 'played_at_dt',  # IDs
            'reason_start', 'reason_end',  # Metadata
            'track_last_played', 'artist_last_played',  # Helper columns
            'year_month', 'listen_seconds', 'skip_status',  # If they exist
            'time_period',  # Categorical - we have hour encoding instead
            # String columns that can't be used directly
            'platform', 'conn_country', 'ip_addr',
            # Null columns (podcasts/audiobooks)
            'episode_name', 'episode_show_name', 'spotify_episode_uri',
            'audiobook_title', 'audiobook_uri', 'audiobook_chapter_uri', 'audiobook_chapter_title',
            # Offline timestamp (mostly null and not useful)
            'offline_timestamp'
        ]

        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        print(f"Using {len(self.feature_names)} features:")
        for feat in sorted(self.feature_names):
            print(f"  - {feat}")

        X = df[self.feature_names]
        y = df['is_skip']

        # Handle any remaining missing values
        X = X.fillna(0)

        # Time-based split (more realistic than random split)
        # Use last 20% of data as test set
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nTrain set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Train skip rate: {y_train.mean():.2%}")
        print(f"Test skip rate: {y_test.mean():.2%}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\nTraining {self.model_type} model...")

        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                tree_method='hist'  # Faster for large datasets
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, y_train)

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        print("✓ Training complete")

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        print("\nEvaluating model...")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        metrics = {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
        }

        # Print results
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Skipped', 'Skipped']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        return metrics, y_pred, y_pred_proba

    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot top N most important features."""
        if self.feature_importance is None:
            print("No feature importance available")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)

        sns.barplot(data=top_features, y='feature', x='importance', ax=ax, palette='viridis')
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")

        plt.close()

    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Skip Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curve to {save_path}")

        plt.close()

    def plot_precision_recall_curve(self, y_test, y_pred_proba, save_path=None):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Skip Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved PR curve to {save_path}")

        plt.close()

    def save_model(self, output_dir: Path):
        """Save trained model and scaler."""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_dir / f'{self.model_type}_model_{timestamp}.joblib'
        scaler_path = output_dir / f'scaler_{timestamp}.joblib'
        features_path = output_dir / f'feature_names_{timestamp}.txt'

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_names))

        print(f"\n✓ Saved model to {model_path}")
        print(f"✓ Saved scaler to {scaler_path}")
        print(f"✓ Saved feature names to {features_path}")

        return model_path


def main():
    """Main training pipeline."""
    # Load engineered features
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    input_file = data_dir / 'features_engineered.csv'

    print(f"Loading features from {input_file}")
    df = pd.read_csv(input_file)
    df['played_at_dt'] = pd.to_datetime(df['played_at_dt'])

    print(f"Loaded {len(df):,} records")

    # Initialize predictor
    predictor = SkipPredictor(model_type='xgboost')

    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, test_size=0.2)

    # Train model
    predictor.train(X_train, y_train)

    # Evaluate
    metrics, y_pred, y_pred_proba = predictor.evaluate(X_test, y_test)

    # Plot results
    plot_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    predictor.plot_feature_importance(top_n=20, save_path=plot_dir / 'feature_importance.png')
    predictor.plot_roc_curve(y_test, y_pred_proba, save_path=plot_dir / 'roc_curve.png')
    predictor.plot_precision_recall_curve(y_test, y_pred_proba, save_path=plot_dir / 'pr_curve.png')

    # Save model
    model_dir = Path(__file__).parent.parent.parent / 'models'
    predictor.save_model(model_dir)

    # Save feature importance
    if predictor.feature_importance is not None:
        importance_file = plot_dir / 'feature_importance.csv'
        predictor.feature_importance.to_csv(importance_file, index=False)
        print(f"✓ Saved feature importance to {importance_file}")

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == '__main__':
    main()
