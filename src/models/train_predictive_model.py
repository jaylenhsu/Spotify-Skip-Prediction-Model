"""
Train TRUE PREDICTIVE skip prediction model (no data leakage).

This model excludes features that are only known AFTER playing the track,
making it suitable for real-time prediction before playback starts.

Leaky features removed:
- ms_played: Only known after listening
- session_id: Assigned during/after the session

This demonstrates understanding of:
1. Data leakage in ML systems
2. Difference between exploratory analysis vs real-time prediction
3. Feature selection for prediction tasks
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class PredictiveSkipModel:
    """Train skip prediction model WITHOUT leaky features."""

    def __init__(self, model_type='random_forest'):
        """Initialize predictive model."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Prepare data for training WITHOUT LEAKY FEATURES.

        Leaky features (excluded):
        - ms_played: Only known after listening
        - session_id: Assigned during/after session
        - session_position: Requires knowing session structure
        - session_length: Requires session to complete

        Args:
            df: DataFrame with engineered features
            test_size: Fraction of data to use for testing

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Base exclusions (metadata, targets, etc.)
        exclude_cols = [
            'is_skip',  # Target
            'skipped',  # Duplicate of target
            'track_name', 'artist_name', 'album_name',  # Metadata
            'track_uri', 'track_id', 'played_at', 'played_at_dt',  # IDs
            'reason_start', 'reason_end',  # Metadata
            'track_last_played', 'artist_last_played',  # Helper columns
            'year_month', 'listen_seconds', 'skip_status',  # If they exist
            'time_period',  # Categorical - we have hour encoding instead
            'platform', 'conn_country', 'ip_addr',  # String columns
            # Null columns (podcasts/audiobooks)
            'episode_name', 'episode_show_name', 'spotify_episode_uri',
            'audiobook_title', 'audiobook_uri', 'audiobook_chapter_uri', 'audiobook_chapter_title',
            'offline_timestamp',
            # LEAKY FEATURES (only known after/during playback)
            'ms_played',  # Main leakage: duration listened
            'session_id',  # Assigned during session
            'session_position',  # Requires knowing session structure
            'session_length',  # Requires session to complete
        ]

        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        print("\n" + "="*70)
        print("TRUE PREDICTIVE MODEL (No Data Leakage)")
        print("="*70)
        print(f"\nUsing {len(self.feature_names)} features (excludes leaky features):")
        print("\nEXCLUDED LEAKY FEATURES:")
        print("  ✗ ms_played - Only known after listening")
        print("  ✗ session_id - Assigned during session")
        print("  ✗ session_position - Requires session structure")
        print("  ✗ session_length - Requires session completion")
        print("\nINCLUDED PREDICTIVE FEATURES:")
        for feat in sorted(self.feature_names):
            print(f"  ✓ {feat}")

        X = df[self.feature_names]
        y = df['is_skip']

        # Handle missing values
        X = X.fillna(0)

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Skip rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train Random Forest model."""
        print("\n" + "="*70)
        print("TRAINING PREDICTIVE MODEL")
        print("="*70)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train Random Forest
        print("\nTraining Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.model.fit(X_train_scaled, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("✓ Training complete!")

        return self

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        print("\n" + "="*70)
        print("EVALUATING PREDICTIVE MODEL")
        print("="*70)

        X_test_scaled = self.scaler.transform(X_test)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"\n{'Metric':<25} {'Score':>10}")
        print("-" * 37)
        print(f"{'ROC AUC Score':<25} {roc_auc:>9.2%}")
        print(f"{'Average Precision':<25} {avg_precision:>9.2%}")

        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_test, y_pred, target_names=['Not Skipped', 'Skipped']))

        return {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def plot_feature_importance(self, save_path=None, top_n=15):
        """Plot feature importance."""
        plt.figure(figsize=(10, 8))

        top_features = self.feature_importance.head(top_n)

        ax = sns.barplot(
            data=top_features,
            y='feature',
            x='importance',
            palette='viridis'
        )

        ax.set_title('Feature Importance - Predictive Model (No Leakage)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
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
        ax.set_title('ROC Curve - Predictive Model', fontsize=14, fontweight='bold')
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
        ax.set_title('Precision-Recall Curve - Predictive Model', fontsize=14, fontweight='bold')
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
        model_path = output_dir / f'predictive_model_{timestamp}.joblib'
        scaler_path = output_dir / f'predictive_scaler_{timestamp}.joblib'
        features_path = output_dir / f'predictive_features_{timestamp}.txt'
        importance_path = output_dir / f'predictive_feature_importance_{timestamp}.csv'

        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # Save feature names
        with open(features_path, 'w') as f:
            for feat in self.feature_names:
                f.write(f"{feat}\n")

        # Save feature importance
        self.feature_importance.to_csv(importance_path, index=False)

        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Features saved to {features_path}")
        print(f"Feature importance saved to {importance_path}")


def main():
    """Main training pipeline for predictive model."""
    # Load engineered features
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    df = pd.read_csv(data_dir / 'features_engineered.csv')

    # Convert datetime
    df['played_at_dt'] = pd.to_datetime(df['played_at_dt'])

    print(f"\nLoaded {len(df):,} listening events")

    # Initialize predictor
    predictor = PredictiveSkipModel(model_type='random_forest')

    # Prepare data (without leaky features)
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, test_size=0.2)

    # Train model
    predictor.train(X_train, y_train)

    # Evaluate
    results = predictor.evaluate(X_test, y_test)

    # Create plots directory
    plots_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    predictor.plot_feature_importance(plots_dir / 'predictive_feature_importance.png', top_n=15)
    predictor.plot_roc_curve(y_test, results['y_pred_proba'], plots_dir / 'predictive_roc_curve.png')
    predictor.plot_precision_recall_curve(y_test, results['y_pred_proba'], plots_dir / 'predictive_pr_curve.png')

    # Save model
    models_dir = Path(__file__).parent.parent.parent / 'models'
    predictor.save_model(models_dir)

    # Print top features
    print("\n" + "="*70)
    print("TOP 10 PREDICTIVE FEATURES (No Leakage)")
    print("="*70)
    for i, row in predictor.feature_importance.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:>7.2%}")

    print("\n" + "="*70)
    print("PREDICTIVE MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nROC AUC: {results['roc_auc']:.2%}")
    print(f"Average Precision: {results['avg_precision']:.2%}")
    print("\nThis model can predict skip behavior BEFORE playing a track!")


if __name__ == '__main__':
    main()
