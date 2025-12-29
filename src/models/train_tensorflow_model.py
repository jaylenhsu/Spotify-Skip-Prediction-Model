"""
Train skip prediction model using TensorFlow/Keras neural network.

This demonstrates deep learning capabilities with TensorFlow and compares
neural network performance against traditional ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class TensorFlowSkipPredictor:
    """TensorFlow/Keras-based skip prediction model."""

    def __init__(self, hidden_dims=[128, 64, 32], dropout=0.3, lr=0.001):
        """Initialize TensorFlow predictor."""
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.history = None

        # Check GPU availability
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, use_leaky=False):
        """
        Prepare data for training.

        Args:
            df: DataFrame with engineered features
            test_size: Fraction of data to use for testing
            use_leaky: Whether to include leaky features

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Base exclusions
        exclude_cols = [
            'is_skip', 'skipped',
            'track_name', 'artist_name', 'album_name',
            'track_uri', 'track_id', 'played_at', 'played_at_dt',
            'reason_start', 'reason_end',
            'track_last_played', 'artist_last_played',
            'year_month', 'listen_seconds', 'skip_status',
            'time_period',
            'platform', 'conn_country', 'ip_addr',
            'episode_name', 'episode_show_name', 'spotify_episode_uri',
            'audiobook_title', 'audiobook_uri', 'audiobook_chapter_uri', 'audiobook_chapter_title',
            'offline_timestamp'
        ]

        # Add leaky features to exclusions if not using them
        if not use_leaky:
            exclude_cols.extend(['ms_played', 'session_id', 'session_position', 'session_length'])
            model_type = "Predictive (No Leakage)"
        else:
            model_type = "Retrospective (With Leakage)"

        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        print("\n" + "="*70)
        print(f"TENSORFLOW/KERAS NEURAL NETWORK - {model_type}")
        print("="*70)
        print(f"\nUsing {len(self.feature_names)} features")

        X = df[self.feature_names].fillna(0)
        y = df['is_skip']

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Skip rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")

        return X_train, X_test, y_train, y_test

    def build_model(self, input_dim):
        """
        Build Keras neural network architecture.

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),

            # Hidden layers
            layers.Dense(self.hidden_dims[0], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-5)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),

            layers.Dense(self.hidden_dims[1], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-5)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),

            layers.Dense(self.hidden_dims[2], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-5)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout),

            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=512, verbose=1):
        """
        Train TensorFlow/Keras neural network.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0, 1, or 2)
        """
        print("\n" + "="*70)
        print("TRAINING TENSORFLOW MODEL")
        print("="*70)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None

        # Build model
        input_dim = X_train_scaled.shape[1]
        self.model = self.build_model(input_dim)

        print(f"\nModel Architecture:")
        self.model.summary()

        # Calculate class weights for imbalanced dataset
        total = len(y_train)
        pos = y_train.sum()
        neg = total - pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}

        print(f"\nClass weights: {{0: {weight_for_0:.2f}, 1: {weight_for_1:.2f}}}")

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=verbose
        )

        print("\n✓ Training complete!")
        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        print("\n" + "="*70)
        print("EVALUATING TENSORFLOW MODEL")
        print("="*70)

        X_test_scaled = self.scaler.transform(X_test)

        # Predictions
        y_pred_proba = self.model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

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

    def plot_training_history(self, save_path=None):
        """Plot training history (loss and metrics)."""
        if self.history is None:
            print("No training history available")
            return

        history_dict = self.history.history

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(history_dict['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history_dict:
            axes[0, 0].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history_dict:
            axes[0, 1].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # AUC
        axes[1, 0].plot(history_dict['auc'], label='Training AUC', linewidth=2)
        if 'val_auc' in history_dict:
            axes[1, 0].plot(history_dict['val_auc'], label='Validation AUC', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Precision & Recall
        axes[1, 1].plot(history_dict['precision'], label='Training Precision', linewidth=2)
        axes[1, 1].plot(history_dict['recall'], label='Training Recall', linewidth=2)
        if 'val_precision' in history_dict:
            axes[1, 1].plot(history_dict['val_precision'], label='Val Precision', linewidth=2, linestyle='--')
        if 'val_recall' in history_dict:
            axes[1, 1].plot(history_dict['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history to {save_path}")

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
        ax.set_title('ROC Curve - TensorFlow Model', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curve to {save_path}")

        plt.close()

    def save_model(self, output_dir: Path):
        """Save trained model and scaler."""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_dir / f'tensorflow_model_{timestamp}.keras'
        scaler_path = output_dir / f'tensorflow_scaler_{timestamp}.joblib'
        features_path = output_dir / f'tensorflow_features_{timestamp}.txt'

        # Save Keras model
        self.model.save(model_path)

        # Save scaler
        joblib.dump(self.scaler, scaler_path)

        # Save feature names
        with open(features_path, 'w') as f:
            for feat in self.feature_names:
                f.write(f"{feat}\n")

        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Features saved to {features_path}")


def main():
    """Main training pipeline for TensorFlow model."""
    # Load engineered features
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    df = pd.read_csv(data_dir / 'features_engineered.csv')
    df['played_at_dt'] = pd.to_datetime(df['played_at_dt'])

    print(f"\nLoaded {len(df):,} listening events")

    # Initialize predictor
    predictor = TensorFlowSkipPredictor(
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        lr=0.001
    )

    # Prepare data (no leaky features for fair comparison)
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, test_size=0.2, use_leaky=False)

    # Train model (reduced epochs and larger batch for stability)
    history = predictor.train(X_train, y_train, epochs=30, batch_size=1024, verbose=1)

    # Evaluate
    results = predictor.evaluate(X_test, y_test)

    # Create plots directory
    plots_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    predictor.plot_training_history(plots_dir / 'tensorflow_training_history.png')
    predictor.plot_roc_curve(y_test, results['y_pred_proba'], plots_dir / 'tensorflow_roc_curve.png')

    # Save model
    models_dir = Path(__file__).parent.parent.parent / 'models'
    predictor.save_model(models_dir)

    print("\n" + "="*70)
    print("TENSORFLOW MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nROC AUC: {results['roc_auc']:.2%}")
    print(f"Average Precision: {results['avg_precision']:.2%}")


if __name__ == '__main__':
    main()
