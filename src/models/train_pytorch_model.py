"""
Train skip prediction model using PyTorch neural network.

This demonstrates deep learning capabilities and compares neural network
performance against traditional ML models (Random Forest).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class SkipPredictionNN(nn.Module):
    """Neural network for skip prediction."""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        """
        Initialize neural network architecture.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(SkipPredictionNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PyTorchSkipPredictor:
    """PyTorch-based skip prediction model."""

    def __init__(self, hidden_dims=[128, 64, 32], dropout=0.3, lr=0.001):
        """Initialize PyTorch predictor."""
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, use_leaky=False):
        """
        Prepare data for training.

        Args:
            df: DataFrame with engineered features
            test_size: Fraction of data to use for testing
            use_leaky: Whether to include leaky features (ms_played, session_id)

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
        print(f"PYTORCH NEURAL NETWORK - {model_type}")
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

    def train(self, X_train, y_train, epochs=50, batch_size=512, verbose=True):
        """
        Train PyTorch neural network.

        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
        """
        print("\n" + "="*70)
        print("TRAINING PYTORCH MODEL")
        print("="*70)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        input_dim = X_train_scaled.shape[1]
        self.model = SkipPredictionNN(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        print(f"\nModel Architecture:")
        print(self.model)
        print(f"\nTotal parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss and optimizer
        # Use weighted loss for imbalanced dataset
        pos_weight = torch.tensor([1.0 / y_train.mean()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # Training loop
        self.model.train()
        history = {'loss': []}

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass (model outputs after sigmoid)
                outputs = self.model(batch_X)

                # For BCELoss (since model has sigmoid)
                loss = nn.BCELoss()(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            scheduler.step(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        print("\n✓ Training complete!")
        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        print("\n" + "="*70)
        print("EVALUATING PYTORCH MODEL")
        print("="*70)

        self.model.eval()
        X_test_scaled = self.scaler.transform(X_test)
        X_tensor = torch.FloatTensor(X_test_scaled).to(self.device)

        with torch.no_grad():
            y_pred_proba = self.model(X_tensor).cpu().numpy().flatten()

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

    def plot_training_history(self, history, save_path=None):
        """Plot training loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('PyTorch Model Training Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
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
        ax.set_title('ROC Curve - PyTorch Model', fontsize=14, fontweight='bold')
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
        model_path = output_dir / f'pytorch_model_{timestamp}.pt'
        scaler_path = output_dir / f'pytorch_scaler_{timestamp}.joblib'
        features_path = output_dir / f'pytorch_features_{timestamp}.txt'

        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'input_dim': len(self.feature_names)
        }, model_path)

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
    """Main training pipeline for PyTorch model."""
    # Load engineered features
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    df = pd.read_csv(data_dir / 'features_engineered.csv')
    df['played_at_dt'] = pd.to_datetime(df['played_at_dt'])

    print(f"\nLoaded {len(df):,} listening events")

    # Initialize predictor
    predictor = PyTorchSkipPredictor(
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        lr=0.001
    )

    # Prepare data (no leaky features for fair comparison)
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, test_size=0.2, use_leaky=False)

    # Train model
    history = predictor.train(X_train, y_train, epochs=50, batch_size=512)

    # Evaluate
    results = predictor.evaluate(X_test, y_test)

    # Create plots directory
    plots_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    predictor.plot_training_history(history, plots_dir / 'pytorch_training_loss.png')
    predictor.plot_roc_curve(y_test, results['y_pred_proba'], plots_dir / 'pytorch_roc_curve.png')

    # Save model
    models_dir = Path(__file__).parent.parent.parent / 'models'
    predictor.save_model(models_dir)

    print("\n" + "="*70)
    print("PYTORCH MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nROC AUC: {results['roc_auc']:.2%}")
    print(f"Average Precision: {results['avg_precision']:.2%}")


if __name__ == '__main__':
    main()
