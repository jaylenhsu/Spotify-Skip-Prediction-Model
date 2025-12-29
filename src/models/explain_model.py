"""
Model explainability using SHAP (SHapley Additive exPlanations).

This script generates SHAP values to understand why the model predicts skips.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_model_and_data():
    """Load trained model, scaler, and test data."""
    # Find most recent model
    models_dir = Path(__file__).parent.parent.parent / 'models'
    model_files = sorted(models_dir.glob('random_forest_model_*.joblib'))

    if not model_files:
        raise FileNotFoundError("No trained model found")

    model_path = model_files[-1]
    # Extract timestamp: random_forest_model_20251229_171013.joblib -> 20251229_171013
    timestamp = '_'.join(model_path.stem.split('_')[-2:])

    scaler_path = models_dir / f'scaler_{timestamp}.joblib'
    features_path = models_dir / f'feature_names_{timestamp}.txt'

    print(f"Loading model from {model_path}")

    # Load model components
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f]

    # Load engineered features
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    df = pd.read_csv(data_dir / 'features_engineered.csv')
    df['played_at_dt'] = pd.to_datetime(df['played_at_dt'])

    # Prepare test data (last 20%)
    split_idx = int(len(df) * 0.8)
    X_test = df[feature_names].iloc[split_idx:]
    y_test = df['is_skip'].iloc[split_idx:]

    X_test_scaled = scaler.transform(X_test.fillna(0))

    return model, X_test_scaled, X_test, y_test, feature_names


def generate_shap_explanations(model, X_test_scaled, X_test, feature_names, n_samples=1000):
    """
    Generate SHAP explanations.

    Args:
        model: Trained model
        X_test_scaled: Scaled test features
        X_test: Unscaled test features (for display)
        feature_names: List of feature names
        n_samples: Number of samples to use for SHAP (more = slower but more accurate)

    Returns:
        SHAP explainer and values
    """
    print(f"\nGenerating SHAP explanations for {n_samples} samples...")
    print("This may take a few minutes...")

    # Sample data for faster computation
    if len(X_test_scaled) > n_samples:
        indices = np.random.choice(len(X_test_scaled), n_samples, replace=False)
        X_sample = X_test_scaled[indices]
    else:
        X_sample = X_test_scaled

    # Create TreeExplainer (faster for tree-based models)
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values might be a list [class_0, class_1]
    # We want explanations for the "skip" class (class 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print("✓ SHAP values calculated")

    return explainer, shap_values, X_sample, indices if len(X_test_scaled) > n_samples else None


def plot_shap_summary(shap_values, X_sample, feature_names, save_path):
    """Plot SHAP summary showing feature importance."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to {save_path}")


def plot_shap_bar(shap_values, feature_names, save_path):
    """Plot bar chart of mean absolute SHAP values."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP bar plot to {save_path}")


def explain_predictions(explainer, shap_values, X_sample, X_test, y_test, feature_names, indices, save_dir):
    """Generate explanations for specific predictions."""
    print("\nGenerating individual prediction explanations...")

    # Select interesting examples
    if indices is not None:
        y_sample = y_test.iloc[indices]
    else:
        y_sample = y_test

    # Find examples of correctly predicted skips and non-skips
    model_preds = (shap_values.sum(axis=1) + explainer.expected_value[1] if isinstance(explainer.expected_value, list) else shap_values.sum(axis=1) + explainer.expected_value) > 0

    correct_skip_idx = np.where((y_sample == True) & (model_preds == True))[0]
    correct_no_skip_idx = np.where((y_sample == False) & (model_preds == False))[0]

    if len(correct_skip_idx) > 0:
        # Explain a correctly predicted skip
        idx = correct_skip_idx[0]
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_sample[idx],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_waterfall_skip.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP waterfall plot (skip) to {save_dir / 'shap_waterfall_skip.png'}")

    if len(correct_no_skip_idx) > 0:
        # Explain a correctly predicted non-skip
        idx = correct_no_skip_idx[0]
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_sample[idx],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_waterfall_no_skip.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP waterfall plot (no skip) to {save_dir / 'shap_waterfall_no_skip.png'}")


def main():
    """Main explainability pipeline."""
    # Load model and data
    model, X_test_scaled, X_test, y_test, feature_names = load_model_and_data()

    # Generate SHAP explanations
    explainer, shap_values, X_sample, indices = generate_shap_explanations(
        model, X_test_scaled, X_test, feature_names, n_samples=1000
    )

    # Create plots directory
    plots_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_shap_summary(shap_values, X_sample, feature_names, plots_dir / 'shap_summary.png')
    plot_shap_bar(shap_values, feature_names, plots_dir / 'shap_importance.png')

    # Explain specific predictions
    explain_predictions(explainer, shap_values, X_sample, X_test, y_test, feature_names, indices, plots_dir)

    print("\n" + "="*50)
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print("="*50)
    print(f"\nAll plots saved to {plots_dir}")

    # Save SHAP values for later use
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(plots_dir / 'shap_values.csv', index=False)
    print(f"✓ Saved SHAP values to {plots_dir / 'shap_values.csv'}")


if __name__ == '__main__':
    main()
