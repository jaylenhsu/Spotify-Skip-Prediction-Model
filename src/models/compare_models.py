"""
Compare Exploratory vs Predictive Models.

This script creates a visual comparison showing the difference between:
1. Exploratory Model: Uses all features including ms_played (95.92% AUC)
2. Predictive Model: Excludes leaky features for pre-playback prediction (86.06% AUC)

This demonstrates understanding of:
- Data leakage in ML systems
- Trade-offs between exploratory analysis and prediction
- Feature selection for different use cases
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def create_comparison_visualization():
    """Create side-by-side comparison of both models."""

    # Model comparison data
    comparison_data = {
        'Model Type': ['Exploratory\nAnalysis', 'Predictive\n(Pre-Playback)'],
        'ROC AUC': [95.92, 86.06],
        'Average Precision': [98.06, 93.45],
        'Accuracy': [91.0, 72.0],
        'Features Used': [33, 29],
        'Use Case': [
            'Why did I skip?\n(Post-hoc analysis)',
            'Will I skip this track?\n(Before playback)'
        ]
    }

    # Leaky features info
    leaky_features = [
        'ms_played (22.9%)',
        'session_id (16.4%)',
        'session_position (1.2%)',
        'session_length (1.6%)'
    ]

    # Top predictive features
    predictive_features = [
        'artist_skip_rate (28.8%)',
        'year (21.1%)',
        'track_skip_rate (14.6%)',
        'prev_track_skipped (10.0%)',
        'skip_streak (8.7%)'
    ]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('Exploratory Analysis vs Predictive Model Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. ROC AUC Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = comparison_data['Model Type']
    roc_scores = comparison_data['ROC AUC']
    colors = ['#ff7f0e', '#2ca02c']

    bars = ax1.barh(models, roc_scores, color=colors, alpha=0.7)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('ROC AUC Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance: ROC AUC', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, roc_scores)):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=11, fontweight='bold')

    # Add annotation
    ax1.text(0.5, -0.25, 'Exploratory model has higher accuracy\nbut uses features only available after playback',
            transform=ax1.transAxes, ha='center', fontsize=9, style='italic', color='gray')

    # 2. Average Precision Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ap_scores = comparison_data['Average Precision']

    bars = ax2.barh(models, ap_scores, color=colors, alpha=0.7)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Average Precision (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Model Performance: Average Precision', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, ap_scores)):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=11, fontweight='bold')

    ax2.text(0.5, -0.25, 'Predictive model still achieves excellent precision\nfor real-time skip prediction',
            transform=ax2.transAxes, ha='center', fontsize=9, style='italic', color='gray')

    # 3. Features Comparison - Exploratory
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'Exploratory Model Features', ha='center', fontsize=12,
             fontweight='bold', transform=ax3.transAxes)

    # Leaky features box
    ax3.add_patch(plt.Rectangle((0.05, 0.55), 0.9, 0.35,
                                facecolor='#ffcccc', edgecolor='red', linewidth=2, alpha=0.3))
    ax3.text(0.5, 0.85, '⚠️  LEAKY FEATURES (Post-Playback Only)', ha='center',
             fontsize=10, fontweight='bold', color='darkred', transform=ax3.transAxes)

    y_pos = 0.78
    for feature in leaky_features:
        ax3.text(0.1, y_pos, f'✗ {feature}', fontsize=9, transform=ax3.transAxes)
        y_pos -= 0.08

    # Predictive features box
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.45,
                                facecolor='#ccffcc', edgecolor='green', linewidth=2, alpha=0.3))
    ax3.text(0.5, 0.48, '✓ PREDICTIVE FEATURES', ha='center',
             fontsize=10, fontweight='bold', color='darkgreen', transform=ax3.transAxes)

    ax3.text(0.1, 0.40, '✓ Plus all features from Predictive Model →',
             fontsize=9, transform=ax3.transAxes, style='italic')

    ax3.text(0.5, -0.05, f'Total: {comparison_data["Features Used"][0]} features',
             ha='center', fontsize=10, fontweight='bold', transform=ax3.transAxes)

    # 4. Features Comparison - Predictive
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.text(0.5, 0.95, 'Predictive Model Features', ha='center', fontsize=12,
             fontweight='bold', transform=ax4.transAxes)

    # Top features box
    ax4.add_patch(plt.Rectangle((0.05, 0.15), 0.9, 0.75,
                                facecolor='#ccffcc', edgecolor='green', linewidth=2, alpha=0.3))
    ax4.text(0.5, 0.85, 'TOP 5 FEATURES (Known Before Playback)', ha='center',
             fontsize=10, fontweight='bold', color='darkgreen', transform=ax4.transAxes)

    y_pos = 0.75
    for i, feature in enumerate(predictive_features, 1):
        ax4.text(0.1, y_pos, f'{i}. {feature}', fontsize=9, transform=ax4.transAxes)
        y_pos -= 0.11

    ax4.text(0.1, 0.20, '+ Historical skip patterns\n+ Temporal features (time, day, etc.)\n+ Track/artist popularity',
             fontsize=8, transform=ax4.transAxes, style='italic')

    ax4.text(0.5, -0.05, f'Total: {comparison_data["Features Used"][1]} features (no leakage)',
             ha='center', fontsize=10, fontweight='bold', transform=ax4.transAxes)

    # 5. Use Case Comparison
    ax5 = fig.add_subplot(gs[2, :])

    use_case_text = [
        ["Use Case", "Exploratory Analysis", "Predictive Model"],
        ["Question", "Why did I skip this track?", "Will I skip this track?"],
        ["Timing", "After playback completed", "Before playback starts"],
        ["Data Available", "All listening data (ms_played, etc.)", "Only pre-playback features"],
        ["Best For", "Understanding behavior\nFeature analysis\nInsights generation",
         "Pre-playback prediction\nRecommendation systems\nPlaylist optimization"],
        ["ROC AUC", "95.92%", "86.06%"],
        ["Data Leakage?", "Yes (uses ms_played)", "No leakage"]
    ]

    # Create table
    table = ax5.table(cellText=use_case_text, cellLoc='left', loc='center',
                     colWidths=[0.2, 0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(use_case_text)):
        table[(i, 0)].set_facecolor('#E7E6E6')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#FFF2CC')
        table[(i, 2)].set_facecolor('#E2EFDA')

    ax5.axis('off')

    # Save figure
    plots_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    save_path = plots_dir / 'model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved model comparison to {save_path}")
    plt.close()

    # Also create a simple metrics comparison CSV
    metrics_df = pd.DataFrame(comparison_data)
    csv_path = plots_dir / 'model_comparison_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics CSV to {csv_path}")


def print_summary():
    """Print summary comparison."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print("\n📊 EXPLORATORY ANALYSIS MODEL")
    print("   Purpose: Understand why tracks were skipped")
    print("   ROC AUC: 95.92% | Avg Precision: 98.06%")
    print("   Features: 33 (includes ms_played, session_id)")
    print("   ⚠️  Contains data leakage - uses post-playback features")

    print("\n🎯 PREDICTIVE MODEL")
    print("   Purpose: Predict skips BEFORE playing a track")
    print("   ROC AUC: 86.06% | Avg Precision: 93.45%")
    print("   Features: 29 (no leaky features)")
    print("   ✅ No data leakage - uses only pre-playback features")

    print("\n📉 Performance Trade-off")
    print("   ROC AUC decrease: 9.86 percentage points")
    print("   Avg Precision decrease: 4.61 percentage points")
    print("   Trade-off justification: Predictive model can run in real-time!")

    print("\n💡 Key Insight for Interviews:")
    print("   'I built two models to demonstrate understanding of data leakage:")
    print("   - Exploratory model (95.92% AUC) for analyzing skip behavior")
    print("   - Predictive model (86.06% AUC) for pre-playback prediction")
    print("   The 10-point AUC drop demonstrates the trade-off between using")
    print("   all available data vs. avoiding leakage. The exploratory model")
    print("   uses post-playback features like listening duration.'")

    print("\n" + "="*80)


if __name__ == '__main__':
    create_comparison_visualization()
    print_summary()
