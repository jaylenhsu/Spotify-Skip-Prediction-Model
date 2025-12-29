"""
Compare all trained models: Random Forest vs PyTorch Neural Network.

This creates a comprehensive comparison showing different modeling approaches
and their performance on skip prediction.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def create_model_comparison():
    """Create comprehensive model comparison visualization."""

    # Model comparison data
    models_data = {
        'Model': [
            'Random Forest\n(Predictive)',
            'PyTorch NN\n(Predictive)',
            'Random Forest\n(Exploratory)'
        ],
        'Type': ['Tree-Based', 'Deep Learning', 'Tree-Based'],
        'ROC AUC': [86.06, 83.39, 95.92],
        'Avg Precision': [93.45, 92.50, 98.06],
        'Accuracy': [72.0, 76.0, 91.0],
        'Features': [29, 29, 33],
        'Leaky Features': ['No', 'No', 'Yes'],
        'Training Time': ['~2 min', '~3 min', '~2 min'],
        'Parameters': ['~100K trees', '~14K params', '~100K trees']
    }

    df = pd.DataFrame(models_data)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('Model Architecture Comparison: Traditional ML vs Deep Learning',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. ROC AUC Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    bars = ax1.barh(df['Model'], df['ROC AUC'], color=colors, alpha=0.7)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('ROC AUC Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance: ROC AUC', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, df['ROC AUC']):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=11, fontweight='bold')

    # 2. Average Precision Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(df['Model'], df['Avg Precision'], color=colors, alpha=0.7)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Average Precision (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Model Performance: Average Precision', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, df['Avg Precision']):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=11, fontweight='bold')

    # 3. Model Characteristics
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'Model Characteristics Comparison', ha='center', fontsize=14,
             fontweight='bold', transform=ax3.transAxes)

    # Random Forest Predictive
    ax3.add_patch(plt.Rectangle((0.02, 0.55), 0.3, 0.35,
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2, alpha=0.3))
    ax3.text(0.17, 0.85, '🌳 Random Forest (Predictive)', ha='center',
             fontsize=11, fontweight='bold', color='#155724', transform=ax3.transAxes)
    ax3.text(0.04, 0.78, '✓ ROC AUC: 86.06%', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.04, 0.72, '✓ Fast training (~2 min)', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.04, 0.66, '✓ Feature importance built-in', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.04, 0.60, '✓ No hyperparameter tuning needed', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.04, 0.54, '⚠️ Less flexible architecture', fontsize=9, transform=ax3.transAxes, color='#856404')

    # PyTorch NN
    ax3.add_patch(plt.Rectangle((0.35, 0.55), 0.3, 0.35,
                                facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2, alpha=0.3))
    ax3.text(0.5, 0.85, '🧠 PyTorch Neural Network', ha='center',
             fontsize=11, fontweight='bold', color='#856404', transform=ax3.transAxes)
    ax3.text(0.37, 0.78, '✓ ROC AUC: 83.39%', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.37, 0.72, '✓ Flexible architecture (14K params)', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.37, 0.66, '✓ GPU acceleration support', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.37, 0.60, '⚠️ Requires hyperparameter tuning', fontsize=9, transform=ax3.transAxes, color='#856404')
    ax3.text(0.37, 0.54, '⚠️ Slightly lower AUC than RF', fontsize=9, transform=ax3.transAxes, color='#856404')

    # Random Forest Exploratory (reference)
    ax3.add_patch(plt.Rectangle((0.68, 0.55), 0.3, 0.35,
                                facecolor='#f8d7da', edgecolor='#dc3545', linewidth=2, alpha=0.3))
    ax3.text(0.83, 0.85, '📊 Random Forest (Exploratory)', ha='center',
             fontsize=11, fontweight='bold', color='#721c24', transform=ax3.transAxes)
    ax3.text(0.70, 0.78, '✓ ROC AUC: 95.92% (highest)', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.70, 0.72, '✓ Best for understanding behavior', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.70, 0.66, '⚠️ Uses leaky features (ms_played)', fontsize=9, transform=ax3.transAxes, color='#721c24')
    ax3.text(0.70, 0.60, '⚠️ Post-playback features only', fontsize=9, transform=ax3.transAxes, color='#721c24')
    ax3.text(0.70, 0.54, '⚠️ Not suitable for prediction', fontsize=9, transform=ax3.transAxes, color='#721c24')

    # 4. When to Use Each Model
    ax4 = fig.add_subplot(gs[2, :])

    use_case_text = [
        ["Model", "Best For", "Data Leakage?", "Key Advantage"],
        [
            "Random Forest\n(Predictive)",
            "Pre-playback prediction\nRecommendation systems\nFeature interpretability",
            "No leakage",
            "Best predictive performance\nwithout leakage (86.06%)"
        ],
        [
            "PyTorch NN\n(Predictive)",
            "GPU-accelerated inference\nDeep learning approach\nComplex patterns",
            "No leakage",
            "Flexible architecture\nGPU support\nModern ML stack"
        ],
        [
            "Random Forest\n(Exploratory)",
            "Understanding behavior\nInsights generation\nFeature discovery",
            "Yes (ms_played)",
            "Highest accuracy (95.92%)\nfor analyzing skip patterns"
        ]
    ]

    table = ax4.table(cellText=use_case_text, cellLoc='left', loc='center',
                     colWidths=[0.2, 0.35, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.8)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    colors_table = ['#E2EFDA', '#FFF2CC', '#F8D7DA']
    for i in range(1, len(use_case_text)):
        for j in range(4):
            table[(i, j)].set_facecolor(colors_table[i-1])
            if j == 0:
                table[(i, j)].set_text_props(weight='bold')

    ax4.axis('off')

    # Save figure
    plots_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    save_path = plots_dir / 'all_models_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive model comparison to {save_path}")
    plt.close()

    # Also save metrics CSV
    metrics_df = df[['Model', 'Type', 'ROC AUC', 'Avg Precision', 'Accuracy', 'Features', 'Leaky Features']]
    csv_path = plots_dir / 'all_models_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics CSV to {csv_path}")


def print_summary():
    """Print comprehensive model comparison summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)

    print("\n🌳 RANDOM FOREST (PREDICTIVE - No Data Leakage)")
    print("   Purpose: Skip prediction before playback")
    print("   ROC AUC: 86.06% | Avg Precision: 93.45% | Accuracy: 72%")
    print("   Features: 29 (no leaky features)")
    print("   ✅ No data leakage | Fast training | Built-in feature importance")
    print("   📊 Best for pre-playback prediction")

    print("\n🧠 PYTORCH NEURAL NETWORK (PREDICTIVE - No Data Leakage)")
    print("   Purpose: Deep learning approach to skip prediction")
    print("   ROC AUC: 83.39% | Avg Precision: 92.50% | Accuracy: 76%")
    print("   Architecture: 3 hidden layers (128→64→32) | 14,657 parameters")
    print("   ✅ No data leakage | GPU support | Flexible architecture")
    print("   💡 Demonstrates deep learning skills")

    print("\n📊 RANDOM FOREST (EXPLORATORY - With Data Leakage)")
    print("   Purpose: Understand why tracks were skipped")
    print("   ROC AUC: 95.92% | Avg Precision: 98.06% | Accuracy: 91%")
    print("   Features: 33 (includes ms_played, session_id)")
    print("   ⚠️ Uses post-playback features (not suitable for prediction)")
    print("   📈 Best for exploratory analysis and insights")

    print("\n" + "="*80)
    print("KEY INSIGHTS FOR INTERNSHIP INTERVIEWS")
    print("="*80)

    print("\n1. 📊 Model Selection Trade-offs:")
    print("   - Random Forest (Predictive): Best performance without leakage (86.06%)")
    print("   - PyTorch NN: Demonstrates modern DL skills, slightly lower performance (83.39%)")
    print("   - Exploratory model: Highest accuracy (95.92%) but uses post-playback features")

    print("\n2. 🎯 Understanding Data Leakage:")
    print("   - 10-point AUC drop when removing leaky features demonstrates trade-off")
    print("   - Predictive models use only pre-playback features")
    print("   - Shows understanding of feature availability in real systems")

    print("\n3. 🔧 Technical Skills Demonstrated:")
    print("   - Traditional ML: scikit-learn, Random Forest, feature engineering")
    print("   - Deep Learning: PyTorch, neural network architecture design")
    print("   - Data awareness: Handling imbalanced data, identifying data leakage")
    print("   - Model evaluation: ROC-AUC, PR curves, proper metrics for imbalanced data")

    print("\n" + "="*80)


if __name__ == '__main__':
    create_model_comparison()
    print_summary()
