"""
Feature Quality Analysis Script

Analyzes feature datasets to identify:
- Constant features
- Features with high NaN rates
- Features with extreme outliers
- Feature distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def analyze_feature_quality(df, output_dir='reports/feature_quality'):
    """
    Comprehensive feature quality analysis.
    
    Args:
        df: DataFrame with features
        output_dir: Output directory for reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(" FEATURE QUALITY ANALYSIS")
    print("="*70)
    
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 
                   'label', 'entry_price', 'exit_price', 'bars_held']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Total samples: {len(df)}")
    
    # Analysis results
    results = []
    
    for col in feature_cols:
        data = df[col]
        
        # Basic stats
        nan_count = data.isna().sum()
        nan_pct = (nan_count / len(data)) * 100
        unique_count = data.nunique()
        
        # Check if constant
        is_constant = unique_count <= 1
        
        # Check for inf
        if np.issubdtype(data.dtype, np.number):
            inf_count = np.isinf(data).sum()
            
            # Stats for numeric features
            mean_val = data.mean()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()
            
            # Outlier detection (values beyond 3 std)
            if std_val > 0:
                outlier_count = ((data - mean_val).abs() > 3 * std_val).sum()
                outlier_pct = (outlier_count / len(data)) * 100
            else:
                outlier_count = 0
                outlier_pct = 0
        else:
            inf_count = 0
            mean_val = None
            std_val = None
            min_val = None
            max_val = None
            outlier_count = 0
            outlier_pct = 0
        
        results.append({
            'feature': col,
            'nan_count': nan_count,
            'nan_pct': nan_pct,
            'inf_count': inf_count,
            'unique_values': unique_count,
            'is_constant': is_constant,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'outlier_count': outlier_count,
            'outlier_pct': outlier_pct
        })
    
    results_df = pd.DataFrame(results)
    
    # Identify problematic features
    print("\n" + "="*70)
    print("PROBLEMATIC FEATURES")
    print("="*70)
    
    # Constant features
    constant_features = results_df[results_df['is_constant']]['feature'].tolist()
    print(f"\nConstant features ({len(constant_features)}):")
    for feat in constant_features:
        print(f"  - {feat}")
    
    # High NaN rate
    high_nan = results_df[results_df['nan_pct'] > 10]
    print(f"\nFeatures with >10% NaN ({len(high_nan)}):")
    for _, row in high_nan.iterrows():
        print(f"  - {row['feature']}: {row['nan_pct']:.1f}%")
    
    # High outlier rate
    high_outliers = results_df[results_df['outlier_pct'] > 5]
    print(f"\nFeatures with >5% outliers ({len(high_outliers)}):")
    for _, row in high_outliers.iterrows():
        print(f"  - {row['feature']}: {row['outlier_pct']:.1f}%")
    
    # Features with inf values
    inf_features = results_df[results_df['inf_count'] > 0]
    print(f"\nFeatures with inf values ({len(inf_features)}):")
    for _, row in inf_features.iterrows():
        print(f"  - {row['feature']}: {row['inf_count']} inf values")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    features_to_remove = set()
    
    # Remove constant features
    features_to_remove.update(constant_features)
    print(f"\n1. Remove {len(constant_features)} constant features")
    
    # Remove high NaN features
    high_nan_features = high_nan['feature'].tolist()
    features_to_remove.update(high_nan_features)
    print(f"2. Remove {len(high_nan_features)} features with >10% NaN")
    
    # Remove inf features
    inf_feature_list = inf_features['feature'].tolist()
    features_to_remove.update(inf_feature_list)
    print(f"3. Remove {len(inf_feature_list)} features with inf values")
    
    print(f"\nTotal features to remove: {len(features_to_remove)}")
    print(f"Remaining features: {len(feature_cols) - len(features_to_remove)}")
    
    # Save results
    results_df.to_csv(output_dir / 'feature_quality_report.csv', index=False)
    print(f"\nSaved detailed report to {output_dir / 'feature_quality_report.csv'}")
    
    # Save recommendations
    pd.DataFrame({
        'feature': list(features_to_remove),
        'reason': ['quality_issue'] * len(features_to_remove)
    }).to_csv(output_dir / 'features_to_remove.csv', index=False)
    
    # Plot distributions for top features
    plot_feature_distributions(df, feature_cols[:20], output_dir)
    
    return results_df, list(features_to_remove)


def plot_feature_distributions(df, features, output_dir, n_cols=4):
    """Plot distributions for selected features"""
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feat in enumerate(features):
        if i < len(axes):
            data = df[feat].dropna()
            
            if len(data) > 0:
                axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{feat}\n(mean={data.mean():.3f}, std={data.std():.3f})')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150)
    print(f"Saved distribution plots to {output_dir / 'feature_distributions.png'}")
    plt.close()


def analyze_all_pairs(data_dir='data/labels', output_dir='reports/feature_quality'):
    """
    Analyze feature quality across all labeled pairs.
    
    Args:
        data_dir: Directory containing labeled CSV files
        output_dir: Output directory for reports
    """
    data_dir = Path(data_dir)
    labeled_files = list(data_dir.glob('*_labeled.csv'))
    
    if not labeled_files:
        print(f"No labeled files found in {data_dir}")
        return
    
    print(f"\nFound {len(labeled_files)} labeled files")
    
    all_problematic_features = {}
    
    for file in labeled_files:
        pair_name = file.stem.replace('_labeled', '')
        print(f"\n{'='*70}")
        print(f"Analyzing {pair_name}")
        print('='*70)
        
        df = pd.read_csv(file)
        
        pair_output_dir = Path(output_dir) / pair_name
        results_df, problematic = analyze_feature_quality(df, pair_output_dir)
        
        all_problematic_features[pair_name] = problematic
    
    # Find features problematic across multiple pairs
    print("\n" + "="*70)
    print("CROSS-PAIR ANALYSIS")
    print("="*70)
    
    feature_problem_count = {}
    for pair, features in all_problematic_features.items():
        for feat in features:
            feature_problem_count[feat] = feature_problem_count.get(feat, 0) + 1
    
    # Features problematic in multiple pairs
    multi_pair_problems = {feat: count for feat, count in feature_problem_count.items() 
                          if count > 1}
    
    if multi_pair_problems:
        print(f"\nFeatures problematic in multiple pairs:")
        for feat, count in sorted(multi_pair_problems.items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"  - {feat}: {count}/{len(labeled_files)} pairs")
        
        # Save
        pd.DataFrame({
            'feature': list(multi_pair_problems.keys()),
            'problem_count': list(multi_pair_problems.values())
        }).to_csv(Path(output_dir) / 'cross_pair_problematic_features.csv', index=False)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            # Analyze all pairs
            analyze_all_pairs()
        else:
            # Analyze single file
            file_path = sys.argv[1]
            df = pd.read_csv(file_path)
            analyze_feature_quality(df)
    else:
        print("Usage:")
        print("  Single file: python src/analyze_features.py path/to/labeled_file.csv")
        print("  All pairs:   python src/analyze_features.py --all")
