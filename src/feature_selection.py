"""
Feature Selection and Analysis Pipeline for ML Trading Bot

This module provides comprehensive feature selection tools including:
- Feature importance analysis
- Correlation-based feature removal
- Recursive Feature Elimination (RFE)
- Automated feature selection pipeline
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Comprehensive feature selection toolkit for trading ML models.
    """
    
    def __init__(self, output_dir='reports/feature_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_importance = None
        self.correlation_matrix = None
        self.selected_features = None
        
    def analyze_feature_importance(self, X, y, feature_names, model_type='rf'):
        """
        Analyze feature importance using tree-based models.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            model_type: 'rf' or 'gbc'
            
        Returns:
            DataFrame with feature importance rankings
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Train model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        print(f"Training {model_type.upper()} model for feature importance...")
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Add cumulative importance
        self.feature_importance['cumulative_importance'] = \
            self.feature_importance['importance'].cumsum()
        
        # Statistics
        print(f"\nTotal features: {len(feature_names)}")
        print(f"Features with importance > 0.01: {(importance > 0.01).sum()}")
        print(f"Features with importance > 0.001: {(importance > 0.001).sum()}")
        print(f"Features with importance = 0: {(importance == 0).sum()}")
        
        # Top features
        print("\nTop 20 Most Important Features:")
        print(self.feature_importance.head(20).to_string(index=False))
        
        # Save to CSV
        output_file = self.output_dir / 'feature_importance.csv'
        self.feature_importance.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
        
        # Plot
        self._plot_feature_importance()
        
        return self.feature_importance
    
    def _plot_feature_importance(self, top_n=30):
        """Plot feature importance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top N features
        top_features = self.feature_importance.head(top_n)
        ax1.barh(range(len(top_features)), top_features['importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Importance')
        ax1.set_title(f'Top {top_n} Features by Importance')
        ax1.invert_yaxis()
        
        # Cumulative importance
        ax2.plot(range(len(self.feature_importance)), 
                self.feature_importance['cumulative_importance'])
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=150)
        print(f"Saved plot to {self.output_dir / 'feature_importance.png'}")
        plt.close()
    
    def analyze_correlations(self, X, feature_names, threshold=0.95):
        """
        Identify highly correlated features.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            threshold: Correlation threshold for removal
            
        Returns:
            List of features to remove
        """
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate correlation matrix
        df = pd.DataFrame(X, columns=feature_names)
        self.correlation_matrix = df.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = np.triu(np.ones_like(self.correlation_matrix), k=1)
        upper_triangle = upper_triangle.astype(bool)
        
        high_corr_pairs = []
        features_to_remove = set()
        
        for i in range(len(self.correlation_matrix)):
            for j in range(i+1, len(self.correlation_matrix)):
                if self.correlation_matrix.iloc[i, j] > threshold:
                    feat1 = feature_names[i]
                    feat2 = feature_names[j]
                    corr_value = self.correlation_matrix.iloc[i, j]
                    high_corr_pairs.append((feat1, feat2, corr_value))
                    
                    # Remove the feature with lower importance (if available)
                    if self.feature_importance is not None:
                        imp1 = self.feature_importance[
                            self.feature_importance['feature'] == feat1
                        ]['importance'].values
                        imp2 = self.feature_importance[
                            self.feature_importance['feature'] == feat2
                        ]['importance'].values
                        
                        if len(imp1) > 0 and len(imp2) > 0:
                            if imp1[0] < imp2[0]:
                                features_to_remove.add(feat1)
                            else:
                                features_to_remove.add(feat2)
                    else:
                        # Remove second feature by default
                        features_to_remove.add(feat2)
        
        print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (>{threshold})")
        print(f"Recommending removal of {len(features_to_remove)} features")
        
        if len(high_corr_pairs) > 0:
            print("\nTop 10 Highly Correlated Pairs:")
            corr_df = pd.DataFrame(high_corr_pairs, 
                                  columns=['Feature 1', 'Feature 2', 'Correlation'])
            corr_df = corr_df.sort_values('Correlation', ascending=False)
            print(corr_df.head(10).to_string(index=False))
            
            # Save
            corr_df.to_csv(self.output_dir / 'high_correlations.csv', index=False)
        
        # Plot correlation heatmap (top features only)
        if self.feature_importance is not None:
            self._plot_correlation_heatmap(top_n=50)
        
        return list(features_to_remove)
    
    def _plot_correlation_heatmap(self, top_n=50):
        """Plot correlation heatmap for top features"""
        top_features = self.feature_importance.head(top_n)['feature'].tolist()
        corr_subset = self.correlation_matrix.loc[top_features, top_features]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_subset, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Heatmap - Top {top_n} Features')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=150)
        print(f"Saved heatmap to {self.output_dir / 'correlation_heatmap.png'}")
        plt.close()
    
    def recursive_feature_elimination(self, X, y, feature_names, 
                                     n_features_to_select=50):
        """
        Perform Recursive Feature Elimination.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            n_features_to_select: Target number of features
            
        Returns:
            List of selected features
        """
        print("\n" + "="*60)
        print("RECURSIVE FEATURE ELIMINATION")
        print("="*60)
        
        # Use RandomForest as estimator
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"Selecting top {n_features_to_select} features...")
        selector = RFE(estimator, n_features_to_select=n_features_to_select, 
                      step=10, verbose=1)
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.support_
        selected_features = [feat for feat, selected in 
                           zip(feature_names, selected_mask) if selected]
        
        print(f"\nSelected {len(selected_features)} features")
        print("\nSelected Features:")
        for i, feat in enumerate(selected_features, 1):
            print(f"{i:3d}. {feat}")
        
        # Save
        pd.DataFrame({'feature': selected_features}).to_csv(
            self.output_dir / 'rfe_selected_features.csv', index=False
        )
        
        return selected_features
    
    def select_features(self, X, y, feature_names, 
                       importance_threshold=0.001,
                       correlation_threshold=0.95,
                       use_rfe=False,
                       rfe_n_features=None):
        """
        Automated feature selection pipeline.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            importance_threshold: Minimum importance to keep
            correlation_threshold: Max correlation before removal
            use_rfe: Whether to use RFE
            rfe_n_features: Number of features for RFE
            
        Returns:
            List of selected features
        """
        print("\n" + "="*70)
        print(" AUTOMATED FEATURE SELECTION PIPELINE")
        print("="*70)
        
        # Step 1: Feature Importance
        self.analyze_feature_importance(X, y, feature_names)
        
        # Remove low-importance features
        important_features = self.feature_importance[
            self.feature_importance['importance'] > importance_threshold
        ]['feature'].tolist()
        
        print(f"\nStep 1: Removed {len(feature_names) - len(important_features)} " +
              f"low-importance features (threshold={importance_threshold})")
        
        # Step 2: Correlation Analysis
        important_indices = [i for i, feat in enumerate(feature_names) 
                           if feat in important_features]
        X_important = X[:, important_indices]
        
        correlated_features = self.analyze_correlations(
            X_important, important_features, correlation_threshold
        )
        
        selected_features = [f for f in important_features 
                           if f not in correlated_features]
        
        print(f"\nStep 2: Removed {len(correlated_features)} highly correlated features")
        
        # Step 3: Optional RFE
        if use_rfe and rfe_n_features:
            selected_indices = [i for i, feat in enumerate(important_features) 
                              if feat in selected_features]
            X_selected = X_important[:, selected_indices]
            
            selected_features = self.recursive_feature_elimination(
                X_selected, y, selected_features, rfe_n_features
            )
            
            print(f"\nStep 3: RFE selected {len(selected_features)} features")
        
        # Final summary
        print("\n" + "="*70)
        print("FEATURE SELECTION SUMMARY")
        print("="*70)
        print(f"Original features:     {len(feature_names)}")
        print(f"After importance:      {len(important_features)}")
        print(f"After correlation:     {len(selected_features)}")
        print(f"Reduction:             {len(feature_names) - len(selected_features)} " +
              f"({(1 - len(selected_features)/len(feature_names))*100:.1f}%)")
        
        # Save final selection
        self.selected_features = selected_features
        pd.DataFrame({'feature': selected_features}).to_csv(
            self.output_dir / 'final_selected_features.csv', index=False
        )
        
        print(f"\nSaved selected features to {self.output_dir / 'final_selected_features.csv'}")
        
        return selected_features


def analyze_labeled_data(labeled_file, output_dir='reports/feature_analysis'):
    """
    Analyze features from a labeled dataset.
    
    Args:
        labeled_file: Path to labeled CSV file
        output_dir: Output directory for reports
    """
    print(f"\nAnalyzing {labeled_file}...")
    
    # Load data
    df = pd.read_csv(labeled_file)
    
    # Separate features and labels
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 
                   'label', 'entry_price', 'exit_price', 'bars_held']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Filter to tradeable labels only (1 and 2)
    tradeable_mask = (y == 1) | (y == 2)
    X = X[tradeable_mask]
    y = y[tradeable_mask]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)} (after filtering to tradeable labels)")
    
    # Run feature selection
    selector = FeatureSelector(output_dir=output_dir)
    selected_features = selector.select_features(
        X, y, feature_cols,
        importance_threshold=0.001,
        correlation_threshold=0.95,
        use_rfe=False
    )
    
    return selected_features


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        labeled_file = sys.argv[1]
        analyze_labeled_data(labeled_file)
    else:
        print("Usage: python feature_selection.py path/to/labeled_file.csv")
        print("\nExample:")
        print("python src/feature_selection.py data/labels/EURUSD_labeled.csv")
