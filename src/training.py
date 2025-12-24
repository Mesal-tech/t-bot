import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class SMCModelTrainer:
    """
    Train ML models optimized for small datasets with overfitting prevention.
    """
    
    def __init__(self, model_type='rf', use_smote=True):
        """
        Args:
            model_type: 'rf' (Random Forest) or 'gbc' (Gradient Boosting)
            use_smote: Whether to apply SMOTE for class balancing
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.model = None
        self.scaler = StandardScaler()  # Feature standardization
        self.feature_columns = None
        
    def train(self, df, validation_split=0.15):
        """
        Train model with small-data optimizations.
        
        Args:
            df: DataFrame with features and labels
            validation_split: Fraction of data for validation
            
        Returns:
            Trained model
        """
        X, y = self._prepare_training_data(df)
        
        # Time-series split (preserve temporal order)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\n{'='*60}")
        print("TRAINING DATA SPLIT")
        print('='*60)
        print(f"Training:   {len(X_train):,} samples")
        print(f"Validation: {len(X_val):,} samples")
        print(f"Training labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # SMOTE for class balance
        if self.use_smote and len(np.unique(y_train)) > 1:
            print("\nApplying SMOTE for class balancing...")
            # Adaptive k_neighbors for small datasets
            min_class_size = min(np.bincount(y_train)[np.bincount(y_train) > 0])
            k = min(5, min_class_size - 1)
            
            if k >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            else:
                print("Skipping SMOTE (insufficient samples per class)")
        
        # Initialize model with overfitting prevention
        print(f"\nInitializing {self.model_type.upper()} model...")
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,      # Reduced for small data
                max_depth=8,           # Prevent deep trees
                min_samples_split=30,  # Require more samples to split
                min_samples_leaf=15,   # Larger leaf nodes
                max_features='sqrt',   # Feature randomness
                max_samples=0.8,       # Bootstrap 80% of data
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif self.model_type == 'gbc':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=30,
                min_samples_leaf=15,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=0
            )
        
        # Train
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        print("Training complete")
        
        # Evaluate
        print("\n" + "="*60)
        print("TRAINING SET PERFORMANCE")
        print("="*60)
        self._evaluate(X_train, y_train)
        
        print("\n" + "="*60)
        print("VALIDATION SET PERFORMANCE")
        print("="*60)
        self._evaluate(X_val, y_val)
        
        # Check for overfitting
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        print(f"\n{'='*60}")
        print("OVERFITTING CHECK")
        print('='*60)
        print(f"Train accuracy:      {train_acc:.3f}")
        print(f"Validation accuracy: {val_acc:.3f}")
        print(f"Gap:                 {train_acc - val_acc:.3f}")
        
        if train_acc - val_acc > 0.15:
            print("\nWARNING: Possible overfitting detected!")
            print("Consider: reducing max_depth, increasing min_samples_leaf")
        else:
            print("\nOverfitting check passed")
        
        self._print_feature_importance()
        return self.model
    
    def _prepare_training_data(self, df):
        """Extract features and labels"""
        # Columns to exclude from features
        drop_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'label', 
            'entry_price', 'exit_price', 'bars_held', 'hour', 
            'day_of_week', 'pair', 'pip_size',
            'open_normalized', 'high_normalized', 'low_normalized', 'close_normalized'
        ]
        
        # Get feature columns
        self.feature_columns = [c for c in df.columns if c not in drop_cols]
        
        X = df[self.feature_columns].values
        y = df['label'].values
        
        # Remove NaN rows
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Standardize features (mean=0, std=1)
        X = self.scaler.fit_transform(X)
        
        print(f"\n{'='*60}")
        print("TRAINING DATA SUMMARY")
        print('='*60)
        print(f"Shape:    {X.shape}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Scaled:   Mean={X.mean():.4f}, Std={X.std():.4f}")
        print(f"Labels:   {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
    
    def _evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.model.predict(X)
        
        print(classification_report(
            y, y_pred, 
            target_names=['No Trade', 'Buy', 'Sell'],
            zero_division=0
        ))
        
        # Trade-only accuracy (excluding label 0)
        trade_mask = y != 0
        if trade_mask.sum() > 0:
            trade_acc = (y[trade_mask] == y_pred[trade_mask]).mean()
            print(f"Trade-Only Accuracy: {trade_acc*100:.2f}%")
            
            # Per-direction accuracy
            buy_mask = y == 1
            sell_mask = y == 2
            if buy_mask.sum() > 0:
                buy_acc = (y[buy_mask] == y_pred[buy_mask]).mean()
                print(f"Buy Accuracy:        {buy_acc*100:.2f}%")
            if sell_mask.sum() > 0:
                sell_acc = (y[sell_mask] == y_pred[sell_mask]).mean()
                print(f"Sell Accuracy:       {sell_acc*100:.2f}%")
    
    def _print_feature_importance(self):
        """Display top features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n" + "="*60)
            print("TOP 15 FEATURES")
            print("="*60)
            print(importances.head(15).to_string(index=False))
    
    def save_model(self, filepath):
        """Save model and metadata"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'scaler': self.scaler  # Save scaler for live trading
        }, filepath)
        print(f"\nModel saved: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load saved model"""
        data = joblib.load(filepath)
        trainer = SMCModelTrainer(model_type=data['model_type'])
        trainer.model = data['model']
        trainer.feature_columns = data['feature_columns']
        trainer.scaler = data.get('scaler', StandardScaler())  # Load scaler
        return trainer


def train_universal_model(labeled_files, output_path, model_type='rf'):
    """
    Train a universal model on combined data from multiple pairs.
    
    Args:
        labeled_files: List of paths to labeled CSV files
        output_path: Path to save trained model
        model_type: 'rf' or 'gbc'
    """
    print(f"\n{'='*60}")
    print("TRAINING UNIVERSAL MODEL")
    print('='*60)
    
    # Combine all pairs
    dfs = []
    for file in labeled_files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs.append(df)
        print(f"Loaded {file}: {len(df):,} samples")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nCombined dataset: {len(df_combined):,} samples")
    
    # Train
    trainer = SMCModelTrainer(model_type=model_type, use_smote=True)
    trainer.train(df_combined, validation_split=0.15)
    
    # Save
    trainer.save_model(output_path)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Train on single pair
        pair = sys.argv[1]
        input_file = f'data/labels/{pair}_labeled.csv'
        output_file = f'models/saved/{pair}_model.pkl'
        
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        trainer = SMCModelTrainer(model_type='rf')
        trainer.train(df, validation_split=0.2)
        trainer.save_model(output_file)
    else:
        print("Usage: python training.py EURUSD")
