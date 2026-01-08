"""
Batch Training Script
Trains and backtests models for all pairs in config_live.py
"""

import sys
import os
import pandas as pd
import glob
from pathlib import Path

# Add parent directory to path to import config_live
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_live
from features import process_pair_features
from labeling import label_pair_data
from training import SMCModelTrainer, train_universal_model
from backtesting import SMCBacktester

def train_all():
    print("="*70)
    print(" STARTING BATCH TRAINING FOR ALL PAIRS")
    print("="*70)
    print(f"Pairs to process: {len(config_live.PAIRS)}")
    print(f"Pairs: {', '.join(config_live.PAIRS)}")
    
    results = []
    labeled_files = []
    
    # Ensure directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/labels', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('models/evaluation', exist_ok=True)
    
    for pair in config_live.PAIRS:
        print(f"\n\n{'#'*70}")
        print(f" PROCESSING {pair}")
        print(f"{'#'*70}")
        
        # 1. Check Data
        input_file = f'data/{pair}_15min_data.csv'
        if not os.path.exists(input_file):
            # Try compressed
            if os.path.exists(input_file + '.gz'):
                input_file += '.gz'
            else:
                print(f"❌ Data file not found for {pair}: {input_file}")
                continue
        
        try:
            # 2. Generate Features
            feature_file = f'data/processed/{pair}_features.csv'
            print(f"\n--- Generating Features ---")
            process_pair_features(pair, input_file, feature_file)
            
            # 3. Label Data
            label_file = f'data/labels/{pair}_labeled.csv'
            print(f"\n--- Labeling Data ---")
            label_pair_data(pair, feature_file, label_file, 
                           tp_pips=config_live.TP_PIPS, 
                           sl_pips=config_live.SL_PIPS)
            
            labeled_files.append(label_file)
            
            # 4. Train Model
            model_file = f'models/saved/{pair}_model.pkl'
            print(f"\n--- Training Model ---")
            
            df_labeled = pd.read_csv(label_file)
            df_labeled['timestamp'] = pd.to_datetime(df_labeled['timestamp'])
            
            trainer = SMCModelTrainer(model_type='rf', use_smote=True)
            trainer.train(df_labeled, validation_split=0.15)
            trainer.save_model(model_file)
            
            # 5. Backtest
            print(f"\n--- Backtesting ---")
            backtester = SMCBacktester(model_path=model_file, min_probability=0.65,
                                     tp_pips=config_live.TP_PIPS, sl_pips=config_live.SL_PIPS)
            
            # Backtest on last 30%
            bt_result = backtester.backtest(df_labeled, out_of_sample_pct=0.3)
            
            if bt_result:
                metrics = bt_result['metrics']
                metrics['pair'] = pair
                results.append(metrics)
                
                # Save equity curve
                pd.DataFrame({'equity': bt_result['equity_curve']}).to_csv(
                    f'models/evaluation/{pair}_equity.csv', index=False
                )
                
        except Exception as e:
            print(f"❌ Error processing {pair}: {e}")
            import traceback
            traceback.print_exc()
    
    # Train Universal Model
    if labeled_files:
        print(f"\n\n{'#'*70}")
        print(f" TRAINING UNIVERSAL MODEL")
        print(f"{'#'*70}")
        try:
            universal_model_file = 'models/saved/universal_smc_model.pkl'
            train_universal_model(labeled_files, universal_model_file)
            print("✓ Universal model trained successfully")
        except Exception as e:
            print(f"❌ Error training universal model: {e}")
            
    # Summary Report
    print(f"\n\n{'='*70}")
    print(" FINAL BATCH RESULTS")
    print("="*70)
    
    if results:
        df_results = pd.DataFrame(results)
        # Reorder columns
        cols = ['pair', 'total_trades', 'win_rate', 'profit_factor', 'total_pnl', 'sharpe', 'max_drawdown']
        df_results = df_results[[c for c in cols if c in df_results.columns] + 
                               [c for c in df_results.columns if c not in cols]]
        
        print(df_results.to_string(index=False))
        
        # Save summary
        df_results.to_csv('models/evaluation/batch_summary.csv', index=False)
        print(f"\nSummary saved to models/evaluation/batch_summary.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    train_all()
