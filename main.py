#!/usr/bin/env python3
"""
ML SMC Trading Bot - Main Orchestration Script

Complete end-to-end pipeline for training and backtesting an ML-based
Smart Money Concepts trading bot.

Usage:
    python main.py --mode full                    # Run complete pipeline
    python main.py --mode features                # Generate features only
    python main.py --mode labels                  # Generate labels only
    python main.py --mode train                   # Train model only
    python main.py --mode backtest                # Backtest only
    python main.py --mode full --pairs EURUSD GBPUSD  # Specific pairs
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from features import SMCFeatureGenerator, process_pair_features
from labeling import TripleBarrierLabeler, label_pair_data
from training import SMCModelTrainer, train_universal_model
from backtesting import SMCBacktester

import pandas as pd


class MLSMCPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, pairs=None, tp_pips=30, sl_pips=10, min_prob=0.65, model_type='rf'):
        self.pairs = pairs or ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBPm', 'GBPJPYm']
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.min_prob = min_prob
        self.model_type = model_type
        
        # Paths
        self.data_dir = Path('data')
        self.processed_dir = self.data_dir / 'processed'
        self.labels_dir = self.data_dir / 'labels'
        self.models_dir = Path('models') / 'saved'
        self.eval_dir = Path('models') / 'evaluation'
        
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("ML SMC TRADING BOT - FULL PIPELINE")
        print("="*70)
        print(f"Pairs: {', '.join(self.pairs)}")
        print(f"TP/SL: {self.tp_pips}/{self.sl_pips} pips")
        print(f"Model: {self.model_type.upper()}")
        print("="*70)
        
        # Phase 1: Features
        print("\n" + "🔧 PHASE 1: FEATURE ENGINEERING")
        self.generate_features()
        
        # Phase 2: Labels
        print("\n" + "🏷️  PHASE 2: LABEL GENERATION")
        self.generate_labels()
        
        # Phase 3: Training
        print("\n" + "🤖 PHASE 3: MODEL TRAINING")
        self.train_model()
        
        # Phase 4: Backtesting
        print("\n" + "📈 PHASE 4: BACKTESTING")
        self.backtest_model()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {self.eval_dir}")
        
    def generate_features(self):
        """Generate features for all pairs"""
        for pair in self.pairs:
            input_file = self.data_dir / f'{pair}_15min_data.csv'
            output_file = self.processed_dir / f'{pair}_features.csv'
            
            if not input_file.exists():
                print(f"⚠️  Skipping {pair}: {input_file} not found")
                continue
            
            print(f"\nProcessing {pair}...")
            process_pair_features(pair, str(input_file), str(output_file))
    
    def generate_labels(self):
        """Generate labels for all pairs"""
        for pair in self.pairs:
            input_file = self.processed_dir / f'{pair}_features.csv'
            output_file = self.labels_dir / f'{pair}_labeled.csv'
            
            if not input_file.exists():
                print(f"⚠️  Skipping {pair}: {input_file} not found")
                continue
            
            print(f"\nLabeling {pair}...")
            label_pair_data(
                pair, 
                str(input_file), 
                str(output_file),
                tp_pips=self.tp_pips,
                sl_pips=self.sl_pips
            )
    
    def train_model(self):
        """Train universal model on all pairs"""
        # Collect labeled files
        labeled_files = []
        for pair in self.pairs:
            file = self.labels_dir / f'{pair}_labeled.csv'
            if file.exists():
                labeled_files.append(str(file))
        
        if not labeled_files:
            print("❌ No labeled data found!")
            return
        
        output_path = self.models_dir / 'universal_smc_model.pkl'
        
        print(f"\nTraining on {len(labeled_files)} pairs...")
        train_universal_model(labeled_files, str(output_path), model_type=self.model_type)
    
    def backtest_model(self):
        """Backtest trained model"""
        model_path = self.models_dir / 'universal_smc_model.pkl'
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        # Backtest on each pair
        for pair in self.pairs:
            data_file = self.labels_dir / f'{pair}_labeled.csv'
            
            if not data_file.exists():
                print(f"⚠️  Skipping {pair}: {data_file} not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"BACKTESTING {pair}")
            print('='*60)
            
            # Load data
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Backtest
            backtester = SMCBacktester(
                model_path=str(model_path),
                min_probability=self.min_prob,
                tp_pips=self.tp_pips,
                sl_pips=self.sl_pips
            )
            
            results = backtester.backtest(df, out_of_sample_pct=0.3)
            
            # Save results
            if results:
                pair_eval_dir = self.eval_dir / pair
                pair_eval_dir.mkdir(parents=True, exist_ok=True)
                backtester.save_results(results, str(pair_eval_dir))


def main():
    parser = argparse.ArgumentParser(
        description='ML SMC Trading Bot Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full
  python main.py --mode features --pairs EURUSD GBPUSD
  python main.py --mode train --model-type gbc
  python main.py --mode backtest --min-prob 0.7
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'features', 'labels', 'train', 'backtest'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--pairs',
        nargs='+',
        help='Currency pairs to process (default: all available)'
    )
    
    parser.add_argument(
        '--tp-pips',
        type=int,
        default=30,
        help='Take profit in pips (default: 30)'
    )
    
    parser.add_argument(
        '--sl-pips',
        type=int,
        default=10,
        help='Stop loss in pips (default: 10)'
    )
    
    parser.add_argument(
        '--min-prob',
        type=float,
        default=0.65,
        help='Minimum probability threshold for trades (default: 0.65)'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['rf', 'gbc'],
        default='rf',
        help='Model type: rf (Random Forest) or gbc (Gradient Boosting)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLSMCPipeline(
        pairs=args.pairs,
        tp_pips=args.tp_pips,
        sl_pips=args.sl_pips,
        min_prob=args.min_prob,
        model_type=args.model_type
    )
    
    # Execute requested mode
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    elif args.mode == 'features':
        pipeline.generate_features()
    elif args.mode == 'labels':
        pipeline.generate_labels()
    elif args.mode == 'train':
        pipeline.train_model()
    elif args.mode == 'backtest':
        pipeline.backtest_model()


if __name__ == "__main__":
    main()
