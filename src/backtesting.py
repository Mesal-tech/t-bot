import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class SMCBacktester:
    """
    Backtest trained ML model on unseen data with realistic trade simulation.
    """
    
    def __init__(self, model_path=None, model=None, feature_columns=None, 
                 min_probability=0.65, tp_pips=30, sl_pips=10):
        """
        Args:
            model_path: Path to saved model file (or provide model directly)
            model: Trained model object (if not loading from file)
            feature_columns: List of feature column names
            min_probability: Minimum confidence threshold for trades
            tp_pips: Take profit in pips
            sl_pips: Stop loss in pips
        """
        self.min_probability = min_probability
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        
        # Load model
        if model_path:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.scaler = model_data.get('scaler', None)  # Load scaler
        elif model and feature_columns:
            self.model = model
            self.feature_columns = feature_columns
            self.scaler = None  # No scaler when model is passed directly
        else:
            raise ValueError("Must provide either model_path or (model + feature_columns)")
    
    def backtest(self, df_labeled, out_of_sample_pct=0.3, n_bars=3000):
        """
        Run walk-forward backtest on out-of-sample data.
        
        Args:
            df_labeled: DataFrame with features and labels
            out_of_sample_pct: Fraction of data to use for testing (ignored if n_bars is set)
            n_bars: Number of bars to use for testing (default: 3000, uses last n bars)
            
        Returns:
            Dictionary with trades, equity_curve, and metrics
        """
        # Use last n_bars for testing (instead of percentage-based split)
        if n_bars is not None:
            # Take the last n_bars
            df_test = df_labeled.iloc[-n_bars:].copy().reset_index(drop=True)
        else:
            # Fallback to percentage-based split
            split_idx = int(len(df_labeled) * (1 - out_of_sample_pct))
            df_test = df_labeled.iloc[split_idx:].copy().reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print("BACKTESTING SETUP")
        print('='*60)
        print(f"Total data:       {len(df_labeled):,} bars")
        if n_bars is not None:
            print(f"Test data:        {len(df_test):,} bars (last {n_bars} bars)")
        else:
            print(f"Test data:        {len(df_test):,} bars ({out_of_sample_pct*100:.0f}%)")
        print(f"Date range:       {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
        print(f"Min probability:  {self.min_probability}")
        print(f"TP/SL:            {self.tp_pips}/{self.sl_pips} pips")
        
        # Extract features
        X = df_test[self.feature_columns].values
        
        # Standardize features using the same scaler from training
        if self.scaler is not None:
            X = self.scaler.transform(X)
            print(f"Features scaled - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
        else:
            print("WARNING: No scaler found! Features are NOT standardized.")
        
        # Predict probabilities
        print("\nGenerating predictions...")
        probas = self.model.predict_proba(X)
        
        # Detect pip size
        avg_price = df_test['close'].mean()
        pip_size = 0.01 if avg_price > 50 else 0.0001
        
        trades = []
        equity_curve = [0]
        current_equity = 0
        
        print("Simulating trades...")
        for i in range(len(df_test) - 100):  # Leave buffer for exit scanning
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(df_test)} ({i/len(df_test)*100:.1f}%)")
            
            # Get probabilities for this bar
            prob_no_trade = probas[i, 0]
            prob_buy = probas[i, 1] if probas.shape[1] > 1 else 0
            prob_sell = probas[i, 2] if probas.shape[1] > 2 else 0
            
            signal = 0
            confidence = 0
            
            # Decision logic
            if prob_buy >= self.min_probability and prob_buy > prob_sell:
                signal = 1
                confidence = prob_buy
            elif prob_sell >= self.min_probability and prob_sell > prob_buy:
                signal = 2
                confidence = prob_sell
            
            if signal == 0:
                continue
            
            # Simulate trade
            entry_price = df_test.iloc[i]['close']
            entry_time = df_test.iloc[i]['timestamp']
            
            if signal == 1:  # Buy
                tp_price = entry_price + (self.tp_pips * pip_size)
                sl_price = entry_price - (self.sl_pips * pip_size)
            else:  # Sell
                tp_price = entry_price - (self.tp_pips * pip_size)
                sl_price = entry_price + (self.sl_pips * pip_size)
            
            # Look forward for exit
            future_data = df_test.iloc[i+1:i+100]
            
            exit_price = None
            exit_time = None
            pnl = 0
            
            for j, row in future_data.iterrows():
                if signal == 1:  # Buy
                    if row['high'] >= tp_price:
                        exit_price = tp_price
                        exit_time = row['timestamp']
                        pnl = self.tp_pips
                        break
                    elif row['low'] <= sl_price:
                        exit_price = sl_price
                        exit_time = row['timestamp']
                        pnl = -self.sl_pips
                        break
                else:  # Sell
                    if row['low'] <= tp_price:
                        exit_price = tp_price
                        exit_time = row['timestamp']
                        pnl = self.tp_pips
                        break
                    elif row['high'] >= sl_price:
                        exit_price = sl_price
                        exit_time = row['timestamp']
                        pnl = -self.sl_pips
                        break
            
            if exit_price is None:
                continue
            
            current_equity += pnl
            equity_curve.append(current_equity)
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'signal': 'BUY' if signal == 1 else 'SELL',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pips': pnl,
                'confidence': confidence
            })
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve)
    
    def _calculate_metrics(self, trades, equity_curve):
        """Calculate comprehensive performance metrics"""
        df_trades = pd.DataFrame(trades)
        
        if len(df_trades) == 0:
            print("\n  No trades executed!")
            return None
        
        wins = df_trades[df_trades['pnl_pips'] > 0]
        losses = df_trades[df_trades['pnl_pips'] < 0]
        
        total_trades = len(df_trades)
        win_rate = len(wins) / total_trades
        
        avg_win = wins['pnl_pips'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pips'].mean() if len(losses) > 0 else 0
        
        total_pnl = df_trades['pnl_pips'].sum()
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Profit factor
        total_win_pips = wins['pnl_pips'].sum() if len(wins) > 0 else 0
        total_loss_pips = abs(losses['pnl_pips'].sum()) if len(losses) > 0 else 1
        profit_factor = total_win_pips / total_loss_pips if total_loss_pips > 0 else 0
        
        # Sharpe Ratio (annualized)
        returns = df_trades['pnl_pips'].values
        sharpe = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
        
        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades:     {total_trades}")
        print(f"Wins:             {len(wins)} ({win_rate*100:.1f}%)")
        print(f"Losses:           {len(losses)} ({(1-win_rate)*100:.1f}%)")
        print(f"Average Win:      {avg_win:.2f} pips")
        print(f"Average Loss:     {avg_loss:.2f} pips")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Total P&L:        {total_pnl:.2f} pips")
        print(f"Max Drawdown:     {max_drawdown:.2f} pips")
        print(f"Sharpe Ratio:     {sharpe:.2f}")
        print(f"Avg Confidence:   {df_trades['confidence'].mean():.3f}")
        print("="*60)
        
        # Buy vs Sell breakdown
        buy_trades = df_trades[df_trades['signal'] == 'BUY']
        sell_trades = df_trades[df_trades['signal'] == 'SELL']
        
        if len(buy_trades) > 0:
            buy_wr = (buy_trades['pnl_pips'] > 0).mean()
            print(f"\nBuy Trades:  {len(buy_trades)} (Win Rate: {buy_wr*100:.1f}%)")
        if len(sell_trades) > 0:
            sell_wr = (sell_trades['pnl_pips'] > 0).mean()
            print(f"Sell Trades: {len(sell_trades)} (Win Rate: {sell_wr*100:.1f}%)")
        
        return {
            'trades': df_trades,
            'equity_curve': equity_curve,
            'metrics': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'avg_confidence': df_trades['confidence'].mean()
            }
        }
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown in pips"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def save_results(self, results, output_dir):
        """Save backtest results to files"""
        if results is None:
            return
        
        # Save trade log
        trades_path = f"{output_dir}/trade_log.csv"
        results['trades'].to_csv(trades_path, index=False)
        print(f"\nTrade log saved: {trades_path}")
        
        # Save equity curve
        equity_path = f"{output_dir}/equity_curve.csv"
        pd.DataFrame({'equity': results['equity_curve']}).to_csv(equity_path, index=False)
        print(f"Equity curve saved: {equity_path}")
        
        # Save metrics
        metrics_path = f"{output_dir}/metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("BACKTEST METRICS\n")
            f.write("="*60 + "\n")
            for key, value in results['metrics'].items():
                f.write(f"{key}: {value}\n")
        print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        data_path = sys.argv[2] if len(sys.argv) > 2 else 'data/labels/EURUSD_labeled.csv'
        
        # Load data
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Backtest
        backtester = SMCBacktester(model_path=model_path, min_probability=0.65)
        results = backtester.backtest(df, out_of_sample_pct=0.3)
        
        # Save results
        if results:
            backtester.save_results(results, 'models/evaluation')
    else:
        print("Usage: python backtesting.py models/saved/model.pkl [data/labels/EURUSD_labeled.csv]")
