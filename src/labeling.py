import numpy as np
import pandas as pd

class TripleBarrierLabeler:
    """
    Labels trades based on triple-barrier method with fixed TP/SL.
    
    Label 1 = Buy wins (TP hit before SL)
    Label 2 = Sell wins (TP hit before SL)
    Label 0 = No clear direction or both hit
    """
    
    def __init__(self, tp_pips=30, sl_pips=10, max_holding_bars=96):
        """
        Args:
            tp_pips: Take profit in pips (default: 30)
            sl_pips: Stop loss in pips (default: 10)
            max_holding_bars: Maximum bars to hold trade (default: 96 = 24 hours on 15m)
        """
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.max_holding_bars = max_holding_bars
    
    def label_data(self, df):
        """
        Generate labels for entire dataset using triple-barrier method.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with added columns: label, entry_price, exit_price, bars_held
        """
        df = df.copy()
        
        # Detect pip size (JPY pairs vs others)
        avg_price = df['close'].mean()
        pip_size = 0.01 if avg_price > 50 else 0.0001
        
        tp_distance = self.tp_pips * pip_size
        sl_distance = self.sl_pips * pip_size
        
        print(f"Pip size: {pip_size} | TP: {tp_distance:.5f} | SL: {sl_distance:.5f}")
        
        labels = []
        entry_prices = []
        exit_prices = []
        bars_held = []
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)
        
        print("Labeling bars...")
        for i in range(n):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{n} ({i/n*100:.1f}%)")
            
            entry_price = closes[i]
            
            # Define barriers
            buy_tp = entry_price + tp_distance
            buy_sl = entry_price - sl_distance
            sell_tp = entry_price - tp_distance
            sell_sl = entry_price + sl_distance
            
            # Look forward
            end_idx = min(i + self.max_holding_bars, n)
            future_highs = highs[i+1:end_idx]
            future_lows = lows[i+1:end_idx]
            
            if len(future_highs) == 0:
                labels.append(0)
                entry_prices.append(entry_price)
                exit_prices.append(entry_price)
                bars_held.append(0)
                continue
            
            # Check BUY scenario
            buy_tp_hit = np.where(future_highs >= buy_tp)[0]
            buy_sl_hit = np.where(future_lows <= buy_sl)[0]
            
            buy_wins = False
            buy_exit_bar = 0
            if len(buy_tp_hit) > 0:
                if len(buy_sl_hit) == 0 or buy_tp_hit[0] < buy_sl_hit[0]:
                    buy_wins = True
                    buy_exit_bar = buy_tp_hit[0]
            
            # Check SELL scenario
            sell_tp_hit = np.where(future_lows <= sell_tp)[0]
            sell_sl_hit = np.where(future_highs >= sell_sl)[0]
            
            sell_wins = False
            sell_exit_bar = 0
            if len(sell_tp_hit) > 0:
                if len(sell_sl_hit) == 0 or sell_tp_hit[0] < sell_sl_hit[0]:
                    sell_wins = True
                    sell_exit_bar = sell_tp_hit[0]
            
            # Assign label
            if buy_wins and not sell_wins:
                label = 1  # Buy
                exit_bar = buy_exit_bar
                exit_price = buy_tp
            elif sell_wins and not buy_wins:
                label = 2  # Sell
                exit_bar = sell_exit_bar
                exit_price = sell_tp
            else:
                label = 0  # No trade or both hit (rare)
                exit_bar = 0
                exit_price = entry_price
            
            labels.append(label)
            entry_prices.append(entry_price)
            exit_prices.append(exit_price)
            bars_held.append(exit_bar)
        
        df['label'] = labels
        df['entry_price'] = entry_prices
        df['exit_price'] = exit_prices
        df['bars_held'] = bars_held
        
        # Statistics
        print(f"\n{'='*60}")
        print("LABEL DISTRIBUTION")
        print('='*60)
        total = len(df)
        no_trade = (df['label'] == 0).sum()
        buy = (df['label'] == 1).sum()
        sell = (df['label'] == 2).sum()
        
        print(f"No Trade (0): {no_trade:,} ({no_trade/total*100:.1f}%)")
        print(f"Buy (1):      {buy:,} ({buy/total*100:.1f}%)")
        print(f"Sell (2):     {sell:,} ({sell/total*100:.1f}%)")
        print(f"Total:        {total:,}")
        print(f"\nTradeable setups: {buy + sell:,} ({(buy + sell)/total*100:.1f}%)")
        
        # Average holding period for winning trades
        winning_trades = df[df['label'] != 0]
        if len(winning_trades) > 0:
            avg_bars = winning_trades['bars_held'].mean()
            print(f"Avg holding period: {avg_bars:.1f} bars ({avg_bars*15:.0f} minutes)")
        
        return df


def label_pair_data(pair_name, input_path, output_path, tp_pips=30, sl_pips=10):
    """
    Convenience function to label a single pair's features.
    
    Args:
        pair_name: Name of the pair (e.g., 'EURUSD')
        input_path: Path to features CSV file
        output_path: Path to save labeled data
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips
    """
    print(f"\n{'='*60}")
    print(f"Labeling {pair_name}")
    print('='*60)
    
    # Load features
    features = pd.read_csv(input_path)
    features['timestamp'] = pd.to_datetime(features['timestamp'])
    
    # Label
    labeler = TripleBarrierLabeler(tp_pips=tp_pips, sl_pips=sl_pips)
    labeled = labeler.label_data(features)
    
    # Save
    labeled.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    
    return labeled


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pair = sys.argv[1]
        input_file = f'data/processed/{pair}_features.csv'
        output_file = f'data/labels/{pair}_labeled.csv'
        label_pair_data(pair, input_file, output_file)
    else:
        print("Usage: python labeling.py EURUSD")
