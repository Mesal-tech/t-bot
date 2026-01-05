import os
import glob
import pandas as pd

root_dir = 'models/evaluation'
metrics = []

for pair_dir in glob.glob(os.path.join(root_dir, '*')):
    if not os.path.isdir(pair_dir):
        continue
        
    pair_name = os.path.basename(pair_dir)
    metrics_file = os.path.join(pair_dir, 'metrics.txt')
    
    if os.path.exists(metrics_file):
        data = {'Pair': pair_name}
        with open(metrics_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.strip().split(':', 1)
                    data[key.strip()] = val.strip()
        metrics.append(data)

df = pd.DataFrame(metrics)

# Rename columns map
col_map = {
    'total_trades': 'Total Trades',
    'win_rate': 'Win Rate', 
    'profit_factor': 'Profit Factor',
    'total_pnl': 'Total P&L',
    'max_drawdown': 'Max Drawdown', 
    'sharpe': 'Sharpe Ratio'
}
df = df.rename(columns=col_map)

# Reorder columns
cols = [c for c in ['Pair', 'Total Trades', 'Win Rate', 'Profit Factor', 'Total P&L', 'Max Drawdown', 'Sharpe Ratio'] if c in df.columns]
df = df[cols]
print(df.to_string(index=False))

# Calculate portfolio stats
if 'Total Trades' in df.columns:
    total_trades = pd.to_numeric(df['Total Trades']).sum()
    print(f"\nPORTFOLIO TOTALS:")
    print(f"Total Trades: {total_trades}")
    
    if 'Total P&L' in df.columns:
        # P&L might just be numbers now, based on file content "3620"
        total_pnl = pd.to_numeric(df['Total P&L']).sum()
        print(f"Total P&L: {total_pnl:.2f} pips")
        print(f"Avg Trades/Day: {total_trades / 30:.1f} (approx)")
