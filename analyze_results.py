import pandas as pd

pairs = ['EURUSD', 'GBPUSDm', 'USDJPYm', 'EURGBPm', 'GBPJPYm']
total_trades = 0
total_wins = 0
total_pnl = 0

print('Multi-Pair Backtest Summary:')
print('='*70)

for pair in pairs:
    try:
        df = pd.read_csv(f'models/evaluation/{pair}/trade_log.csv')
        wins = (df['pnl_pips'] > 0).sum()
        trades = len(df)
        pnl = df['pnl_pips'].sum()
        wr = wins/trades*100 if trades > 0 else 0
        print(f'{pair:10} | Trades: {trades:3} | Win Rate: {wr:5.1f}% | P&L: {pnl:7.1f} pips')
        total_trades += trades
        total_wins += wins
        total_pnl += pnl
    except Exception as e:
        print(f'{pair:10} | No data - {e}')

print('='*70)
if total_trades > 0:
    print(f'TOTAL      | Trades: {total_trades:3} | Win Rate: {total_wins/total_trades*100:5.1f}% | P&L: {total_pnl:7.1f} pips')
else:
    print('No trades found')
