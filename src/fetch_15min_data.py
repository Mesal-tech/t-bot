"""Fetch EURUSD 15-minute data from MT5."""

import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import os

import time

def fetch_symbol_data(symbol):
    print(f"\nExample processing for: {symbol}")
    print("-" * 40)
    
    # Check symbol
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found!")
        return

    print(f"Description: {symbol_info.description}")

    # Enable if needed
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to enable symbol {symbol}")
            return

    # Fetch 15-minute data
    print(f"Fetching 15-minute bars...")
    
    rates = mt5.copy_rates_from_pos(symbol, 15, 0, 10000)  # 15 = M15 timeframe

    if rates is None or len(rates) == 0:
        print(f"Failed to fetch data for {symbol}: {mt5.last_error()}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Rename columns
    df = df.rename(columns={
        'time': 'timestamp',
        'tick_volume': 'volume'
    })

    # Select columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"[OK] Fetched {len(df):,} bars")
    
    duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
    
    # Save
    os.makedirs("data", exist_ok=True)

    csv_file = f"data/{symbol}_15min_data.csv"
    df.to_csv(csv_file, index=False)
    file_size = os.path.getsize(csv_file) / (1024 * 1024)

    print(f"[OK] Saved to: {csv_file}")
    print(f"Size: {file_size:.2f} MB")

    # Compressed version
    gz_file = f"{csv_file}.gz"
    df.to_csv(gz_file, index=False, compression='gzip')
    gz_size = os.path.getsize(gz_file) / (1024 * 1024)

    print(f"[OK] Compressed: {gz_file}")
    
def main():
    print("="*80)
    print("Fetching 15-Minute Data for Multiple Symbols")
    print("="*80)

    # Initialize MT5
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return

    try:
        print("MT5 Connected")
        print(f"Version: {mt5.version()}")
        
        symbols_to_fetch = [
            'EURUSDm', 
            'GBPUSDm', 
            'USDJPYm',
            'EURGBPm',
            'GBPJPYm', 
            'XAUUSDm', 
            'USDCHFm', 
            'AUDUSDm',
            'USDCADm'
        ]

        for i, symbol in enumerate(symbols_to_fetch, 1):
            print(f"\n[{i}/{len(symbols_to_fetch)}] Processing {symbol}...")
            fetch_symbol_data(symbol)
            
            if i < len(symbols_to_fetch):
                print("Waiting 1 second...")
                time.sleep(1)

        print("\n" + "="*80)
        print("[OK] Batch Data Fetch Complete!")
        print("="*80)

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
