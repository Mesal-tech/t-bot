"""List all available symbols in MT5."""

import MetaTrader5 as mt5

def main():
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return

    try:
        print("All available symbols:")
        print("=" * 60)

        symbols = mt5.symbols_get()
        print(f"Total symbols: {len(symbols)}\n")

        # Group by currency pairs
        eur_pairs = []
        forex_pairs = []
        other_symbols = []

        for s in symbols:
            name = s.name
            if 'EUR' in name:
                eur_pairs.append(name)
            elif any(curr in name for curr in ['USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
                forex_pairs.append(name)
            else:
                other_symbols.append(name)

        print(f"EUR pairs ({len(eur_pairs)}):")
        for pair in sorted(eur_pairs):
            print(f"  - {pair}")

        print(f"\nOther Forex pairs ({len(forex_pairs)}):")
        for pair in sorted(forex_pairs)[:30]:  # Show first 30
            print(f"  - {pair}")

        if len(forex_pairs) > 30:
            print(f"  ... and {len(forex_pairs) - 30} more")
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
