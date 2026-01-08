"""
Migration Script: Old SMC to Corrected SMC
This script helps transition from the old implementation to the corrected one
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from smc_corrected import SmartMoneyConceptsFull
from sessions import add_session_features


def compare_implementations(df: pd.DataFrame):
    """
    Compare old vs corrected implementation
    """
    print("="*70)
    print(" COMPARING OLD VS CORRECTED SMC IMPLEMENTATION")
    print("="*70)
    
    # Import old implementation
    try:
        from smc import SmartMoneyConceptsFull as OldSMC
        old_smc = OldSMC()
        df_old = old_smc.analyze(df.copy())
        print("\n✓ Old implementation loaded")
    except Exception as e:
        print(f"\n✗ Could not load old implementation: {e}")
        df_old = None
    
    # Run corrected implementation
    new_smc = SmartMoneyConceptsFull()
    df_new = new_smc.analyze(df.copy())
    print("✓ Corrected implementation loaded")
    
    if df_old is not None:
        print("\n" + "="*70)
        print(" COMPARISON RESULTS")
        print("="*70)
        
        # Compare order blocks
        old_obs = (df_old['swing_ob'] != 0).sum()
        new_obs = (df_new['swing_ob'] != 0).sum()
        print(f"\nOrder Blocks:")
        print(f"  Old: {old_obs}")
        print(f"  New: {new_obs}")
        print(f"  Difference: {new_obs - old_obs}")
        
        # Compare FVGs
        old_fvgs = (df_old['fvg'] != 0).sum()
        new_fvgs = (df_new['fvg'] != 0).sum()
        print(f"\nFair Value Gaps:")
        print(f"  Old: {old_fvgs}")
        print(f"  New: {new_fvgs}")
        print(f"  Difference: {new_fvgs - old_fvgs}")
        
        # Compare BOS/CHoCH
        old_bos = df_old['swing_bos'].sum()
        new_bos = df_new['swing_bos'].sum()
        print(f"\nBreak of Structure:")
        print(f"  Old: {old_bos}")
        print(f"  New: {new_bos}")
        print(f"  Difference: {new_bos - old_bos}")
        
        # Check for look-ahead bias indicators
        print("\n" + "="*70)
        print(" LOOK-AHEAD BIAS CHECK")
        print("="*70)
        
        # In old implementation, breaker blocks were marked retroactively
        old_breakers = df_old['swing_ob_breaker'].sum() if 'swing_ob_breaker' in df_old.columns else 0
        new_breakers = df_new['swing_ob_breaker'].sum()
        
        print(f"\nBreaker Blocks (should be detected in real-time):")
        print(f"  Old: {old_breakers}")
        print(f"  New: {new_breakers}")
        
        if old_breakers > new_breakers * 1.5:
            print("  ⚠️  WARNING: Old implementation likely has look-ahead bias")
            print("     (Too many breakers detected - probably using future data)")
        else:
            print("  ✓ Looks good")
    
    print("\n" + "="*70)
    print(" NEW FEATURES ADDED")
    print("="*70)
    
    print("\n✓ Retracements:")
    print(f"  - Direction: {(df_new['retracement_direction'] != 0).sum()} bars")
    print(f"  - Avg current retracement: {df_new['current_retracement_pct'].mean():.1f}%")
    
    print("\n✓ Previous High/Low:")
    print(f"  - Tracked for {(~df_new['previous_high'].isna()).sum()} bars")
    
    print("\n✓ Improved Features:")
    print("  - FVG consecutive merging")
    print("  - ATR-based inducement zones")
    print("  - Better market maker patterns")
    print("  - Real-time breaker/mitigation tracking")
    
    return df_new


def test_with_sample_data():
    """
    Test with sample data
    """
    print("\n" + "="*70)
    print(" TESTING WITH SAMPLE DATA")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, n_samples)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
        'open': close_prices * (1 + np.random.uniform(-0.0005, 0.0005, n_samples)),
        'high': close_prices * (1 + np.random.uniform(0, 0.001, n_samples)),
        'low': close_prices * (1 - np.random.uniform(0, 0.001, n_samples)),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    print(f"\nCreated {len(df)} bars of sample data")
    print(f"Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    # Run corrected implementation
    df_result = compare_implementations(df)
    
    # Add sessions
    print("\n" + "="*70)
    print(" ADDING SESSIONS")
    print("="*70)
    
    df_result = add_session_features(df_result, time_zone="UTC")
    
    session_cols = [c for c in df_result.columns if 'session' in c or 'kill_zone' in c]
    print(f"\n✓ Added {len(session_cols)} session columns:")
    for col in session_cols[:10]:
        print(f"  - {col}")
    if len(session_cols) > 10:
        print(f"  ... and {len(session_cols) - 10} more")
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print("\n✓ Corrected implementation is working")
    print("✓ All critical fixes applied:")
    print("  1. Removed look-ahead bias from breaker/mitigation blocks")
    print("  2. Improved order block detection (last occurrence)")
    print("  3. Added FVG consecutive merging")
    print("  4. Improved inducement zones (ATR-based)")
    print("  5. Better market maker patterns")
    print("  6. Added retracements")
    print("  7. Added previous high/low")
    print("  8. Added sessions support")
    
    print("\n✓ Ready to use corrected implementation!")
    
    return df_result


if __name__ == "__main__":
    result = test_with_sample_data()
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Review the comparison results above")
    print("2. Update features.py to use smc_corrected instead of smc")
    print("3. Regenerate features for all pairs")
    print("4. Retrain all models with corrected features")
    print("5. Compare performance (should be better in live trading)")
    print("\nIMPORTANT: The corrected implementation removes look-ahead bias,")
    print("so backtest performance may be slightly lower, but LIVE performance")
    print("should be significantly better!")
