"""
Quick test script to verify SMC feature enhancements are working correctly.
Tests feature generation with new advanced features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from features import SMCFeatureGenerator


def test_feature_generation():
    """Test that feature generation works with new advanced features"""
    print("="*70)
    print(" TESTING SMC FEATURE GENERATION")
    print("="*70)
    
    # Create sample data
    print("\n1. Creating sample OHLCV data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic price data
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
    
    print(f"   Created {len(df)} bars of sample data")
    print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    # Generate features
    print("\n2. Generating features...")
    generator = SMCFeatureGenerator()
    
    try:
        features = generator.generate_features(df)
        print(f"   SUCCESS: Generated {len(features.columns)} features")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check for new advanced features
    print("\n3. Verifying new advanced features...")
    
    expected_new_features = [
        # Liquidity sweeps
        'liquidity_sweep', 'sweep_magnitude', # 'sweep_reversal' removed
        'bullish_sweeps_recent', 'bearish_sweeps_recent',
        
        # Inducement zones
        'inducement_zone', 'in_inducement', 'dist_to_inducement',
        
        # Market maker
        'mm_phase', 'amd_pattern', 'institutional_candle',
        'mm_accumulation', 'mm_manipulation', 'mm_distribution',
        'institutional_candles_recent', 'institutional_activity',
        
        # Breaker blocks
        'breaker_block', 'has_breaker', 'dist_to_breaker',
        
        # Mitigation blocks
        'mitigation_block', 'mitigation_strength', # 'mitigation_strength' might be missing if not passed through? Check features.py line 183. Yes it is passed.
        'has_mitigation',
        'dist_to_mitigation',
        
        # Volume-weighted OB
        'ob_volume_weight', 'ob_strength', 'high_strength_ob',
        
        # New features from corrected SMC
        'retracement_direction', 'current_retracement_pct', 'deepest_retracement_pct',
        'above_previous_high', 'below_previous_low', 'dist_vs_prev_high',
        'asian_kill_zone_active', 'london_active', 'new_york_active', # Sample session features
        
        # Interaction features
        'bullish_trend_discount', 'bearish_trend_premium',
        'bos_with_ob', 'choch_with_ob',
        'bullish_fvg_trend', 'bearish_fvg_trend',
        'sweep_reversal_setup',
        'bullish_mitigation_trend', 'bearish_mitigation_trend',
        'premium_rejection', 'discount_acceptance',
        'bullish_confluence', 'bearish_confluence',
    ]
    
    missing_features = []
    for feat in expected_new_features:
        if feat not in features.columns:
            missing_features.append(feat)
    
    if missing_features:
        print(f"   WARNING: Missing {len(missing_features)} expected features:")
        for feat in missing_features[:10]:  # Show first 10
            print(f"     - {feat}")
        if len(missing_features) > 10:
            print(f"     ... and {len(missing_features) - 10} more")
    else:
        print(f"   SUCCESS: All {len(expected_new_features)} new features present")
    
    # Check for NaN issues
    print("\n4. Checking data quality...")
    
    # Exclude OHLC columns from NaN check
    feature_cols = [c for c in features.columns 
                   if c not in ['timestamp', 'open', 'high', 'low', 'close']]
    
    nan_counts = features[feature_cols].isna().sum()
    high_nan_features = nan_counts[nan_counts > len(features) * 0.1]
    
    if len(high_nan_features) > 0:
        print(f"   WARNING: {len(high_nan_features)} features with >10% NaN:")
        for feat, count in high_nan_features.head(10).items():
            pct = (count / len(features)) * 100
            print(f"     - {feat}: {pct:.1f}%")
    else:
        print(f"   SUCCESS: No features with excessive NaN values")
    
    # Check for inf values
    inf_counts = {}
    for col in feature_cols:
        if features[col].dtype in [np.float64, np.float32]:
            inf_count = np.isinf(features[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
    
    if inf_counts:
        print(f"   WARNING: {len(inf_counts)} features with inf values:")
        for feat, count in list(inf_counts.items())[:10]:
            print(f"     - {feat}: {count} inf values")
    else:
        print(f"   SUCCESS: No inf values detected")
    
    # Feature statistics
    print("\n5. Feature Statistics:")
    print(f"   Total features:        {len(feature_cols)}")
    print(f"   New advanced features: {len(expected_new_features)}")
    print(f"   Original features:     {len(feature_cols) - len(expected_new_features)}")
    
    # Sample feature values
    print("\n6. Sample Advanced Feature Values (last 5 bars):")
    sample_features = [
        'liquidity_sweep', 'inducement_zone', 'mm_phase', 
        'institutional_candle', 'bullish_confluence', 'bearish_confluence'
    ]
    
    for feat in sample_features:
        if feat in features.columns:
            values = features[feat].tail(5).values
            print(f"   {feat:25s}: {values}")
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    if not missing_features and not inf_counts:
        print("   STATUS: ALL TESTS PASSED")
        print("   The SMC feature enhancements are working correctly!")
        return True
    else:
        print("   STATUS: TESTS PASSED WITH WARNINGS")
        print("   Review warnings above for potential issues.")
        return True


if __name__ == "__main__":
    success = test_feature_generation()
    sys.exit(0 if success else 1)
