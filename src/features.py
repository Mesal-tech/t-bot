import pandas as pd
import numpy as np
from smc_corrected import SmartMoneyConceptsFull
from sessions import add_session_features

class SMCFeatureGenerator:
    """
    Generate ML-ready features from raw OHLCV data using SMC analysis.
    Produces 30+ features per bar for machine learning models.
    """
    
    def __init__(self, swing_length=50, internal_length=5):
        self.smc = SmartMoneyConceptsFull(
            swing_length=swing_length,
            internal_length=internal_length,
            order_block_filter='atr',
            eq_threshold=0.1
        )
    
    def generate_features(self, df):
        """
        Generate all features for ML from OHLCV data.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            DataFrame with 40+ engineered features
        """
        df = df.copy()
        
        # print("Analyzing SMC structure...")
        smc_df = self.smc.analyze(df)
        
        # Add Session Features
        # print("Adding session features...")
        session_df = add_session_features(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy(), time_zone="UTC")
        
        # Dictionary to collect all features (prevents fragmentation)
        feature_dict = {}
        
        # ===== 1. SMC STRUCTURE FEATURES =====
        # print("Extracting SMC structure features...")
        feature_dict['swing_trend'] = smc_df['swing_trend']
        feature_dict['internal_trend'] = smc_df['internal_trend']
        
        # Structure alignment (both trends agree)
        feature_dict['structure_aligned'] = (
            (smc_df['swing_trend'] == smc_df['internal_trend']) & 
            (smc_df['swing_trend'] != 0)
        ).astype(int)
        
        # Structure breaks (momentum indicators)
        feature_dict['swing_bos'] = smc_df['swing_bos'].astype(int)
        feature_dict['swing_choch'] = smc_df['swing_choch'].astype(int)
        feature_dict['internal_bos'] = smc_df['internal_bos'].astype(int)
        feature_dict['internal_choch'] = smc_df['internal_choch'].astype(int)
        
        # BOS momentum (rolling count)
        feature_dict['bos_momentum'] = smc_df['swing_bos'].rolling(10).sum()
        feature_dict['choch_count'] = smc_df['swing_choch'].rolling(10).sum()
        
        # ===== 2. ORDER BLOCK & ZONE FEATURES =====
        # print("Processing order blocks and zones...")
        feature_dict['swing_ob'] = smc_df['swing_ob']
        feature_dict['internal_ob'] = smc_df['internal_ob']
        
        # Distance to structure levels (normalized by ATR)
        atr = self._calculate_atr(df)
        feature_dict['dist_to_swing_high'] = (smc_df['swing_high_level'] - df['close']) / (atr + 1e-10)
        feature_dict['dist_to_swing_low'] = (df['close'] - smc_df['swing_low_level']) / (atr + 1e-10)
        feature_dict['dist_to_internal_high'] = (smc_df['internal_high_level'] - df['close']) / (atr + 1e-10)
        feature_dict['dist_to_internal_low'] = (df['close'] - smc_df['internal_low_level']) / (atr + 1e-10)
        
        # Premium/Discount/Equilibrium zones
        feature_dict['in_premium'] = (df['close'] > smc_df['premium_bottom']).astype(int)
        feature_dict['in_discount'] = (df['close'] < smc_df['discount_top']).astype(int)
        feature_dict['in_equilibrium'] = (
            (df['close'] >= smc_df['equilibrium_bottom']) & 
            (df['close'] <= smc_df['equilibrium_top'])
        ).astype(int)
        
        # ===== 3. PRICE ACTION FEATURES =====
        # print("Calculating price action features...")
        
        # Returns at multiple timeframes
        for lag in [1, 3, 5, 10, 20, 50]:
            feature_dict[f'return_{lag}'] = df['close'].pct_change(lag)
        
        # Volatility (rolling std of returns)
        returns = df['close'].pct_change()
        for window in [10, 20, 50]:
            feature_dict[f'volatility_{window}'] = returns.rolling(window).std()
        
        # ATR-based features
        feature_dict['atr'] = atr
        feature_dict['atr_ratio'] = atr / (atr.rolling(50).mean() + 1e-10)
        
        # Price position in recent range
        for window in [20, 50, 100]:
            rolling_high = df['high'].rolling(window).max()
            rolling_low = df['low'].rolling(window).min()
            feature_dict[f'price_position_{window}'] = (
                (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            )
        
        # ===== 4. CANDLE PATTERN FEATURES =====
        # print("Analyzing candle patterns...")
        
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        feature_dict['body_ratio'] = body / (total_range + 1e-10)
        
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        feature_dict['upper_wick_ratio'] = upper_wick / (total_range + 1e-10)
        feature_dict['lower_wick_ratio'] = lower_wick / (total_range + 1e-10)
        
        # Bullish/Bearish candle
        feature_dict['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # ===== 5. TIME FEATURES (SESSION AWARENESS) =====
        # print("Adding time features...")
        
        feature_dict['hour'] = df['timestamp'].dt.hour
        feature_dict['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Cyclical encoding
        feature_dict['hour_sin'] = np.sin(2 * np.pi * feature_dict['hour'] / 24)
        feature_dict['hour_cos'] = np.cos(2 * np.pi * feature_dict['hour'] / 24)
        feature_dict['dow_sin'] = np.sin(2 * np.pi * feature_dict['day_of_week'] / 7)
        feature_dict['dow_cos'] = np.cos(2 * np.pi * feature_dict['day_of_week'] / 7)
        
        # Session Features
        for col in session_df.columns:
            if '_active' in col:
                feature_dict[col] = session_df[col]
        
        # ===== 6. FVG FEATURES =====
        feature_dict['has_fvg'] = (smc_df['fvg'] != 0).astype(int)
        feature_dict['fvg_direction'] = smc_df['fvg']
        
        # ===== 7. EQUAL HIGHS/LOWS =====
        feature_dict['has_eqh'] = smc_df['eqh'].astype(int)
        feature_dict['has_eql'] = smc_df['eql'].astype(int)
        
        # ===== 8. LIQUIDITY SWEEP FEATURES =====
        feature_dict['liquidity_sweep'] = smc_df['liquidity_sweep']
        feature_dict['sweep_magnitude'] = smc_df['sweep_magnitude']
        # 'sweep_reversal' removed from smc_corrected as detection is now integrated into liquidity_sweep
        
        # Sweep momentum (rolling count)
        feature_dict['bullish_sweeps_recent'] = (smc_df['liquidity_sweep'] == 1).rolling(20).sum()
        feature_dict['bearish_sweeps_recent'] = (smc_df['liquidity_sweep'] == -1).rolling(20).sum()
        
        # ===== 9. INDUCEMENT ZONE FEATURES =====
        feature_dict['inducement_zone'] = smc_df['inducement_zone']
        feature_dict['in_inducement'] = (smc_df['inducement_zone'] != 0).astype(int)
        
        # Distance to inducement zones
        feature_dict['dist_to_inducement'] = np.where(
            ~smc_df['inducement_top'].isna(),
            (smc_df['inducement_top'] - df['close']) / (atr + 1e-10),
            0
        )
        
        # ===== 10. MARKET MAKER FEATURES =====
        feature_dict['mm_phase'] = smc_df['mm_phase']
        feature_dict['amd_pattern'] = smc_df['amd_pattern'].astype(int)
        feature_dict['institutional_candle'] = smc_df['institutional_candle'].astype(int)
        
        # Market maker phase indicators
        feature_dict['mm_accumulation'] = (smc_df['mm_phase'] == 1).astype(int)
        feature_dict['mm_manipulation'] = (smc_df['mm_phase'] == 2).astype(int)
        feature_dict['mm_distribution'] = (smc_df['mm_phase'] == 3).astype(int)
        
        # Count of institutional candles in recent period
        feature_dict['institutional_candles_recent'] = smc_df['institutional_candle'].rolling(10).sum()
        
        # ===== 11. BREAKER BLOCK FEATURES =====
        feature_dict['breaker_block'] = smc_df['swing_ob_breaker'].astype(int) # Updated naming in corrected
        feature_dict['has_breaker'] = (smc_df['swing_ob_breaker']).astype(int)
        
        # ===== 12. MITIGATION BLOCK FEATURES =====
        feature_dict['mitigation_block'] = smc_df['swing_ob_mitigated'].astype(int) # Updated naming
        feature_dict['has_mitigation'] = (smc_df['swing_ob_mitigated']).astype(int)
        
        # ===== 13. VOLUME-WEIGHTED ORDER BLOCK FEATURES =====
        feature_dict['ob_volume_weight'] = smc_df['ob_volume_weight']
        feature_dict['ob_strength'] = smc_df['ob_strength']
        
        # High-strength order blocks
        feature_dict['high_strength_ob'] = (smc_df['ob_strength'] > 2.0).astype(int)
        
        # ===== 14. NEW FEATURES FROM CORRECTED SMC =====
        
        # Retracements
        feature_dict['retracement_direction'] = smc_df['retracement_direction']
        feature_dict['current_retracement_pct'] = smc_df['current_retracement_pct']
        feature_dict['deepest_retracement_pct'] = smc_df['deepest_retracement_pct']
        
        # Previous High/Low
        feature_dict['above_previous_high'] = (df['close'] > smc_df['previous_high']).astype(int)
        feature_dict['below_previous_low'] = (df['close'] < smc_df['previous_low']).astype(int)
        feature_dict['dist_vs_prev_high'] = (df['close'] - smc_df['previous_high']) / (atr + 1e-10) # Normalized
        
        # ===== 15. INTERACTION FEATURES (POWERFUL COMBINATIONS) =====
        # Trend + Zone alignment
        feature_dict['bullish_trend_discount'] = (
            (smc_df['swing_trend'] == 1) & feature_dict['in_discount']
        ).astype(int)
        feature_dict['bearish_trend_premium'] = (
            (smc_df['swing_trend'] == -1) & feature_dict['in_premium']
        ).astype(int)
        
        # Structure + Order Block confluence
        feature_dict['bos_with_ob'] = (
            smc_df['swing_bos'] & (smc_df['swing_ob'] != 0)
        ).astype(int)
        feature_dict['choch_with_ob'] = (
            smc_df['swing_choch'] & (smc_df['swing_ob'] != 0)
        ).astype(int)
        
        # FVG + Trend alignment
        feature_dict['bullish_fvg_trend'] = (
            (smc_df['fvg'] == 1) & (smc_df['swing_trend'] == 1)
        ).astype(int)
        feature_dict['bearish_fvg_trend'] = (
            (smc_df['fvg'] == -1) & (smc_df['swing_trend'] == -1)
        ).astype(int)
        
        # Liquidity sweep + Setup
        # Note: Reversal is now implicit in sweep detection in corrected SMC
        feature_dict['sweep_reversal_setup'] = (
            (feature_dict['liquidity_sweep'] != 0) & (feature_dict['structure_aligned'])
        ).astype(int)
        
        # Mitigation + Trend alignment
        feature_dict['bullish_mitigation_trend'] = (
            (feature_dict['mitigation_block'] == 1) & (smc_df['swing_trend'] == 1)
        ).astype(int)
        feature_dict['bearish_mitigation_trend'] = (
            (feature_dict['mitigation_block'] == 1) & (smc_df['swing_trend'] == -1) 
        ).astype(int)
        
        # Premium rejection / Discount acceptance
        feature_dict['premium_rejection'] = (
            feature_dict['in_premium'] & (df['close'] < df['open'])
        ).astype(int)
        feature_dict['discount_acceptance'] = (
            feature_dict['in_discount'] & (df['close'] > df['open'])
        ).astype(int)
        
        # Multi-confluence score
        feature_dict['bullish_confluence'] = (
            feature_dict['bullish_trend_discount'] +
            feature_dict['bullish_fvg_trend'] +
            feature_dict['bullish_mitigation_trend'] +
            (smc_df['liquidity_sweep'] == 1).astype(int) +
            (smc_df['swing_ob'] == 1).astype(int)
        )
        feature_dict['bearish_confluence'] = (
            feature_dict['bearish_trend_premium'] +
            feature_dict['bearish_fvg_trend'] +
            feature_dict['bearish_mitigation_trend'] +
            (smc_df['liquidity_sweep'] == -1).astype(int) +
            (smc_df['swing_ob'] == -1).astype(int)
        )
        
        # Institutional activity indicator
        feature_dict['institutional_activity'] = (
            feature_dict['institutional_candle'] +
            feature_dict['amd_pattern'] +
            feature_dict['high_strength_ob']
        )
        
        # ===== ADD ORIGINAL PRICE DATA FOR LABELING =====
        feature_dict['open'] = df['open']
        feature_dict['high'] = df['high']
        feature_dict['low'] = df['low']
        feature_dict['close'] = df['close']
        feature_dict['timestamp'] = df['timestamp']
        
        # Create features DataFrame at once to prevent fragmentation
        features = pd.DataFrame(feature_dict, index=df.index)
        
        # Add Technical Indicators (this returns a new DF, need to ensure it's also optimized)
        # We can actually just merge the dict from _add_technical_indicators
        # But for now let's use the optimized _add_technical_indicators which returns a DF
        # and concat it.
        
        tech_features = self._calculate_technical_indicators(df)
        features = pd.concat([features, tech_features], axis=1)
        
        return features.fillna(0) # Handle NaN
    
    def _calculate_technical_indicators(self, df):
        """
        Calculate technical indicators and return a DataFrame.
        Used to be _add_technical_indicators but renamed to calculate to imply returning new data.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        tech_dict = {}
        
        # 1. RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        tech_dict['rsi_14'] = 100 - (100 / (1 + rs))
        tech_dict['rsi_overbought'] = (tech_dict['rsi_14'] > 70).astype(int)
        tech_dict['rsi_oversold'] = (tech_dict['rsi_14'] < 30).astype(int)
        
        # 2. MACD (Moving Average Convergence Divergence)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        tech_dict['macd'] = exp12 - exp26
        tech_dict['macd_signal'] = tech_dict['macd'].ewm(span=9, adjust=False).mean()
        tech_dict['macd_hist'] = tech_dict['macd'] - tech_dict['macd_signal']
        
        # 3. Bollinger Bands
        bb_window = 20
        bb_ma = close.rolling(bb_window).mean()
        bb_std = close.rolling(bb_window).std()
        tech_dict['bb_upper'] = bb_ma + (bb_std * 2)
        tech_dict['bb_lower'] = bb_ma - (bb_std * 2)
        # %B Indicator (position within bands)
        tech_dict['bb_position'] = (close - tech_dict['bb_lower']) / (tech_dict['bb_upper'] - tech_dict['bb_lower'] + 1e-10)
        # Band Width (volatility)
        tech_dict['bb_width'] = (tech_dict['bb_upper'] - tech_dict['bb_lower']) / bb_ma
        
        # 4. Stochastic Oscillator
        stoch_k = 14
        low_min = low.rolling(stoch_k).min()
        high_max = high.rolling(stoch_k).max()
        tech_dict['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
        tech_dict['stoch_d'] = tech_dict['stoch_k'].rolling(3).mean()
        
        # 5. Simple Moving Averages & Trends
        tech_dict['sma_50'] = close.rolling(50).mean()
        tech_dict['sma_200'] = close.rolling(200).mean()
        tech_dict['above_sma200'] = (close > tech_dict['sma_200']).astype(int)
        tech_dict['sma_cross'] = (tech_dict['sma_50'] > tech_dict['sma_200']).astype(int)
        
        return pd.DataFrame(tech_dict, index=df.index)
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr


def process_pair_features(pair_name, input_path, output_path):
    """
    Convenience function to process a single pair.
    
    Args:
        pair_name: Name of the pair (e.g., 'EURUSD')
        input_path: Path to raw CSV file
        output_path: Path to save processed features
    """
    print(f"\n{'='*60}")
    print(f"Processing {pair_name}")
    print('='*60)
    
    # Load data
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate features
    generator = SMCFeatureGenerator()
    features = generator.generate_features(df)
    
    # Save
    features.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return features


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pair = sys.argv[1]
        input_file = f'data/{pair}_15min_data.csv'
        output_file = f'data/processed/{pair}_features.csv'
        process_pair_features(pair, input_file, output_file)
    else:
        print("Usage: python features.py EURUSD")
