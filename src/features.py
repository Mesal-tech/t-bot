import pandas as pd
import numpy as np
from smc import SmartMoneyConceptsFull

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
        
        print("Analyzing SMC structure...")
        smc_df = self.smc.analyze(df)
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # ===== 1. SMC STRUCTURE FEATURES =====
        print("Extracting SMC structure features...")
        features['swing_trend'] = smc_df['swing_trend']
        features['internal_trend'] = smc_df['internal_trend']
        
        # Structure alignment (both trends agree)
        features['structure_aligned'] = (
            (smc_df['swing_trend'] == smc_df['internal_trend']) & 
            (smc_df['swing_trend'] != 0)
        ).astype(int)
        
        # Structure breaks (momentum indicators)
        features['swing_bos'] = smc_df['swing_bos'].astype(int)
        features['swing_choch'] = smc_df['swing_choch'].astype(int)
        features['internal_bos'] = smc_df['internal_bos'].astype(int)
        features['internal_choch'] = smc_df['internal_choch'].astype(int)
        
        # BOS momentum (rolling count)
        features['bos_momentum'] = smc_df['swing_bos'].rolling(10).sum()
        features['choch_count'] = smc_df['swing_choch'].rolling(10).sum()
        
        # ===== 2. ORDER BLOCK & ZONE FEATURES =====
        print("Processing order blocks and zones...")
        features['swing_ob'] = smc_df['swing_ob']
        features['internal_ob'] = smc_df['internal_ob']
        
        # Distance to structure levels (normalized by ATR)
        atr = self._calculate_atr(df)
        features['dist_to_swing_high'] = (smc_df['swing_high_level'] - df['close']) / (atr + 1e-10)
        features['dist_to_swing_low'] = (df['close'] - smc_df['swing_low_level']) / (atr + 1e-10)
        features['dist_to_internal_high'] = (smc_df['internal_high_level'] - df['close']) / (atr + 1e-10)
        features['dist_to_internal_low'] = (df['close'] - smc_df['internal_low_level']) / (atr + 1e-10)
        
        # Premium/Discount/Equilibrium zones
        features['in_premium'] = (df['close'] > smc_df['premium_bottom']).astype(int)
        features['in_discount'] = (df['close'] < smc_df['discount_top']).astype(int)
        features['in_equilibrium'] = (
            (df['close'] >= smc_df['equilibrium_bottom']) & 
            (df['close'] <= smc_df['equilibrium_top'])
        ).astype(int)
        
        # ===== 3. PRICE ACTION FEATURES =====
        print("Calculating price action features...")
        
        # Returns at multiple timeframes
        for lag in [1, 3, 5, 10, 20, 50]:
            features[f'return_{lag}'] = df['close'].pct_change(lag)
        
        # Volatility (rolling std of returns)
        returns = df['close'].pct_change()
        for window in [10, 20, 50]:
            features[f'volatility_{window}'] = returns.rolling(window).std()
        
        # ATR-based features
        features['atr'] = atr
        features['atr_ratio'] = atr / (atr.rolling(50).mean() + 1e-10)
        
        # Price position in recent range
        for window in [20, 50, 100]:
            rolling_high = df['high'].rolling(window).max()
            rolling_low = df['low'].rolling(window).min()
            features[f'price_position_{window}'] = (
                (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            )
        
        # ===== 4. CANDLE PATTERN FEATURES =====
        print("Analyzing candle patterns...")
        
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        features['body_ratio'] = body / (total_range + 1e-10)
        
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        features['upper_wick_ratio'] = upper_wick / (total_range + 1e-10)
        features['lower_wick_ratio'] = lower_wick / (total_range + 1e-10)
        
        # Bullish/Bearish candle
        features['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # ===== 5. TIME FEATURES (SESSION AWARENESS) =====
        print("Adding time features...")
        
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # ===== 6. FVG FEATURES =====
        features['has_fvg'] = (smc_df['fvg'] != 0).astype(int)
        features['fvg_direction'] = smc_df['fvg']
        
        # ===== 7. EQUAL HIGHS/LOWS =====
        features['has_eqh'] = smc_df['eqh'].astype(int)
        features['has_eql'] = smc_df['eql'].astype(int)
        
        # ===== ADD ORIGINAL PRICE DATA FOR LABELING =====
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['timestamp'] = df['timestamp']
        
        print(f"✅ Generated {len(features.columns)} features for {len(features)} bars")
        
        return features
    
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
    print(f"✅ Saved to {output_path}")
    
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
