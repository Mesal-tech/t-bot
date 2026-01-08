import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import IntEnum

class Bias(IntEnum):
    """Market structure bias"""
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1

class StructureType(IntEnum):
    """Structure break type"""
    NONE = 0
    BOS = 1
    CHOCH = 2

@dataclass
class Pivot:
    """Represents a swing pivot point"""
    current_level: float = float('nan')
    last_level: float = float('nan')
    crossed: bool = False
    bar_time: int = 0
    bar_index: int = 0

@dataclass
class OrderBlock:
    """Represents an order block"""
    bar_high: float
    bar_low: float
    bar_time: int
    bar_index: int
    bias: Bias
    mitigated: bool = False

@dataclass
class FairValueGap:
    """Represents a Fair Value Gap"""
    top: float
    bottom: float
    bar_time: int
    bar_index: int
    bias: Bias
    filled: bool = False

@dataclass
class StructureEvent:
    """Represents a structure break event"""
    bar_index: int
    bar_time: int
    structure_type: StructureType
    bias: Bias
    level: float

class SmartMoneyConceptsFull:
    """
    Complete Python implementation of LuxAlgo's Smart Money Concepts indicator.
    Faithfully recreates the Pine Script logic for:
    - Market Structure (BOS/CHoCH) - both Swing and Internal
    - Order Blocks with volatility filtering
    - Fair Value Gaps
    - Equal Highs/Lows
    - Premium/Discount Zones
    """
    
    def __init__(self, 
                 swing_length: int = 50,
                 internal_length: int = 5,
                 order_block_filter: str = 'atr',  # 'atr' or 'cmr'
                 order_block_mitigation: str = 'high_low',  # 'high_low' or 'close'
                 eq_threshold: float = 0.1,
                 eq_length: int = 3,
                 fvg_auto_threshold: bool = True,
                 internal_confluence_filter: bool = False):
        
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.order_block_filter = order_block_filter
        self.order_block_mitigation = order_block_mitigation
        self.eq_threshold = eq_threshold
        self.eq_length = eq_length
        self.fvg_auto_threshold = fvg_auto_threshold
        self.internal_confluence_filter = internal_confluence_filter
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main analysis function. Takes OHLCV data and returns enriched DataFrame
        with all SMC indicators.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            DataFrame with added SMC indicator columns
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize output columns
        self._initialize_columns(df)
        
        # Calculate volatility measures
        df = self._calculate_volatility(df)
        
        # Parse highs/lows (filter high volatility bars)
        df = self._parse_price_levels(df)
        
        # Process swing structure
        df = self._process_structure(df, self.swing_length, is_internal=False)
        
        # Process internal structure
        df = self._process_structure(df, self.internal_length, is_internal=True)
        
        # Detect Fair Value Gaps
        df = self._detect_fvgs(df)
        
        # Detect Equal Highs/Lows
        df = self._detect_equal_highs_lows(df)
        
        # Calculate Premium/Discount Zones
        df = self._calculate_zones(df)
        
        # Detect Liquidity Sweeps
        df = self._detect_liquidity_sweeps(df)
        
        # Detect Inducement Zones
        df = self._detect_inducement_zones(df)
        
        # Detect Market Maker Patterns
        df = self._detect_market_maker_patterns(df)
        
        # Detect Breaker Blocks
        df = self._detect_breaker_blocks(df)
        
        # Detect Mitigation Blocks
        df = self._detect_mitigation_blocks(df)
        
        # Calculate Volume-Weighted Order Block Strength
        df = self._calculate_ob_volume_weight(df)
        
        return df
    
    def _initialize_columns(self, df: pd.DataFrame):
        """Initialize all output columns"""
        # Swing Structure
        df['swing_bos'] = False
        df['swing_choch'] = False
        df['swing_trend'] = 0
        df['swing_high_level'] = float('nan')
        df['swing_low_level'] = float('nan')
        
        # Internal Structure
        df['internal_bos'] = False
        df['internal_choch'] = False
        df['internal_trend'] = 0
        df['internal_high_level'] = float('nan')
        df['internal_low_level'] = float('nan')
        
        # Order Blocks
        df['swing_ob'] = 0  # 1 for bullish, -1 for bearish
        df['internal_ob'] = 0
        df['swing_ob_top'] = float('nan')
        df['swing_ob_bottom'] = float('nan')
        df['internal_ob_top'] = float('nan')
        df['internal_ob_bottom'] = float('nan')
        
        # Fair Value Gaps
        df['fvg'] = 0  # 1 for bullish, -1 for bearish
        df['fvg_top'] = float('nan')
        df['fvg_bottom'] = float('nan')
        
        # Equal Highs/Lows
        df['eqh'] = False
        df['eql'] = False
        df['eqh_level'] = float('nan')
        df['eql_level'] = float('nan')
        
        # Zones
        df['premium_top'] = float('nan')
        df['premium_bottom'] = float('nan')
        df['equilibrium_top'] = float('nan')
        df['equilibrium_bottom'] = float('nan')
        df['discount_top'] = float('nan')
        df['discount_bottom'] = float('nan')
        
        # Liquidity Sweeps
        df['liquidity_sweep'] = 0  # 1 for bullish sweep, -1 for bearish sweep
        df['sweep_magnitude'] = 0.0  # How far beyond level (in ATR)
        df['sweep_reversal'] = False  # Did price reverse after sweep
        
        # Inducement Zones
        df['inducement_zone'] = 0  # 1 for bullish trap, -1 for bearish trap
        df['inducement_top'] = float('nan')
        df['inducement_bottom'] = float('nan')
        
        # Market Maker Models
        df['mm_phase'] = 0  # 0=none, 1=accumulation, 2=manipulation, 3=distribution
        df['amd_pattern'] = False  # Accumulation-Manipulation-Distribution
        df['institutional_candle'] = False  # Large body, small wicks
        
        # Breaker Blocks
        df['breaker_block'] = 0  # 1 for bullish, -1 for bearish
        df['breaker_top'] = float('nan')
        df['breaker_bottom'] = float('nan')
        
        # Mitigation Blocks
        df['mitigation_block'] = 0  # 1 for bullish, -1 for bearish
        df['mitigation_top'] = float('nan')
        df['mitigation_bottom'] = float('nan')
        df['mitigation_strength'] = 0.0  # 0-1 score
        
        # Volume-Weighted Order Blocks
        df['ob_volume_weight'] = 0.0  # Volume weight for order blocks
        df['ob_strength'] = 0.0  # Combined strength score
        
        
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility measures for order block filtering"""
        # ATR(200)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = tr.rolling(window=200, min_periods=1).mean()
        
        # Cumulative Mean Range
        df['cmr'] = tr.expanding().mean()
        
        # Select volatility measure
        if self.order_block_filter == 'atr':
            df['volatility'] = df['atr']
        else:
            df['volatility'] = df['cmr']
            
        return df
    
    def _parse_price_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse highs/lows to filter out high volatility bars.
        High volatility bars have their high/low swapped.
        """
        # High volatility bar: range >= 2 * volatility
        high_vol_bar = (df['high'] - df['low']) >= (2 * df['volatility'])
        
        df['parsed_high'] = np.where(high_vol_bar, df['low'], df['high'])
        df['parsed_low'] = np.where(high_vol_bar, df['high'], df['low'])
        
        return df
    
    def _leg(self, highs: np.ndarray, lows: np.ndarray, size: int, current_idx: int) -> int:
        """
        Determine the current leg direction (0 = bearish, 1 = bullish).
        Matches Pine Script's leg() function logic.
        
        A new bearish leg starts when high[size] > highest(size)
        A new bullish leg starts when low[size] < lowest(size)
        """
        if current_idx < size * 2:
            return 0
            
        lag_idx = current_idx - size
        
        # Check if the lagged bar was a new high
        # highest(size) means highest of last 'size' bars (not including current)
        window_start = max(0, lag_idx - size)
        window_highs = highs[window_start:lag_idx]
        window_lows = lows[window_start:lag_idx]
        
        if len(window_highs) > 0:
            new_leg_high = highs[lag_idx] > np.max(window_highs)
            new_leg_low = lows[lag_idx] < np.min(window_lows)
            
            # Return the leg that was triggered (bearish leg = 0, bullish leg = 1)
            if new_leg_high:
                return 0  # Start of bearish leg
            elif new_leg_low:
                return 1  # Start of bullish leg
        
        return -1  # No change
    
    def _process_structure(self, df: pd.DataFrame, length: int, is_internal: bool = False) -> pd.DataFrame:
        """
        Process market structure for given length.
        Implements the Pine Script state machine logic for BOS/CHoCH detection.
        """
        prefix = 'internal' if is_internal else 'swing'
        
        # State variables
        swing_high = Pivot()
        swing_low = Pivot()
        trend_bias = Bias.NEUTRAL
        last_leg = -1
        
        # Storage for order blocks
        order_blocks: List[OrderBlock] = []
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        parsed_highs = df['parsed_high'].values
        parsed_lows = df['parsed_low'].values
        
        n = len(df)
        
        for i in range(length * 2, n):
            # 1. Update swing points (leg detection)
            current_leg = self._leg(highs, lows, length, i)
            
            if current_leg != -1 and current_leg != last_leg:
                lag_idx = i - length
                
                if current_leg == 0:  # Start of bearish leg (new high detected)
                    swing_high.last_level = swing_high.current_level
                    swing_high.current_level = highs[lag_idx]
                    swing_high.crossed = False
                    swing_high.bar_time = lag_idx
                    swing_high.bar_index = lag_idx
                    
                elif current_leg == 1:  # Start of bullish leg (new low detected)
                    swing_low.last_level = swing_low.current_level
                    swing_low.current_level = lows[lag_idx]
                    swing_low.crossed = False
                    swing_low.bar_time = lag_idx
                    swing_low.bar_index = lag_idx
                
                last_leg = current_leg
            
            # 2. Check for structure breaks at current bar
            # Apply confluence filter for internal structure
            if is_internal and self.internal_confluence_filter:
                upper_wick = highs[i] - max(closes[i], opens[i])
                lower_wick = min(closes[i], opens[i]) - lows[i]
                body = abs(closes[i] - opens[i])
                
                bullish_bar = lower_wick > upper_wick
                bearish_bar = upper_wick > lower_wick
            else:
                bullish_bar = True
                bearish_bar = True
            
            # Check for break of swing high (bullish structure)
            if (not np.isnan(swing_high.current_level) and 
                closes[i] > swing_high.current_level and 
                not swing_high.crossed and
                bullish_bar):
                
                swing_high.crossed = True
                
                if trend_bias == Bias.BEARISH:
                    # Change of Character
                    df.at[i, f'{prefix}_choch'] = True
                    structure_type = StructureType.CHOCH
                else:
                    # Break of Structure
                    df.at[i, f'{prefix}_bos'] = True
                    structure_type = StructureType.BOS
                
                trend_bias = Bias.BULLISH
                df.at[i, f'{prefix}_trend'] = int(trend_bias)
                
                # Find and store order block (bullish OB)
                if not np.isnan(swing_low.current_level):
                    ob = self._find_order_block(
                        parsed_highs, parsed_lows, opens, closes,
                        swing_low.bar_index, i, Bias.BULLISH
                    )
                    if ob is not None:
                        order_blocks.append(ob)
                        df.at[ob.bar_index, f'{prefix}_ob'] = 1
                        df.at[ob.bar_index, f'{prefix}_ob_top'] = ob.bar_high
                        df.at[ob.bar_index, f'{prefix}_ob_bottom'] = ob.bar_low
            
            # Check for break of swing low (bearish structure)
            elif (not np.isnan(swing_low.current_level) and 
                  closes[i] < swing_low.current_level and 
                  not swing_low.crossed and
                  bearish_bar):
                
                swing_low.crossed = True
                
                if trend_bias == Bias.BULLISH:
                    # Change of Character
                    df.at[i, f'{prefix}_choch'] = True
                    structure_type = StructureType.CHOCH
                else:
                    # Break of Structure
                    df.at[i, f'{prefix}_bos'] = True
                    structure_type = StructureType.BOS
                
                trend_bias = Bias.BEARISH
                df.at[i, f'{prefix}_trend'] = int(trend_bias)
                
                # Find and store order block (bearish OB)
                if not np.isnan(swing_high.current_level):
                    ob = self._find_order_block(
                        parsed_highs, parsed_lows, opens, closes,
                        swing_high.bar_index, i, Bias.BEARISH
                    )
                    if ob is not None:
                        order_blocks.append(ob)
                        df.at[ob.bar_index, f'{prefix}_ob'] = -1
                        df.at[ob.bar_index, f'{prefix}_ob_top'] = ob.bar_high
                        df.at[ob.bar_index, f'{prefix}_ob_bottom'] = ob.bar_low
            
            # 3. Update current structure levels in dataframe
            df.at[i, f'{prefix}_high_level'] = swing_high.current_level
            df.at[i, f'{prefix}_low_level'] = swing_low.current_level
            
            # 4. Check for order block mitigation
            self._check_ob_mitigation(df, order_blocks, i, prefix)
        
        return df
    
    def _find_order_block(self, 
                         parsed_highs: np.ndarray, 
                         parsed_lows: np.ndarray,
                         opens: np.ndarray,
                         closes: np.ndarray,
                         start_idx: int, 
                         end_idx: int, 
                         bias: Bias) -> Optional[OrderBlock]:
        """
        Find the order block in the range between structure points.
        For bullish OB: Find the lowest low in the range
        For bearish OB: Find the highest high in the range
        """
        if start_idx >= end_idx or start_idx < 0:
            return None
        
        if bias == Bias.BULLISH:
            # Find lowest parsed low
            range_lows = parsed_lows[start_idx:end_idx]
            if len(range_lows) == 0:
                return None
            min_low = np.min(range_lows)
            ob_idx = start_idx + np.argmin(range_lows)
        else:  # BEARISH
            # Find highest parsed high
            range_highs = parsed_highs[start_idx:end_idx]
            if len(range_highs) == 0:
                return None
            max_high = np.max(range_highs)
            ob_idx = start_idx + np.argmax(range_highs)
        
        return OrderBlock(
            bar_high=parsed_highs[ob_idx],
            bar_low=parsed_lows[ob_idx],
            bar_time=ob_idx,
            bar_index=ob_idx,
            bias=bias,
            mitigated=False
        )
    
    def _check_ob_mitigation(self, df: pd.DataFrame, order_blocks: List[OrderBlock], 
                            current_idx: int, prefix: str):
        """Check if any order blocks have been mitigated"""
        if self.order_block_mitigation == 'close':
            test_high = df.at[current_idx, 'close']
            test_low = df.at[current_idx, 'close']
        else:  # high_low
            test_high = df.at[current_idx, 'high']
            test_low = df.at[current_idx, 'low']
        
        for ob in order_blocks:
            if ob.mitigated:
                continue
                
            if ob.bias == Bias.BEARISH and test_high > ob.bar_high:
                ob.mitigated = True
            elif ob.bias == Bias.BULLISH and test_low < ob.bar_low:
                ob.mitigated = True
    
    def _detect_fvgs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps.
        Bullish FVG: low[0] > high[2]
        Bearish FVG: high[0] < low[2]
        """
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        
        n = len(df)
        
        # Calculate auto threshold if enabled
        if self.fvg_auto_threshold:
            # Delta percent threshold (cumulative mean of body%)
            body_pct = np.abs(closes - opens) / (opens + 1e-10) * 100
            threshold = np.nancumsum(body_pct) / (np.arange(n) + 1) * 2
        else:
            threshold = np.zeros(n)
        
        for i in range(2, n):
            # Bullish FVG
            if lows[i] > highs[i-2]:
                gap_size = lows[i] - highs[i-2]
                close_delta_pct = abs(closes[i-1] - opens[i-1]) / (opens[i-1] + 1e-10) * 100
                
                if close_delta_pct > threshold[i]:
                    df.at[i, 'fvg'] = 1
                    df.at[i, 'fvg_top'] = lows[i]
                    df.at[i, 'fvg_bottom'] = highs[i-2]
            
            # Bearish FVG
            elif highs[i] < lows[i-2]:
                gap_size = lows[i-2] - highs[i]
                close_delta_pct = abs(closes[i-1] - opens[i-1]) / (opens[i-1] + 1e-10) * 100
                
                if close_delta_pct > threshold[i]:
                    df.at[i, 'fvg'] = -1
                    df.at[i, 'fvg_top'] = lows[i-2]
                    df.at[i, 'fvg_bottom'] = highs[i]
        
        return df
    
    def _detect_equal_highs_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Equal Highs and Equal Lows using the swing length approach.
        EQH: Two highs within threshold distance
        EQL: Two lows within threshold distance
        """
        highs = df['high'].values
        lows = df['low'].values
        atr = df['atr'].values
        
        # State for tracking pivots for equal detection
        eq_high_pivot = Pivot()
        eq_low_pivot = Pivot()
        last_leg = -1
        
        n = len(df)
        length = self.eq_length
        
        for i in range(length * 2, n):
            current_leg = self._leg(highs, lows, length, i)
            
            if current_leg != -1 and current_leg != last_leg:
                lag_idx = i - length
                
                if current_leg == 0:  # New high (start of bearish leg)
                    # Check for equal high
                    if not np.isnan(eq_high_pivot.current_level):
                        diff = abs(eq_high_pivot.current_level - highs[lag_idx])
                        if diff < self.eq_threshold * atr[i]:
                            df.at[i, 'eqh'] = True
                            df.at[i, 'eqh_level'] = highs[lag_idx]
                    
                    eq_high_pivot.last_level = eq_high_pivot.current_level
                    eq_high_pivot.current_level = highs[lag_idx]
                    eq_high_pivot.bar_index = lag_idx
                    
                elif current_leg == 1:  # New low (start of bullish leg)
                    # Check for equal low
                    if not np.isnan(eq_low_pivot.current_level):
                        diff = abs(eq_low_pivot.current_level - lows[lag_idx])
                        if diff < self.eq_threshold * atr[i]:
                            df.at[i, 'eql'] = True
                            df.at[i, 'eql_level'] = lows[lag_idx]
                    
                    eq_low_pivot.last_level = eq_low_pivot.current_level
                    eq_low_pivot.current_level = lows[lag_idx]
                    eq_low_pivot.bar_index = lag_idx
                
                last_leg = current_leg
        
        return df
    
    def _calculate_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Premium/Discount/Equilibrium zones based on trailing highs/lows.
        Uses the swing structure to define ranges.
        """
        # Get trailing high/low from swing structure
        # We'll use a rolling window to find the range
        window = self.swing_length * 4  # Reasonable lookback
        
        for i in range(window, len(df)):
            window_high = df['high'].iloc[i-window:i+1].max()
            window_low = df['low'].iloc[i-window:i+1].min()
            
            range_size = window_high - window_low
            
            if range_size > 0:
                # Premium zone: 95-100% of range
                df.at[i, 'premium_top'] = window_high
                df.at[i, 'premium_bottom'] = window_high - 0.05 * range_size
                
                # Equilibrium: 47.5-52.5% of range
                mid = (window_high + window_low) / 2
                df.at[i, 'equilibrium_top'] = mid + 0.025 * range_size
                df.at[i, 'equilibrium_bottom'] = mid - 0.025 * range_size
                
                # Discount zone: 0-5% of range
                df.at[i, 'discount_top'] = window_low + 0.05 * range_size
                df.at[i, 'discount_bottom'] = window_low
        
        return df
    
    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity sweeps - when price sweeps above/below key levels and reverses.
        A sweep is a false breakout designed to grab liquidity before reversing.
        """
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        atr = df['atr'].values
        
        swing_highs = df['swing_high_level'].values
        swing_lows = df['swing_low_level'].values
        
        n = len(df)
        lookback = 5  # Bars to check for reversal
        
        for i in range(self.swing_length, n - lookback):
            if np.isnan(swing_highs[i]) or np.isnan(swing_lows[i]):
                continue
            
            # Bearish liquidity sweep (sweep above high, then reverse down)
            if highs[i] > swing_highs[i]:
                sweep_distance = (highs[i] - swing_highs[i]) / (atr[i] + 1e-10)
                
                # Check if price reversed down after sweep
                future_lows = lows[i+1:i+lookback+1]
                if len(future_lows) > 0 and np.min(future_lows) < closes[i]:
                    df.at[i, 'liquidity_sweep'] = -1  # Bearish sweep
                    df.at[i, 'sweep_magnitude'] = sweep_distance
                    df.at[i, 'sweep_reversal'] = True
            
            # Bullish liquidity sweep (sweep below low, then reverse up)
            if lows[i] < swing_lows[i]:
                sweep_distance = (swing_lows[i] - lows[i]) / (atr[i] + 1e-10)
                
                # Check if price reversed up after sweep
                future_highs = highs[i+1:i+lookback+1]
                if len(future_highs) > 0 and np.max(future_highs) > closes[i]:
                    df.at[i, 'liquidity_sweep'] = 1  # Bullish sweep
                    df.at[i, 'sweep_magnitude'] = sweep_distance
                    df.at[i, 'sweep_reversal'] = True
        
        return df
    
    def _detect_inducement_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect inducement zones - areas designed to trap retail traders.
        These are typically small moves against the trend before a strong reversal.
        """
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        swing_trend = df['swing_trend'].values
        
        n = len(df)
        lookback = 10
        
        for i in range(lookback, n - 5):
            # In bullish trend, look for bearish inducement (fake bearish move)
            if swing_trend[i] == 1:
                # Check for small bearish move followed by strong bullish continuation
                recent_low = np.min(lows[i-lookback:i])
                if lows[i] <= recent_low:
                    # Check if price rallies strongly after
                    future_highs = highs[i+1:i+6]
                    if len(future_highs) > 0 and np.max(future_highs) > highs[i] * 1.001:
                        df.at[i, 'inducement_zone'] = 1  # Bullish trap (trapped bears)
                        df.at[i, 'inducement_top'] = highs[i]
                        df.at[i, 'inducement_bottom'] = lows[i]
            
            # In bearish trend, look for bullish inducement (fake bullish move)
            elif swing_trend[i] == -1:
                # Check for small bullish move followed by strong bearish continuation
                recent_high = np.max(highs[i-lookback:i])
                if highs[i] >= recent_high:
                    # Check if price drops strongly after
                    future_lows = lows[i+1:i+6]
                    if len(future_lows) > 0 and np.min(future_lows) < lows[i] * 0.999:
                        df.at[i, 'inducement_zone'] = -1  # Bearish trap (trapped bulls)
                        df.at[i, 'inducement_top'] = highs[i]
                        df.at[i, 'inducement_bottom'] = lows[i]
        
        return df
    
    def _detect_market_maker_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market maker patterns including AMD (Accumulation-Manipulation-Distribution).
        Also identify institutional candles (large body, small wicks).
        """
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        
        n = len(df)
        
        for i in range(20, n):
            # Institutional candle detection
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]
            
            # Large body (>70% of range), small wicks, above average volume
            avg_volume = np.mean(volumes[i-20:i])
            if (body / (total_range + 1e-10) > 0.7 and 
                upper_wick / (total_range + 1e-10) < 0.15 and
                lower_wick / (total_range + 1e-10) < 0.15 and
                volumes[i] > avg_volume * 1.2):
                df.at[i, 'institutional_candle'] = True
            
            # AMD Pattern Detection (simplified)
            # Look for: consolidation -> spike -> reversal
            lookback = 15
            
            # Accumulation: tight range
            recent_range = np.max(highs[i-lookback:i]) - np.min(lows[i-lookback:i])
            avg_range = np.mean(highs[i-lookback:i] - lows[i-lookback:i])
            
            if recent_range < avg_range * 3:  # Consolidation
                df.at[i, 'mm_phase'] = 1  # Accumulation
                
                # Check for manipulation (spike out of range)
                if i < n - 5:
                    next_5_high = np.max(highs[i:i+5])
                    next_5_low = np.min(lows[i:i+5])
                    
                    if next_5_high > np.max(highs[i-lookback:i]) * 1.002:
                        df.at[i+1, 'mm_phase'] = 2  # Manipulation
                        df.at[i+1, 'amd_pattern'] = True
                    elif next_5_low < np.min(lows[i-lookback:i]) * 0.998:
                        df.at[i+1, 'mm_phase'] = 2  # Manipulation
                        df.at[i+1, 'amd_pattern'] = True
        
        return df
    
    def _detect_breaker_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect breaker blocks - order blocks that have been broken and flip polarity.
        A bullish OB that gets broken becomes a bearish breaker block and vice versa.
        """
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        swing_ob = df['swing_ob'].values
        swing_ob_top = df['swing_ob_top'].values
        swing_ob_bottom = df['swing_ob_bottom'].values
        
        n = len(df)
        
        # Track order blocks and check if they get broken
        for i in range(n):
            if swing_ob[i] != 0:  # Found an order block
                ob_top = swing_ob_top[i]
                ob_bottom = swing_ob_bottom[i]
                ob_type = swing_ob[i]  # 1 = bullish, -1 = bearish
                
                # Look forward to see if it gets broken
                for j in range(i+1, min(i+100, n)):
                    # Bullish OB broken to downside = bearish breaker
                    if ob_type == 1 and closes[j] < ob_bottom:
                        df.at[j, 'breaker_block'] = -1
                        df.at[j, 'breaker_top'] = ob_top
                        df.at[j, 'breaker_bottom'] = ob_bottom
                        break
                    
                    # Bearish OB broken to upside = bullish breaker
                    elif ob_type == -1 and closes[j] > ob_top:
                        df.at[j, 'breaker_block'] = 1
                        df.at[j, 'breaker_top'] = ob_top
                        df.at[j, 'breaker_bottom'] = ob_bottom
                        break
        
        return df
    
    def _detect_mitigation_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect mitigation blocks - when price returns to an order block to mitigate unfilled orders.
        These are high-probability entry zones.
        """
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        swing_ob = df['swing_ob'].values
        swing_ob_top = df['swing_ob_top'].values
        swing_ob_bottom = df['swing_ob_bottom'].values
        swing_trend = df['swing_trend'].values
        
        n = len(df)
        
        for i in range(n):
            if swing_ob[i] != 0:  # Found an order block
                ob_top = swing_ob_top[i]
                ob_bottom = swing_ob_bottom[i]
                ob_type = swing_ob[i]
                
                # Look forward for price returning to mitigate
                for j in range(i+1, min(i+50, n)):
                    # Bullish OB mitigation (price returns to test from above)
                    if ob_type == 1 and swing_trend[j] == 1:
                        # Price touches OB zone
                        if lows[j] <= ob_top and lows[j] >= ob_bottom:
                            # Calculate mitigation strength
                            penetration = (ob_top - lows[j]) / (ob_top - ob_bottom + 1e-10)
                            df.at[j, 'mitigation_block'] = 1
                            df.at[j, 'mitigation_top'] = ob_top
                            df.at[j, 'mitigation_bottom'] = ob_bottom
                            df.at[j, 'mitigation_strength'] = penetration
                            break
                    
                    # Bearish OB mitigation (price returns to test from below)
                    elif ob_type == -1 and swing_trend[j] == -1:
                        # Price touches OB zone
                        if highs[j] >= ob_bottom and highs[j] <= ob_top:
                            # Calculate mitigation strength
                            penetration = (highs[j] - ob_bottom) / (ob_top - ob_bottom + 1e-10)
                            df.at[j, 'mitigation_block'] = -1
                            df.at[j, 'mitigation_top'] = ob_top
                            df.at[j, 'mitigation_bottom'] = ob_bottom
                            df.at[j, 'mitigation_strength'] = penetration
                            break
        
        return df
    
    def _calculate_ob_volume_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-weighted strength for order blocks.
        Higher volume = stronger order block.
        """
        volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        swing_ob = df['swing_ob'].values
        
        n = len(df)
        
        # Calculate rolling average volume
        avg_volume = pd.Series(volumes).rolling(window=50, min_periods=1).mean().values
        
        for i in range(n):
            if swing_ob[i] != 0:
                # Volume weight: current volume / average volume
                vol_weight = volumes[i] / (avg_volume[i] + 1e-10)
                df.at[i, 'ob_volume_weight'] = min(vol_weight, 5.0)  # Cap at 5x
                
                # Combined strength score (volume + position in range)
                body = abs(df.at[i, 'close'] - df.at[i, 'open'])
                total_range = df.at[i, 'high'] - df.at[i, 'low']
                body_ratio = body / (total_range + 1e-10)
                
                # Strength = volume weight * body ratio
                df.at[i, 'ob_strength'] = vol_weight * body_ratio
        
        return df



# Example usage and helper functions
def get_smc_signals(df: pd.DataFrame, 
                    swing_length: int = 50,
                    internal_length: int = 5) -> pd.DataFrame:
    """
    Convenience function to get SMC analysis with default settings.
    
    Args:
        df: DataFrame with OHLCV data
        swing_length: Length for swing structure detection
        internal_length: Length for internal structure detection
    
    Returns:
        DataFrame with all SMC indicators
    """
    smc = SmartMoneyConceptsFull(
        swing_length=swing_length,
        internal_length=internal_length
    )
    return smc.analyze(df)


def get_current_structure(df: pd.DataFrame) -> dict:
    """
    Extract current market structure state from analyzed DataFrame.
    
    Returns:
        Dictionary with current structure information
    """
    if len(df) == 0:
        return {}
    
    last_idx = len(df) - 1
    
    return {
        'swing_trend': df.at[last_idx, 'swing_trend'],
        'internal_trend': df.at[last_idx, 'internal_trend'],
        'swing_high': df.at[last_idx, 'swing_high_level'],
        'swing_low': df.at[last_idx, 'swing_low_level'],
        'internal_high': df.at[last_idx, 'internal_high_level'],
        'internal_low': df.at[last_idx, 'internal_low_level'],
        'in_premium': df.at[last_idx, 'close'] > df.at[last_idx, 'premium_bottom'],
        'in_discount': df.at[last_idx, 'close'] < df.at[last_idx, 'discount_top'],
        'recent_bos': df['swing_bos'].iloc[-10:].any(),
        'recent_choch': df['swing_choch'].iloc[-10:].any(),
    }

# -------------------------------------------------------------------------
# COMPATIBILITY ADAPTER
# -------------------------------------------------------------------------

class SmartMoneyConcepts(SmartMoneyConceptsFull):
    """
    Adapter class to maintain compatibility with london_strategy.py and features.py
    which expect specific method names and column outputs.
    """
    def get_structure_and_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps analyze() output to legacy naming convention.
        """
        # Run full analysis
        df = self.analyze(df)
        
        # Map columns
        # london_strategy expects: 'BOS', 'CHOCH', 'trend', 'is_ob'
        # Full class provides: 'swing_bos', 'swing_choch', 'swing_trend', 'swing_ob'
        
        df['BOS'] = df['swing_bos']
        df['CHOCH'] = df['swing_choch']
        df['trend'] = df['swing_trend'] # 1 / -1
        
        # 'swing_ob' is 1/-1/0. 'is_ob' expected 1/-1/0.
        df['is_ob'] = df['swing_ob'] 
        
        return df

    def find_fvgs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper for FVG detection.
        Expects: 'bull_fvg', 'bear_fvg' (boolean) for features.py
        """
        # Ensure we have the base FVG calc
        if 'fvg' not in df.columns:
            df = self._detect_fvgs(df)
            
        # Map to booleans for features.py compatibility
        df['bull_fvg'] = (df['fvg'] == 1)
        df['bear_fvg'] = (df['fvg'] == -1)
        
        return df