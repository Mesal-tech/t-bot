"""
Trading Sessions Detection
Based on reference implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Tuple


class TradingSessions:
    """
    Detect trading sessions (Sydney, Tokyo, London, NY, Kill Zones)
    """
    
    # Session definitions (UTC time)
    SESSIONS = {
        "Sydney": {"start": "21:00", "end": "06:00"},
        "Tokyo": {"start": "00:00", "end": "09:00"},
        "London": {"start": "07:00", "end": "16:00"},
        "New York": {"start": "13:00", "end": "22:00"},
        "Asian kill zone": {"start": "00:00", "end": "04:00"},
        "London open kill zone": {"start": "06:00", "end": "09:00"},
        "New York kill zone": {"start": "11:00", "end": "14:00"},
        "London close kill zone": {"start": "14:00", "end": "16:00"},
    }
    
    @staticmethod
    def detect_session(df: pd.DataFrame, 
                       session: str,
                       start_time: str = "",
                       end_time: str = "",
                       time_zone: str = "UTC") -> pd.DataFrame:
        """
        Detect which candles are within a trading session
        
        Args:
            df: DataFrame with timestamp index
            session: Session name or "Custom"
            start_time: Start time for custom session (HH:MM)
            end_time: End time for custom session (HH:MM)
            time_zone: Timezone of the data (UTC, GMT+X, etc.)
            
        Returns:
            DataFrame with session columns
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Handle timezone
        if time_zone != "UTC":
            time_zone = time_zone.replace("GMT", "Etc/GMT")
            time_zone = time_zone.replace("UTC", "Etc/GMT")
            try:
                df.index = df.index.tz_localize(time_zone).tz_convert("UTC")
            except:
                pass  # Already has timezone
        
        # Get session times
        if session == "Custom":
            if not start_time or not end_time:
                raise ValueError("Custom session requires start_time and end_time")
            session_start = datetime.strptime(start_time, "%H:%M").time()
            session_end = datetime.strptime(end_time, "%H:%M").time()
        else:
            if session not in TradingSessions.SESSIONS:
                raise ValueError(f"Unknown session: {session}")
            session_start = datetime.strptime(
                TradingSessions.SESSIONS[session]["start"], "%H:%M"
            ).time()
            session_end = datetime.strptime(
                TradingSessions.SESSIONS[session]["end"], "%H:%M"
            ).time()
        
        # Detect active session
        n = len(df)
        active = np.zeros(n, dtype=np.int32)
        session_high = np.zeros(n, dtype=np.float32)
        session_low = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            current_time = df.index[i].time()
            
            # Check if in session (handle overnight sessions)
            if session_start < session_end:
                in_session = session_start <= current_time <= session_end
            else:  # Overnight session
                in_session = session_start <= current_time or current_time <= session_end
            
            if in_session:
                active[i] = 1
                session_high[i] = max(
                    df['high'].iloc[i],
                    session_high[i - 1] if i > 0 and active[i - 1] else 0
                )
                session_low[i] = min(
                    df['low'].iloc[i],
                    session_low[i - 1] if i > 0 and active[i - 1] and session_low[i - 1] > 0 else float('inf')
                )
        
        df[f'{session.lower().replace(" ", "_")}_active'] = active
        df[f'{session.lower().replace(" ", "_")}_high'] = np.where(
            active == 1, session_high, np.nan
        )
        df[f'{session.lower().replace(" ", "_")}_low'] = np.where(
            active == 1, session_low, np.nan
        )
        
        return df
    
    @staticmethod
    def add_all_sessions(df: pd.DataFrame, time_zone: str = "UTC") -> pd.DataFrame:
        """
        Add all major sessions to DataFrame
        
        Args:
            df: DataFrame with OHLC data
            time_zone: Timezone of the data
            
        Returns:
            DataFrame with all session columns
        """
        for session_name in ["Sydney", "Tokyo", "London", "New York"]:
            df = TradingSessions.detect_session(df, session_name, time_zone=time_zone)
        
        return df
    
    @staticmethod
    def add_kill_zones(df: pd.DataFrame, time_zone: str = "UTC") -> pd.DataFrame:
        """
        Add ICT kill zones to DataFrame
        
        Args:
            df: DataFrame with OHLC data
            time_zone: Timezone of the data
            
        Returns:
            DataFrame with kill zone columns
        """
        kill_zones = [
            "Asian kill zone",
            "London open kill zone", 
            "New York kill zone",
            "London close kill zone"
        ]
        
        for kz in kill_zones:
            df = TradingSessions.detect_session(df, kz, time_zone=time_zone)
        
        return df


def add_session_features(df: pd.DataFrame, time_zone: str = "UTC") -> pd.DataFrame:
    """
    Convenience function to add all session features
    
    Args:
        df: DataFrame with OHLC data and timestamp
        time_zone: Timezone of the data
        
    Returns:
        DataFrame with session features
    """
    df = TradingSessions.add_all_sessions(df, time_zone)
    df = TradingSessions.add_kill_zones(df, time_zone)
    return df
