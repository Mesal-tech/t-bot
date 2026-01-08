# ML SMC Live Trading Configuration
import os
from dotenv import load_dotenv

load_dotenv()

## Trading Parameters
# Top 5 pairs with >50% win rate (ordered by performance)
# XAUUSDm: 94.4% WR, GBPJPYm: 56.1% WR, USDCADm: 54.1% WR, GBPUSDm: 52.6% WR, USDJPYm: 52.1% WR
PAIRS = [
    'XAUUSDm', 'GBPJPYm', 'USDCADm', 'GBPUSDm', 'USDJPYm', 
    'EURUSDm', 'AUDUSDm', 'NZDUSDm', 'EURJPYm', 'EURGBPm',
    'USDCHFm', 'AUDJPYm', 'BTCUSDm', 'ETHUSDm'
]
TP_PIPS = 30
SL_PIPS = 10
MIN_PROBABILITY = 0.40

## Risk Management
RISK_PERCENT = float(os.getenv('RISK_PERCENT', '1.0'))  # Risk % of balance per trade (configurable via .env)
MAX_POSITIONS = 10   # Maximum concurrent positions across all pairs
MAX_DAILY_LOSS = 20.0  # Stop trading if daily loss exceeds 20%
MIN_PROB_INCREASE = 0.10  # Minimum probability increase (10%) required for additional positions on same pair

## Model
MODEL_PATH = 'models/saved/EURUSD_model.pkl'

## MT5 Settings (from .env)
# MT5_LOGIN
# MT5_PASSWORD
# MT5_SERVER
# MT5_TERMINAL_PATH

## Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = 'logs/mt5_trading.log'

## Timeframe
TIMEFRAME = 'M15'  # 15-minute bars
CHECK_INTERVAL = 60  # Check for signals every 60 seconds

## Safety Features
DRY_RUN = False  # Set to True to test without placing real orders
TRADING_HOURS = {
    'start': 0,  # 00:00 UTC
    'end': 23    # 23:00 UTC (trade 24/7)
}

## FX-Tools Integration (from .env)
# ENABLE_JOURNAL_LOGGING=True
# FIREBASE_SERVICE_ACCOUNT_PATH=path/to/serviceAccountKey.json
# FXTOOLS_USER_ID=your_clerk_user_id
JOURNAL_STRATEGY_NAME = "ML SMC Bot"  # Strategy name in journal
