# ML SMC Live Trading Configuration

## Trading Parameters
PAIRS = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'EURGBPm', 'GBPJPYm']
TP_PIPS = 40
SL_PIPS = 10
MIN_PROBABILITY = 0.40

## Risk Management
RISK_PERCENT = 1.0  # Risk 1% of balance per trade
MAX_POSITIONS = 5   # Maximum concurrent positions
MAX_DAILY_LOSS = 5.0  # Stop trading if daily loss exceeds 5%

## Model
MODEL_PATH = 'models/saved/universal_smc_model.pkl'

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
