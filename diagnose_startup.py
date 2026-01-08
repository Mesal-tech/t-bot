
import sys
import os

print("1. Starting diagnostic...")
try:
    import pandas as pd
    print("2. Pandas imported")
except ImportError as e:
    print(f"FAILED: Pandas - {e}")

try:
    import MetaTrader5 as mt5
    print("3. MT5 imported")
except ImportError as e:
    print(f"FAILED: MT5 - {e}")

sys.path.append('src')

print("4. Importing feature dependencies...")
try:
    import smc_corrected
    print("5. smc_corrected imported")
except ImportError as e:
    print(f"FAILED: smc_corrected - {e}")
except Exception as e:
    print(f"CRASH: smc_corrected - {e}")

try:
    import features
    print("6. features imported")
except ImportError as e:
    print(f"FAILED: features - {e}")
except Exception as e:
    print(f"CRASH: features - {e}")

try:
    import firebase_logger
    print("7. firebase_logger imported")
except ImportError as e:
    print(f"FAILED: firebase_logger - {e}")
except Exception as e:
    print(f"CRASH: firebase_logger - {e}")

print("8. Importing live_trading...")
try:
    import live_trading
    print("9. live_trading imported successfully")
except ImportError as e:
    print(f"FAILED: live_trading - {e}")
except Exception as e:
    print(f"CRASH: live_trading - {e}")

print("DIAGNOSTIC COMPLETE")
