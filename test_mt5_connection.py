# Quick Test Script for MT5 Connection

import MetaTrader5 as mt5
from dotenv import load_dotenv
import os

load_dotenv()

print("Testing MT5 Connection...")
print("="*60)

# Initialize
if not mt5.initialize():
    print(f"Failed to initialize: {mt5.last_error()}")
    exit()

print("MT5 initialized successfully")

# Login
login = int(os.getenv('MT5_LOGIN'))
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')

if mt5.login(login, password=password, server=server):
    print(f"Logged in successfully")
    
    # Get account info
    account = mt5.account_info()
    print(f"Account: {account.login}")
    print(f"Balance: ${account.balance:.2f}")
    print(f"Server: {account.server}")
    
    # Test getting bars for EURUSDm
    print("\nTesting EURUSDm data retrieval...")
    
    # Enable symbol
    symbol_info = mt5.symbol_info("EURUSDm")
    if symbol_info is None:
        print("EURUSDm not found!")
    else:
        if not symbol_info.visible:
            print("Enabling EURUSDm...")
            mt5.symbol_select("EURUSDm", True)
        
        # Get bars
        rates = mt5.copy_rates_from_pos("EURUSDm", mt5.TIMEFRAME_M15, 0, 100)
        if rates is not None:
            print(f"Successfully retrieved {len(rates)} bars for EURUSDm")
            print(f"Latest close: {rates[-1]['close']}")
        else:
            print(f"Failed to get bars: {mt5.last_error()}")
else:
    print(f"Login failed: {mt5.last_error()}")

mt5.shutdown()
print("\n" + "="*60)
print("Test complete")
