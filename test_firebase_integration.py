"""
Test Firebase Integration

Run this script to test the Firebase journal logger before using it with the live bot.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from firebase_logger import get_firebase_logger

def test_firebase_connection():
    """Test Firebase connection and basic operations"""
    print("="*70)
    print("Testing Firebase Journal Logger")
    print("="*70)
    
    # Get logger instance
    logger = get_firebase_logger()
    
    if not logger.enabled:
        print("\n❌ Firebase logging is not enabled!")
        print("\nPlease check:")
        print("1. ENABLE_JOURNAL_LOGGING=True in .env")
        print("2. FIREBASE_SERVICE_ACCOUNT_PATH is set correctly")
        print("3. FXTOOLS_USER_ID is set")
        print("4. Service account JSON file exists")
        return False
    
    print("\n✅ Firebase logger initialized successfully!")
    print(f"User ID: {logger.user_id}")
    
    # Test logging a trade entry
    print("\n" + "-"*70)
    print("Test 1: Logging a test trade entry")
    print("-"*70)
    
    try:
        journal_id = logger.log_trade_entry(
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.0850,
            lot_size=0.01,
            take_profit=1.0890,
            stop_loss=1.0840,
            strategy="Test Trade",
            notes="This is a test trade from the integration script",
            confidence=0.75
        )
        
        if journal_id:
            print(f"✅ Trade logged successfully!")
            print(f"Journal ID: {journal_id}")
            
            # Test updating the trade
            print("\n" + "-"*70)
            print("Test 2: Updating trade status to TP")
            print("-"*70)
            
            success = logger.update_trade_result(
                entry_id=journal_id,
                result_status="TP",
                exit_price=1.0890,
                profit_pips=40.0
            )
            
            if success:
                print("✅ Trade status updated successfully!")
            else:
                print("❌ Failed to update trade status")
            
            # Test retrieving trades
            print("\n" + "-"*70)
            print("Test 3: Retrieving user trades")
            print("-"*70)
            
            trades = logger.get_user_trades(limit=5)
            print(f"✅ Retrieved {len(trades)} trades")
            
            if trades:
                print("\nMost recent trade:")
                latest = trades[0]
                print(f"  Symbol: {latest.get('symbol')}")
                print(f"  Direction: {latest.get('direction')}")
                print(f"  Entry: {latest.get('entryPrice')}")
                print(f"  Status: {latest.get('resultStatus')}")
            
            # Clean up test trade
            print("\n" + "-"*70)
            print("Test 4: Cleaning up test trade")
            print("-"*70)
            
            if logger.delete_trade(journal_id):
                print("✅ Test trade deleted successfully!")
            else:
                print("❌ Failed to delete test trade")
            
            print("\n" + "="*70)
            print("✅ All tests passed! Firebase integration is working correctly.")
            print("="*70)
            print("\nYou can now enable journal logging in the live bot.")
            print("The bot will automatically log all trades to your FX-Tools journal.")
            return True
            
        else:
            print("❌ Failed to log trade entry")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_firebase_connection()
    sys.exit(0 if success else 1)
