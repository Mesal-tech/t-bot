"""
Firebase Journal Logger for ML SMC Trading Bot

Automatically logs trades to FX-Tools journal system using Firebase Firestore.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("firebase-admin not installed. Journal logging disabled.")

logger = logging.getLogger(__name__)


class FirebaseJournalLogger:
    """Logs trading activity to FX-Tools Firebase database"""
    
    def __init__(self):
        load_dotenv()
        self.enabled = os.getenv('ENABLE_JOURNAL_LOGGING', 'False').lower() == 'true'
        self.user_id = os.getenv('FXTOOLS_USER_ID')
        self.service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
        self.db = None
        self.initialized = False
        
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase Admin SDK not available. Journal logging disabled.")
            self.enabled = False
            return
        
        if self.enabled:
            self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            if not self.service_account_path:
                logger.error("FIREBASE_SERVICE_ACCOUNT_PATH not set in .env")
                self.enabled = False
                return
            
            if not self.user_id:
                logger.error("FXTOOLS_USER_ID not set in .env")
                self.enabled = False
                return
            
            if not os.path.exists(self.service_account_path):
                logger.error(f"Service account file not found: {self.service_account_path}")
                self.enabled = False
                return
            
            # Initialize Firebase Admin SDK
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.service_account_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            self.initialized = True
            logger.info("Firebase Journal Logger initialized successfully")
            logger.info(f"Logging trades for user: {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}", exc_info=True)
            self.enabled = False
    
    def log_trade_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        lot_size: float,
        take_profit: float,
        stop_loss: float,
        strategy: str = "ML SMC Bot",
        notes: str = "",
        confidence: float = 0.0
    ) -> Optional[str]:
        """
        Log a new trade entry to Firebase
        
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.enabled or not self.initialized:
            return None
        
        try:
            # Remove 'm' suffix from symbol for display (e.g., EURUSDm -> EURUSD)
            display_symbol = symbol.replace('m', '') if symbol.endswith('m') else symbol
            
            # Prepare trade entry data
            entry_data = {
                'userId': self.user_id,
                'symbol': display_symbol,
                'tradeDate': datetime.now().strftime('%Y-%m-%d'),
                'direction': 'LONG' if direction.upper() == 'BUY' else 'SHORT',
                'entryPrice': float(entry_price),
                'amount': float(lot_size),
                'takeProfit': float(take_profit),
                'stopLoss': float(stop_loss),
                'strategy': strategy,
                'notes': notes or f"Confidence: {confidence:.2%}",
                'resultStatus': 'ONGOING',
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            # Add to Firestore
            doc_ref = self.db.collection('journalEntries').add(entry_data)
            doc_id = doc_ref[1].id
            
            logger.info(f"Trade logged to Firebase: {display_symbol} {direction} @ {entry_price} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to log trade entry: {e}", exc_info=True)
            return None
    
    def update_trade_result(
        self,
        entry_id: str,
        result_status: str,
        exit_price: Optional[float] = None,
        profit_pips: Optional[float] = None,
        profit: Optional[float] = None
    ) -> bool:
        """
        Update trade result when it closes (TP or SL)
        
        Args:
            entry_id: Firebase document ID
            result_status: 'TP', 'SL', or 'CLOSED'
            exit_price: Actual exit price (optional)
            profit_pips: Profit in pips (optional)
            profit: Actual profit in currency (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.initialized:
            return False
        
        try:
            update_data = {
                'resultStatus': result_status,
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            # Add optional fields
            if exit_price is not None:
                update_data['exitPrice'] = float(exit_price)
            if profit_pips is not None:
                update_data['profitPips'] = float(profit_pips)
            if profit is not None:
                update_data['profit'] = float(profit)
            
            # Update document
            self.db.collection('journalEntries').document(entry_id).update(update_data)
            
            logger.info(f"Trade result updated: {entry_id} -> {result_status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update trade result: {e}", exc_info=True)
            return False
    
    def get_user_trades(self, limit: int = 100) -> list:
        """
        Retrieve user's trade history from Firebase
        
        Args:
            limit: Maximum number of trades to retrieve
        
        Returns:
            List of trade dictionaries
        """
        if not self.enabled or not self.initialized:
            return []
        
        try:
            trades_ref = self.db.collection('journalEntries')
            query = trades_ref.where('userId', '==', self.user_id).limit(limit)
            docs = query.stream()
            
            trades = []
            for doc in docs:
                trade_data = doc.to_dict()
                trade_data['id'] = doc.id
                trades.append(trade_data)
            
            logger.info(f"Retrieved {len(trades)} trades from Firebase")
            return trades
            
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {e}", exc_info=True)
            return []
    
    def delete_trade(self, entry_id: str) -> bool:
        """
        Delete a trade entry from Firebase
        
        Args:
            entry_id: Firebase document ID
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.initialized:
            return False
        
        try:
            self.db.collection('journalEntries').document(entry_id).delete()
            logger.info(f"Trade deleted: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete trade: {e}", exc_info=True)
            return False


# Singleton instance
_firebase_logger = None

def get_firebase_logger() -> FirebaseJournalLogger:
    """Get or create Firebase logger singleton"""
    global _firebase_logger
    if _firebase_logger is None:
        _firebase_logger = FirebaseJournalLogger()
    return _firebase_logger
