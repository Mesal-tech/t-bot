"""
Telegram Signal Parser using NLP

Uses sentence transformers and regex to intelligently parse trading signals
from Telegram messages, extracting:
- Trade direction (BUY/SELL/CLOSE)
- Currency pair/symbol
- Entry price or market order
- Take profit levels
- Stop loss levels
- Order type (market/limit)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class TelegramSignalParser:
    """Parse trading signals from Telegram messages using NLP"""
    
    def __init__(self):
        # Load sentence transformer model for semantic understanding
        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Template signals for classification
        self.templates = {
            'buy_signal': "Buy EURUSD at 1.0850 take profit 1.0900 stop loss 1.0800",
            'sell_signal': "Sell GBPUSD at 1.2500 take profit 1.2450 stop loss 1.2550",
            'close_signal': "Close EURUSD trade now exit position",
            'update_tp': "Move take profit to breakeven adjust TP",
            'update_sl': "Trail stop loss move SL to entry",
            'market_order': "Market order buy USDJPY now",
            'limit_order': "Limit order sell EURJPY at 160.50"
        }
        
        # Encode templates once
        self.template_embeddings = {
            key: self.model.encode(text)
            for key, text in self.templates.items()
        }
        
        # Known currency pairs
        self.known_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'EURAUD', 'EURCHF', 'AUDJPY', 'GBPAUD',
            'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD',
            # With 'm' suffix for mini lots
            'EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 'AUDUSDm', 'USDCADm', 'NZDUSDm',
            'EURGBPm', 'EURJPYm', 'GBPJPYm', 'EURAUDm', 'EURCHFm', 'AUDJPYm', 'GBPAUDm',
            'XAUUSDm', 'XAGUSDm', 'BTCUSDm', 'ETHUSDm'
        ]
        
        logger.info("Signal parser initialized")
    
    def parse_message(self, message_text: str, reply_to_id: Optional[int] = None) -> Optional[Dict]:
        """
        Parse a Telegram message to extract trading signal
        
        Args:
            message_text: The message text
            reply_to_id: ID of message being replied to (for updates)
            
        Returns:
            Dict with signal info or None if not a valid signal
        """
        if not message_text or len(message_text.strip()) < 5:
            return None
        
        message_text = message_text.strip()
        
        # Classify message type
        signal_type = self._classify_message(message_text)
        
        if signal_type == 'unknown':
            return None
        
        # Extract information based on type
        if signal_type in ['buy_signal', 'sell_signal']:
            return self._parse_entry_signal(message_text, signal_type, reply_to_id)
        elif signal_type == 'close_signal':
            return self._parse_close_signal(message_text, reply_to_id)
        elif signal_type in ['update_tp', 'update_sl']:
            return self._parse_update_signal(message_text, signal_type, reply_to_id)
        
        return None
    
    def _classify_message(self, message_text: str) -> str:
        """Classify message type using sentence similarity"""
        # Encode message
        message_embedding = self.model.encode(message_text)
        
        # Calculate similarities
        similarities = {}
        for key, template_embedding in self.template_embeddings.items():
            similarity = np.dot(message_embedding, template_embedding) / (
                np.linalg.norm(message_embedding) * np.linalg.norm(template_embedding)
            )
            similarities[key] = similarity
        
        # Get best match
        best_match = max(similarities, key=similarities.get)
        best_score = similarities[best_match]
        
        # Threshold for classification
        if best_score < 0.4:
            return 'unknown'
        
        logger.debug(f"Message classified as '{best_match}' (score: {best_score:.3f})")
        return best_match
    
    def _parse_entry_signal(self, text: str, signal_type: str, reply_to_id: Optional[int]) -> Optional[Dict]:
        """Parse BUY/SELL entry signal"""
        direction = 'BUY' if signal_type == 'buy_signal' else 'SELL'
        
        # Extract pair
        pair = self._extract_pair(text)
        if not pair:
            logger.warning(f"Could not extract pair from: {text}")
            return None
        
        # Extract prices
        prices = self._extract_prices(text)
        
        # Determine entry price
        entry_price = None
        order_type = 'MARKET'
        
        # Check for limit order keywords
        if any(word in text.lower() for word in ['limit', 'pending', 'at']):
            order_type = 'LIMIT'
            # First price is likely entry for limit orders
            if prices:
                entry_price = prices[0]
        
        # Extract TP and SL
        tp_levels = self._extract_tp(text, prices)
        sl_level = self._extract_sl(text, prices)
        
        signal = {
            'type': 'ENTRY',
            'direction': direction,
            'symbol': pair,
            'entry_price': entry_price,
            'order_type': order_type,
            'tp_levels': tp_levels,
            'sl': sl_level,
            'reply_to_id': reply_to_id,
            'raw_message': text
        }
        
        logger.info(f"Parsed {direction} signal for {pair}: {signal}")
        return signal
    
    def _parse_close_signal(self, text: str, reply_to_id: Optional[int]) -> Optional[Dict]:
        """Parse CLOSE signal"""
        pair = self._extract_pair(text)
        
        signal = {
            'type': 'CLOSE',
            'symbol': pair,  # May be None if closing all
            'reply_to_id': reply_to_id,
            'raw_message': text
        }
        
        logger.info(f"Parsed CLOSE signal: {signal}")
        return signal
    
    def _parse_update_signal(self, text: str, signal_type: str, reply_to_id: Optional[int]) -> Optional[Dict]:
        """Parse TP/SL update signal"""
        update_type = 'TP' if signal_type == 'update_tp' else 'SL'
        
        pair = self._extract_pair(text)
        prices = self._extract_prices(text)
        
        # Check for breakeven
        is_breakeven = any(word in text.lower() for word in ['breakeven', 'be', 'entry'])
        
        signal = {
            'type': 'UPDATE',
            'update_type': update_type,
            'symbol': pair,
            'new_value': prices[0] if prices else None,
            'is_breakeven': is_breakeven,
            'reply_to_id': reply_to_id,
            'raw_message': text
        }
        
        logger.info(f"Parsed UPDATE signal: {signal}")
        return signal
    
    def _extract_pair(self, text: str) -> Optional[str]:
        """Extract currency pair from text"""
        text_upper = text.upper()
        
        # Check known pairs
        for pair in self.known_pairs:
            if pair.upper() in text_upper:
                return pair
        
        # Try regex for standard forex pairs (6 letters)
        match = re.search(r'\b([A-Z]{6}m?)\b', text_upper)
        if match:
            return match.group(1)
        
        # Try with slash separator
        match = re.search(r'\b([A-Z]{3})[/\s]([A-Z]{3})\b', text_upper)
        if match:
            return match.group(1) + match.group(2)
        
        return None
    
    def _extract_prices(self, text: str) -> List[float]:
        """Extract all price values from text"""
        # Match decimal numbers (e.g., 1.0850, 160.50, 1850.00)
        pattern = r'\b\d+\.\d+\b'
        matches = re.findall(pattern, text)
        
        prices = []
        for match in matches:
            try:
                prices.append(float(match))
            except ValueError:
                continue
        
        return prices
    
    def _extract_tp(self, text: str, prices: List[float]) -> List[float]:
        """Extract take profit levels"""
        tp_levels = []
        
        # Look for TP keywords
        tp_pattern = r'(?:tp|take profit|target)[\s:]*(\d+\.\d+)'
        matches = re.finditer(tp_pattern, text.lower())
        
        for match in matches:
            try:
                tp_levels.append(float(match.group(1)))
            except (ValueError, IndexError):
                continue
        
        # If no TP found with keywords, try to infer from prices
        if not tp_levels and len(prices) >= 2:
            # Assume second price might be TP
            tp_levels.append(prices[1])
        
        return tp_levels
    
    def _extract_sl(self, text: str, prices: List[float]) -> Optional[float]:
        """Extract stop loss level"""
        # Look for SL keywords
        sl_pattern = r'(?:sl|stop loss|stop)[\s:]*(\d+\.\d+)'
        match = re.search(sl_pattern, text.lower())
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # If no SL found with keywords, try to infer from prices
        if len(prices) >= 3:
            # Assume third price might be SL
            return prices[2]
        
        return None


if __name__ == "__main__":
    # Test the parser
    logging.basicConfig(level=logging.INFO)
    
    parser = TelegramSignalParser()
    
    # Test messages
    test_messages = [
        "🔥 BUY EURUSD @ 1.0850\nTP: 1.0900\nSL: 1.0800",
        "SELL GBPUSDm at 1.2500, TP 1.2450, SL 1.2550",
        "Close XAUUSD position now!",
        "Move TP to breakeven for USDJPY",
        "Trail SL to 1.0870",
        "Market buy BTCUSD now, TP 45000, SL 42000"
    ]
    
    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"Message: {msg}")
        result = parser.parse_message(msg)
        print(f"Result: {result}")
