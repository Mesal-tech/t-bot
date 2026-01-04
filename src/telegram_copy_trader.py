"""
Telegram Copy Trader

Monitors Telegram channels for trading signals and executes them via MT5.
Uses NLP-based signal parser to understand messages and track signals via message IDs.
"""

import asyncio
import logging
import os
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from telethon import TelegramClient, events
from telethon.tl.types import Message

from telegram_signal_parser import TelegramSignalParser

load_dotenv()

logger = logging.getLogger(__name__)


class TelegramCopyTrader:
    """Copy trades from Telegram channels"""
    
    def __init__(self, mt5_connector, risk_percent=2.0, max_positions=25):
        """
        Args:
            mt5_connector: MT5Connector instance for trade execution
            risk_percent: Risk per copy trade as % of balance
            max_positions: Max positions from copy trading
        """
        self.mt5 = mt5_connector
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        
        # Telegram credentials
        self.api_id = int(os.getenv('TG_API_ID'))
        self.api_hash = os.getenv('TG_API_HASH')
        self.session_name = os.getenv('TG_SESSION', 'copybot_session')
        
        # Parse channels from env (comma-separated)
        channels_str = os.getenv('TG_CHANNEL', '')
        self.channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]
        
        if not self.channels:
            logger.warning("No Telegram channels configured in TG_CHANNEL")
        
        # Initialize signal parser
        self.parser = TelegramSignalParser()
        
        # Track active signals: message_id -> signal_info
        self.active_signals = {}
        
        # Track MT5 tickets: message_id -> mt5_ticket
        self.signal_tickets = {}
        
        # Telegram client
        self.client = None
        self.running = False
        
        logger.info(f"Telegram Copy Trader initialized for channels: {self.channels}")
    
    async def start(self):
        """Start the Telegram copy trader"""
        if not self.channels:
            logger.warning("No channels to monitor, copy trading disabled")
            return
        
        logger.info("Starting Telegram Copy Trader...")
        
        # Create Telegram client
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        
        # Register message handler
        @self.client.on(events.NewMessage(chats=self.channels))
        async def handle_message(event):
            await self._handle_new_message(event)
        
        # Start client
        await self.client.start()
        logger.info("Telegram client started")
        
        # Get channel info
        for channel in self.channels:
            try:
                entity = await self.client.get_entity(channel)
                # Strip emojis from channel name to avoid encoding issues
                channel_name = ''.join(c for c in entity.title if ord(c) < 0x10000)
                logger.info(f"Monitoring channel: {channel_name} ({channel})")
            except Exception as e:
                logger.error(f"Failed to get channel {channel}: {e}")
        
        self.running = True
        logger.info("Telegram Copy Trader is running")
        
        # Keep running
        await self.client.run_until_disconnected()
    
    async def stop(self):
        """Stop the Telegram copy trader"""
        logger.info("Stopping Telegram Copy Trader...")
        self.running = False
        
        if self.client:
            await self.client.disconnect()
        
        logger.info("Telegram Copy Trader stopped")
    
    async def _handle_new_message(self, event):
        """Handle new message from Telegram channel"""
        try:
            message: Message = event.message
            text = message.message
            
            if not text:
                return
            
            logger.info(f"New message from {event.chat.title}: {text[:100]}...")
            
            # Get reply_to_msg_id if this is a reply
            reply_to_id = message.reply_to_msg_id if message.reply_to else None
            
            # Parse the message
            signal = self.parser.parse_message(text, reply_to_id)
            
            if not signal:
                logger.debug("Message is not a valid trading signal")
                return
            
            # Handle based on signal type
            if signal['type'] == 'ENTRY':
                await self._handle_entry_signal(message.id, signal)
            elif signal['type'] == 'CLOSE':
                await self._handle_close_signal(message.id, signal)
            elif signal['type'] == 'UPDATE':
                await self._handle_update_signal(message.id, signal)
        
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _handle_entry_signal(self, message_id: int, signal: Dict):
        """Handle entry signal (BUY/SELL)"""
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry_price = signal.get('entry_price')
            order_type = signal.get('order_type', 'MARKET')
            tp_levels = signal.get('tp_levels', [])
            sl = signal.get('sl')
            
            logger.info(f"Processing {direction} signal for {symbol}")
            
            # Check if we already have too many copy trade positions
            copy_positions = len([t for t in self.signal_tickets.values() if t])
            if copy_positions >= self.max_positions:
                logger.warning(f"Max copy trading positions ({self.max_positions}) reached, skipping")
                return
            
            # For market orders, execute immediately
            if order_type == 'MARKET':
                ticket = self._execute_market_order(symbol, direction, tp_levels, sl)
                if ticket:
                    self.active_signals[message_id] = signal
                    self.signal_tickets[message_id] = ticket
                    logger.warning(f"🔔 COPY TRADE OPENED: Ticket {ticket} from Telegram")
            else:
                # For limit orders, would need to place pending order
                # For now, log and skip (can be implemented later)
                logger.info(f"Limit order detected at {entry_price}, skipping (not implemented)")
        
        except Exception as e:
            logger.error(f"Error handling entry signal: {e}", exc_info=True)
    
    async def _handle_close_signal(self, message_id: int, signal: Dict):
        """Handle close signal"""
        try:
            symbol = signal.get('symbol')
            reply_to_id = signal.get('reply_to_id')
            
            # If replying to a message, close that specific signal
            if reply_to_id and reply_to_id in self.signal_tickets:
                ticket = self.signal_tickets[reply_to_id]
                if ticket:
                    logger.warning(f"✅ COPY TRADE CLOSED: Position {ticket} (reply to message {reply_to_id})")
                    self._close_position_by_ticket(ticket)
                    del self.signal_tickets[reply_to_id]
                    if reply_to_id in self.active_signals:
                        del self.active_signals[reply_to_id]
            
            # If symbol specified, close all positions for that symbol
            elif symbol:
                logger.warning(f"✅ CLOSING ALL {symbol} COPY TRADES")
                self._close_positions_by_symbol(symbol)
            
            # Otherwise, close all copy trade positions
            else:
                logger.warning("✅ CLOSING ALL COPY TRADES")
                for ticket in list(self.signal_tickets.values()):
                    if ticket:
                        self._close_position_by_ticket(ticket)
                self.signal_tickets.clear()
                self.active_signals.clear()
        
        except Exception as e:
            logger.error(f"Error handling close signal: {e}", exc_info=True)
    
    async def _handle_update_signal(self, message_id: int, signal: Dict):
        """Handle TP/SL update signal"""
        try:
            update_type = signal.get('update_type')
            new_value = signal.get('new_value')
            is_breakeven = signal.get('is_breakeven', False)
            reply_to_id = signal.get('reply_to_id')
            
            if not reply_to_id or reply_to_id not in self.signal_tickets:
                logger.warning("Update signal without valid reply_to_id, skipping")
                return
            
            ticket = self.signal_tickets[reply_to_id]
            if not ticket:
                return
            
            logger.info(f"Updating {update_type} for position {ticket}")
            
            # Get position
            import MetaTrader5 as mt5
            positions = mt5.positions_get(ticket=ticket)
            
            if not positions or len(positions) == 0:
                logger.warning(f"Position {ticket} not found")
                return
            
            position = positions[0]
            
            # Calculate new TP/SL
            if is_breakeven:
                new_value = position.price_open
            
            # Modify position
            if update_type == 'TP':
                self._modify_position(position, tp=new_value)
            elif update_type == 'SL':
                self._modify_position(position, sl=new_value)
        
        except Exception as e:
            logger.error(f"Error handling update signal: {e}", exc_info=True)
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol by adding 'm' suffix if not present"""
        # List of symbols that typically need 'm' suffix for MT5
        if symbol and not symbol.endswith('m'):
            # Add 'm' suffix for standard forex pairs and metals
            if len(symbol) == 6 or symbol.startswith('XAU') or symbol.startswith('XAG'):
                return symbol + 'm'
        return symbol
    
    def _execute_market_order(self, symbol: str, direction: str, tp_levels: list, sl: Optional[float]) -> Optional[int]:
        """Execute market order via MT5"""
        try:
            import MetaTrader5 as mt5
            
            # Normalize symbol (add 'm' suffix if needed)
            symbol = self._normalize_symbol(symbol)
            logger.info(f"Normalized symbol: {symbol}")
            
            # Get symbol info
            symbol_info = self.mt5.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick for {symbol}")
                return None
            
            # Calculate lot size based on risk
            account_info = mt5.account_info()
            balance = account_info.balance
            risk_amount = balance * (self.risk_percent / 100)
            
            logger.info(f"Balance: ${balance:.2f}, Risk Amount (10%): ${risk_amount:.2f}")
            
            # Calculate lot size based on SL
            lot_size = 0.01  # Default minimum
            if sl:
                # Calculate based on SL distance
                entry_price = tick.ask if direction == 'BUY' else tick.bid
                sl_distance = abs(entry_price - sl)
                point = symbol_info['point']
                
                # Calculate SL in pips
                pip_size = point * 10 if symbol_info['digits'] == 5 or symbol_info['digits'] == 3 else point
                sl_pips = sl_distance / pip_size
                
                logger.info(f"SL Distance: {sl_distance:.5f}, SL in pips: {sl_pips:.1f}")
                
                if sl_pips > 0:
                    # For forex pairs: Risk = Lot Size * SL in pips * Pip Value
                    # Pip Value ≈ 10 for standard lots on most pairs
                    # Simplified: lot_size = risk_amount / (sl_pips * 10)
                    # More accurate for forex: account for contract size
                    contract_size = symbol_info.get('trade_contract_size', 100000)
                    pip_value_per_lot = (pip_size * contract_size) / entry_price if 'USD' not in symbol[-3:] else (pip_size * contract_size)
                    
                    # Calculate lot size: risk_amount = lot_size * sl_pips * pip_value_per_lot
                    lot_size = risk_amount / (sl_pips * pip_value_per_lot)
                    
                    logger.info(f"Calculated lot size before rounding: {lot_size:.4f}")
                    
                    # Round to volume step
                    volume_step = symbol_info['volume_step']
                    lot_size = round(lot_size / volume_step) * volume_step
                    
                    # Ensure within limits
                    lot_size = max(symbol_info['volume_min'], min(lot_size, symbol_info['volume_max']))
                    
                    logger.info(f"Final lot size: {lot_size:.2f}")
            else:
                # No SL provided, use fixed percentage of balance
                # Assume 100 pip risk for calculation
                logger.warning("No SL provided, using minimum lot size")
                lot_size = symbol_info['volume_min']
            
            # Use first TP if multiple
            tp = tp_levels[0] if tp_levels else None
            
            # Determine order type
            order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # Place order
            logger.info(f"Placing {direction} order: {symbol} {lot_size} lots, TP: {tp}, SL: {sl}")
            
            result = self.mt5.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=lot_size,
                sl=sl,
                tp=tp,
                comment="Telegram Copy Trade"
            )
            
            if result:
                return result.order
            
            return None
        
        except Exception as e:
            logger.error(f"Error executing market order: {e}", exc_info=True)
            return None
    
    def _close_position_by_ticket(self, ticket: int):
        """Close position by ticket number"""
        try:
            import MetaTrader5 as mt5
            
            positions = mt5.positions_get(ticket=ticket)
            if positions and len(positions) > 0:
                self.mt5.close_position(positions[0])
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
    
    def _close_positions_by_symbol(self, symbol: str):
        """Close all positions for a symbol"""
        try:
            # Normalize symbol
            symbol = self._normalize_symbol(symbol)
            
            # Close positions that match this symbol and are from copy trading
            for msg_id, ticket in list(self.signal_tickets.items()):
                if ticket:
                    import MetaTrader5 as mt5
                    positions = mt5.positions_get(ticket=ticket)
                    if positions and len(positions) > 0:
                        pos = positions[0]
                        if pos.symbol == symbol:
                            self.mt5.close_position(pos)
                            del self.signal_tickets[msg_id]
                            if msg_id in self.active_signals:
                                del self.active_signals[msg_id]
        except Exception as e:
            logger.error(f"Error closing positions for {symbol}: {e}")
    
    def _modify_position(self, position, tp=None, sl=None):
        """Modify position TP/SL"""
        try:
            import MetaTrader5 as mt5
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": sl if sl is not None else position.sl,
                "tp": tp if tp is not None else position.tp,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to modify position: {result.retcode}")
            else:
                logger.info(f"Position {position.ticket} modified successfully")
        
        except Exception as e:
            logger.error(f"Error modifying position: {e}")


if __name__ == "__main__":
    # Test the copy trader
    logging.basicConfig(level=logging.INFO)
    
    # This would normally be run with MT5 connector
    print("Telegram Copy Trader module loaded")
