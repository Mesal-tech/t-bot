import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
from dotenv import load_dotenv
import joblib
import logging
import asyncio
import threading

# Import features from same directory
try:
    from features import SMCFeatureGenerator
    from firebase_logger import get_firebase_logger
    from telegram_copy_trader import TelegramCopyTrader
except ImportError:
    # If running from parent directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from features import SMCFeatureGenerator
    from firebase_logger import get_firebase_logger
    from telegram_copy_trader import TelegramCopyTrader

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'mt5_trading.log')

# Create formatters
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
simple_formatter = logging.Formatter('%(asctime)s - %(message)s')

# File handler - logs everything at INFO level
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(detailed_formatter)

# Console handler - only logs WARNING and above (for trade events)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(simple_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)


class MT5Connector:
    """Handle MT5 connection and data retrieval"""
    
    def __init__(self):
        load_dotenv()
        self.login = int(os.getenv('MT5_LOGIN'))
        self.password = os.getenv('MT5_PASSWORD')
        self.server = os.getenv('MT5_SERVER')
        self.terminal_path = os.getenv('MT5_TERMINAL_PATH')
        self.connected = False
    
    def connect(self):
        """Connect to MT5"""
        # Check if terminal path exists
        if not os.path.exists(self.terminal_path):
            logger.error(f"MT5 terminal not found at: {self.terminal_path}")
            logger.info("Please update MT5_TERMINAL_PATH in .env file")
            return False
        
        # Initialize MT5
        logger.info(f"Initializing MT5 from: {self.terminal_path}")
        if not mt5.initialize(path=self.terminal_path):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            logger.info("Trying to initialize without path...")
            
            # Try without path
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed again: {mt5.last_error()}")
                logger.info("Please ensure MT5 terminal is installed and try running it manually first")
                return False
        
        # Login
        logger.info(f"Logging in to account {self.login} on server {self.server}")
        if not mt5.login(self.login, password=self.password, server=self.server):
            error = mt5.last_error()
            logger.error(f"MT5 login failed: {error}")
            logger.info("Please check your credentials in .env file")
            mt5.shutdown()
            return False
        
        self.connected = True
        account_info = mt5.account_info()
        if account_info:
            logger.info("Connected to MT5 successfully")
            logger.info(f"Account: {account_info.login}")
            logger.info(f"Balance: ${account_info.balance:.2f}")
            logger.info(f"Server: {account_info.server}")
        else:
            logger.warning("Connected but couldn't get account info")
        
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def get_bars(self, symbol, timeframe, count=1000):
        """Get historical bars"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        # Ensure symbol is selected and visible
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            logger.info(f"Enabling symbol {symbol}...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None
        
        # Get rates
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"Failed to get bars for {symbol}: {error}")
            logger.info(f"Trying alternative method...")
            
            # Try with datetime
            from datetime import datetime, timedelta
            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Still failed to get bars for {symbol}")
                return None
        
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
        
        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
    
    def get_symbol_info(self, symbol):
        """Get symbol information"""
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        return {
            'point': info.point,
            'digits': info.digits,
            'trade_contract_size': info.trade_contract_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step
        }
    
    def place_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment="ML SMC Bot"):
        """Place a market order"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add price for pending orders
        if price is not None:
            request["price"] = price
        
        # Add SL/TP
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Order placed: {symbol} {order_type} {volume} lots @ {result.price}")
        return result
    
    def get_positions(self, symbol=None):
        """Get open positions"""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return list(positions)
    
    def close_position(self, position):
        """Close a position"""
        symbol = position.symbol
        volume = position.volume
        position_type = position.type
        
        # Determine close order type
        if position_type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
        else:
            close_type = mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": position.ticket,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close by ML SMC Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position: {result.retcode}")
            return False
        
        logger.warning(f"POSITION CLOSED: {symbol} {volume} lots")
        return True


class LiveTradingBot:
    """Live trading bot using trained ML model"""
    
    def __init__(self, model_path, pairs, tp_pips=40, sl_pips=10, min_prob=0.4, 
                 risk_percent=1.0, max_positions=5, min_prob_increase=0.25,
                 enable_copy_trading=True, copy_risk_percent=2.0, copy_max_positions=25):
        """
        Args:
            model_path: Path to universal model (used as base/fallback)
            pairs: List of symbols to trade
            tp_pips: Take profit in pips
            sl_pips: Stop loss in pips
            min_prob: Minimum confidence threshold
            risk_percent: Risk per trade as % of balance
            max_positions: Maximum concurrent positions
            min_prob_increase: Minimum probability increase for additional positions on same pair
            enable_copy_trading: Enable Telegram copy trading
            copy_risk_percent: Risk per copy trade as % of balance
            copy_max_positions: Max positions from copy trading
        """
        self.mt5 = MT5Connector()
        self.pairs = pairs
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.min_prob = min_prob
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        self.min_prob_increase = min_prob_increase
        self.enable_copy_trading = enable_copy_trading
        self.copy_risk_percent = copy_risk_percent
        self.copy_max_positions = copy_max_positions
        
        self.feature_columns = None
        
        # Load models
        self.models = {}
        self.universal_model = None
        self.scalers = {}
        self.universal_scaler = None
        
        # Check if individual models exist
        models_dir = os.path.dirname(model_path)
        for pair in self.pairs:
            pair_model_path = os.path.join(models_dir, f'{pair}_model.pkl')
            if os.path.exists(pair_model_path):
                logger.info(f"Loading specific model for {pair} from {pair_model_path}")
                try:
                    model_data = joblib.load(pair_model_path)
                    self.models[pair] = model_data['model']
                    self.scalers[pair] = model_data.get('scaler', None)
                    # We assume feature columns are standard across models
                    if self.feature_columns is None:
                        self.feature_columns = model_data['feature_columns']
                except Exception as e:
                    logger.error(f"Failed to load model for {pair}: {e}")
        
        # Load universal model as fallback
        if os.path.exists(model_path):
            logger.info(f"Loading universal model from {model_path}")
            model_data = joblib.load(model_path)
            self.universal_model = model_data['model']
            self.universal_scaler = model_data.get('scaler', None)
            if self.feature_columns is None:
                self.feature_columns = model_data['feature_columns']
        
        if not self.models and not self.universal_model:
            logger.error("No models loaded! Please train models first.")
            raise ValueError("No models found")
            
        logger.info(f"Loaded {len(self.models)} specific models and universal fallback: {self.universal_model is not None}")
        
        # Feature generator
        self.feature_gen = SMCFeatureGenerator()
        
        # Firebase journal logger
        self.firebase_logger = get_firebase_logger()
        
        # Trading state
        self.running = False
        self.positions = {}  # Maps MT5 ticket to Firebase entry ID
        self.position_journal_ids = {}  # Maps MT5 ticket to Firebase document ID
        self.position_probabilities = {}  # Maps MT5 ticket to confidence probability
        self.position_sources = {}  # Maps MT5 ticket to source ('ML' or 'COPY')
        
        # Telegram copy trader
        self.copy_trader = None
        if self.enable_copy_trading:
            self.copy_trader = TelegramCopyTrader(
                mt5_connector=self.mt5,
                risk_percent=self.copy_risk_percent,
                max_positions=self.copy_max_positions
            )
    
    def start(self):
        """Start the trading bot"""
        if not self.mt5.connect():
            logger.error("Failed to connect to MT5")
            return
        
        self.running = True
        logger.info("="*70)
        logger.info("ML SMC LIVE TRADING BOT STARTED")
        logger.info("="*70)
        logger.info(f"Pairs: {', '.join(self.pairs)}")
        logger.info(f"TP/SL: {self.tp_pips}/{self.sl_pips} pips")
        logger.info(f"Min Probability: {self.min_prob}")
        logger.info(f"Risk per Trade: {self.risk_percent}%")
        logger.info(f"Max Positions: {self.max_positions}")
        logger.info(f"Min Prob Increase for Multi-Position: {self.min_prob_increase*100:.0f}%")
        if self.enable_copy_trading:
            logger.info(f"Copy Trading: ENABLED (Risk: {self.copy_risk_percent}%, Max: {self.copy_max_positions})")
        else:
            logger.info("Copy Trading: DISABLED")
        logger.info("="*70)
        
        try:
            # Start Telegram copy trader first if enabled (wait for authentication)
            if self.enable_copy_trading and self.copy_trader:
                logger.info("Starting Telegram copy trader (please complete authentication)...")
                telegram_thread = threading.Thread(target=self._run_telegram_copy_trader, daemon=True)
                telegram_thread.start()
                
                # Wait for Telegram authentication to complete
                import time
                logger.info("Waiting 5 minutes for Telegram authentication to complete...")
                logger.info("Please enter your phone number and verification code when prompted.")
                time.sleep(300)  # Wait 5 minutes for user authentication
                logger.info("Telegram copy trader running in background")
                logger.info("Starting ML SMC market monitoring...")
            
            # Run main ML trading loop
            self.run_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def _run_telegram_copy_trader(self):
        """Run Telegram copy trader in async event loop"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.copy_trader.start())
        except Exception as e:
            logger.error(f"Telegram copy trader error: {e}", exc_info=True)
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        
        # Stop Telegram copy trader
        if self.enable_copy_trading and self.copy_trader:
            try:
                asyncio.run(self.copy_trader.stop())
            except Exception as e:
                logger.error(f"Error stopping copy trader: {e}")
        
        self.mt5.disconnect()
        logger.info("Bot stopped")
    
    def run_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check each pair
                for symbol in self.pairs:
                    self.process_symbol(symbol)
                
                # Check existing positions
                self.monitor_positions()
                
                # Wait before next iteration (15 minutes for 15m timeframe)
                logger.info("Waiting for next bar...")
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)
    
    def process_symbol(self, symbol):
        """Process a single symbol for trading signals"""
        try:
            # Get historical data (10000 bars to match training)
            df = self.mt5.get_bars(symbol, mt5.TIMEFRAME_M15, count=10000)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return
            
            # Generate features
            features = self.feature_gen.generate_features(df)
            
            # Get latest bar features
            latest = features.iloc[-1]
            X = latest[self.feature_columns].values.reshape(1, -1)
            
            # Convert to float64 to handle mixed types
            try:
                X = X.astype(np.float64)
            except (ValueError, TypeError) as e:
                logger.error(f"Cannot convert features to numeric for {symbol}: {e}")
                logger.error(f"Feature columns: {self.feature_columns}")
                logger.error(f"Feature dtypes: {latest[self.feature_columns].dtypes.to_dict()}")
                return
            
            # Check for NaN
            if np.isnan(X).any():
                logger.warning(f"NaN in features for {symbol}, skipping")
                return
            
            # Select model and scaler
            if symbol in self.models:
                model = self.models[symbol]
                scaler = self.scalers.get(symbol)
                logger.debug(f"Using specific model for {symbol}")
            elif self.universal_model:
                model = self.universal_model
                scaler = self.universal_scaler
                logger.debug(f"Using universal model for {symbol}")
            else:
                logger.warning(f"No model available for {symbol}, skipping")
                return

            # Standardize features using the same scaler from training
            if scaler is not None:
                X = scaler.transform(X)
                logger.debug(f"{symbol} Features scaled - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
            
            # Predict
            probas = model.predict_proba(X)[0]
            prob_no_trade = probas[0]
            prob_buy = probas[1] if len(probas) > 1 else 0
            prob_sell = probas[2] if len(probas) > 2 else 0
            
            # Log predictions for transparency
            logger.info(f"{symbol} Predictions - NO_TRADE: {prob_no_trade:.3f}, BUY: {prob_buy:.3f}, SELL: {prob_sell:.3f}")
            
            # Determine signal
            signal = None
            confidence = 0
            
            if prob_buy >= self.min_prob and prob_buy > prob_sell:
                signal = 'BUY'
                confidence = prob_buy
            elif prob_sell >= self.min_prob and prob_sell > prob_buy:
                signal = 'SELL'
                confidence = prob_sell
            
            if signal:
                logger.info(f"{symbol}: {signal} signal (confidence: {confidence:.3f})")
                
                # Check if we already have positions for this symbol
                positions = self.mt5.get_positions(symbol)
                
                # Filter positions managed by this bot
                bot_positions = [pos for pos in positions if pos.magic == 234000]
                
                if len(bot_positions) > 0:
                    # Get probabilities of existing positions for this symbol
                    existing_probs = [
                        self.position_probabilities.get(pos.ticket, 0) 
                        for pos in bot_positions
                    ]
                    max_existing_prob = max(existing_probs) if existing_probs else 0
                    
                    # Check if new signal probability is significantly higher
                    prob_difference = confidence - max_existing_prob
                    
                    if prob_difference < self.min_prob_increase:
                        logger.info(
                            f"{symbol}: Already have {len(bot_positions)} position(s) "
                            f"(max prob: {max_existing_prob:.3f}). "
                            f"New signal prob ({confidence:.3f}) not high enough "
                            f"(needs +{self.min_prob_increase:.2f}, got +{prob_difference:.3f}). Skipping."
                        )
                        return
                    else:
                        logger.info(
                            f"{symbol}: New signal prob ({confidence:.3f}) is {prob_difference:.3f} "
                            f"higher than max existing ({max_existing_prob:.3f}). "
                            f"Opening additional position."
                        )
                
                # Check max positions
                all_positions = self.mt5.get_positions()
                if len(all_positions) >= self.max_positions:
                    logger.info(f"Max positions ({self.max_positions}) reached, skipping")
                    return
                
                # Place trade
                self.place_trade(symbol, signal, confidence)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    def place_trade(self, symbol, signal, confidence):
        """Place a trade"""
        try:
            # Get symbol info
            symbol_info = self.mt5.get_symbol_info(symbol)
            if symbol_info is None:
                return
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get tick for {symbol}")
                return
            
            # Calculate lot size based on risk
            account_info = mt5.account_info()
            balance = account_info.balance
            risk_amount = balance * (self.risk_percent / 100)
            
            # Pip value calculation
            point = symbol_info['point']
            pip_size = point * 10 if symbol_info['digits'] == 5 or symbol_info['digits'] == 3 else point
            
            # Calculate lot size
            # Risk = Lot Size * SL in pips * Pip Value
            # For forex: Pip Value = Lot Size * Contract Size * Pip Size
            sl_pips_value = self.sl_pips * pip_size
            lot_size = risk_amount / (self.sl_pips * 10)  # Simplified calculation
            
            # Round to volume step
            volume_step = symbol_info['volume_step']
            lot_size = round(lot_size / volume_step) * volume_step
            
            # Ensure within limits
            lot_size = max(symbol_info['volume_min'], min(lot_size, symbol_info['volume_max']))
            
            # Calculate SL/TP
            if signal == 'BUY':
                entry_price = tick.ask
                sl = entry_price - (self.sl_pips * pip_size)
                tp = entry_price + (self.tp_pips * pip_size)
                order_type = mt5.ORDER_TYPE_BUY
            else:  # SELL
                entry_price = tick.bid
                sl = entry_price + (self.sl_pips * pip_size)
                tp = entry_price - (self.tp_pips * pip_size)
                order_type = mt5.ORDER_TYPE_SELL
            
            # Place order
            logger.warning(f"PLACING {signal} ORDER: {symbol} {lot_size} lots @ {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
            
            result = self.mt5.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=lot_size,
                sl=sl,
                tp=tp,
                comment=f"ML SMC {signal} {confidence:.2f}"
            )
            
            if result:
                logger.warning(f"TRADE OPENED: Ticket {result.order}")
                
                # Store position probability and source for multi-position logic
                self.position_probabilities[result.order] = confidence
                self.position_sources[result.order] = 'ML'
                
                # Log to Firebase journal
                try:
                    journal_id = self.firebase_logger.log_trade_entry(
                        symbol=symbol,
                        direction=signal,
                        entry_price=entry_price,
                        lot_size=lot_size,
                        take_profit=tp,
                        stop_loss=sl,
                        strategy=os.getenv('JOURNAL_STRATEGY_NAME', 'ML SMC Bot'),
                        confidence=confidence
                    )
                    
                    if journal_id:
                        self.position_journal_ids[result.order] = journal_id
                        logger.info(f"Trade logged to journal: {journal_id}")
                except Exception as e:
                    logger.error(f"Failed to log trade to journal: {e}")
            
        except Exception as e:
            logger.error(f"Error placing trade for {symbol}: {e}", exc_info=True)
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        positions = self.mt5.get_positions()
        
        # Get current position tickets
        current_tickets = {pos.ticket for pos in positions if pos.magic == 234000}
        
        # Check for closed positions
        tracked_tickets = set(self.position_journal_ids.keys())
        closed_tickets = tracked_tickets - current_tickets
        
        # Update journal for closed positions
        for ticket in closed_tickets:
            if ticket in self.position_journal_ids:
                journal_id = self.position_journal_ids[ticket]
                try:
                    # Get position history to determine if TP or SL
                    # For now, we'll check the last known position state
                    # In a production system, you'd query MT5 history
                    logger.info(f"Position {ticket} closed, updating journal {journal_id}")
                    
                    # Since we can't easily determine TP vs SL after closure,
                    # we'll mark it as closed and let the user update manually if needed
                    # A better approach would be to track position state before closure
                    
                    # Remove from tracking
                    del self.position_journal_ids[ticket]
                    
                    # Clean up probability tracking
                    if ticket in self.position_probabilities:
                        del self.position_probabilities[ticket]
                    
                except Exception as e:
                    logger.error(f"Error updating journal for closed position {ticket}: {e}")
        
        if len(positions) == 0:
            return
        
        logger.info(f"Monitoring {len(positions)} open positions")
        
        for position in positions:
            # Check if managed by this bot
            if position.magic != 234000:
                continue
            
            # Log position status
            profit = position.profit
            symbol = position.symbol
            ticket = position.ticket
            logger.info(f"{symbol}: P&L = ${profit:.2f}")
            
            # Check if position is close to TP or SL
            if ticket in self.position_journal_ids:
                try:
                    journal_id = self.position_journal_ids[ticket]
                    
                    # Get position details
                    current_price = position.price_current
                    entry_price = position.price_open
                    tp = position.tp
                    sl = position.sl
                    
                    # Calculate distance to TP/SL
                    if position.type == 0:  # Buy
                        distance_to_tp = tp - current_price if tp > 0 else float('inf')
                        distance_to_sl = current_price - sl if sl > 0 else float('inf')
                    else:  # Sell
                        distance_to_tp = current_price - tp if tp > 0 else float('inf')
                        distance_to_sl = sl - current_price if sl > 0 else float('inf')
                    
                    # If very close to TP or SL, prepare for update
                    # This is a predictive update - actual update happens when position closes
                    
                except Exception as e:
                    logger.error(f"Error monitoring position {ticket}: {e}")


if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Import configuration from config_live.py
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import config_live
    
    # Start bot with configuration
    bot = LiveTradingBot(
        model_path=config_live.MODEL_PATH,
        pairs=config_live.PAIRS,
        tp_pips=config_live.TP_PIPS,
        sl_pips=config_live.SL_PIPS,
        min_prob=config_live.MIN_PROBABILITY,
        risk_percent=config_live.RISK_PERCENT,
        max_positions=config_live.MAX_POSITIONS,
        min_prob_increase=config_live.MIN_PROB_INCREASE,
        enable_copy_trading=config_live.ENABLE_COPY_TRADING,
        copy_risk_percent=config_live.COPY_TRADING_RISK_PERCENT,
        copy_max_positions=config_live.COPY_TRADING_MAX_POSITIONS
    )
    
    bot.start()
