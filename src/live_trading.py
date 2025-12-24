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

# Import features from same directory
try:
    from features import SMCFeatureGenerator
except ImportError:
    # If running from parent directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from features import SMCFeatureGenerator

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'mt5_trading.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
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
        
        logger.info(f"Position closed: {symbol} {volume} lots")
        return True


class LiveTradingBot:
    """Live trading bot using trained ML model"""
    
    def __init__(self, model_path, pairs, tp_pips=40, sl_pips=10, min_prob=0.4, 
                 risk_percent=1.0, max_positions=5):
        """
        Args:
            model_path: Path to trained model
            pairs: List of symbols to trade
            tp_pips: Take profit in pips
            sl_pips: Stop loss in pips
            min_prob: Minimum confidence threshold
            risk_percent: Risk per trade as % of balance
            max_positions: Maximum concurrent positions
        """
        self.mt5 = MT5Connector()
        self.pairs = pairs
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.min_prob = min_prob
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data.get('scaler', None)  # Load scaler
        
        if self.scaler is None:
            logger.warning("No scaler found in model! Features will NOT be standardized.")
            logger.warning("Consider retraining the model with the updated training.py")
        else:
            logger.info("Scaler loaded successfully")
        
        # Feature generator
        self.feature_gen = SMCFeatureGenerator()
        
        # Trading state
        self.running = False
        self.positions = {}
    
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
        logger.info("="*70)
        
        try:
            self.run_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
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
            
            # Standardize features using the same scaler from training
            if self.scaler is not None:
                X = self.scaler.transform(X)
                logger.debug(f"{symbol} Features scaled - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
            
            # Predict
            probas = self.model.predict_proba(X)[0]
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
                
                # Check if we already have a position
                positions = self.mt5.get_positions(symbol)
                if len(positions) > 0:
                    logger.info(f"{symbol}: Already have position, skipping")
                    return
                
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
            logger.info(f"Placing {signal} order: {symbol} {lot_size} lots @ {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
            
            result = self.mt5.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=lot_size,
                sl=sl,
                tp=tp,
                comment=f"ML SMC {signal} {confidence:.2f}"
            )
            
            if result:
                logger.info(f"Trade placed successfully: Ticket {result.order}")
            
        except Exception as e:
            logger.error(f"Error placing trade for {symbol}: {e}", exc_info=True)
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        positions = self.mt5.get_positions()
        
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
            logger.info(f"{symbol}: P&L = ${profit:.2f}")


if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configuration
    MODEL_PATH = 'models/saved/universal_smc_model.pkl'
    PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Start bot
    bot = LiveTradingBot(
        model_path=MODEL_PATH,
        pairs=PAIRS,
        tp_pips=40,
        sl_pips=10,
        min_prob=0.4,
        risk_percent=1.0,
        max_positions=5
    )
    
    bot.start()
