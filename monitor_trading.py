"""
Monitor live trading performance and generate reports
"""

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv


class TradingMonitor:
    """Monitor and report on live trading performance"""
    
    def __init__(self):
        load_dotenv()
        self.login = int(os.getenv('MT5_LOGIN'))
        self.password = os.getenv('MT5_PASSWORD')
        self.server = os.getenv('MT5_SERVER')
        self.terminal_path = os.getenv('MT5_TERMINAL_PATH')
    
    def connect(self):
        """Connect to MT5"""
        if not mt5.initialize(path=self.terminal_path):
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        if not mt5.login(self.login, password=self.password, server=self.server):
            print(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        return True
    
    def get_account_summary(self):
        """Get account summary"""
        account_info = mt5.account_info()
        
        if account_info is None:
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'margin_level': account_info.margin_level
        }
    
    def get_open_positions(self):
        """Get all open positions"""
        positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        data = []
        for pos in positions:
            data.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'comment': pos.comment
            })
        
        return pd.DataFrame(data)
    
    def get_trade_history(self, days=7):
        """Get trade history for last N days"""
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()
        
        deals = mt5.history_deals_get(from_date, to_date)
        
        if deals is None:
            return pd.DataFrame()
        
        data = []
        for deal in deals:
            # Only include actual trades (not balance operations)
            if deal.entry == 1:  # Entry deal
                data.append({
                    'ticket': deal.ticket,
                    'time': pd.to_datetime(deal.time, unit='s'),
                    'symbol': deal.symbol,
                    'type': 'BUY' if deal.type == 0 else 'SELL',
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'comment': deal.comment
                })
        
        return pd.DataFrame(data)
    
    def generate_report(self):
        """Generate performance report"""
        if not self.connect():
            return
        
        print("\n" + "="*70)
        print("ML SMC LIVE TRADING - PERFORMANCE REPORT")
        print("="*70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Account summary
        account = self.get_account_summary()
        if account:
            print("\n📊 ACCOUNT SUMMARY")
            print("-" * 70)
            print(f"Balance:      ${account['balance']:,.2f}")
            print(f"Equity:       ${account['equity']:,.2f}")
            print(f"Profit:       ${account['profit']:,.2f}")
            print(f"Margin:       ${account['margin']:,.2f}")
            print(f"Margin Free:  ${account['margin_free']:,.2f}")
            if account['margin'] > 0:
                print(f"Margin Level: {account['margin_level']:.2f}%")
        
        # Open positions
        positions = self.get_open_positions()
        if len(positions) > 0:
            print("\n📈 OPEN POSITIONS")
            print("-" * 70)
            print(positions.to_string(index=False))
            print(f"\nTotal Open P&L: ${positions['profit'].sum():,.2f}")
        else:
            print("\n📈 OPEN POSITIONS: None")
        
        # Trade history
        history = self.get_trade_history(days=7)
        if len(history) > 0:
            print("\n📜 TRADE HISTORY (Last 7 Days)")
            print("-" * 70)
            
            wins = history[history['profit'] > 0]
            losses = history[history['profit'] < 0]
            
            print(f"Total Trades:  {len(history)}")
            print(f"Wins:          {len(wins)} ({len(wins)/len(history)*100:.1f}%)")
            print(f"Losses:        {len(losses)} ({len(losses)/len(history)*100:.1f}%)")
            print(f"Total Profit:  ${history['profit'].sum():,.2f}")
            
            if len(wins) > 0:
                print(f"Avg Win:       ${wins['profit'].mean():,.2f}")
            if len(losses) > 0:
                print(f"Avg Loss:      ${losses['profit'].mean():,.2f}")
            
            # Per-symbol breakdown
            print("\nPer-Symbol Performance:")
            symbol_stats = history.groupby('symbol').agg({
                'profit': ['count', 'sum', 'mean']
            }).round(2)
            print(symbol_stats)
        else:
            print("\n📜 TRADE HISTORY: No trades in last 7 days")
        
        print("\n" + "="*70)
        
        mt5.shutdown()


if __name__ == "__main__":
    monitor = TradingMonitor()
    monitor.generate_report()
