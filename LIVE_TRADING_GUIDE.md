# ML SMC Trading Bot - Live Trading Guide

## 🚀 Quick Start

### 1. Test Connection
```bash
python -c "import MetaTrader5 as mt5; from dotenv import load_dotenv; import os; load_dotenv(); print('MT5 installed:', mt5.__version__ if hasattr(mt5, '__version__') else 'Yes')"
```

### 2. Start Paper Trading
```bash
# Start with default settings
python start_live_trading.py

# Test mode (no real orders)
python start_live_trading.py --dry-run

# Custom settings
python start_live_trading.py --pairs EURUSD GBPUSD --risk-percent 0.5 --max-positions 3
```

### 3. Monitor Performance
```bash
# Generate performance report
python monitor_trading.py
```

---

## 📋 Configuration

Edit `config_live.py` to customize:

```python
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'GBPJPY']
TP_PIPS = 30
SL_PIPS = 10
MIN_PROBABILITY = 0.65
RISK_PERCENT = 1.0
MAX_POSITIONS = 5
```

---

## 🎯 How It Works

1. **Data Collection**: Bot fetches last 1000 bars of 15-minute data from MT5
2. **Feature Generation**: Applies SMC analysis to generate 51 features
3. **Prediction**: Uses trained ML model to predict BUY/SELL/NO_TRADE
4. **Signal Filtering**: Only trades when confidence ≥ 65%
5. **Risk Management**: Calculates lot size based on 1% risk per trade
6. **Order Execution**: Places market order with 30-pip TP and 10-pip SL
7. **Monitoring**: Tracks open positions and logs all activity

---

## ⚙️ Command Line Options

```bash
python start_live_trading.py [OPTIONS]

Options:
  --pairs EURUSD GBPUSD      # Currency pairs to trade
  --tp-pips 30               # Take profit in pips
  --sl-pips 10               # Stop loss in pips
  --min-prob 0.65            # Minimum confidence threshold
  --risk-percent 1.0         # Risk per trade (% of balance)
  --max-positions 5          # Maximum concurrent positions
  --dry-run                  # Test mode (no real orders)
```

---

## 📊 Monitoring

### Real-Time Logs
```bash
# View live logs
tail -f logs/mt5_trading.log   # Linux/Mac
Get-Content logs/mt5_trading.log -Wait  # Windows PowerShell
```

### Performance Report
```bash
python monitor_trading.py
```

Shows:
- Account balance, equity, profit
- Open positions with P&L
- Trade history (last 7 days)
- Win rate and profit factor
- Per-symbol performance

---

## 🛡️ Safety Features

### Built-in Protections
- ✅ Maximum positions limit (default: 5)
- ✅ Risk management (1% per trade)
- ✅ Confidence threshold (65% minimum)
- ✅ Duplicate position prevention
- ✅ Automatic SL/TP on every trade

### Manual Controls
- **Stop Bot**: Press `Ctrl+C`
- **Close All Positions**: Use MT5 terminal
- **Pause Trading**: Stop the bot, positions remain open

---

## ⚠️ Important Warnings

> [!CAUTION]
> **This is Paper Trading**
> 
> You're using a demo account. Results will differ in live trading due to:
> - Slippage
> - Spread variations
> - Order execution delays
> - Market conditions

> [!WARNING]
> **Monitor Closely**
> 
> For the first 30 days:
> - Check performance daily
> - Review all trades
> - Adjust parameters if needed
> - Don't go live until proven profitable

> [!IMPORTANT]
> **Expected Performance**
> 
> Realistic targets for paper trading:
> - Win Rate: 50-70% (not 90%!)
> - Profit Factor: 1.5-2.5
> - Monthly Return: 5-15%
> - Max Drawdown: 10-20%

---

## 🔧 Troubleshooting

### Bot Won't Start
**Error:** "MT5 initialization failed"
- Check MT5 is installed at path in `.env`
- Verify MT5 terminal is closed before starting bot
- Try running MT5 manually first

**Error:** "MT5 login failed"
- Check credentials in `.env`
- Verify demo account is still active
- Ensure server name is correct

### No Trades Being Placed
- Check confidence threshold (lower to 0.5 for testing)
- Verify pairs are available on your broker
- Check if max positions limit reached
- Review logs for error messages

### High Loss Rate
- **Expected** - Backtest results don't guarantee live performance
- Lower risk to 0.5% per trade
- Reduce max positions to 2-3
- Increase confidence threshold to 0.7

---

## 📈 Performance Tracking

### Daily Checklist
- [ ] Check account balance
- [ ] Review open positions
- [ ] Analyze closed trades
- [ ] Check win rate (target: 50-70%)
- [ ] Monitor drawdown (max: 20%)

### Weekly Review
- [ ] Calculate profit factor
- [ ] Review per-pair performance
- [ ] Adjust parameters if needed
- [ ] Update trading journal

### Monthly Evaluation
- [ ] Compare to backtest results
- [ ] Decide: continue, adjust, or stop
- [ ] If profitable: consider live trading
- [ ] If unprofitable: retrain model or collect more data

---

## 🎓 Best Practices

1. **Start Small**
   - Use minimum lot sizes
   - Trade only 1-2 pairs initially
   - Gradually increase as confidence grows

2. **Keep Records**
   - Save all logs
   - Screenshot important trades
   - Maintain a trading journal

3. **Stay Disciplined**
   - Don't override the bot manually
   - Don't change parameters mid-session
   - Trust the system or stop it

4. **Know When to Stop**
   - If daily loss > 5%, stop trading
   - If win rate < 40% after 50 trades, reassess
   - If drawdown > 20%, pause and analyze

---

## 📞 Support Files

- `start_live_trading.py` - Main bot launcher
- `monitor_trading.py` - Performance monitoring
- `config_live.py` - Configuration settings
- `src/live_trading.py` - Core trading logic
- `logs/mt5_trading.log` - Activity logs

---

## 🚦 Next Steps

1. **Test the bot** in dry-run mode
2. **Monitor for 7 days** to verify it works
3. **Enable live trading** on demo account
4. **Track for 30 days** to validate performance
5. **Decide**: Go live, retrain, or improve

**Current Status:** Ready for paper trading ✅
