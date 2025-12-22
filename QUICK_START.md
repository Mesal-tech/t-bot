# 🚀 Quick Start - Live Trading Bot

## ✅ Ready to Start!

Your ML SMC Trading Bot is configured and ready for paper trading on your Exness demo account.

---

## 📋 Pre-Start Checklist

- ✅ MT5 installed and accessible
- ✅ Demo account active (297437818)
- ✅ Model trained (90% win rate on backtest)
- ✅ Correct symbol names configured (EURUSDm, GBPUSDm, etc.)

---

## 🎯 Start Trading Now

### Option 1: Test Mode (Recommended First)
```bash
# Stop current bot with Ctrl+C first, then:
python start_live_trading.py --dry-run --pairs EURUSDm
```

This will:
- Connect to MT5
- Fetch live data
- Generate predictions
- **NOT place any real orders**

### Option 2: Live Paper Trading
```bash
python start_live_trading.py --pairs EURUSDm GBPUSDm USDJPYm
```

This will:
- Connect to MT5
- Monitor 3 pairs
- Place real orders on demo account
- Risk 1% per trade

---

## 📊 Monitor Performance

### While Bot is Running
```bash
# In another terminal
python monitor_trading.py
```

Shows:
- Current balance
- Open positions
- Recent trades
- Win rate

---

## ⚙️ Configuration

Edit `config_live.py` to customize:

```python
PAIRS = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'EURGBPm', 'GBPJPYm']
TP_PIPS = 30
SL_PIPS = 10
MIN_PROBABILITY = 0.65
RISK_PERCENT = 1.0
MAX_POSITIONS = 5
```

---

## 🎓 What to Expect

### First Hour
- Bot connects to MT5
- Starts monitoring pairs
- May take 15-60 minutes for first signal
- Check logs to confirm it's working

### First Day
- Expect 2-5 trades total
- Win rate may vary (don't judge yet)
- Monitor for any errors
- Verify trades are placed correctly

### First Week
- Need 10+ trades for meaningful data
- Track win rate (target: 50-70%)
- Monitor drawdown (max: 20%)
- Adjust parameters if needed

---

## 🛑 How to Stop

1. Press `Ctrl+C` in the terminal
2. Bot will disconnect from MT5
3. Open positions remain (close manually in MT5 if needed)

---

## 📝 Important Notes

> **Balance is $0.00**
> Your demo account shows $0 balance. You may need to:
> 1. Deposit demo funds in MT5 terminal
> 2. Or create a new demo account with initial balance

> **Symbol Names**
> Exness uses "m" suffix: EURUSDm, GBPUSDm, etc.
> The bot is now configured correctly for this.

> **Realistic Expectations**
> - Backtest: 90% win rate
> - Live (expected): 50-70% win rate
> - This is normal and still profitable with 1:3 RR

---

## 🚦 Next Steps

1. **Fund Demo Account** (if balance is $0)
   - Open MT5
   - Request demo funds or create new demo account

2. **Start Bot**
   ```bash
   python start_live_trading.py --pairs EURUSDm GBPUSDm
   ```

3. **Monitor for 24 Hours**
   - Check every few hours
   - Review logs for errors
   - Verify trades are placed

4. **Evaluate After 1 Week**
   - Run `python monitor_trading.py`
   - Check win rate and P&L
   - Decide: continue, adjust, or retrain

---

## 📞 Quick Commands

| Action | Command |
|--------|---------|
| Start (test) | `python start_live_trading.py --dry-run --pairs EURUSDm` |
| Start (live) | `python start_live_trading.py` |
| Monitor | `python monitor_trading.py` |
| Stop | Press `Ctrl+C` |
| View logs | `Get-Content logs/mt5_trading.log -Wait` |

---

**You're all set! 🎉**

Start with test mode, then move to live paper trading once you're comfortable.
