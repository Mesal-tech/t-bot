#!/usr/bin/env python3
"""
Start the ML SMC Live Trading Bot

Usage:
    python start_live_trading.py              # Start with default config
    python start_live_trading.py --dry-run    # Test mode (no real orders)
    python start_live_trading.py --pairs EURUSD GBPUSD  # Specific pairs
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Now import
from live_trading import LiveTradingBot

# Import config
config_path = os.path.dirname(__file__)
sys.path.insert(0, config_path)
import config_live as config


def main():
    parser = argparse.ArgumentParser(description='ML SMC Live Trading Bot')
    
    parser.add_argument(
        '--pairs',
        nargs='+',
        default=config.PAIRS,
        help='Currency pairs to trade'
    )
    
    parser.add_argument(
        '--tp-pips',
        type=int,
        default=config.TP_PIPS,
        help='Take profit in pips'
    )
    
    parser.add_argument(
        '--sl-pips',
        type=int,
        default=config.SL_PIPS,
        help='Stop loss in pips'
    )
    
    parser.add_argument(
        '--min-prob',
        type=float,
        default=config.MIN_PROBABILITY,
        help='Minimum probability threshold'
    )
    
    parser.add_argument(
        '--risk-percent',
        type=float,
        default=config.RISK_PERCENT,
        help='Risk per trade as % of balance'
    )
    
    parser.add_argument(
        '--max-positions',
        type=int,
        default=config.MAX_POSITIONS,
        help='Maximum concurrent positions'
    )
    
    parser.add_argument(
        '--min-prob-increase',
        type=float,
        default=config.MIN_PROB_INCREASE,
        help='Minimum probability increase for additional positions on same pair'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test mode - no real orders'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed logging (INFO level) to console'
    )
    
    args = parser.parse_args()

    # Configure logging based on verbose flag
    if args.verbose:
        import logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
             # Check if it's a StreamHandler (console) and not FileHandler
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.INFO)
                print("Detailed logging enabled (INFO level)")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Display configuration
    print("\n" + "="*70)
    print("ML SMC LIVE TRADING BOT")
    print("="*70)
    print(f"Model: {config.MODEL_PATH}")
    print(f"Pairs: {', '.join(args.pairs)}")
    print(f"TP/SL: {args.tp_pips}/{args.sl_pips} pips")
    print(f"Min Probability: {args.min_prob}")
    print(f"Risk per Trade: {args.risk_percent}%")
    print(f"Max Positions: {args.max_positions}")
    print(f"Min Prob Increase for Multi-Position: {args.min_prob_increase*100:.0f}%")
    print(f"Mode: {'DRY RUN (Testing)' if args.dry_run else 'LIVE TRADING'}")
    print("="*70)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No real orders will be placed")
    else:
        print("\n⚠️  LIVE TRADING MODE - Real orders will be placed!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    print("\nStarting bot...")
    print("Press Ctrl+C to stop\n")
    
    # Initialize bot
    bot = LiveTradingBot(
        model_path=config.MODEL_PATH,
        pairs=args.pairs,
        tp_pips=args.tp_pips,
        sl_pips=args.sl_pips,
        min_prob=args.min_prob,
        risk_percent=args.risk_percent,
        max_positions=args.max_positions,
        min_prob_increase=args.min_prob_increase
    )
    
    # Start trading
    bot.start()


if __name__ == "__main__":
    main()
