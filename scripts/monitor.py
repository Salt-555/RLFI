import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def plot_training_progress(log_dir='logs'):
    print("="*60)
    print("RL TRADING BOT - TRAINING MONITOR")
    print("="*60)
    
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return
    
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir={log_dir}")
    print("\nThen open http://localhost:6006 in your browser")


def analyze_backtest_results(results_dir='backtest_results'):
    print("\n" + "="*60)
    print("BACKTEST RESULTS ANALYSIS")
    print("="*60)
    
    report_path = os.path.join(results_dir, 'report.txt')
    
    if os.path.exists(report_path):
        print("\nBacktest Report:")
        with open(report_path, 'r') as f:
            print(f.read())
    else:
        print(f"No backtest report found at {report_path}")
    
    plot_path = os.path.join(results_dir, 'performance.png')
    if os.path.exists(plot_path):
        print(f"\nPerformance plot saved at: {plot_path}")
    else:
        print("No performance plot found")


def analyze_paper_trading(log_dir='paper_trading_logs'):
    print("\n" + "="*60)
    print("PAPER TRADING ANALYSIS")
    print("="*60)
    
    trades_path = os.path.join(log_dir, 'trades.csv')
    
    if not os.path.exists(trades_path):
        print(f"No trade log found at {trades_path}")
        return
    
    df = pd.read_csv(trades_path)
    
    print(f"\nTotal Trades: {len(df)}")
    print(f"Buy Orders: {len(df[df['action'] == 'BUY'])}")
    print(f"Sell Orders: {len(df[df['action'] == 'SELL'])}")
    print(f"Stop Loss Triggers: {len(df[df['action'] == 'STOP_LOSS'])}")
    
    print("\nTrades by Ticker:")
    print(df['ticker'].value_counts())
    
    print("\nRecent Trades:")
    print(df.tail(10))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor RL Trading Bot')
    parser.add_argument('--mode', choices=['training', 'backtest', 'paper'], 
                       default='training', help='What to monitor')
    
    args = parser.parse_args()
    
    if args.mode == 'training':
        plot_training_progress()
    elif args.mode == 'backtest':
        analyze_backtest_results()
    elif args.mode == 'paper':
        analyze_paper_trading()


if __name__ == '__main__':
    main()
