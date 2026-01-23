#!/usr/bin/env python3
"""
Strategy Analyzer - Analyze and compare AI trading strategies across weeks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.autotest.strategy_analyzer import StrategyAnalyzer
from src.autotest.strategy_database import StrategyDatabase


def main():
    parser = argparse.ArgumentParser(description='Analyze AI Trading Strategies')
    parser.add_argument('--weeks', type=int, default=12,
                       help='Number of weeks to analyze (default: 12)')
    parser.add_argument('--export', action='store_true',
                       help='Export all data to CSV files')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization charts')
    parser.add_argument('--compare', nargs=2, metavar=('MODEL1', 'MODEL2'),
                       help='Compare two specific models')
    parser.add_argument('--top', type=int, default=10,
                       help='Show top N strategies (default: 10)')
    parser.add_argument('--db', default='autotest_strategies.db',
                       help='Path to strategy database')
    
    args = parser.parse_args()
    
    print("="*80)
    print("STRATEGY ANALYZER")
    print("="*80)
    
    analyzer = StrategyAnalyzer(args.db)
    
    if args.compare:
        # Compare two specific strategies
        model1, model2 = args.compare
        print(f"\nComparing {model1} vs {model2}...")
        comparison = analyzer.compare_strategies(model1, model2)
        
        if 'error' in comparison:
            print(f"Error: {comparison['error']}")
        else:
            print("\nStrategy 1:")
            print(f"  Model: {comparison['strategy1']['model_id']}")
            print(f"  Tickers: {comparison['strategy1']['tickers']}")
            print(f"  Learning Rate: {comparison['strategy1']['learning_rate']}")
            print(f"  Sharpe Ratio: {comparison['strategy1']['sharpe_ratio']:.3f}")
            print(f"  Total Return: {comparison['strategy1']['total_return']*100:.2f}%")
            
            print("\nStrategy 2:")
            print(f"  Model: {comparison['strategy2']['model_id']}")
            print(f"  Tickers: {comparison['strategy2']['tickers']}")
            print(f"  Learning Rate: {comparison['strategy2']['learning_rate']}")
            print(f"  Sharpe Ratio: {comparison['strategy2']['sharpe_ratio']:.3f}")
            print(f"  Total Return: {comparison['strategy2']['total_return']*100:.2f}%")
            
            print("\nComparison:")
            print(f"  Sharpe Difference: {comparison['comparison']['sharpe_diff']:.3f}")
            print(f"  Return Difference: {comparison['comparison']['return_diff']*100:.2f}%")
            print(f"  Better Strategy: {comparison['comparison']['better_strategy']}")
    
    else:
        # Generate weekly report
        report = analyzer.generate_weekly_report(args.weeks)
        
        if args.export:
            print("\nExporting data to CSV...")
            output_path = analyzer.export_analysis_report(weeks_back=args.weeks)
            print(f"Analysis exported to: {output_path}")
        
        if args.visualize:
            print("\nGenerating visualizations...")
            analyzer.visualize_performance_trends(weeks_back=args.weeks)
    
    # Show top strategies
    print(f"\n{'='*80}")
    print(f"TOP {args.top} STRATEGIES BY SHARPE RATIO")
    print("="*80)
    
    db = StrategyDatabase(args.db)
    top_strategies = db.get_top_strategies(weeks_back=args.weeks, top_n=args.top)
    
    if not top_strategies.empty:
        for i, row in top_strategies.iterrows():
            print(f"\n{i+1}. {row['model_id']}")
            print(f"   Week: {row['week_number']}")
            print(f"   Tickers: {row['tickers']}")
            print(f"   Learning Rate: {row['learning_rate']}")
            print(f"   Sharpe: {row['sharpe_ratio']:.3f}")
            print(f"   Return: {row['total_return']*100:.2f}%")
            print(f"   Rank: #{row['rank_position']}")
    else:
        print("No strategies found in database")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
