"""
Basic Usage Example - RL Trading Bot

This script demonstrates the complete workflow:
1. Load data
2. Create environment
3. Train agent
4. Backtest
5. Evaluate performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.environment.trading_env import StockTradingEnv
from src.agents.trainer import RLTrainer
from src.evaluation.backtester import Backtester


def main():
    print("="*60)
    print("BASIC USAGE EXAMPLE")
    print("="*60)
    
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['tickers'] = ['AAPL', 'MSFT']
    config['training']['total_timesteps'] = 50000
    
    print("\n1. Loading and processing data...")
    data_loader = DataLoader(
        tickers=config['data']['tickers'],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    df = data_loader.download_data()
    
    feature_engineer = FeatureEngineer(
        tech_indicator_list=config['features']['technical_indicators'],
        use_turbulence=True
    )
    df = feature_engineer.preprocess_data(df)
    
    train_df, val_df, test_df = data_loader.split_data(df, 0.7, 0.15, 0.15)
    
    print("\n2. Creating environment...")
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=len(config['data']['tickers']),
        hmax=100,
        initial_amount=100000,
        transaction_cost_pct=0.001,
        reward_scaling=1e-4,
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=120
    )
    
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=len(config['data']['tickers']),
        hmax=100,
        initial_amount=100000,
        transaction_cost_pct=0.001,
        reward_scaling=1e-4,
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=120
    )
    
    print("\n3. Training PPO agent...")
    trainer = RLTrainer(train_env, config, model_name='ppo')
    model = trainer.train(total_timesteps=50000)
    
    print("\n4. Backtesting...")
    backtester = Backtester(test_env, model, config)
    results = backtester.run_backtest()
    
    print("\n5. Evaluating performance...")
    metrics = backtester.calculate_metrics()
    
    from src.evaluation.metrics import print_metrics
    print_metrics(metrics)
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE!")
    print("="*60)
    print("\nKey Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Total Return: {metrics['cumulative_return']:.2%}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")


if __name__ == '__main__':
    main()
