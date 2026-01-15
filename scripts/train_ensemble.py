import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.environment.trading_env import StockTradingEnv
from src.agents.ensemble import EnsembleTrainer


def main():
    print("="*60)
    print("RL TRADING BOT - ENSEMBLE TRAINING")
    print("="*60)
    
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Loading processed data...")
    processed_path = 'data/processed/stock_data_processed.csv'
    
    if not os.path.exists(processed_path):
        print("Processed data not found. Run train_single.py first or process data manually.")
        return
    
    data_loader = DataLoader(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    df = data_loader.load_data(processed_path)
    
    print("\n2. Splitting data...")
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    print("\n3. Creating environments...")
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=len(config['data']['tickers']),
        hmax=config['environment']['hmax'],
        initial_amount=config['environment']['initial_amount'],
        transaction_cost_pct=config['environment']['transaction_cost_pct'],
        reward_scaling=config['environment']['reward_scaling'],
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=config['features']['turbulence_threshold']
    )
    
    val_env = StockTradingEnv(
        df=val_df,
        stock_dim=len(config['data']['tickers']),
        hmax=config['environment']['hmax'],
        initial_amount=config['environment']['initial_amount'],
        transaction_cost_pct=config['environment']['transaction_cost_pct'],
        reward_scaling=config['environment']['reward_scaling'],
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=config['features']['turbulence_threshold']
    )
    
    print("\n4. Training ensemble models...")
    ensemble = EnsembleTrainer(train_env, val_env, config)
    
    algorithms = config['training']['algorithms']
    print(f"Training algorithms: {algorithms}")
    
    ensemble.train_all_models(algorithms)
    
    print("\n5. Evaluating models on validation set...")
    ensemble.evaluate_models(val_env)
    
    print("\n6. Performance Summary:")
    summary = ensemble.get_performance_summary()
    print(summary)
    
    best_algo, best_model = ensemble.select_best_model('sharpe_ratio')
    
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: {best_algo.upper()}")
    print(f"All models saved to: ./models/")
    print("\nNext steps:")
    print("  1. Run backtesting: python scripts/backtest.py")
    print("  2. Paper trade: python scripts/paper_trade.py")


if __name__ == '__main__':
    main()
