import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.environment.trading_env import StockTradingEnv
from src.agents.trainer import RLTrainer


def main():
    print("="*60)
    print("RL TRADING BOT - SINGLE AGENT TRAINING")
    print("="*60)
    
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Loading data...")
    data_loader = DataLoader(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    data_path = 'data/raw/stock_data.csv'
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        df = data_loader.load_data(data_path)
    else:
        print("Downloading fresh data...")
        df = data_loader.download_data()
        data_loader.save_data(df, data_path)
    
    print("\n2. Engineering features...")
    feature_engineer = FeatureEngineer(
        tech_indicator_list=config['features']['technical_indicators'],
        use_turbulence=config['features']['use_turbulence']
    )
    df = feature_engineer.preprocess_data(df)
    
    processed_path = 'data/processed/stock_data_processed.csv'
    data_loader.save_data(df, processed_path)
    
    print("\n3. Splitting data...")
    train_df, val_df, test_df = data_loader.split_data(
        df,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    print("\n4. Creating training environment...")
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=len(config['data']['tickers']),
        initial_amount=config['environment']['initial_amount'],
        transaction_cost_pct=config['environment']['transaction_cost_pct'],
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=config['features']['turbulence_threshold'],
        max_position_pct=config['environment'].get('max_position_pct', 0.3)
    )
    
    val_env = StockTradingEnv(
        df=val_df,
        stock_dim=len(config['data']['tickers']),
        initial_amount=config['environment']['initial_amount'],
        transaction_cost_pct=config['environment']['transaction_cost_pct'],
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=config['features']['turbulence_threshold'],
        max_position_pct=config['environment'].get('max_position_pct', 0.3)
    )
    
    algorithm = input("\nSelect algorithm (ppo/a2c/ddpg) [default: ppo]: ").strip().lower() or 'ppo'
    
    print(f"\n5. Training {algorithm.upper()} agent...")
    trainer = RLTrainer(train_env, config, model_name=algorithm)
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    eval_env = DummyVecEnv([lambda: Monitor(val_env)])
    
    model = trainer.train(eval_env=eval_env)
    
    print("\n6. Saving model...")
    os.makedirs('models', exist_ok=True)
    trainer.save_model(f'models/{algorithm}_trained.zip')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: models/{algorithm}_trained.zip")
    print("\nNext steps:")
    print("  1. Run backtesting: python scripts/backtest.py")
    print("  2. Train ensemble: python scripts/train_ensemble.py")
    print("  3. Paper trade: python scripts/paper_trade.py")


if __name__ == '__main__':
    main()
