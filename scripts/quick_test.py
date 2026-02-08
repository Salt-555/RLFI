import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.environment.trading_env import StockTradingEnv
from src.agents.trainer import RLTrainer
from src.evaluation.backtester import Backtester

print("="*60)
print("QUICK WORKFLOW TEST")
print("="*60)

config = {
    'data': {
        'tickers': ['AAPL', 'MSFT'],
        'start_date': '2022-01-01',
        'end_date': '2024-01-01',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    },
    'features': {
        'technical_indicators': ['macd', 'rsi_30', 'cci_30', 'dx_30'],
        'use_turbulence': False,
        'turbulence_threshold': 120
    },
    'environment': {
        'initial_amount': 100000,
        'transaction_cost_pct': 0.001,
        'hmax': 100,
        'reward_scaling': 1e-4
    },
    'training': {
        'total_timesteps': 10000,
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99
        }
    },
    'monitoring': {
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints'
    }
}

print("\n1. Downloading data...")
data_loader = DataLoader(
    tickers=config['data']['tickers'],
    start_date=config['data']['start_date'],
    end_date=config['data']['end_date']
)

df = data_loader.download_data()
print(f"Downloaded {len(df)} rows")

print("\n2. Engineering features...")
feature_engineer = FeatureEngineer(
    tech_indicator_list=config['features']['technical_indicators'],
    use_turbulence=config['features']['use_turbulence']
)
df = feature_engineer.preprocess_data(df)

print("\n3. Splitting data...")
holdout_days = config['data'].get('holdout_days', 0) if config['data'].get('holdout_enabled', False) else 0
train_df, val_df, test_df, _ = data_loader.split_data(
    df,
    train_ratio=config['data']['train_ratio'],
    val_ratio=config['data']['val_ratio'],
    test_ratio=config['data']['test_ratio'],
    holdout_days=holdout_days
)

print("\n4. Creating environment...")
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

test_env = StockTradingEnv(
    df=test_df,
    stock_dim=len(config['data']['tickers']),
    hmax=config['environment']['hmax'],
    initial_amount=config['environment']['initial_amount'],
    transaction_cost_pct=config['environment']['transaction_cost_pct'],
    reward_scaling=config['environment']['reward_scaling'],
    tech_indicator_list=config['features']['technical_indicators'],
    turbulence_threshold=config['features']['turbulence_threshold'],
    store_history_every=1  # Store every step for accurate backtest metrics
)

print("\n5. Training PPO agent (10k timesteps - quick test)...")
trainer = RLTrainer(train_env, config, model_name='ppo')
model = trainer.train(total_timesteps=10000)

print("\n6. Saving model...")
os.makedirs('models', exist_ok=True)
trainer.save_model('models/ppo_quick_test.zip')

print("\n7. Running backtest...")
backtester = Backtester(test_env, model, config)
results = backtester.run_backtest()

print("\n8. Calculating metrics...")
metrics = backtester.calculate_metrics()

from src.evaluation.metrics import print_metrics
print_metrics(metrics)

print("\n" + "="*60)
print("WORKFLOW TEST COMPLETE!")
print("="*60)
print(f"\nKey Results:")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"  Total Return: {metrics['cumulative_return']:.2%}")
print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"  Final Value: ${metrics['final_value']:,.2f}")
