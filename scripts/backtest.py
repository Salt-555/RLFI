import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.data_loader import DataLoader
from src.environment.trading_env import StockTradingEnv
from src.agents.trainer import RLTrainer
from src.evaluation.backtester import Backtester


def main():
    print("="*60)
    print("RL TRADING BOT - BACKTESTING")
    print("="*60)
    
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Loading processed data...")
    processed_path = 'data/processed/stock_data_processed.csv'
    
    if not os.path.exists(processed_path):
        print("Processed data not found. Run train_single.py first.")
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
    
    print("\n3. Creating test environment...")
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=len(config['data']['tickers']),
        hmax=config['environment']['hmax'],
        initial_amount=config['environment']['initial_amount'],
        transaction_cost_pct=config['environment']['transaction_cost_pct'],
        reward_scaling=config['environment']['reward_scaling'],
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=config['features']['turbulence_threshold']
    )
    
    print("\n4. Loading trained model...")
    available_models = [f for f in os.listdir('models') if f.endswith('.zip')]
    
    if not available_models:
        print("No trained models found. Run train_single.py or train_ensemble.py first.")
        return
    
    print("Available models:")
    for i, model in enumerate(available_models):
        print(f"  {i+1}. {model}")
    
    choice = input("\nSelect model number [default: 1]: ").strip() or "1"
    model_path = f"models/{available_models[int(choice)-1]}"
    
    algorithm = available_models[int(choice)-1].split('_')[0]
    
    trainer = RLTrainer(test_env, config, model_name=algorithm)
    model = trainer.load_model(model_path)
    
    print("\n5. Running backtest...")
    backtester = Backtester(test_env, model, config)
    results = backtester.run_backtest(deterministic=True)
    
    print("\n6. Calculating metrics...")
    metrics = backtester.calculate_metrics()
    
    print("\n7. Generating report...")
    os.makedirs('backtest_results', exist_ok=True)
    backtester.generate_report(save_path='backtest_results/report.txt')
    
    print("\n8. Plotting results...")
    backtester.plot_results(save_path='backtest_results/performance.png')
    
    print("\n" + "="*60)
    print("BACKTESTING COMPLETE!")
    print("="*60)
    print("Results saved to: backtest_results/")


if __name__ == '__main__':
    main()
