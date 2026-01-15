import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.agents.trainer import RLTrainer
from src.trading.paper_trading import PaperTrader
from src.environment.trading_env import StockTradingEnv


def main():
    print("="*60)
    print("RL TRADING BOT - PAPER TRADING")
    print("="*60)
    
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if not config['paper_trading']['enabled']:
        print("\nWARNING: Paper trading is disabled in config.")
        proceed = input("Enable paper trading? (yes/no): ").strip().lower()
        if proceed != 'yes':
            print("Exiting...")
            return
    
    print("\n1. Loading trained model...")
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
    
    dummy_env = StockTradingEnv(
        df=None,
        stock_dim=len(config['data']['tickers']),
        hmax=config['environment']['hmax'],
        initial_amount=config['environment']['initial_amount'],
        transaction_cost_pct=config['environment']['transaction_cost_pct'],
        reward_scaling=config['environment']['reward_scaling'],
        tech_indicator_list=config['features']['technical_indicators'],
        turbulence_threshold=config['features']['turbulence_threshold']
    )
    
    trainer = RLTrainer(dummy_env, config, model_name=algorithm)
    model = trainer.load_model(model_path)
    
    print("\n2. Initializing paper trader...")
    print("\nIMPORTANT: Make sure you have set up your Alpaca API keys in .env file")
    print("Get free paper trading keys at: https://alpaca.markets/")
    
    proceed = input("\nHave you configured your API keys? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("\nPlease configure your API keys in .env file:")
        print("  ALPACA_API_KEY=your_key_here")
        print("  ALPACA_SECRET_KEY=your_secret_here")
        return
    
    try:
        paper_trader = PaperTrader(model, config)
        
        print("\n3. Starting paper trading...")
        duration = input("Enter duration in hours [default: 24]: ").strip() or "24"
        duration = int(duration)
        
        paper_trader.run_trading_loop(duration_hours=duration)
        
        print("\n4. Saving trade log...")
        os.makedirs('paper_trading_logs', exist_ok=True)
        paper_trader.save_trade_log('paper_trading_logs/trades.csv')
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease set up your Alpaca API credentials in .env file")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n" + "="*60)
    print("PAPER TRADING SESSION ENDED")
    print("="*60)


if __name__ == '__main__':
    main()
