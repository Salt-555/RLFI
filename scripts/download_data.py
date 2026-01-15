import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer


def main():
    print("="*60)
    print("RL TRADING BOT - DATA DOWNLOAD & PREPROCESSING")
    print("="*60)
    
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Downloading stock data...")
    data_loader = DataLoader(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    df = data_loader.download_data()
    
    print("\n2. Saving raw data...")
    os.makedirs('data/raw', exist_ok=True)
    data_loader.save_data(df, 'data/raw/stock_data.csv')
    
    print("\n3. Engineering features...")
    feature_engineer = FeatureEngineer(
        tech_indicator_list=config['features']['technical_indicators'],
        use_turbulence=config['features']['use_turbulence']
    )
    df_processed = feature_engineer.preprocess_data(df)
    
    print("\n4. Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    data_loader.save_data(df_processed, 'data/processed/stock_data_processed.csv')
    
    print("\n" + "="*60)
    print("DATA DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Raw data: data/raw/stock_data.csv")
    print(f"Processed data: data/processed/stock_data_processed.csv")
    print(f"\nData shape: {df_processed.shape}")
    print(f"Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
    print(f"Tickers: {df_processed['tic'].unique().tolist()}")


if __name__ == '__main__':
    main()
