import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
import os


class DataLoader:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def download_data(self) -> pd.DataFrame:
        print(f"Downloading data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}")
        
        data_frames = []
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if df.empty:
                    print(f"Warning: No data for {ticker}")
                    continue
                
                df = df.reset_index()
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
                
                df.columns = [col.lower() for col in df.columns]
                df['tic'] = ticker
                
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
                if all(col in df.columns for col in required_cols):
                    data_frames.append(df[required_cols])
                else:
                    print(f"Warning: Missing columns for {ticker}. Found: {df.columns.tolist()}")
                    
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
        
        if not data_frames:
            raise ValueError("No data downloaded for any ticker")
            
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        print(f"Downloaded {len(combined_df)} rows for {len(combined_df['tic'].unique())} tickers")
        return combined_df
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        unique_dates = sorted(df['date'].unique())
        n_dates = len(unique_dates)
        
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))
        
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        test_dates = unique_dates[val_end_idx:]
        
        train_df = df[df['date'].isin(train_dates)].reset_index(drop=True)
        val_df = df[df['date'].isin(val_dates)].reset_index(drop=True)
        test_df = df[df['date'].isin(test_dates)].reset_index(drop=True)
        
        print(f"Train: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} rows)")
        print(f"Val: {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df)} rows)")
        print(f"Test: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} rows)")
        
        return train_df, val_df, test_df
