import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
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
                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                   holdout_days: int = 0):
        """
        Split data into train/val/test/holdout sets.
        
        Args:
            df: DataFrame with stock data
            train_ratio, val_ratio, test_ratio: Ratios for splitting remaining data after holdout
            holdout_days: Number of days to reserve as untouched holdout set (final validation only)
        
        Returns:
            Tuple of (train_df, val_df, test_df, holdout_df)
            - holdout_df: Never touched during model selection/training (true out-of-sample)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        unique_dates = sorted(df['date'].unique())
        n_dates = len(unique_dates)
        
        # First, split off holdout period (most recent dates) if enabled
        if holdout_days > 0:
            holdout_start_idx = max(0, n_dates - holdout_days)
            holdout_dates = unique_dates[holdout_start_idx:]
            remaining_dates = unique_dates[:holdout_start_idx]
            n_remaining = len(remaining_dates)
        else:
            holdout_dates = []
            remaining_dates = unique_dates
            n_remaining = n_dates
        
        # Split remaining data into train/val/test
        train_end_idx = int(n_remaining * train_ratio)
        val_end_idx = int(n_remaining * (train_ratio + val_ratio))
        
        train_dates = remaining_dates[:train_end_idx]
        val_dates = remaining_dates[train_end_idx:val_end_idx]
        test_dates = remaining_dates[val_end_idx:]
        
        train_df = df[df['date'].isin(train_dates)].reset_index(drop=True)
        val_df = df[df['date'].isin(val_dates)].reset_index(drop=True)
        test_df = df[df['date'].isin(test_dates)].reset_index(drop=True)
        holdout_df = df[df['date'].isin(holdout_dates)].reset_index(drop=True) if holdout_dates else pd.DataFrame()
        
        print(f"Train: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} rows)")
        print(f"Val: {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df)} rows)")
        print(f"Test: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} rows)")
        if holdout_days > 0 and not holdout_df.empty:
            print(f"Holdout: {holdout_df['date'].min()} to {holdout_df['date'].max()} ({len(holdout_df)} rows) [FINAL VALIDATION ONLY]")
        
        return train_df, val_df, test_df, holdout_df
