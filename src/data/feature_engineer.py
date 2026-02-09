import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
from typing import List, Dict
import json
import os


class FeatureEngineer:
    def __init__(self, tech_indicator_list: List[str], use_turbulence: bool = True, 
                 external_stats: Dict = None):
        """
        Args:
            tech_indicator_list: List of technical indicators to calculate
            use_turbulence: Whether to calculate turbulence index
            external_stats: Optional dict with 'means' and 'stds' for z-score normalization.
                          If provided, uses these instead of computing from data.
                          CRITICAL for live trading to match training distribution.
        """
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.external_stats = external_stats  # Pre-computed from training
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by=['tic', 'date'])
        
        stock_dfs = []
        for tic in df['tic'].unique():
            tic_df = df[df['tic'] == tic].copy()
            tic_df = tic_df.reset_index(drop=True)
            
            # Save date and tic columns before StockDataFrame processing
            date_col = tic_df['date'].copy()
            tic_col = tic_df['tic'].copy()
            
            # Remove date column before Sdf.retype to prevent corruption
            tic_df_no_date = tic_df.drop('date', axis=1)
            stock = Sdf.retype(tic_df_no_date)
            
            for indicator in self.tech_indicator_list:
                try:
                    stock[indicator]
                except Exception as e:
                    print(f"Warning: Could not calculate {indicator} for {tic}: {e}")
            
            # Convert back to regular dataframe and restore date
            tic_df = pd.DataFrame(stock)
            tic_df.insert(0, 'date', date_col)
            tic_df['tic'] = tic_col
            stock_dfs.append(tic_df)
        
        df = pd.concat(stock_dfs, ignore_index=True)
        df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        
        # Fill NaN values in technical indicators only (not date)
        tech_cols = [col for col in df.columns if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
        df[tech_cols] = df[tech_cols].ffill().bfill().fillna(0)
        
        # Capture raw stats BEFORE normalization so they can be used for
        # consistent z-score normalization during live inference.
        # Previously these were captured after normalization, making them
        # useless (~0 mean, ~1 std) for normalizing raw live data.
        self._raw_indicator_stats = {}
        for col in tech_cols:
            if col in df.columns:
                self._raw_indicator_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        
        # Normalize technical indicators to roughly -1 to 1 range for better ML training
        # CRITICAL: Use external_stats (from training) if provided, otherwise compute from data
        for col in tech_cols:
            if col in df.columns:
                if self.external_stats and col in self.external_stats.get('means', {}):
                    # Use training statistics for consistent normalization
                    col_mean = self.external_stats['means'][col]
                    col_std = self.external_stats['stds'][col]
                else:
                    # Fallback: compute from current data (training only)
                    col_mean = df[col].mean()
                    col_std = df[col].std()
                
                if col_std > 0:
                    df[col] = (df[col] - col_mean) / (col_std + 1e-8)
                    # Clip extreme values
                    df[col] = df[col].clip(-5, 5)
        
        return df
    
    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_turbulence:
            return df
            
        df = df.copy()
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        unique_dates = df['date'].unique()
        turbulence_index = []
        
        for date in unique_dates:
            current_price = df[df['date'] == date]['close'].values
            
            hist_data = df[df['date'] < date]
            if len(hist_data) < 252:
                turbulence_index.append(0)
                continue
            
            hist_price = hist_data.pivot(index='date', columns='tic', values='close')
            hist_price = hist_price.tail(252)
            
            if hist_price.shape[0] < 2:
                turbulence_index.append(0)
                continue
            
            returns = hist_price.pct_change().dropna()
            
            if returns.shape[0] < 2 or returns.shape[1] != len(current_price):
                turbulence_index.append(0)
                continue
            
            try:
                cov_matrix = returns.cov()
                mean_returns = returns.mean()
                
                current_returns = (current_price / hist_price.iloc[-1].values) - 1
                
                diff = current_returns - mean_returns.values
                turbulence = np.dot(np.dot(diff, np.linalg.pinv(cov_matrix)), diff.T)
                turbulence_index.append(turbulence)
            except Exception as e:
                turbulence_index.append(0)
        
        turbulence_df = pd.DataFrame({'date': unique_dates, 'turbulence': turbulence_index})
        df = df.merge(turbulence_df, on='date', how='left')
        df['turbulence'] = df['turbulence'].fillna(0)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding technical indicators...")
        df = self.add_technical_indicators(df)
        
        if self.use_turbulence:
            print("Calculating turbulence index...")
            df = self.add_turbulence(df)
        
        print(f"Feature engineering complete. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def get_indicator_stats(self, df: pd.DataFrame = None) -> Dict[str, Dict[str, float]]:
        """
        Get mean and std for each technical indicator from BEFORE normalization.
        These stats are needed for consistent z-score normalization during live inference.
        
        Uses the raw stats captured during add_technical_indicators() before z-scoring.
        If raw stats aren't available (e.g., add_technical_indicators wasn't called yet),
        falls back to computing from the provided DataFrame.
        
        Args:
            df: Optional processed DataFrame (used as fallback only)
        
        Returns:
            Dict with 'means' and 'stds' for each indicator
        """
        means = {}
        stds = {}
        
        # Use raw pre-normalization stats if available
        if hasattr(self, '_raw_indicator_stats') and self._raw_indicator_stats:
            for col, stats in self._raw_indicator_stats.items():
                means[col] = stats['mean']
                stds[col] = stats['std']
        elif df is not None:
            # Fallback: compute from provided DataFrame (may be post-normalization)
            tech_cols = [col for col in df.columns if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'turbulence']]
            for col in tech_cols:
                if col in df.columns:
                    means[col] = float(df[col].mean())
                    stds[col] = float(df[col].std())
        
        return {'means': means, 'stds': stds}
    
    def save_indicator_stats(self, stats: Dict, filepath: str):
        """Save indicator statistics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Indicator stats saved to {filepath}")
    
    @staticmethod
    def load_indicator_stats(filepath: str) -> Dict:
        """Load indicator statistics from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
