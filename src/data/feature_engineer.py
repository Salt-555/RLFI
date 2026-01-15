import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
from typing import List


class FeatureEngineer:
    def __init__(self, tech_indicator_list: List[str], use_turbulence: bool = True):
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by=['tic', 'date'])
        
        stock_dfs = []
        for tic in df['tic'].unique():
            tic_df = df[df['tic'] == tic].copy()
            tic_df = tic_df.reset_index(drop=True)
            
            date_col = tic_df['date'].copy()
            
            stock = Sdf.retype(tic_df.copy())
            
            for indicator in self.tech_indicator_list:
                try:
                    stock[indicator]
                except Exception as e:
                    print(f"Warning: Could not calculate {indicator} for {tic}: {e}")
            
            tic_df = stock.copy()
            tic_df['date'] = date_col
            stock_dfs.append(tic_df)
        
        df = pd.concat(stock_dfs, ignore_index=True)
        df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        
        date_col = df['date']
        df = df.drop('date', axis=1)
        df = df.ffill().bfill()
        df = df.fillna(0)
        df.insert(0, 'date', date_col)
        
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
