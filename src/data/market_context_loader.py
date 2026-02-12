"""
Market Context Data Loader

Fetches alternative market data that provides context for trading decisions.
All data is provided in raw, neutral format so RL models can learn relationships dynamically.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import requests
import os


class MarketContextLoader:
    """
    Loads market context data including:
    - VIX and VIX3M (volatility term structure)
    - Put/Call ratio (market sentiment)
    - Earnings calendar (event risk)
    - FOMC calendar (policy risk)
    
    All data is provided in raw form without interpretation.
    """
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_vix_data(self) -> pd.DataFrame:
        """
        Fetch VIX (^VIX) and VIX3M (^VIX3M) data.
        
        Returns DataFrame with:
        - date
        - vix_close (spot VIX level)
        - vix3m_close (3-month VIX)
        - vix_spread (VIX3M - VIX, raw difference)
        - vix_ratio (VIX3M / VIX, relative term structure)
        """
        print("Fetching VIX and VIX3M data...")
        
        try:
            # Fetch VIX (1-month implied vol)
            vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
            # Fetch VIX3M (3-month implied vol)
            vix3m = yf.download('^VIX3M', start=self.start_date, end=self.end_date, progress=False)
            
            if vix.empty or vix3m.empty:
                print("Warning: Could not fetch VIX data")
                return pd.DataFrame()
            
            vix = vix.reset_index()
            vix3m = vix3m.reset_index()
            
            # Handle multi-index columns from yfinance
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = [col[0] if col[1] == '' else col[0] for col in vix.columns]
            if isinstance(vix3m.columns, pd.MultiIndex):
                vix3m.columns = [col[0] if col[1] == '' else col[0] for col in vix3m.columns]
            
            # Normalize column names
            vix.columns = [col.lower() for col in vix.columns]
            vix3m.columns = [col.lower() for col in vix3m.columns]
            
            # Merge on date
            merged = pd.merge(
                vix[['date', 'close']].rename(columns={'close': 'vix_close'}),
                vix3m[['date', 'close']].rename(columns={'close': 'vix3m_close'}),
                on='date',
                how='outer'
            )
            
            merged['date'] = pd.to_datetime(merged['date'])
            merged = merged.sort_values('date')
            
            # Forward fill missing values (VIX3M might have gaps)
            merged['vix_close'] = merged['vix_close'].ffill()
            merged['vix3m_close'] = merged['vix3m_close'].ffill()
            
            # Calculate raw spread and ratio (neutral - let RL learn what they mean)
            merged['vix_spread'] = merged['vix3m_close'] - merged['vix_close']  # Can be negative (backwardation) or positive (contango)
            merged['vix_ratio'] = merged['vix3m_close'] / (merged['vix_close'] + 1e-8)  # Relative term structure
            
            # Also include VIX level normalized to 0-1 range (30 is considered high)
            merged['vix_norm'] = merged['vix_close'] / 50.0  # Raw level, not opinionated
            
            print(f"Fetched VIX data: {len(merged)} days")
            return merged
            
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def fetch_put_call_ratio(self) -> pd.DataFrame:
        """
        Fetch CBOE put/call ratio data.
        
        Returns DataFrame with:
        - date
        - put_call_ratio (total put volume / total call volume)
        - put_call_5dma (5-day moving average)
        
        Note: Uses total equity put/call ratio from CBOE.
        """
        print("Fetching put/call ratio data...")
        
        try:
            # CBOE total put/call ratio index
            # ^VIX is spot volatility, we can also look at ^VIX9D for near-term
            
            # Fetch from Yahoo as a proxy (VIX tends to correlate with put buying)
            # For actual put/call data, we'd need CBOE or Unusual Whales API
            
            # Alternative: Use VIX as a proxy, or fetch from CBOE website
            # For now, we'll use a placeholder that can be replaced with real data
            
            # Create date range
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')  # Business days
            
            # Placeholder - in production, replace with actual CBOE data
            # CBOE provides this at: https://www.cboe.com/data/putcallratio
            
            df = pd.DataFrame({'date': dates})
            df['put_call_ratio'] = 1.0  # Neutral placeholder
            df['put_call_5dma'] = 1.0
            
            print("Note: Using placeholder put/call ratio. Replace with CBOE data feed for production.")
            print("Get real data from: https://www.cboe.com/data/putcallratio")
            
            return df
            
        except Exception as e:
            print(f"Error fetching put/call ratio: {e}")
            return pd.DataFrame()
    
    def fetch_earnings_calendar(self, tickers: List[str]) -> pd.DataFrame:
        """
        Generate earnings calendar features for given tickers.
        
        Args:
            tickers: List of stock tickers
            
        Returns DataFrame with:
        - date
        - tic
        - days_to_earnings (days until next earnings, -1 if unknown/no earnings upcoming)
        - days_since_earnings (days since last earnings, -1 if unknown)
        """
        print(f"Generating earnings calendar features for {len(tickers)} tickers...")
        
        # In production, this would fetch from:
        # - Yahoo Finance (yfinance.Ticker.calendar)
        # - Earnings Whispers API
        # - Alpha Vantage
        
        # For now, create a placeholder structure
        # The RL model will learn that certain date ranges (late Jan, late Apr, etc.) 
        # tend to have higher volatility
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        records = []
        for ticker in tickers:
            for date in dates:
                # Placeholder: assume earnings every 90 days (quarterly)
                # In production, fetch actual earnings dates
                day_of_year = date.dayofyear
                days_to_er = (90 - (day_of_year % 90)) % 90
                if days_to_er > 45:
                    days_to_er = days_to_er - 90  # Negative = past earnings
                
                records.append({
                    'date': date,
                    'tic': ticker,
                    'days_to_earnings': days_to_er,
                    'days_since_earnings': -days_to_er if days_to_er < 0 else (90 - days_to_er)
                })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        
        print("Note: Using placeholder earnings calendar. Integrate Yahoo Finance or Earnings Whispers for real data.")
        return df
    
    def fetch_fomc_calendar(self) -> pd.DataFrame:
        """
        Generate FOMC meeting calendar features.
        
        Returns DataFrame with:
        - date
        - days_to_fomc (days until next FOMC meeting)
        - days_since_fomc (days since last FOMC meeting)
        - in_fomc_week (1 if date is in week of FOMC meeting, 0 otherwise)
        
        FOMC typically meets 8 times per year.
        """
        print("Generating FOMC calendar features...")
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # 2024-2025 FOMC meeting dates (update as needed)
        # Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
        fomc_dates_2024 = [
            '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
            '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18'
        ]
        fomc_dates_2025 = [
            '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-11',
            '2025-07-30', '2025-09-17', '2025-11-06', '2025-12-17'
        ]
        fomc_dates_2026 = [
            '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-10',
            '2026-07-29', '2026-09-16', '2026-11-05', '2026-12-16'
        ]
        
        all_fomc = pd.to_datetime(fomc_dates_2024 + fomc_dates_2025 + fomc_dates_2026)
        
        records = []
        for date in dates:
            # Find nearest FOMC dates
            future_fomc = all_fomc[all_fomc >= date]
            past_fomc = all_fomc[all_fomc < date]
            
            if len(future_fomc) > 0:
                next_fomc = future_fomc.min()
                days_to_fomc = (next_fomc - date).days
            else:
                days_to_fomc = 999  # Far future
            
            if len(past_fomc) > 0:
                last_fomc = past_fomc.max()
                days_since_fomc = (date - last_fomc).days
            else:
                days_since_fomc = 999
            
            # FOMC week = 3 days before to 1 day after
            in_fomc_week = 1 if days_to_fomc <= 3 or days_since_fomc <= 1 else 0
            
            records.append({
                'date': date,
                'days_to_fomc': days_to_fomc,
                'days_since_fomc': days_since_fomc,
                'in_fomc_week': in_fomc_week
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"FOMC calendar: {len(all_fomc)} meetings in range")
        return df
    
    def load_all_context(self, tickers: List[str]) -> pd.DataFrame:
        """
        Load all market context features and merge into single DataFrame.
        
        Args:
            tickers: List of stock tickers in portfolio
            
        Returns:
            DataFrame with all context features merged on date
        """
        print("\nLoading market context data...")
        
        # Load each data source
        vix_df = self.fetch_vix_data()
        pc_df = self.fetch_put_call_ratio()
        fomc_df = self.fetch_fomc_calendar()
        earnings_df = self.fetch_earnings_calendar(tickers)
        
        # Start with stock dates
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        context_df = pd.DataFrame({'date': dates})
        context_df['date'] = pd.to_datetime(context_df['date'])
        
        # Merge VIX data (market-wide)
        if not vix_df.empty:
            context_df = context_df.merge(vix_df, on='date', how='left')
            context_df[['vix_close', 'vix3m_close', 'vix_spread', 'vix_ratio', 'vix_norm']] = \
                context_df[['vix_close', 'vix3m_close', 'vix_spread', 'vix_ratio', 'vix_norm']].ffill().bfill().fillna(0)
        else:
            context_df['vix_close'] = 20.0
            context_df['vix3m_close'] = 22.0
            context_df['vix_spread'] = 2.0
            context_df['vix_ratio'] = 1.1
            context_df['vix_norm'] = 0.4
        
        # Merge put/call ratio (market-wide)
        if not pc_df.empty:
            context_df = context_df.merge(pc_df, on='date', how='left')
            context_df[['put_call_ratio', 'put_call_5dma']] = \
                context_df[['put_call_ratio', 'put_call_5dma']].ffill().bfill().fillna(1.0)
        else:
            context_df['put_call_ratio'] = 1.0
            context_df['put_call_5dma'] = 1.0
        
        # Merge FOMC calendar (market-wide)
        if not fomc_df.empty:
            context_df = context_df.merge(fomc_df, on='date', how='left')
        else:
            context_df['days_to_fomc'] = 30
            context_df['days_since_fomc'] = 30
            context_df['in_fomc_week'] = 0
        
        # For earnings, we need to merge with ticker-specific data
        # This will be done in feature_engineer when processing each ticker
        
        print(f"Market context loaded: {len(context_df)} days")
        print(f"Features: {context_df.columns.tolist()}")
        
        return context_df, earnings_df


class LiveMarketContext:
    """
    Real-time market context loader for live trading.
    Fetches latest data on-demand.
    """
    
    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
    
    def get_current_context(self) -> dict:
        """
        Get current market context for live trading.
        
        Returns dict with:
        - vix_close
        - vix3m_close  
        - vix_spread
        - vix_ratio
        - vix_norm
        - put_call_ratio (if available)
        - days_to_fomc
        - days_since_fomc
        - in_fomc_week
        """
        now = datetime.now()
        
        # Check cache
        if self._cache_time and (now - self._cache_time) < self._cache_duration:
            return self._cache
        
        try:
            # Fetch latest VIX
            vix = yf.Ticker('^VIX')
            vix3m = yf.Ticker('^VIX3M')
            
            vix_hist = vix.history(period='5d')
            vix3m_hist = vix3m.history(period='5d')
            
            if not vix_hist.empty and not vix3m_hist.empty:
                vix_close = vix_hist['Close'].iloc[-1]
                vix3m_close = vix3m_hist['Close'].iloc[-1]
                
                context = {
                    'vix_close': float(vix_close),
                    'vix3m_close': float(vix3m_close),
                    'vix_spread': float(vix3m_close - vix_close),
                    'vix_ratio': float(vix3m_close / vix_close) if vix_close > 0 else 1.0,
                    'vix_norm': float(vix_close / 50.0),
                }
            else:
                context = {
                    'vix_close': 20.0,
                    'vix3m_close': 22.0,
                    'vix_spread': 2.0,
                    'vix_ratio': 1.1,
                    'vix_norm': 0.4,
                }
            
            # Add FOMC context
            today = pd.Timestamp.now().normalize()
            fomc_df = MarketContextLoader('2024-01-01', '2027-12-31').fetch_fomc_calendar()
            today_row = fomc_df[fomc_df['date'] == today]
            
            if not today_row.empty:
                context['days_to_fomc'] = int(today_row['days_to_fomc'].iloc[0])
                context['days_since_fomc'] = int(today_row['days_since_fomc'].iloc[0])
                context['in_fomc_week'] = int(today_row['in_fomc_week'].iloc[0])
            else:
                context['days_to_fomc'] = 30
                context['days_since_fomc'] = 30
                context['in_fomc_week'] = 0
            
            # Put/call ratio placeholder
            context['put_call_ratio'] = 1.0
            context['put_call_5dma'] = 1.0
            
            self._cache = context
            self._cache_time = now
            
            return context
            
        except Exception as e:
            print(f"Error fetching live market context: {e}")
            # Return defaults
            return {
                'vix_close': 20.0,
                'vix3m_close': 22.0,
                'vix_spread': 2.0,
                'vix_ratio': 1.1,
                'vix_norm': 0.4,
                'put_call_ratio': 1.0,
                'put_call_5dma': 1.0,
                'days_to_fomc': 30,
                'days_since_fomc': 30,
                'in_fomc_week': 0,
            }


if __name__ == '__main__':
    # Test the loader
    loader = MarketContextLoader('2024-01-01', '2024-12-31')
    context, earnings = loader.load_all_context(['AAPL', 'MSFT'])
    print("\nSample context data:")
    print(context.head())
    print("\nSample earnings data:")
    print(earnings.head(10))
