"""
Walk-Forward Validation System

Implements rolling window cross-validation for time series data.
Instead of static train/test splits, this creates overlapping windows
that move forward in time, always using recent data for validation.

Key features:
- Rolling windows based on distance from current date
- Expanding window option (grows over time)
- Multiple validation folds for robustness
- Automatic window sizing based on data availability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import json


class WalkForwardValidator:
    """
    Implements walk-forward validation for time series trading data.
    
    Instead of fixed train/test dates, uses rolling windows relative to
current date. This ensures the system works correctly as time progresses.
    """
    
    def __init__(
        self,
        train_window_days: int = 1260,  # 5 years of trading days
        val_window_days: int = 252,     # 1 year for validation
        test_window_days: int = 126,    # 6 months for testing
        min_train_days: int = 504,      # Minimum 2 years training data
        step_size_days: int = 63,       # Move forward 3 months each fold
        n_folds: int = 3,               # Number of validation folds
        expanding_window: bool = True,  # Use expanding window (grows over time)
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_window_days: Size of training window in trading days
            val_window_days: Size of validation window in trading days  
            test_window_days: Size of test window (can overlap with recent data)
            min_train_days: Minimum training data required
            step_size_days: How much to slide window forward each fold
            n_folds: Number of validation folds to generate
            expanding_window: If True, training window grows; if False, fixed size
        """
        self.train_window_days = train_window_days
        self.val_window_days = val_window_days
        self.test_window_days = test_window_days
        self.min_train_days = min_train_days
        self.step_size_days = step_size_days
        self.n_folds = n_folds
        self.expanding_window = expanding_window
    
    def create_windows(
        self, 
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Create walk-forward validation windows based on reference date.
        
        Args:
            df: DataFrame with stock data (must have 'date' column)
            reference_date: Date to use as "today" (defaults to actual today)
            
        Returns:
            List of window dictionaries with train/val/test indices
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Get unique trading days from data
        all_dates = sorted(df['date'].unique())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        
        # Find reference date in data (or closest previous date)
        ref_date = pd.Timestamp(reference_date).normalize()
        if ref_date not in date_to_idx:
            # Find closest date before reference
            valid_dates = [d for d in all_dates if d <= ref_date]
            if not valid_dates:
                raise ValueError(f"Reference date {ref_date} is before any data")
            ref_date = max(valid_dates)
        
        ref_idx = date_to_idx[ref_date]
        
        # Check if we have enough data
        if ref_idx < self.min_train_days:
            raise ValueError(
                f"Not enough historical data. Have {ref_idx} days, "
                f"need at least {self.min_train_days}"
            )
        
        windows = []
        
        # Create n_folds walk-forward windows
        for fold in range(self.n_folds):
            # Calculate window boundaries
            # Test window ends at reference date
            test_end_idx = ref_idx
            test_start_idx = max(0, test_end_idx - self.test_window_days)
            
            # Validation window ends before test window
            val_end_idx = test_start_idx - 1 if test_start_idx > 0 else test_end_idx
            val_start_idx = max(0, val_end_idx - self.val_window_days)
            
            # Training window
            if self.expanding_window:
                # Expanding window: starts from beginning, grows over time
                train_start_idx = 0
                train_end_idx = val_start_idx - 1 if val_start_idx > 0 else val_end_idx
            else:
                # Fixed window: slides forward
                train_end_idx = val_start_idx - 1 if val_start_idx > 0 else val_end_idx
                train_start_idx = max(0, train_end_idx - self.train_window_days)
            
            # Check minimum training size
            train_size = train_end_idx - train_start_idx + 1
            if train_size < self.min_train_days:
                print(f"Warning: Fold {fold+1} has only {train_size} training days")
                continue
            
            # Get actual dates
            train_dates = all_dates[train_start_idx:train_end_idx+1]
            val_dates = all_dates[val_start_idx:val_end_idx+1]
            test_dates = all_dates[test_start_idx:test_end_idx+1]
            
            window = {
                'fold': fold + 1,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'train_days': len(train_dates),
                'val_start': val_dates[0],
                'val_end': val_dates[-1],
                'val_days': len(val_dates),
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'test_days': len(test_dates),
                'reference_date': ref_date,
            }
            windows.append(window)
            
            # Slide reference date back for next fold
            ref_idx = max(0, ref_idx - self.step_size_days)
            ref_date = all_dates[ref_idx] if ref_idx < len(all_dates) else all_dates[0]
        
        return windows
    
    def split_data_for_window(
        self, 
        df: pd.DataFrame, 
        window: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train/val/test based on window specification.
        
        Args:
            df: Full dataframe with all data
            window: Window dictionary from create_windows()
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = df[
            (df['date'] >= window['train_start']) & 
            (df['date'] <= window['train_end'])
        ].copy()
        
        val_df = df[
            (df['date'] >= window['val_start']) & 
            (df['date'] <= window['val_end'])
        ].copy()
        
        test_df = df[
            (df['date'] >= window['test_start']) & 
            (df['date'] <= window['test_end'])
        ].copy()
        
        return train_df, val_df, test_df
    
    def get_optimal_train_val_split(
        self,
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get a single optimal train/val/test split based on current date.
        
        This is a simplified version that creates one good split rather than
        multiple folds. Uses the most recent data for validation/testing.
        
        Args:
            df: DataFrame with stock data
            reference_date: Reference date (default: today)
            train_ratio: Fraction of available data for training
            val_ratio: Fraction of available data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        all_dates = sorted(df['date'].unique())
        ref_date = pd.Timestamp(reference_date).normalize()
        
        # Find reference date in data
        if ref_date not in all_dates:
            valid_dates = [d for d in all_dates if d <= ref_date]
            if not valid_dates:
                raise ValueError(f"Reference date {ref_date} is before any data")
            ref_date = max(valid_dates)
        
        ref_idx = all_dates.index(ref_date)
        total_days = ref_idx + 1
        
        # Calculate split indices
        train_end_idx = int(total_days * train_ratio)
        val_end_idx = train_end_idx + int(total_days * val_ratio)
        
        # Ensure minimum sizes
        if train_end_idx < self.min_train_days:
            train_end_idx = min(self.min_train_days, total_days - 20)
            val_end_idx = train_end_idx + min(63, total_days - train_end_idx - 10)
        
        # Get dates
        train_dates = all_dates[:train_end_idx]
        val_dates = all_dates[train_end_idx:val_end_idx]
        test_dates = all_dates[val_end_idx:ref_idx+1]
        
        # Split data
        train_df = df[df['date'].isin(train_dates)].copy()
        val_df = df[df['date'].isin(val_dates)].copy()
        test_df = df[df['date'].isin(test_dates)].copy()
        
        print(f"Walk-forward split from {all_dates[0]} to {ref_date}:")
        print(f"  Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
        print(f"  Val: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")
        print(f"  Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
        
        return train_df, val_df, test_df


def print_window_summary(windows: List[Dict]):
    """Print a summary of walk-forward windows."""
    print("\n=== WALK-FORWARD VALIDATION WINDOWS ===\n")
    
    for w in windows:
        print(f"Fold {w['fold']}:")
        print(f"  Reference date: {w['reference_date']}")
        print(f"  Train: {w['train_start']} to {w['train_end']} ({w['train_days']} days)")
        print(f"  Val:   {w['val_start']} to {w['val_end']} ({w['val_days']} days)")
        print(f"  Test:  {w['test_start']} to {w['test_end']} ({w['test_days']} days)")
        print()


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='B')  # Business days
    df = pd.DataFrame({
        'date': np.repeat(dates, 5),
        'tic': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'] * len(dates),
        'close': np.random.randn(len(dates) * 5) * 10 + 100
    })
    
    # Create validator
    validator = WalkForwardValidator(
        train_window_days=1260,  # 5 years
        val_window_days=252,     # 1 year
        test_window_days=126,    # 6 months
        n_folds=3,
        expanding_window=True
    )
    
    # Create windows based on "today"
    windows = validator.create_windows(df, reference_date=datetime(2024, 12, 31))
    print_window_summary(windows)
    
    # Get single optimal split
    train_df, val_df, test_df = validator.get_optimal_train_val_split(df)
    print(f"\nOptimal split sizes:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
