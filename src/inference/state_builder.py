"""
State Builder for Inference
Ensures consistent state construction between training and live trading.
Must match exactly what the training environment produces.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


class StateBuilder:
    """
    Builds observation states for inference that match training exactly.
    Handles feature normalization with zero lag for live trading.
    """
    
    def __init__(
        self,
        tickers: List[str],
        tech_indicators: List[str],
        initial_amount: float = 100000
    ):
        """
        Initialize state builder.
        
        Args:
            tickers: List of stock tickers (must match training order)
            tech_indicators: List of technical indicators (must match training)
            initial_amount: Initial portfolio value for normalization
        """
        self.tickers = tickers
        self.tech_indicators = tech_indicators
        self.initial_amount = initial_amount
        self.stock_dim = len(tickers)
        
        # Running statistics for technical indicators (for z-score normalization)
        # These should be computed from training data or loaded from saved stats
        self.indicator_means = {}
        self.indicator_stds = {}
        
        # Track peak portfolio value for drawdown calculation
        self.peak_portfolio_value = initial_amount
        
        # Track price history for momentum calculation (matches training env)
        self.price_history = []
        
        # State dimension calculation (must match trading_env.py)
        # cash + prices + holdings + position_ratio + portfolio_growth + drawdown + momentum + indicators + turbulence
        self.state_dim = 1 + 2 * self.stock_dim + 4 + len(tech_indicators) * self.stock_dim + 1
    
    def set_indicator_stats(self, means: Dict[str, float], stds: Dict[str, float]):
        """
        Set pre-computed indicator statistics from training data.
        This ensures consistent normalization between training and inference.
        
        Args:
            means: Dict of indicator_name -> mean value
            stds: Dict of indicator_name -> std value
        """
        self.indicator_means = means
        self.indicator_stds = stds
    
    def compute_indicator_stats_from_data(self, historical_df: pd.DataFrame):
        """
        Compute indicator statistics from historical data.
        Call this once with training data to get consistent normalization.
        
        Args:
            historical_df: DataFrame with technical indicators already computed
        """
        for indicator in self.tech_indicators:
            if indicator in historical_df.columns:
                self.indicator_means[indicator] = historical_df[indicator].mean()
                self.indicator_stds[indicator] = historical_df[indicator].std()
    
    def build_state(
        self,
        cash: float,
        portfolio_value: float,
        stock_prices: Dict[str, float],
        stocks_owned: Dict[str, float],
        tech_indicator_values: Dict[str, Dict[str, float]],
        turbulence: float = 0.0
    ) -> np.ndarray:
        """
        Build state vector matching training environment exactly.
        
        Args:
            cash: Current cash balance
            portfolio_value: Total portfolio value
            stock_prices: Dict of ticker -> current price
            stocks_owned: Dict of ticker -> shares owned
            tech_indicator_values: Dict of ticker -> {indicator: value}
            turbulence: Current turbulence index value
        
        Returns:
            State vector matching training format
        """
        state = []
        
        # 1. Normalized cash (same as training)
        state.append(cash / self.initial_amount)
        
        # 2. Normalized stock prices (relative to first stock, capped)
        prices = [stock_prices.get(ticker, 0.0) for ticker in self.tickers]
        if prices[0] > 0:
            normalized_prices = np.clip(np.array(prices) / prices[0], 0, 10).tolist()
        else:
            normalized_prices = [1.0] * self.stock_dim
        state.extend(normalized_prices)
        
        # 3. Normalized holdings (as fraction of max reasonable position)
        holdings = [stocks_owned.get(ticker, 0.0) for ticker in self.tickers]
        normalized_holdings = np.clip(np.array(holdings) / 100, 0, 10).tolist()
        state.extend(normalized_holdings)
        
        # 4. Position-aware features
        total_stock_value = sum(
            stocks_owned.get(t, 0) * stock_prices.get(t, 0) 
            for t in self.tickers
        )
        position_ratio = total_stock_value / (portfolio_value + 1e-6)
        state.append(position_ratio)
        
        portfolio_growth = portfolio_value / self.initial_amount
        state.append(portfolio_growth)
        
        # Update peak for drawdown
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / (self.peak_portfolio_value + 1e-6)
        state.append(current_drawdown)
        
        # 4b. Momentum feature: 5-day price momentum (matches training env)
        prices = np.array([stock_prices.get(ticker, 0.0) for ticker in self.tickers])
        self.price_history.append(prices.copy())
        if len(self.price_history) > 10:
            self.price_history.pop(0)
        
        if len(self.price_history) >= 5:
            old_prices = self.price_history[-5]
            momentum = np.mean((prices - old_prices) / (old_prices + 1e-8))
            state.append(float(np.clip(momentum, -1, 1)))
        else:
            state.append(0.0)
        
        # 5. Technical indicators (z-score normalized, same as training)
        for indicator in self.tech_indicators:
            for ticker in self.tickers:
                raw_value = tech_indicator_values.get(ticker, {}).get(indicator, 0.0)
                
                # Z-score normalize using training statistics
                mean = self.indicator_means.get(indicator, 0.0)
                std = self.indicator_stds.get(indicator, 1.0)
                
                if std > 0:
                    normalized_value = (raw_value - mean) / (std + 1e-8)
                    normalized_value = np.clip(normalized_value, -5, 5)
                else:
                    normalized_value = 0.0
                
                state.append(normalized_value)
        
        # 6. Turbulence (normalized)
        state.append(turbulence / 100)
        
        # Convert to numpy array
        state_array = np.array(state, dtype=np.float32)
        
        return state_array
    
    def reset_peak(self, initial_value: float = None):
        """Reset peak portfolio value and price history (call at start of new trading session)."""
        if initial_value is not None:
            self.peak_portfolio_value = initial_value
        else:
            self.peak_portfolio_value = self.initial_amount
        self.price_history = []


class IndicatorCalculator:
    """
    Calculate technical indicators for live data with zero lag.
    Uses the same stockstats library as training.
    """
    
    def __init__(self, tech_indicators: List[str]):
        self.tech_indicators = tech_indicators
    
    def calculate_from_bars(
        self, 
        historical_bars: pd.DataFrame,
        current_price: float = None
    ) -> Dict[str, float]:
        """
        Calculate technical indicators from historical OHLCV bars.
        
        Args:
            historical_bars: DataFrame with columns [open, high, low, close, volume]
            current_price: Optional current price to append as latest bar
        
        Returns:
            Dict of indicator_name -> value for the latest bar
        """
        from stockstats import StockDataFrame as Sdf
        
        df = historical_bars.copy()
        
        # Append current price as latest bar if provided
        if current_price is not None:
            latest = pd.DataFrame([{
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 0
            }])
            df = pd.concat([df, latest], ignore_index=True)
        
        # Convert to StockDataFrame and calculate indicators
        stock = Sdf.retype(df)
        
        indicators = {}
        for indicator in self.tech_indicators:
            try:
                # Access indicator to trigger calculation
                stock[indicator]
                indicators[indicator] = float(stock[indicator].iloc[-1])
            except Exception as e:
                indicators[indicator] = 0.0
        
        return indicators
