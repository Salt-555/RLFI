"""
Live Paper Trading with Alpaca API
Real-time market data streaming and automated trading
"""
import os
import time
import signal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from stable_baselines3 import PPO, A2C, DDPG, SAC

# Use our own FeatureEngineer for consistency with training
from src.data.feature_engineer import FeatureEngineer

load_dotenv()

# Rate limiting for Alpaca API (200 requests/minute = 0.3 seconds between calls to be safe)
API_RATE_LIMIT_DELAY = 0.35  # seconds between API calls


class LivePaperTrader:
    """
    Live paper trading system using Alpaca API
    Streams real-time data and executes trades based on RL model predictions
    """
    
    def __init__(
        self,
        model,
        tickers: List[str],
        initial_capital: float,
        tech_indicators: List[str],
        model_metadata: Dict,
        update_frequency: str = "1Min"
    ):
        """
        Initialize live paper trader
        
        Args:
            model: Trained RL model (PPO, A2C, or DDPG)
            tickers: List of stock tickers to trade
            initial_capital: Starting capital
            tech_indicators: Technical indicators to calculate
            model_metadata: Model configuration metadata
            update_frequency: How often to update ('1Min', '5Min', '15Min', '1Hour')
        """
        self.model = model
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.tech_indicators = tech_indicators
        self.model_metadata = model_metadata
        self.update_frequency = update_frequency
        
        # Initialize Alpaca clients
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key or api_key == 'your_api_key_here':
            raise ValueError("Alpaca API keys not configured in .env file")
        
        print(f"Connecting to Alpaca with API key: {api_key[:10]}...")
        
        try:
            self.trading_client = TradingClient(api_key, secret_key, paper=True)
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
            print("Alpaca clients initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize Alpaca clients: {str(e)}")
        
        # Trading state
        self.positions = {ticker: 0 for ticker in tickers}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.trades_executed = 0
        self.transaction_costs = 0.0
        
        # History tracking
        self.portfolio_history = []
        self.trade_log = []
        self.timestamp_history = []
        
        # Control flag for stopping
        self.stop_trading = False
        
        # Feature engineering - use same class as training for consistency
        self.fe = FeatureEngineer(
            tech_indicator_list=tech_indicators,
            use_turbulence=False  # Disable turbulence for live trading (needs 252 days history)
        )
        
        # Position sizing limit - must match training env's max_position_pct
        # Training env limits each buy to portfolio_value * max_position_pct * action
        self.max_position_pct = model_metadata.get('max_position_pct', 0.3)
        
        # State dimensions - must match training env: 1 + 2*stock_dim + 4 + indicators*stock_dim + 1
        self.stock_dim = len(tickers)
        self.state_space = 1 + 2 * self.stock_dim + 4 + len(tech_indicators) * self.stock_dim + 1
        
        # Track peak portfolio value for drawdown calculation (matches training env)
        self.peak_portfolio_value = initial_capital
        
        # Track price history for momentum calculation (matches training env)
        self.price_history = []
        
        # Track initial prices for per-stock normalization (matches trading_env._initial_prices)
        self._initial_prices = None
        
    def get_account_info(self) -> Dict:
        """Get current account information from Alpaca"""
        try:
            time.sleep(API_RATE_LIMIT_DELAY)  # Rate limiting
            account = self.trading_client.get_account()
            print(f"Account retrieved: {account.account_number}")
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity)
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to connect to Alpaca: {str(e)}")
    
    def get_current_positions(self) -> Dict:
        """Get current positions from Alpaca"""
        try:
            time.sleep(API_RATE_LIMIT_DELAY)  # Rate limiting
            positions = self.trading_client.get_all_positions()
            position_dict = {}
            for pos in positions:
                if pos.symbol in self.tickers:
                    position_dict[pos.symbol] = {
                        'qty': float(pos.qty),
                        'market_value': float(pos.market_value),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'unrealized_pl': float(pos.unrealized_pl)
                    }
            return position_dict
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all tickers"""
        try:
            time.sleep(API_RATE_LIMIT_DELAY)  # Rate limiting
            request = StockLatestQuoteRequest(symbol_or_symbols=self.tickers)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            prices = {}
            for ticker in self.tickers:
                if ticker in quotes:
                    ask_price = float(quotes[ticker].ask_price)
                    # Use bid price if ask is 0
                    if ask_price == 0:
                        ask_price = float(quotes[ticker].bid_price)
                    prices[ticker] = ask_price if ask_price > 0 else 0.0
                else:
                    prices[ticker] = 0.0
            
            return prices
        except Exception as e:
            print(f"Error getting latest prices: {e}")
            return {ticker: 0.0 for ticker in self.tickers}
    
    def get_historical_data(self, days: int = 120) -> pd.DataFrame:
        """
        Get historical data for feature engineering
        
        Args:
            days: Number of days of historical data to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            time.sleep(API_RATE_LIMIT_DELAY)  # Rate limiting
            request = StockBarsRequest(
                symbol_or_symbols=self.tickers,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            # Convert to DataFrame format expected by FinRL
            data_frames = []
            for ticker in self.tickers:
                if ticker in bars.data:
                    ticker_bars = bars.data[ticker]
                    df = pd.DataFrame([{
                        'date': bar.timestamp.strftime('%Y-%m-%d'),
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': float(bar.volume),
                        'tic': ticker
                    } for bar in ticker_bars])
                    data_frames.append(df)
            
            if data_frames:
                combined_df = pd.concat(data_frames, ignore_index=True)
                combined_df = combined_df.sort_values(['date', 'tic']).reset_index(drop=True)
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, historical_df: pd.DataFrame, current_prices: Dict[str, float]) -> np.ndarray:
        """
        Calculate technical indicators and construct state vector.
        
        CRITICAL: This must produce a state vector identical in structure, dimension,
        and normalization to StockTradingEnv._get_state() used during training.
        
        Training state layout:
        [cash/initial] + [prices_normalized] + [holdings_normalized] +
        [position_ratio, portfolio_growth, drawdown, momentum] +
        [z-score_indicators...] + [turbulence/100]
        
        Args:
            historical_df: Historical OHLCV data
            current_prices: Current prices for each ticker
        
        Returns:
            State vector for model prediction
        """
        try:
            # Add current prices as latest row
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_rows = []
            
            for ticker in self.tickers:
                # Get last known OHLCV for this ticker
                ticker_hist = historical_df[historical_df['tic'] == ticker]
                
                # Skip ticker if no historical data or current price is 0
                if ticker_hist.empty or current_prices.get(ticker, 0) <= 0:
                    print(f"Skipping {ticker}: no data or invalid price")
                    continue
                
                ticker_data = ticker_hist.iloc[-1]
                
                current_rows.append({
                    'date': current_date,
                    'open': current_prices.get(ticker, ticker_data['close']),
                    'high': current_prices.get(ticker, ticker_data['close']),
                    'low': current_prices.get(ticker, ticker_data['close']),
                    'close': current_prices.get(ticker, ticker_data['close']),
                    'volume': ticker_data['volume'],
                    'tic': ticker
                })
            
            current_df = pd.DataFrame(current_rows)
            extended_df = pd.concat([historical_df, current_df], ignore_index=True)
            
            # Ensure date column is datetime
            extended_df['date'] = pd.to_datetime(extended_df['date'])
            
            # Calculate technical indicators using same method as training
            # NOTE: FeatureEngineer.add_technical_indicators() already applies z-score
            # normalization and clips to [-5, 5], matching training preprocessing
            processed = self.fe.add_technical_indicators(extended_df)
            
            # Get latest row for each ticker (sorted to match training order)
            latest_date = processed['date'].max()
            latest_data = processed[processed['date'] == latest_date].sort_values('tic')
            
            # Build ordered price array matching self.tickers order
            prices = np.array([current_prices.get(t, 0.0) for t in self.tickers])
            holdings = np.array([self.positions.get(t, 0) for t in self.tickers])
            
            # === BUILD STATE VECTOR (must match StockTradingEnv._get_state exactly) ===
            state = []
            
            # 1. Normalized cash (same as training)
            state.append(self.cash / self.initial_capital)
            
            # 2. Log returns: stationary, scale-invariant features
            #    Matches trading_env._get_state() log return calculation
            if len(self.price_history) >= 2:
                prev_prices = self.price_history[-2]  # Yesterday's prices
                for i in range(self.stock_dim):
                    if prev_prices[i] > 0:
                        log_ret = np.log(prices[i] / prev_prices[i])
                        state.append(np.clip(log_ret, -1, 1))
                    else:
                        state.append(0.0)
            else:
                # First call - no previous price yet
                state.extend([0.0] * self.stock_dim)
            
            # 3. Normalized holdings: (shares * price) / (initial_amount * 0.1)
            #    Matches trading_env value-based normalization
            for i in range(self.stock_dim):
                holding_value = holdings[i] * prices[i]
                state.append(np.clip(holding_value / (self.initial_capital * 0.1), 0, 10))
            
            # 4. Position-aware features (NEW - was missing before!)
            total_stock_value = np.sum(holdings * prices)
            position_ratio = total_stock_value / (self.portfolio_value + 1e-6)
            state.append(position_ratio)
            
            portfolio_growth = self.portfolio_value / self.initial_capital
            state.append(portfolio_growth)
            
            # Update peak for drawdown
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / (self.peak_portfolio_value + 1e-6)
            state.append(current_drawdown)
            
            # 5. Momentum feature: 5-day price momentum (average across stocks)
            self.price_history.append(prices.copy())
            if len(self.price_history) > 10:
                self.price_history.pop(0)
            
            if len(self.price_history) >= 5:
                old_prices = self.price_history[-5]
                momentum = np.mean((prices - old_prices) / (old_prices + 1e-8))
                state.append(np.clip(momentum, -1, 1))
            else:
                state.append(0.0)
            
            # 6. Technical indicators (already z-score normalized by FeatureEngineer)
            for indicator in self.tech_indicators:
                for ticker in self.tickers:
                    ticker_data = latest_data[latest_data['tic'] == ticker]
                    if not ticker_data.empty and indicator in ticker_data.columns:
                        state.append(float(ticker_data[indicator].iloc[0]))
                    else:
                        state.append(0.0)
            
            # 7. Turbulence (normalized, 0 for live trading since we don't calculate it)
            state.append(0.0)
            
            return np.array(state, dtype=np.float32)
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            import traceback
            traceback.print_exc()
            # Return zero state on error
            return np.zeros(self.state_space, dtype=np.float32)
    
    def execute_trade(self, ticker: str, action: float, current_price: float, available_cash: float = None):
        """
        Execute trade based on model action
        
        Args:
            ticker: Stock ticker
            action: Model action (positive = buy, negative = sell)
            current_price: Current price of the stock
            available_cash: Available buying power (if None, will fetch from Alpaca)
        """
        try:
            if current_price <= 0:
                return
            
            # Get available cash if not provided
            if available_cash is None:
                account_info = self.get_account_info()
                available_cash = float(account_info['buying_power'])
            
            if action > 0.01:
                # Buy order - match training environment position sizing:
                # Training env: max_position_value = portfolio_value * max_position_pct * abs(action)
                # Then also limited by available cash
                max_position_value = self.portfolio_value * self.max_position_pct * abs(action)
                cash_to_use = min(max_position_value, available_cash)
                shares_to_buy = int(cash_to_use / (current_price * 1.001))  # Include 0.1% transaction cost
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * 1.001
                    
                    order_request = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares_to_buy,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    time.sleep(API_RATE_LIMIT_DELAY)  # Rate limiting
                    order = self.trading_client.submit_order(order_request)
                    
                    # Update local state
                    self.cash -= cost
                    self.positions[ticker] = self.positions.get(ticker, 0) + shares_to_buy
                    self.trades_executed += 1
                    self.transaction_costs += cost * 0.001
                    
                    self.trade_log.append({
                        'timestamp': datetime.now(),
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost,
                        'order_id': order.id
                    })
                    
                    print(f"BUY {shares_to_buy} shares of {ticker} @ ${current_price:.2f}")
                    
            elif action < -0.01:
                # Sell order - calculate shares based on current position and action magnitude
                current_position = self.positions.get(ticker, 0)
                
                if current_position > 0:
                    shares_to_sell = int(current_position * abs(action))
                    shares_to_sell = min(shares_to_sell, current_position)
                    
                    if shares_to_sell > 0:
                        order_request = MarketOrderRequest(
                            symbol=ticker,
                            qty=shares_to_sell,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        time.sleep(API_RATE_LIMIT_DELAY)  # Rate limiting
                        order = self.trading_client.submit_order(order_request)
                        
                        # Update local state
                        proceeds = shares_to_sell * current_price * 0.999  # Subtract 0.1% transaction cost
                        self.cash += proceeds
                        self.positions[ticker] -= shares_to_sell
                        self.trades_executed += 1
                        self.transaction_costs += proceeds * 0.001
                        
                        self.trade_log.append({
                            'timestamp': datetime.now(),
                            'ticker': ticker,
                            'action': 'SELL',
                            'shares': shares_to_sell,
                            'price': current_price,
                            'proceeds': proceeds,
                            'order_id': order.id
                        })
                        
                        print(f"SELL {shares_to_sell} shares of {ticker} @ ${current_price:.2f}")
                    
        except Exception as e:
            print(f"Error executing trade for {ticker}: {e}")
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current prices"""
        holdings_value = sum(
            self.positions.get(ticker, 0) * current_prices.get(ticker, 0)
            for ticker in self.tickers
        )
        self.portfolio_value = self.cash + holdings_value
        
        self.portfolio_history.append(self.portfolio_value)
        self.timestamp_history.append(datetime.now())
    
    def run_live_trading(self, duration_minutes: int = 60, update_interval_seconds: int = 60):
        """
        Run live paper trading for specified duration
        
        Args:
            duration_minutes: How long to run trading (in minutes)
            update_interval_seconds: How often to check prices and make decisions
        """
        print("="*60)
        print("STARTING LIVE PAPER TRADING")
        print("="*60)
        print(f"Tickers: {', '.join(self.tickers)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Update Interval: {update_interval_seconds} seconds")
        print("="*60)
        
        # Get initial account info
        account_info = self.get_account_info()
        if account_info:
            print(f"Alpaca Account Connected:")
            print(f"  Cash: ${account_info['cash']:,.2f}")
            print(f"  Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        
        # Get historical data for feature engineering (120 days for indicator lookback)
        print("\nFetching historical data for technical indicators...")
        historical_df = self.get_historical_data(days=120)
        
        if historical_df.empty:
            print("ERROR: Could not fetch historical data")
            return
        
        print(f"Loaded {len(historical_df)} rows of historical data")
        
        # Trading loop
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        iteration = 0
        
        print(f"\nStarting trading loop until {end_time.strftime('%H:%M:%S')}")
        print("-"*60)
        
        while datetime.now() < end_time and not self.stop_trading:
            iteration += 1
            current_time = datetime.now()
            
            print(f"\n[{current_time.strftime('%H:%M:%S')}] Iteration {iteration}")
            
            if self.stop_trading:
                print("\nStopping trading as requested...")
                break
            
            # Get latest prices
            current_prices = self.get_latest_prices()
            print(f"Current Prices: {current_prices}")
            
            # Calculate state with technical indicators
            state = self.calculate_features(historical_df, current_prices)
            
            # Get model prediction
            action, _ = self.model.predict(state, deterministic=True)
            print(f"Model Actions: {action}")
            
            # Fetch account info once for all trades
            account_info = self.get_account_info()
            available_cash = float(account_info['buying_power'])
            
            # Execute trades for each ticker
            for idx, ticker in enumerate(self.tickers):
                if idx < len(action):
                    self.execute_trade(ticker, action[idx], current_prices.get(ticker, 0), available_cash)
                    # Update available cash after each trade (approximate)
                    if action[idx] > 0.01:
                        cash_used = available_cash * abs(action[idx])
                        available_cash = max(0, available_cash - cash_used)
            
            # Update portfolio value
            self.update_portfolio_value(current_prices)
            
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Positions: {self.positions}")
            
            # Wait for next update (check stop flag periodically)
            if datetime.now() < end_time and not self.stop_trading:
                # Sleep in smaller chunks to be more responsive to interrupts
                sleep_remaining = update_interval_seconds
                while sleep_remaining > 0 and not self.stop_trading:
                    sleep_chunk = min(1, sleep_remaining)  # Sleep 1 second at a time
                    time.sleep(sleep_chunk)
                    sleep_remaining -= sleep_chunk
        
        print("\n" + "="*60)
        if self.stop_trading:
            print("LIVE PAPER TRADING STOPPED BY USER")
        else:
            print("LIVE PAPER TRADING COMPLETED")
        print("="*60)
        print(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Total Return: ${self.portfolio_value - self.initial_capital:,.2f} ({((self.portfolio_value/self.initial_capital - 1)*100):.2f}%)")
        print(f"Total Trades: {self.trades_executed}")
        print(f"Transaction Costs: ${self.transaction_costs:,.2f}")
        print("="*60)
        
        # Reset stop flag
        self.stop_trading = False
    
    def get_results(self) -> Dict:
        """Get trading results summary"""
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio_value,
            'total_return': self.portfolio_value - self.initial_capital,
            'return_pct': (self.portfolio_value / self.initial_capital - 1) * 100,
            'trades_executed': self.trades_executed,
            'transaction_costs': self.transaction_costs,
            'portfolio_history': self.portfolio_history,
            'timestamp_history': self.timestamp_history,
            'trade_log': self.trade_log,
            'final_positions': self.positions,
            'final_cash': self.cash
        }
