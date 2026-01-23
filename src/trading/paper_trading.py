import os
import time
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import pandas as pd
import yfinance as yf
from src.inference.state_builder import StateBuilder, IndicatorCalculator


class PaperTrader:
    def __init__(
        self, 
        model, 
        config: Dict[str, Any], 
        api_key: str = None, 
        secret_key: str = None,
        indicator_stats: Optional[Dict] = None
    ):
        load_dotenv()
        
        self.model = model
        self.config = config
        
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        
        self.client = TradingClient(self.api_key, self.secret_key, paper=True)
        
        self.tickers = config['data']['tickers']
        self.tech_indicators = config['features']['technical_indicators']
        self.check_interval = config['paper_trading'].get('check_interval', 300)
        self.max_position_size = config['environment'].get('max_position_pct', 0.3)
        self.stop_loss_pct = config['paper_trading'].get('stop_loss_pct', 0.02)
        
        # Initialize state builder for consistent state construction
        self.state_builder = StateBuilder(
            tickers=self.tickers,
            tech_indicators=self.tech_indicators,
            initial_amount=config['environment']['initial_amount']
        )
        
        # Set indicator statistics if provided (from training data)
        if indicator_stats:
            self.state_builder.set_indicator_stats(
                means=indicator_stats.get('means', {}),
                stds=indicator_stats.get('stds', {})
            )
        
        # Initialize indicator calculator
        self.indicator_calculator = IndicatorCalculator(self.tech_indicators)
        
        # Cache for historical data (refreshed periodically)
        self.historical_data_cache = {}
        self.cache_refresh_interval = 3600  # Refresh every hour
        self.last_cache_refresh = 0
        
        self.trade_log = []
        
    def get_account_info(self) -> Dict[str, Any]:
        account = self.client.get_account()
        return {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'equity': float(account.equity)
        }
    
    def get_positions(self) -> Dict[str, Any]:
        positions = self.client.get_all_positions()
        position_dict = {}
        
        for position in positions:
            position_dict[position.symbol] = {
                'qty': float(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc)
            }
        
        return position_dict
    
    def get_current_prices(self) -> Dict[str, float]:
        prices = {}
        for ticker in self.tickers:
            try:
                bars = self.client.get_latest_bar(ticker)
                prices[ticker] = float(bars.c)
            except Exception as e:
                print(f"Error getting price for {ticker}: {e}")
                prices[ticker] = 0.0
        
        return prices
    
    def execute_trade(self, ticker: str, action: float, current_price: float):
        account_info = self.get_account_info()
        positions = self.get_positions()
        
        current_qty = positions.get(ticker, {}).get('qty', 0)
        
        if action > 0.1:
            max_buy_value = account_info['cash'] * self.max_position_size * abs(action)
            shares_to_buy = int(max_buy_value / current_price)
            
            if shares_to_buy > 0:
                try:
                    order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares_to_buy,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    order = self.client.submit_order(order_data)
                    
                    self.trade_log.append({
                        'timestamp': pd.Timestamp.now(),
                        'ticker': ticker,
                        'action': 'BUY',
                        'qty': shares_to_buy,
                        'price': current_price,
                        'order_id': order.id
                    })
                    
                    print(f"BUY {shares_to_buy} shares of {ticker} at ${current_price:.2f}")
                    
                except Exception as e:
                    print(f"Error executing BUY order for {ticker}: {e}")
        
        elif action < -0.1 and current_qty > 0:
            shares_to_sell = int(current_qty * abs(action))
            shares_to_sell = min(shares_to_sell, current_qty)
            
            if shares_to_sell > 0:
                try:
                    order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    order = self.client.submit_order(order_data)
                    
                    self.trade_log.append({
                        'timestamp': pd.Timestamp.now(),
                        'ticker': ticker,
                        'action': 'SELL',
                        'qty': shares_to_sell,
                        'price': current_price,
                        'order_id': order.id
                    })
                    
                    print(f"SELL {shares_to_sell} shares of {ticker} at ${current_price:.2f}")
                    
                except Exception as e:
                    print(f"Error executing SELL order for {ticker}: {e}")
    
    def check_stop_loss(self):
        positions = self.get_positions()
        
        for ticker, position in positions.items():
            unrealized_plpc = position['unrealized_plpc']
            
            if unrealized_plpc < -self.stop_loss_pct:
                print(f"STOP LOSS triggered for {ticker}: {unrealized_plpc:.2%} loss")
                
                try:
                    qty = int(position['qty'])
                    order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    order = self.client.submit_order(order_data)
                    
                    self.trade_log.append({
                        'timestamp': pd.Timestamp.now(),
                        'ticker': ticker,
                        'action': 'STOP_LOSS',
                        'qty': qty,
                        'price': position['current_price'],
                        'order_id': order.id
                    })
                    
                except Exception as e:
                    print(f"Error executing stop loss for {ticker}: {e}")
    
    def run_trading_loop(self, duration_hours: int = 24):
        print(f"Starting paper trading for {duration_hours} hours...")
        print(f"Monitoring tickers: {self.tickers}")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        while time.time() < end_time:
            try:
                account_info = self.get_account_info()
                print(f"\nPortfolio Value: ${account_info['portfolio_value']:,.2f}")
                
                prices = self.get_current_prices()
                
                state = self._build_state(account_info, prices)
                
                action, _ = self.model.predict(state, deterministic=True)
                
                for i, ticker in enumerate(self.tickers):
                    if i < len(action):
                        self.execute_trade(ticker, action[i], prices[ticker])
                
                self.check_stop_loss()
                
                print(f"Sleeping for {self.check_interval} seconds...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\nTrading interrupted by user")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)
        
        print("\nPaper trading session complete")
        self.print_summary()
    
    def _refresh_historical_cache(self):
        """Refresh historical data cache for indicator calculation."""
        current_time = time.time()
        if current_time - self.last_cache_refresh < self.cache_refresh_interval:
            return  # Cache still fresh
        
        print("Refreshing historical data cache...")
        for ticker in self.tickers:
            try:
                # Get 60 days of historical data for indicator calculation
                df = yf.download(ticker, period='60d', progress=False)
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
                    self.historical_data_cache[ticker] = df
            except Exception as e:
                print(f"Error fetching historical data for {ticker}: {e}")
        
        self.last_cache_refresh = current_time
    
    def _calculate_indicators(self, ticker: str, current_price: float) -> Dict[str, float]:
        """Calculate technical indicators for a ticker using cached historical data."""
        if ticker not in self.historical_data_cache:
            return {ind: 0.0 for ind in self.tech_indicators}
        
        hist_df = self.historical_data_cache[ticker]
        return self.indicator_calculator.calculate_from_bars(hist_df, current_price)
    
    def _build_state(self, account_info: Dict, prices: Dict[str, float]) -> np.ndarray:
        """Build state vector matching training environment exactly."""
        # Refresh historical data cache if needed
        self._refresh_historical_cache()
        
        # Get current positions
        positions = self.get_positions()
        stocks_owned = {ticker: positions.get(ticker, {}).get('qty', 0) for ticker in self.tickers}
        
        # Calculate technical indicators for each ticker
        tech_indicator_values = {}
        for ticker in self.tickers:
            current_price = prices.get(ticker, 0)
            tech_indicator_values[ticker] = self._calculate_indicators(ticker, current_price)
        
        # Use StateBuilder for consistent state construction
        state = self.state_builder.build_state(
            cash=account_info['cash'],
            portfolio_value=account_info['portfolio_value'],
            stock_prices=prices,
            stocks_owned=stocks_owned,
            tech_indicator_values=tech_indicator_values,
            turbulence=0.0  # Could calculate from market data if needed
        )
        
        return state
    
    def print_summary(self):
        account_info = self.get_account_info()
        
        print("\n" + "="*60)
        print("PAPER TRADING SUMMARY")
        print("="*60)
        print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"Cash: ${account_info['cash']:,.2f}")
        print(f"Equity: ${account_info['equity']:,.2f}")
        print(f"Total Trades: {len(self.trade_log)}")
        print("="*60 + "\n")
    
    def save_trade_log(self, filepath: str):
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            df.to_csv(filepath, index=False)
            print(f"Trade log saved to {filepath}")
