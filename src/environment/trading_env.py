import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple


class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int = None,  # Deprecated, kept for backwards compatibility
        initial_amount: float = 100000,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1.0,
        tech_indicator_list: List[str] = None,
        turbulence_threshold: float = 120,
        lookback_window: int = 1,
        day: int = 0,
        max_position_pct: float = 0.1  # NEW: Max % of portfolio per position
    ):
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax  # Deprecated
        self.max_position_pct = max_position_pct  # NEW: 10% default
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or []
        self.turbulence_threshold = turbulence_threshold
        self.lookback_window = lookback_window
        self.day = day
        
        self.data = self._prepare_data()
        self.terminal = False
        
        # State: cash + prices + holdings + position_ratio + portfolio_growth + drawdown + momentum + indicators + turbulence
        # Added: 5-day momentum feature for trend awareness
        self.state_dim = 1 + 2 * stock_dim + 4 + len(tech_indicator_list) * stock_dim + 1
        
        # Track price history for momentum calculation
        self.price_history = []
        
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.stock_dim,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self.reset()
        
    def _prepare_data(self) -> pd.DataFrame:
        data = self.df.copy()
        data = data.sort_values(['date', 'tic'], ignore_index=True)
        data.index = data['date'].factorize()[0]
        
        # Pre-index data by day for O(1) lookups instead of O(n) filtering
        self.data_by_day = {}
        for day_idx in data.index.unique():
            self.data_by_day[day_idx] = data[data.index == day_idx]
        
        return data
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.day = 0
        self.terminal = False
        
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.stocks_owned = np.zeros(self.stock_dim)
        self.stock_prices = self._get_stock_prices(self.day)
        
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = []
        self.date_memory = [self._get_date(self.day)]
        
        # Track for reward calculation
        self.episode_returns = []
        self.peak_portfolio_value = self.initial_amount
        self.trades_this_episode = 0
        self.price_history = []  # Reset price history
        
        state = self._get_state()
        info = {}
        
        return state, info
    
    def step(self, actions):
        self.terminal = self.day >= len(self.data.index.unique()) - 1
        
        if self.terminal:
            state = self._get_state()
            reward = 0
            info = {
                'portfolio_value': self.portfolio_value,
                'total_return': (self.portfolio_value - self.initial_amount) / self.initial_amount
            }
            return state, reward, True, False, info
        
        begin_portfolio_value = self.portfolio_value
        
        actions = np.clip(actions, -1, 1)
        
        turbulence = self._get_turbulence(self.day)
        if turbulence >= self.turbulence_threshold:
            actions = np.array([-1] * self.stock_dim)
        
        self._execute_trades(actions)
        
        self.day += 1
        self.stock_prices = self._get_stock_prices(self.day)
        
        # Track price history for momentum (keep last 10 days)
        self.price_history.append(self.stock_prices.copy())
        if len(self.price_history) > 10:
            self.price_history.pop(0)
        
        self.portfolio_value = self.cash + np.sum(self.stocks_owned * self.stock_prices)
        
        portfolio_return = (self.portfolio_value - begin_portfolio_value) / begin_portfolio_value
        self.episode_returns.append(portfolio_return)
        
        # Update peak for drawdown calculation
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        # Calculate multi-component reward
        reward = self._calculate_reward(portfolio_return, actions, begin_portfolio_value)
        
        # Only store history every 100 steps to reduce memory overhead
        if self.day % 100 == 0 or self.terminal:
            self.asset_memory.append(self.portfolio_value)
            self.portfolio_return_memory.append(portfolio_return)
            self.actions_memory.append(actions)
            self.date_memory.append(self._get_date(self.day))
        
        state = self._get_state()
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return
        }
        
        return state, reward, self.terminal, False, info
    
    def _execute_trades(self, actions):
        for i, action in enumerate(actions):
            current_price = self.stock_prices[i]
            
            if action > 0:  # Buy signal
                # Calculate max position value as % of current portfolio
                max_position_value = self.portfolio_value * self.max_position_pct * abs(action)
                
                # Calculate how many shares we can afford
                max_shares = int(max_position_value / (current_price * (1 + self.transaction_cost_pct)))
                
                # Also limit by available cash
                max_shares_by_cash = int(self.cash / (current_price * (1 + self.transaction_cost_pct)))
                max_shares = min(max_shares, max_shares_by_cash)
                
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_cost_pct)
                    if cost <= self.cash:
                        self.stocks_owned[i] += max_shares
                        self.cash -= cost
                        
            elif action < 0:  # Sell signal
                # Sell percentage of current position based on action strength
                shares_to_sell = int(self.stocks_owned[i] * abs(action))
                shares_to_sell = min(shares_to_sell, self.stocks_owned[i])
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price * (1 - self.transaction_cost_pct)
                    self.stocks_owned[i] -= shares_to_sell
                    self.cash += proceeds
    
    def _calculate_reward(self, portfolio_return: float, actions: np.ndarray, begin_value: float) -> float:
        """
        Optimal reward function for training a profitable trader.
        
        Design principles:
        1. PRIMARY signal = portfolio performance (not penalties)
        2. SPARSE penalties only for truly bad behavior
        3. Risk-adjusted returns
        4. Positive rewards must be achievable
        """
        
        # 1. PRIMARY: Log return scaled to meaningful range
        #    Log returns are better for compounding and standard in finance
        #    Scale so 0.1% daily return ≈ 1.0 reward
        log_return = np.log((self.portfolio_value / begin_value) + 1e-8)
        return_reward = log_return * 1000
        
        # 2. RISK ADJUSTMENT: Reduce reward during high volatility periods
        #    This encourages consistent returns over erratic gains
        risk_multiplier = 1.0
        if len(self.episode_returns) > 20:
            recent_volatility = np.std(self.episode_returns[-20:])
            if recent_volatility > 0.02:  # High volatility threshold (2% daily std)
                risk_multiplier = 0.5  # Halve rewards during volatile periods
        
        # 3. SPARSE PENALTY: Excessive trading (churning)
        #    Only penalize when trading intensity is extreme, not every trade
        trade_intensity = np.abs(actions).mean()  # 0 to 1 range
        churn_penalty = -2.0 if trade_intensity > 0.8 else 0
        
        # 4. SPARSE PENALTY: Catastrophic drawdown (>15% from peak)
        #    Only penalize severe drawdowns, not normal fluctuations
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / (self.peak_portfolio_value + 1e-6)
        drawdown_penalty = -3.0 if drawdown > 0.15 else 0
        
        # 5. BONUS: Reward for notably profitable steps
        #    Positive reinforcement for good trading decisions
        profit_bonus = 0.5 if portfolio_return > 0.003 else 0  # >0.3% gain
        
        # 6. SMALL CONTINUOUS INCENTIVE: Encourage being invested
        #    Very small bonus for having skin in the game (not a penalty for not)
        position_value = np.sum(self.stocks_owned * self.stock_prices)
        position_ratio = position_value / (self.portfolio_value + 1e-6)
        investment_bonus = 0.1 if position_ratio > 0.3 else 0  # Small bonus if >30% invested
        
        # Combine: Primary signal + sparse penalties + bonuses
        total_reward = (return_reward * risk_multiplier) + churn_penalty + drawdown_penalty + profit_bonus + investment_bonus
        
        # Clip for stability but allow meaningful positive/negative range
        total_reward = np.clip(total_reward, -10, 10)
        
        return total_reward
    
    def _get_state(self) -> np.ndarray:
        state = [self.cash / self.initial_amount]
        
        # Normalize stock prices (relative to first stock, capped)
        if self.stock_prices[0] > 0:
            normalized_prices = np.clip(self.stock_prices / self.stock_prices[0], 0, 10).tolist()
        else:
            normalized_prices = [1.0] * self.stock_dim
        state.extend(normalized_prices)
        
        # Normalize stocks owned (as fraction of max reasonable position)
        # Assume max ~1000 shares per position for normalization
        normalized_holdings = np.clip(self.stocks_owned / 100, 0, 10).tolist()
        state.extend(normalized_holdings)
        
        # Position-aware features (NEW)
        total_stock_value = np.sum(self.stocks_owned * self.stock_prices)
        state.append(total_stock_value / (self.portfolio_value + 1e-6))  # Position ratio
        state.append(self.portfolio_value / self.initial_amount)  # Portfolio growth
        
        # Drawdown feature
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / (self.peak_portfolio_value + 1e-6)
        state.append(current_drawdown)
        
        # Momentum feature: 5-day price momentum (average across stocks)
        if len(self.price_history) >= 5:
            old_prices = self.price_history[-5]
            momentum = np.mean((self.stock_prices - old_prices) / (old_prices + 1e-8))
            state.append(np.clip(momentum, -1, 1))
        else:
            state.append(0.0)
        
        # Add technical indicators
        tech_indicators = self._get_tech_indicators(self.day)
        state.extend(tech_indicators)
        
        # Add turbulence
        turbulence = self._get_turbulence(self.day)
        state.append(turbulence / 100)
        
        return np.array(state, dtype=np.float32)
    
    def _get_stock_prices(self, day: int) -> np.ndarray:
        day_data = self.data_by_day.get(day)
        if day_data is None or day_data.empty:
            return np.zeros(self.stock_dim)
        prices = day_data['close'].values
        return prices
    
    def _get_tech_indicators(self, day: int) -> List[float]:
        day_data = self.data_by_day.get(day)
        indicators = []
        
        if day_data is None or day_data.empty:
            return [0] * (len(self.tech_indicator_list) * self.stock_dim)
        
        for indicator in self.tech_indicator_list:
            if indicator in day_data.columns:
                values = day_data[indicator].values
                indicators.extend(values)
            else:
                indicators.extend([0] * self.stock_dim)
        
        return indicators
    
    def _get_turbulence(self, day: int) -> float:
        day_data = self.data_by_day.get(day)
        if day_data is not None and not day_data.empty and 'turbulence' in day_data.columns:
            return day_data['turbulence'].values[0]
        return 0
    
    def _get_date(self, day: int) -> str:
        day_data = self.data_by_day.get(day)
        if day_data is not None and not day_data.empty:
            return str(day_data['date'].values[0])
        return ""
    
    def render(self, mode='human'):
        return self.portfolio_value
    
    def get_portfolio_history(self) -> Dict:
        return {
            'dates': self.date_memory,
            'portfolio_values': self.asset_memory,
            'returns': self.portfolio_return_memory,
            'actions': self.actions_memory
        }
