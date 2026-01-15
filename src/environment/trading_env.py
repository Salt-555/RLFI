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
        hmax: int,
        initial_amount: float,
        transaction_cost_pct: float,
        reward_scaling: float,
        tech_indicator_list: List[str],
        turbulence_threshold: float = 120,
        lookback_window: int = 1,
        day: int = 0
    ):
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.lookback_window = lookback_window
        self.day = day
        
        self.data = self._prepare_data()
        self.terminal = False
        
        self.state_dim = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim + 1
        
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
        
        self.portfolio_value = self.cash + np.sum(self.stocks_owned * self.stock_prices)
        
        portfolio_return = (self.portfolio_value - begin_portfolio_value) / begin_portfolio_value
        reward = portfolio_return * self.reward_scaling
        
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
            
            if action > 0:
                available_cash = self.cash * abs(action)
                max_shares = min(
                    int(available_cash / (current_price * (1 + self.transaction_cost_pct))),
                    self.hmax
                )
                
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_cost_pct)
                    if cost <= self.cash:
                        self.stocks_owned[i] += max_shares
                        self.cash -= cost
                        
            elif action < 0:
                shares_to_sell = min(
                    int(self.stocks_owned[i] * abs(action)),
                    self.stocks_owned[i]
                )
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price * (1 - self.transaction_cost_pct)
                    self.stocks_owned[i] -= shares_to_sell
                    self.cash += proceeds
    
    def _get_state(self) -> np.ndarray:
        state = [self.cash / self.initial_amount]
        
        state.extend(self.stock_prices / self.stock_prices[0])
        
        state.extend(self.stocks_owned)
        
        tech_indicators = self._get_tech_indicators(self.day)
        state.extend(tech_indicators)
        
        turbulence = self._get_turbulence(self.day)
        state.append(turbulence / 100)
        
        return np.array(state, dtype=np.float32)
    
    def _get_stock_prices(self, day: int) -> np.ndarray:
        day_data = self.data[self.data.index == day]
        prices = day_data['close'].values
        return prices
    
    def _get_tech_indicators(self, day: int) -> List[float]:
        day_data = self.data[self.data.index == day]
        indicators = []
        
        for indicator in self.tech_indicator_list:
            if indicator in day_data.columns:
                values = day_data[indicator].values
                indicators.extend(values)
            else:
                indicators.extend([0] * self.stock_dim)
        
        return indicators
    
    def _get_turbulence(self, day: int) -> float:
        day_data = self.data[self.data.index == day]
        if 'turbulence' in day_data.columns:
            return day_data['turbulence'].values[0]
        return 0
    
    def _get_date(self, day: int) -> str:
        day_data = self.data[self.data.index == day]
        if not day_data.empty:
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
