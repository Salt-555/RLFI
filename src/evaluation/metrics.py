import numpy as np
import pandas as pd
from typing import List, Dict


def calculate_sharpe_ratio(portfolio_values: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    return sharpe


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    
    values = pd.Series(portfolio_values)
    cumulative_max = values.cummax()
    drawdown = (values - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)


def calculate_cumulative_return(portfolio_values: List[float]) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]


def calculate_annual_return(portfolio_values: List[float], periods_per_year: int = 252) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    
    total_return = calculate_cumulative_return(portfolio_values)
    n_periods = len(portfolio_values)
    years = n_periods / periods_per_year
    
    if years <= 0:
        return 0.0
    
    annual_return = (1 + total_return) ** (1 / years) - 1
    return annual_return


def calculate_win_rate(returns: List[float]) -> float:
    if len(returns) == 0:
        return 0.0
    
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def calculate_sortino_ratio(portfolio_values: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
    return sortino


def calculate_calmar_ratio(portfolio_values: List[float], periods_per_year: int = 252) -> float:
    annual_return = calculate_annual_return(portfolio_values, periods_per_year)
    max_dd = calculate_max_drawdown(portfolio_values)
    
    if max_dd == 0:
        return 0.0
    
    return annual_return / max_dd


def calculate_all_metrics(portfolio_values: List[float], benchmark_values: List[float] = None) -> Dict[str, float]:
    returns = pd.Series(portfolio_values).pct_change().dropna().tolist()
    
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_values),
        'sortino_ratio': calculate_sortino_ratio(portfolio_values),
        'max_drawdown': calculate_max_drawdown(portfolio_values),
        'cumulative_return': calculate_cumulative_return(portfolio_values),
        'annual_return': calculate_annual_return(portfolio_values),
        'win_rate': calculate_win_rate(returns),
        'calmar_ratio': calculate_calmar_ratio(portfolio_values),
        'total_trades': len(returns),
        'final_value': portfolio_values[-1] if portfolio_values else 0,
        'initial_value': portfolio_values[0] if portfolio_values else 0
    }
    
    if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
        benchmark_return = calculate_cumulative_return(benchmark_values)
        metrics['benchmark_return'] = benchmark_return
        metrics['excess_return'] = metrics['cumulative_return'] - benchmark_return
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    print(f"\nReturns:")
    print(f"  Cumulative Return:    {metrics['cumulative_return']:>10.2%}")
    print(f"  Annual Return:        {metrics['annual_return']:>10.2%}")
    
    if 'benchmark_return' in metrics:
        print(f"  Benchmark Return:     {metrics['benchmark_return']:>10.2%}")
        print(f"  Excess Return:        {metrics['excess_return']:>10.2%}")
    
    print(f"\nRisk-Adjusted:")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:>10.3f}")
    print(f"  Calmar Ratio:         {metrics['calmar_ratio']:>10.3f}")
    
    print(f"\nRisk:")
    print(f"  Max Drawdown:         {metrics['max_drawdown']:>10.2%}")
    
    print(f"\nTrading:")
    print(f"  Win Rate:             {metrics['win_rate']:>10.2%}")
    print(f"  Total Trades:         {metrics['total_trades']:>10.0f}")
    
    print(f"\nPortfolio:")
    print(f"  Initial Value:        ${metrics['initial_value']:>10,.2f}")
    print(f"  Final Value:          ${metrics['final_value']:>10,.2f}")
    
    print("="*60 + "\n")
