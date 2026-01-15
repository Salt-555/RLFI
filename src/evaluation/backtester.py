import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from src.evaluation.metrics import calculate_all_metrics, print_metrics


class Backtester:
    def __init__(self, env, model, config: Dict[str, Any]):
        self.env = env
        self.model = model
        self.config = config
        self.results = None
        
    def run_backtest(self, deterministic: bool = True) -> Dict[str, Any]:
        print("Running backtest...")
        
        obs, _ = self.env.reset()
        done = False
        
        portfolio_values = []
        actions_taken = []
        dates = []
        
        step = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            if 'portfolio_value' in info:
                portfolio_values.append(info['portfolio_value'])
                actions_taken.append(action)
            
            step += 1
            
            if step % 100 == 0:
                print(f"Step {step}, Portfolio Value: ${info.get('portfolio_value', 0):,.2f}")
        
        history = self.env.get_portfolio_history()
        
        self.results = {
            'portfolio_values': history['portfolio_values'],
            'returns': history['returns'],
            'actions': history['actions'],
            'dates': history['dates']
        }
        
        print(f"Backtest complete. Total steps: {step}")
        return self.results
    
    def calculate_metrics(self, benchmark_values: List[float] = None) -> Dict[str, float]:
        if self.results is None:
            raise ValueError("Run backtest first")
        
        metrics = calculate_all_metrics(
            self.results['portfolio_values'],
            benchmark_values
        )
        
        return metrics
    
    def plot_results(self, benchmark_values: List[float] = None, save_path: str = None):
        if self.results is None:
            raise ValueError("Run backtest first")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        dates = pd.to_datetime(self.results['dates'])
        portfolio_values = self.results['portfolio_values']
        
        axes[0].plot(dates, portfolio_values, label='Portfolio Value', linewidth=2)
        if benchmark_values is not None:
            axes[0].plot(dates, benchmark_values, label='Benchmark', linewidth=2, alpha=0.7)
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        returns = pd.Series(portfolio_values).pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod() - 1
        axes[1].plot(dates, cumulative_returns * 100, color='green', linewidth=2)
        axes[1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Return (%)')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        
        cummax = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - cummax) / cummax * 100
        axes[2].fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        axes[2].plot(dates, drawdown, color='red', linewidth=2)
        axes[2].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, benchmark_values: List[float] = None, save_path: str = None):
        if self.results is None:
            raise ValueError("Run backtest first")
        
        metrics = self.calculate_metrics(benchmark_values)
        
        print_metrics(metrics)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("BACKTEST REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write("Configuration:\n")
                f.write(f"  Tickers: {self.config['data']['tickers']}\n")
                f.write(f"  Period: {self.config['data']['start_date']} to {self.config['data']['end_date']}\n")
                f.write(f"  Initial Capital: ${self.config['environment']['initial_amount']:,.2f}\n\n")
                
                f.write("Performance Metrics:\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        if 'return' in key or 'drawdown' in key or 'rate' in key:
                            f.write(f"  {key.replace('_', ' ').title()}: {value:.2%}\n")
                        else:
                            f.write(f"  {key.replace('_', ' ').title()}: {value:.3f}\n")
                    else:
                        f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            
            print(f"Report saved to {save_path}")
        
        return metrics
    
    def compare_with_baseline(self, baseline_strategy: str = 'buy_and_hold'):
        if self.results is None:
            raise ValueError("Run backtest first")
        
        initial_value = self.results['portfolio_values'][0]
        
        if baseline_strategy == 'buy_and_hold':
            print("Comparing with Buy-and-Hold strategy...")
        
        print("\nComparison complete")
