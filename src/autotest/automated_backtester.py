import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.environment.trading_env import StockTradingEnv
from src.agents.trainer import RLTrainer
from src.evaluation.backtester import Backtester
from src.autotest.strategy_database import StrategyDatabase


class AutomatedBacktester:
    """
    Automated backtesting system for evaluating and ranking trained models
    """
    
    def __init__(self, base_config_path: str = 'config/default_config.yaml',
                 autotest_config_path: str = 'config/autotest_config.yaml'):
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        with open(autotest_config_path, 'r') as f:
            self.autotest_config = yaml.safe_load(f)['autotest']
        
        self.results_dir = self.autotest_config['storage']['results_dir']
        self.logs_dir = self.autotest_config['storage']['logs_dir']
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize strategy database
        db_path = self.autotest_config['storage'].get('strategy_db', 'autotest_strategies.db')
        self.strategy_db = StrategyDatabase(db_path)
        
        self.backtest_results = []
        
        # Data cache: avoid re-downloading the same tickers during a backtest session
        # Key: frozenset of tickers, Value: processed DataFrame
        self._data_cache = {}
    
    def backtest_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest a single model
        
        Args:
            model_info: Dictionary containing model_path, metadata, and parameters
        
        Returns:
            Dictionary with backtest results and metrics
        """
        model_id = model_info['model_id']
        model_path = model_info['model_path']
        params = model_info['params']
        algorithm = params['algorithm']
        tickers = params['tickers']
        
        print("\n" + "="*80)
        print(f"BACKTESTING MODEL: {model_id}")
        print("="*80)
        print(f"Algorithm: {algorithm}")
        print(f"Tickers: {tickers}")
        print("="*80)
        
        result = {
            'model_id': model_id,
            'algorithm': algorithm,
            'tickers': tickers,
            'success': False,
            'error': None,
            'metrics': {}
        }
        
        try:
            # Load data for backtesting (with cache to avoid duplicate downloads)
            print(f"[{model_id}] Loading backtest data...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.autotest_config['backtesting']['test_period_days'] + 60)
            
            cache_key = frozenset(tickers)
            if cache_key in self._data_cache:
                print(f"[{model_id}] Using cached data for {tickers}")
                df = self._data_cache[cache_key].copy()
            else:
                data_loader = DataLoader(
                    tickers=tickers,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                df = data_loader.download_data()
                self._data_cache[cache_key] = df.copy()
            
            # Feature engineering - use model-specific indicators if available
            # This is critical: the backtest must use the same indicators the model
            # was trained with, otherwise the state vector dimensions won't match
            tech_indicators = model_info.get('tech_indicators', self.base_config['features']['technical_indicators'])
            print(f"[{model_id}] Engineering features ({len(tech_indicators)} indicators)...")
            
            # CRITICAL: Load training statistics for consistent normalization
            # Without this, backtest uses different feature distributions than training
            indicator_stats_path = model_info.get('indicator_stats_path')
            external_stats = None
            if indicator_stats_path and os.path.exists(indicator_stats_path):
                try:
                    external_stats = FeatureEngineer.load_indicator_stats(indicator_stats_path)
                    print(f"[{model_id}] Loaded training indicator stats for consistent normalization")
                except Exception as e:
                    print(f"[{model_id}] Warning: Could not load indicator stats: {e}")
            else:
                print(f"[{model_id}] Warning: No indicator stats found - backtest may use different normalization than training")
            
            feature_engineer = FeatureEngineer(
                tech_indicator_list=tech_indicators,
                use_turbulence=self.base_config['features']['use_turbulence'],
                external_stats=external_stats  # Use training stats for consistent normalization
            )
            df = feature_engineer.preprocess_data(df)
            
            # Use last N days for testing
            test_days = self.autotest_config['backtesting']['test_period_days']
            unique_dates = sorted(df['date'].unique())
            
            if len(unique_dates) < test_days:
                test_dates = unique_dates
            else:
                test_dates = unique_dates[-test_days:]
            
            test_df = df[df['date'].isin(test_dates)].copy()
            test_df.index = test_df['date'].factorize()[0]
            
            print(f"[{model_id}] Backtest period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
            
            # Create test environment with same config as training
            print(f"[{model_id}] Creating test environment...")
            # Use model-specific tech indicators if available
            tech_indicators = model_info.get('tech_indicators', self.base_config['features']['technical_indicators'])
            
            test_env = StockTradingEnv(
                df=test_df,
                stock_dim=len(tickers),
                initial_amount=self.base_config['environment']['initial_amount'],
                transaction_cost_pct=self.base_config['environment']['transaction_cost_pct'],
                reward_scaling=1.0,  # Not used with multi-component rewards
                tech_indicator_list=tech_indicators,
                turbulence_threshold=self.base_config['features']['turbulence_threshold'],
                max_position_pct=self.base_config['environment'].get('max_position_pct', 0.3),
                store_history_every=1  # Store every step for accurate backtest metrics
            )
            
            # Load model
            print(f"[{model_id}] Loading model...")
            model_config = self.base_config.copy()
            trainer = RLTrainer(test_env, model_config, model_name=algorithm)
            model = trainer.load_model(model_path)
            
            # Run backtest
            print(f"[{model_id}] Running backtest...")
            backtester = Backtester(test_env, model, model_config)
            backtest_results = backtester.run_backtest(deterministic=True)
            
            # Calculate metrics
            print(f"[{model_id}] Calculating metrics...")
            portfolio_values = backtest_results['portfolio_values']
            
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate returns
            returns = pd.Series(portfolio_values).pct_change().dropna()
            
            # Sharpe ratio (annualized)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Max drawdown
            cumulative = pd.Series(portfolio_values)
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Win rate
            positive_returns = (returns > 0).sum()
            total_days = len(returns)
            win_rate = (positive_returns / total_days) if total_days > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Calmar ratio
            annualized_return = (1 + returns.mean()) ** 252 - 1
            calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'sortino_ratio': sortino_ratio,
                'volatility': volatility,
                'calmar_ratio': calmar_ratio,
                'final_value': final_value,
                'initial_value': initial_value
            }
            
            result['success'] = True
            result['metrics'] = metrics
            result['portfolio_values'] = portfolio_values
            result['dates'] = backtest_results['dates']
            
            # Note: ranking_score and rank_position will be added later during ranking
            # For now, save with placeholder values
            self.strategy_db.save_backtest_result(
                model_id=model_id,
                metrics=metrics,
                ranking_score=0.0,  # Will be updated during ranking
                rank_position=999   # Will be updated during ranking
            )
            
            print(f"\n[{model_id}] ✓ Backtest complete")
            print(f"[{model_id}] Total Return: {total_return*100:.2f}%")
            print(f"[{model_id}] Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"[{model_id}] Max Drawdown: {max_drawdown*100:.2f}%")
            print(f"[{model_id}] Win Rate: {win_rate*100:.1f}%")
            print(f"[{model_id}] Results logged to database")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            
            print(f"\n[{model_id}] ✗ Backtest failed")
            print(f"[{model_id}] Error: {e}")
            
            # Save error log
            error_log_path = os.path.join(self.logs_dir, f"{model_id}_backtest_error.log")
            with open(error_log_path, 'w') as f:
                f.write(f"Model ID: {model_id}\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
        
        return result
    
    def backtest_all_models(self, trained_models: List[Dict[str, Any]], 
                            only_selected: bool = True) -> List[Dict[str, Any]]:
        """
        Backtest trained models
        
        Args:
            trained_models: List of trained model information
            only_selected: If True, only backtest models marked selected_for_testing
        
        Returns:
            List of backtest results
        """
        # Filter to only selected models if requested
        if only_selected:
            models_to_test = []
            for model_info in trained_models:
                # Check metadata for selected_for_testing flag
                metadata_path = model_info.get('metadata_path')
                if metadata_path and os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    if metadata.get('selected_for_testing', False):
                        # Add tech_indicators and indicator_stats_path from metadata
                        model_info['tech_indicators'] = metadata.get('tech_indicators')
                        model_info['indicator_stats_path'] = metadata.get('indicator_stats_path')
                        models_to_test.append(model_info)
                else:
                    # If no metadata, include model (backwards compatibility)
                    models_to_test.append(model_info)
            
            print(f"\nFiltered to {len(models_to_test)}/{len(trained_models)} models marked for testing")
        else:
            models_to_test = trained_models
        
        print("\n" + "="*80)
        print(f"AUTOMATED BACKTESTING SESSION")
        print(f"Backtesting {len(models_to_test)} models")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        results = []
        successful = 0
        failed = 0
        
        for i, model_info in enumerate(models_to_test, 1):
            print(f"\n\nProgress: {i}/{len(models_to_test)}")
            
            result = self.backtest_model(model_info)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Save results
        results_path = os.path.join(self.results_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        with open(results_path, 'w') as f:
            # Remove large arrays for YAML storage
            results_for_yaml = []
            for r in results:
                r_copy = r.copy()
                if 'portfolio_values' in r_copy:
                    del r_copy['portfolio_values']
                if 'dates' in r_copy:
                    del r_copy['dates']
                results_for_yaml.append(r_copy)
            
            yaml.dump({
                'total_models': len(trained_models),
                'successful': successful,
                'failed': failed,
                'results': results_for_yaml
            }, f)
        
        print("\n" + "="*80)
        print("AUTOMATED BACKTESTING COMPLETE")
        print("="*80)
        print(f"Total Models Tested: {len(models_to_test)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {results_path}")
        print("="*80)
        
        self.backtest_results = results
        return results
    
    def rank_models(self, backtest_results: List[Dict[str, Any]]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rank models based on backtest metrics
        
        Args:
            backtest_results: List of backtest results
        
        Returns:
            List of tuples (model_id, score, metrics) sorted by score (highest first)
        """
        print("\n" + "="*80)
        print("RANKING MODELS")
        print("="*80)
        
        weights = self.autotest_config['backtesting']['ranking_weights']
        
        successful_results = [r for r in backtest_results if r['success']]
        
        if not successful_results:
            print("No successful backtests to rank")
            return []
        
        # Collect all metric values for min-max normalization
        # Without normalization, Sharpe (0-2) and Sortino (0-3) dominate over
        # total_return (0.0-0.5) and win_rate (0.4-0.6) despite the config weights
        metric_keys = ['sharpe_ratio', 'total_return', 'win_rate', 'sortino_ratio']
        metric_values = {k: [] for k in metric_keys}
        dd_values = []  # max_drawdown handled separately (inverted)
        
        for result in successful_results:
            metrics = result['metrics']
            for k in metric_keys:
                metric_values[k].append(metrics.get(k, 0))
            dd_values.append(1 - metrics.get('max_drawdown', 1))  # Inverted: lower DD = higher score
        
        # Compute min-max ranges
        metric_ranges = {}
        for k in metric_keys:
            vals = metric_values[k]
            min_v, max_v = min(vals), max(vals)
            metric_ranges[k] = (min_v, max_v - min_v) if (max_v - min_v) > 1e-8 else (min_v, 1.0)
        
        dd_min, dd_range = min(dd_values), max(dd_values) - min(dd_values)
        if dd_range < 1e-8:
            dd_range = 1.0
        
        ranked_models = []
        
        for result in successful_results:
            metrics = result['metrics']
            
            # Normalize each metric to 0-1 range, then apply weights
            normalized = {}
            for k in metric_keys:
                min_v, range_v = metric_ranges[k]
                normalized[k] = (metrics.get(k, 0) - min_v) / range_v
            
            dd_score = ((1 - metrics.get('max_drawdown', 1)) - dd_min) / dd_range
            
            score = (
                weights['sharpe_ratio'] * normalized['sharpe_ratio'] +
                weights['total_return'] * normalized['total_return'] +
                weights['max_drawdown'] * dd_score +
                weights['win_rate'] * normalized['win_rate'] +
                weights['sortino_ratio'] * normalized['sortino_ratio']
            )
            
            ranked_models.append((result['model_id'], score, result))
        
        # Sort by score (highest first)
        ranked_models.sort(key=lambda x: x[1], reverse=True)
        
        # Update database with ranking scores
        for rank_position, (model_id, score, result) in enumerate(ranked_models, 1):
            # Update the backtest result with actual ranking
            cursor = self.strategy_db.conn.cursor()
            cursor.execute('''
                UPDATE backtest_results 
                SET ranking_score = ?, rank_position = ?
                WHERE model_id = ? AND week_number = ?
            ''', (score, rank_position, model_id, self.strategy_db.get_current_week_number()))
            self.strategy_db.conn.commit()
        
        print("\nTop Models:")
        for i, (model_id, score, result) in enumerate(ranked_models[:10], 1):
            metrics = result['metrics']
            print(f"\n{i}. {model_id} (Score: {score:.4f})")
            print(f"   Algorithm: {result['algorithm']}")
            print(f"   Tickers: {result['tickers']}")
            print(f"   Return: {metrics['total_return']*100:.2f}%")
            print(f"   Sharpe: {metrics['sharpe_ratio']:.3f}")
            print(f"   Max DD: {metrics['max_drawdown']*100:.2f}%")
            print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        
        # Save rankings
        rankings_path = os.path.join(self.results_dir, f"model_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        rankings_data = []
        for model_id, score, result in ranked_models:
            metrics = result['metrics']
            rankings_data.append({
                'model_id': model_id,
                'score': score,
                'algorithm': result['algorithm'],
                'tickers': ','.join(result['tickers']),
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'sortino_ratio': metrics['sortino_ratio']
            })
        
        rankings_df = pd.DataFrame(rankings_data)
        rankings_df.to_csv(rankings_path, index=False)
        print(f"\nRankings saved to: {rankings_path}")
        
        return ranked_models
    
    def get_top_models(self, ranked_models: List[Tuple[str, float, Dict[str, Any]]], 
                       top_n: int = None) -> List[Dict[str, Any]]:
        """
        Get top N models from ranked list
        
        Args:
            ranked_models: List of ranked models
            top_n: Number of top models to return
        
        Returns:
            List of top model results
        """
        if top_n is None:
            top_n = self.autotest_config['backtesting']['top_n_models']
        
        top_models = [result for _, _, result in ranked_models[:top_n]]
        
        print(f"\nSelected top {len(top_models)} models for paper trading")
        
        return top_models
