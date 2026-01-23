import os
import sys
import yaml
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback
from threading import Thread
import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.trainer import RLTrainer
from src.trading.live_paper_trading import LivePaperTrader
from src.environment.trading_env import StockTradingEnv
from src.autotest.strategy_database import StrategyDatabase


class PaperTradeOrchestrator:
    """
    Orchestrates paper trading for multiple top-performing models simultaneously
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
        
        self.paper_trade_results = []
        self.active_traders = []
    
    def setup_model_for_trading(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and prepare a model for paper trading
        
        Args:
            model_info: Dictionary containing model information
        
        Returns:
            Dictionary with loaded model and configuration
        """
        model_id = model_info['model_id']
        model_path = model_info['model_path']
        params = model_info['params']
        algorithm = params['algorithm']
        tickers = params['tickers']
        
        print(f"\n[{model_id}] Setting up for paper trading...")
        
        try:
            # Get model-specific tech indicators from metadata if available
            tech_indicators = model_info.get('tech_indicators', self.base_config['features']['technical_indicators'])
            
            # Create dummy environment for loading model (matches training config)
            dummy_env = StockTradingEnv(
                df=None,
                stock_dim=len(tickers),
                initial_amount=self.autotest_config['paper_trading']['initial_capital'],
                transaction_cost_pct=self.base_config['environment']['transaction_cost_pct'],
                reward_scaling=1.0,  # Not used with multi-component rewards
                tech_indicator_list=tech_indicators,
                turbulence_threshold=self.base_config['features']['turbulence_threshold'],
                max_position_pct=self.base_config['environment'].get('max_position_pct', 0.3)
            )
            
            # Load model
            model_config = self.base_config.copy()
            trainer = RLTrainer(dummy_env, model_config, model_name=algorithm)
            model = trainer.load_model(model_path)
            
            print(f"[{model_id}] ✓ Model loaded successfully")
            
            return {
                'model_id': model_id,
                'model': model,
                'tickers': tickers,
                'algorithm': algorithm,
                'params': params,
                'success': True
            }
            
        except Exception as e:
            print(f"[{model_id}] ✗ Failed to load model: {e}")
            return {
                'model_id': model_id,
                'success': False,
                'error': str(e)
            }
    
    def run_paper_trading_session(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run paper trading session for a single model
        
        Args:
            model_info: Dictionary containing model and configuration
        
        Returns:
            Dictionary with paper trading results
        """
        model_id = model_info['model_id']
        model = model_info['model']
        tickers = model_info['tickers']
        
        print("\n" + "="*80)
        print(f"PAPER TRADING: {model_id}")
        print("="*80)
        print(f"Algorithm: {model_info['algorithm']}")
        print(f"Tickers: {tickers}")
        print("="*80)
        
        result = {
            'model_id': model_id,
            'algorithm': model_info['algorithm'],
            'tickers': tickers,
            'success': False,
            'error': None,
            'performance': {}
        }
        
        try:
            # Create metadata for LivePaperTrader
            metadata = {
                'tickers': tickers,
                'tech_indicators': self.base_config['features']['technical_indicators'],
                'state_space': None  # Will be calculated by LivePaperTrader
            }
            
            # Initialize paper trader
            print(f"[{model_id}] Initializing paper trader...")
            trader = LivePaperTrader(
                model=model,
                tickers=tickers,
                initial_capital=self.autotest_config['paper_trading']['initial_capital'],
                tech_indicators=self.base_config['features']['technical_indicators'],
                model_metadata=metadata,
                update_frequency="5Min"
            )
            
            # Store trader reference
            self.active_traders.append({
                'model_id': model_id,
                'trader': trader
            })
            
            # Run trading session
            duration_hours = self.autotest_config['paper_trading']['duration_hours']
            check_interval = self.autotest_config['paper_trading']['check_interval']
            
            print(f"[{model_id}] Starting paper trading session...")
            print(f"[{model_id}] Duration: {duration_hours} hours")
            print(f"[{model_id}] Check interval: {check_interval} seconds")
            
            trader.run_live_trading(
                duration_minutes=int(duration_hours * 60),
                update_interval_seconds=check_interval
            )
            
            # Get results
            trading_results = trader.get_results()
            
            result['success'] = True
            result['performance'] = {
                'initial_capital': trading_results['initial_capital'],
                'final_value': trading_results['final_value'],
                'total_return': trading_results['return_pct'],
                'trades_executed': trading_results['trades_executed'],
                'transaction_costs': trading_results['transaction_costs']
            }
            result['trade_log'] = trading_results['trade_log']
            result['portfolio_history'] = trading_results['portfolio_history']
            
            # Save to strategy database
            self.strategy_db.save_paper_trade_result(
                model_id=model_id,
                performance=result['performance'],
                was_top_model=True
            )
            
            print(f"\n[{model_id}] ✓ Paper trading complete")
            print(f"[{model_id}] Final Value: ${trading_results['final_value']:,.2f}")
            print(f"[{model_id}] Return: {trading_results['return_pct']:.2f}%")
            print(f"[{model_id}] Trades: {trading_results['trades_executed']}")
            print(f"[{model_id}] Results logged to database")
            
            # Save individual model results
            model_results_path = os.path.join(
                self.results_dir, 
                f"{model_id}_paper_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            )
            with open(model_results_path, 'w') as f:
                # Remove large arrays for YAML
                result_copy = result.copy()
                if 'trade_log' in result_copy:
                    result_copy['trade_log'] = f"{len(result_copy['trade_log'])} trades"
                if 'portfolio_history' in result_copy:
                    result_copy['portfolio_history'] = f"{len(result_copy['portfolio_history'])} data points"
                yaml.dump(result_copy, f)
            
            # Save trade log as CSV
            if trading_results['trade_log']:
                trade_log_path = os.path.join(
                    self.results_dir,
                    f"{model_id}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                pd.DataFrame(trading_results['trade_log']).to_csv(trade_log_path, index=False)
                print(f"[{model_id}] Trade log saved to: {trade_log_path}")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            
            print(f"\n[{model_id}] ✗ Paper trading failed")
            print(f"[{model_id}] Error: {e}")
            
            # Save error log
            error_log_path = os.path.join(self.logs_dir, f"{model_id}_paper_trade_error.log")
            with open(error_log_path, 'w') as f:
                f.write(f"Model ID: {model_id}\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
        
        return result
    
    def run_all_paper_trading(self, top_models: List[Dict[str, Any]], 
                             sequential: bool = True) -> List[Dict[str, Any]]:
        """
        Run paper trading for all top models
        
        Args:
            top_models: List of top model information
            sequential: If True, run sequentially; if False, run in parallel
        
        Returns:
            List of paper trading results
        """
        print("\n" + "="*80)
        print(f"AUTOMATED PAPER TRADING SESSION")
        print(f"Trading with {len(top_models)} top models")
        print(f"Mode: {'Sequential' if sequential else 'Parallel'}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Setup all models
        print("\nSetting up models...")
        setup_models = []
        for model_info in top_models:
            setup_result = self.setup_model_for_trading(model_info)
            if setup_result['success']:
                setup_models.append(setup_result)
        
        print(f"\n{len(setup_models)}/{len(top_models)} models ready for trading")
        
        if not setup_models:
            print("No models available for paper trading")
            return []
        
        results = []
        
        if sequential:
            # Run models one at a time
            for i, model_info in enumerate(setup_models, 1):
                print(f"\n\nProgress: {i}/{len(setup_models)}")
                result = self.run_paper_trading_session(model_info)
                results.append(result)
        else:
            # Run models in parallel (not recommended for paper trading with same account)
            print("\nWARNING: Parallel paper trading may cause conflicts with the same Alpaca account")
            print("Running sequentially instead...")
            
            for i, model_info in enumerate(setup_models, 1):
                print(f"\n\nProgress: {i}/{len(setup_models)}")
                result = self.run_paper_trading_session(model_info)
                results.append(result)
        
        # Save summary
        summary_path = os.path.join(
            self.results_dir, 
            f"paper_trade_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        
        summary_data = {
            'total_models': len(setup_models),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': []
        }
        
        for r in results:
            r_copy = r.copy()
            if 'trade_log' in r_copy:
                del r_copy['trade_log']
            if 'portfolio_history' in r_copy:
                del r_copy['portfolio_history']
            summary_data['results'].append(r_copy)
        
        with open(summary_path, 'w') as f:
            yaml.dump(summary_data, f)
        
        print("\n" + "="*80)
        print("AUTOMATED PAPER TRADING COMPLETE")
        print("="*80)
        print(f"Total Models: {len(setup_models)}")
        print(f"Successful: {summary_data['successful']}")
        print(f"Failed: {summary_data['failed']}")
        print(f"Summary saved to: {summary_path}")
        
        # Print performance summary
        print("\nPerformance Summary:")
        for result in results:
            if result['success']:
                perf = result['performance']
                print(f"\n{result['model_id']}:")
                print(f"  Return: {perf['total_return']:.2f}%")
                print(f"  Final Value: ${perf['final_value']:,.2f}")
                print(f"  Trades: {perf['trades_executed']}")
        
        print("="*80)
        
        self.paper_trade_results = results
        return results
    
    def stop_all_trading(self):
        """Stop all active trading sessions"""
        print("\nStopping all active trading sessions...")
        for trader_info in self.active_traders:
            trader_info['trader'].stop_trading = True
        print("All trading sessions stopped")
