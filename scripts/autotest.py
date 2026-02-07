#!/usr/bin/env python3
"""
AUTOTEST - Automated AI Trading System
Automatically trains, backtests, and paper trades AI models on Sunday nights (2AM-4AM)
"""
import sys
import os

# Ensure we're using the project's venv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(project_root, 'venv', 'bin', 'python3')

# If not already running in venv, restart with venv python
if sys.executable != venv_python and os.path.exists(venv_python):
    print(f"Restarting with project venv: {venv_python}")
    os.execv(venv_python, [venv_python] + sys.argv)

sys.path.append(project_root)

import yaml
import time
from datetime import datetime, timedelta
import argparse
import signal

from src.autotest.parameter_generator import ParameterGenerator
from src.autotest.automated_trainer import AutomatedTrainer
from src.autotest.automated_backtester import AutomatedBacktester
from src.autotest.paper_trade_orchestrator import PaperTradeOrchestrator
from src.autotest.strategy_analyzer import StrategyAnalyzer
from src.autotest.strategy_database import StrategyDatabase
from src.autotest.model_manager import ModelManager
from src.autotest.model_lifecycle import ModelLifecycleManager, ModelState
from src.agents.trainer import request_shutdown, reset_shutdown


class AutoTestSystem:
    """
    Main AUTOTEST system coordinator
    """
    
    def __init__(self, config_path: str = 'config/autotest_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['autotest']
        
        self.param_generator = ParameterGenerator(config_path)
        self.trainer = AutomatedTrainer()
        self.backtester = AutomatedBacktester()
        self.orchestrator = PaperTradeOrchestrator()
        
        # Initialize strategy tracking
        db_path = self.config['storage'].get('strategy_db', 'autotest_strategies.db')
        self.strategy_db = StrategyDatabase(db_path)
        self.strategy_analyzer = StrategyAnalyzer(db_path)
        
        # Initialize model manager for retention and continuation
        self.model_manager = ModelManager(config_path='config/training_strategy.yaml',
                                         autotest_config_path=config_path)
        
        # Initialize lifecycle manager for colosseum mode
        self.lifecycle_manager = ModelLifecycleManager(db_path)
        
        # Colosseum config
        self.colosseum_config = self.config.get('colosseum', {})
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n\nReceived shutdown signal. Stopping AUTOTEST...")
        self.running = False
        request_shutdown()  # Signal trainer to stop current training
        self.orchestrator.stop_all_trading()
    
    def is_scheduled_time(self) -> bool:
        """
        Check if current time is within the scheduled window (Sunday 2AM-4AM)
        
        Returns:
            True if within scheduled window, False otherwise
        """
        now = datetime.now()
        
        # Check day of week (6 = Sunday)
        if now.weekday() != self.config['schedule']['day_of_week']:
            return False
        
        # Check hour range
        start_hour = self.config['schedule']['start_hour']
        end_hour = self.config['schedule']['end_hour']
        
        if start_hour <= now.hour < end_hour:
            return True
        
        return False
    
    def wait_for_scheduled_time(self):
        """Wait until the scheduled time window"""
        print("Waiting for scheduled time window...")
        print(f"Schedule: Day {self.config['schedule']['day_of_week']} (0=Mon, 6=Sun), "
              f"{self.config['schedule']['start_hour']:02d}:00-{self.config['schedule']['end_hour']:02d}:00")
        
        while not self.is_scheduled_time() and self.running:
            now = datetime.now()
            print(f"Current time: {now.strftime('%A %Y-%m-%d %H:%M:%S')}")
            print("Not in scheduled window. Checking again in 1 hour...")
            time.sleep(3600)  # Check every hour
        
        if self.running:
            print(f"\n{'='*80}")
            print("SCHEDULED TIME WINDOW REACHED - STARTING AUTOTEST")
            print(f"{'='*80}\n")
    
    def run_full_cycle(self, force: bool = False):
        """
        Run complete AUTOTEST cycle: train -> backtest -> paper trade
        
        Args:
            force: If True, skip schedule check and run immediately
        """
        if not force:
            self.wait_for_scheduled_time()
        
        if not self.running:
            print("AUTOTEST stopped before execution")
            return
        
        start_time = datetime.now()
        print(f"\n{'='*80}")
        print(f"AUTOTEST FULL CYCLE STARTED")
        print(f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        try:
            # PHASE 1: Generate parameter sets and train models
            print("\n" + "="*80)
            print("PHASE 1: TRAINING")
            print("="*80)
            
            # Get hybrid training plan (exploration + exploitation + champions)
            print("Generating hybrid training plan...")
            training_plan = self.model_manager.get_training_plan()
            
            print(f"\nTraining Plan:")
            print(f"  Exploration (new): {len(training_plan['exploration'])} models")
            print(f"  Exploitation (continued): {len(training_plan['exploitation'])} models")
            print(f"  Champions (elite): {len(training_plan['champions'])} models")
            print(f"  Total: {training_plan['total_models']} models")
            print(f"  Total timesteps: {training_plan['total_timesteps']:,}")
            
            # Generate parameter sets for new models
            num_new_models = len(training_plan['exploration'])
            parameter_sets = self.param_generator.generate_balanced_sets(num_new_models) if num_new_models > 0 else []
            
            # Save parameter sets
            param_file = os.path.join(
                self.config['storage']['logs_dir'],
                f"parameters_{start_time.strftime('%Y%m%d_%H%M%S')}.yaml"
            )
            os.makedirs(self.config['storage']['logs_dir'], exist_ok=True)
            self.param_generator.save_parameter_sets(parameter_sets, param_file)
            
            # Train all models (new + continued + champions)
            all_training_results = []
            
            # Train new exploration models
            if parameter_sets:
                print(f"\nTraining {len(parameter_sets)} new exploration models...")
                exploration_results = self.trainer.train_all_models(parameter_sets)
                all_training_results.extend(exploration_results)
            
            # Continue training exploitation models
            if training_plan['exploitation']:
                print(f"\nContinuing {len(training_plan['exploitation'])} exploitation models...")
                for exploit_spec in training_plan['exploitation']:
                    # Load parent model and continue training
                    result = self.trainer.continue_training_model(exploit_spec)
                    all_training_results.append(result)
            
            # Continue training champion models
            if training_plan['champions']:
                print(f"\nContinuing {len(training_plan['champions'])} champion models...")
                for champ_spec in training_plan['champions']:
                    # Load champion and continue training
                    result = self.trainer.continue_training_model(champ_spec)
                    all_training_results.append(result)
            
            training_results = all_training_results
            
            successful_models = [r for r in training_results if r['success']]
            print(f"\nTraining complete: {len(successful_models)}/{len(training_results)} models successful")
            
            if not successful_models:
                print("ERROR: No models trained successfully. Aborting AUTOTEST.")
                return
            
            # PHASE 2: Backtest all trained models
            print("\n" + "="*80)
            print("PHASE 2: BACKTESTING")
            print("="*80)
            
            print(f"Backtesting {len(successful_models)} models...")
            backtest_results = self.backtester.backtest_all_models(successful_models)
            
            successful_backtests = [r for r in backtest_results if r['success']]
            print(f"\nBacktesting complete: {len(successful_backtests)}/{len(successful_models)} models successful")
            
            if not successful_backtests:
                print("ERROR: No successful backtests. Aborting AUTOTEST.")
                return
            
            # Rank models and select top performers
            print("\nRanking models...")
            ranked_models = self.backtester.rank_models(backtest_results)
            
            top_n = self.config['backtesting']['top_n_models']
            top_models = self.backtester.get_top_models(ranked_models, top_n)
            
            print(f"\nSelected top {len(top_models)} models for paper trading")
            
            # Archive top models for future continuation training
            print("\nArchiving top models...")
            self.model_manager.archive_top_models()
            
            # PHASE 3: Paper trade with top models
            print("\n" + "="*80)
            print("PHASE 3: PAPER TRADING")
            print("="*80)
            
            # Check if we should wait for market hours
            now = datetime.now()
            market_open_hour = 9  # 9:30 AM EST (adjust for your timezone)
            
            if now.hour < market_open_hour:
                wait_minutes = (market_open_hour - now.hour) * 60 - now.minute
                print(f"\nWaiting {wait_minutes} minutes for market open...")
                print("You can cancel and run paper trading manually later if needed.")
                
                # Wait in chunks to allow for interruption
                for _ in range(wait_minutes):
                    if not self.running:
                        print("AUTOTEST stopped during market wait")
                        return
                    time.sleep(60)
            
            print(f"Starting paper trading with top {len(top_models)} models...")
            
            # Reconstruct model info with paths for paper trading
            top_models_with_paths = []
            for model_result in top_models:
                model_id = model_result['model_id']
                # Find corresponding training result to get model path
                for train_result in successful_models:
                    if train_result['model_id'] == model_id:
                        top_models_with_paths.append(train_result)
                        break
            
            paper_trade_results = self.orchestrator.run_all_paper_trading(
                top_models_with_paths,
                sequential=True
            )
            
            # PHASE 4: Generate final report and strategy analysis
            print("\n" + "="*80)
            print("PHASE 4: FINAL REPORT & STRATEGY ANALYSIS")
            print("="*80)
            
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds() / 3600
            
            # Calculate best model metrics
            best_sharpe = max([r['metrics']['sharpe_ratio'] for r in successful_backtests if r['success']])
            best_return = max([r['metrics']['total_return'] for r in successful_backtests if r['success']])
            avg_sharpe = sum([r['metrics']['sharpe_ratio'] for r in successful_backtests if r['success']]) / len(successful_backtests)
            avg_return = sum([r['metrics']['total_return'] for r in successful_backtests if r['success']]) / len(successful_backtests)
            best_model_id = ranked_models[0][0] if ranked_models else None
            
            print(f"\nAUTOTEST CYCLE COMPLETE")
            print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {total_duration:.2f} hours")
            print(f"\nSummary:")
            print(f"  Models Trained: {len(successful_models)}/{len(parameter_sets)}")
            print(f"  Models Backtested: {len(successful_backtests)}/{len(successful_models)}")
            print(f"  Models Paper Traded: {len([r for r in paper_trade_results if r['success']])}/{len(top_models)}")
            print(f"\nBest Performance:")
            print(f"  Best Sharpe Ratio: {best_sharpe:.3f}")
            print(f"  Best Return: {best_return*100:.2f}%")
            print(f"  Avg Sharpe Ratio: {avg_sharpe:.3f}")
            print(f"  Avg Return: {avg_return*100:.2f}%")
            
            # Save weekly summary to database
            self.strategy_db.save_weekly_summary({
                'models_trained': len(successful_models),
                'models_backtested': len(successful_backtests),
                'models_paper_traded': len([r for r in paper_trade_results if r['success']]),
                'best_model_id': best_model_id,
                'best_sharpe_ratio': best_sharpe,
                'best_total_return': best_return,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_total_return': avg_return,
                'execution_time_hours': total_duration
            })
            
            # Generate strategy analysis
            print("\n" + "="*80)
            print("STRATEGY ANALYSIS - COMPARING WITH PAST WEEKS")
            print("="*80)
            
            if self.config['strategy_tracking']['enabled']:
                analysis_report = self.strategy_analyzer.generate_weekly_report(
                    weeks_back=self.config['strategy_tracking']['compare_weeks']
                )
                
                # Export analysis
                self.strategy_analyzer.export_analysis_report(
                    output_dir=self.config['storage']['results_dir'],
                    weeks_back=self.config['strategy_tracking']['compare_weeks']
                )
            
            # Save final summary
            summary_path = os.path.join(
                self.config['storage']['results_dir'],
                f"autotest_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.yaml"
            )
            
            summary = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': total_duration,
                'models_trained': len(successful_models),
                'models_backtested': len(successful_backtests),
                'models_paper_traded': len([r for r in paper_trade_results if r['success']]),
                'top_models': [m['model_id'] for m in top_models],
                'best_model_id': best_model_id,
                'best_sharpe_ratio': float(best_sharpe),
                'best_total_return': float(best_return),
                'avg_sharpe_ratio': float(avg_sharpe),
                'avg_total_return': float(avg_return)
            }
            
            os.makedirs(self.config['storage']['results_dir'], exist_ok=True)
            with open(summary_path, 'w') as f:
                yaml.dump(summary, f)
            
            print(f"\nFinal summary saved to: {summary_path}")
            print("="*80)
            
        except Exception as e:
            print(f"\nERROR in AUTOTEST cycle: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error log
            error_path = os.path.join(
                self.config['storage']['logs_dir'],
                f"autotest_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            with open(error_path, 'w') as f:
                f.write(f"AUTOTEST Error at {datetime.now().isoformat()}\n\n")
                f.write(str(e) + "\n\n")
                f.write(traceback.format_exc())
            
            print(f"Error log saved to: {error_path}")
    
    def run_training_only(self):
        """Run only the training phase"""
        print("Running training phase only...")
        
        reset_shutdown()  # Clear any previous shutdown flag
        num_models = self.config['training']['num_models_to_train']
        parameter_sets = self.param_generator.generate_balanced_sets(num_models)
        training_results = self.trainer.train_all_models(parameter_sets)
        
        print(f"\nTraining complete: {len([r for r in training_results if r['success']])}/{len(parameter_sets)} successful")
    
    def run_backtest_only(self):
        """Run only the backtesting phase on existing models"""
        print("Running backtesting phase only...")
        
        models_dir = self.config['storage']['models_dir']
        
        # Find all trained models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        
        if not model_files:
            print(f"No models found in {models_dir}")
            return
        
        print(f"Found {len(model_files)} models to backtest")
        
        # Load model metadata
        models_to_test = []
        for model_file in model_files:
            model_id = model_file.replace('.zip', '').rsplit('_', 1)[0]
            metadata_file = os.path.join(models_dir, f"{model_id}_metadata.yaml")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                models_to_test.append({
                    'model_id': metadata['model_id'],
                    'model_path': metadata['model_path'],
                    'params': metadata['parameters']
                })
        
        if not models_to_test:
            print("No valid model metadata found")
            return
        
        backtest_results = self.backtester.backtest_all_models(models_to_test)
        ranked_models = self.backtester.rank_models(backtest_results)
        
        print(f"\nBacktesting complete: {len([r for r in backtest_results if r['success']])}/{len(models_to_test)} successful")
    
    def _get_model_generation(self, model_id: str) -> int:
        """Get generation number from model ID."""
        if '_g' in model_id:
            try:
                return int(model_id.split('_g')[-1])
            except:
                pass
        return 1
    
    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open using Alpaca API.
        Falls back to simple time check if API unavailable.
        """
        try:
            from alpaca.trading.client import TradingClient
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and secret_key:
                client = TradingClient(api_key, secret_key, paper=True)
                clock = client.get_clock()
                return clock.is_open
        except Exception as e:
            # Fall back to simple time-based check
            pass
        
        # Fallback: Simple time-based check (Pacific time)
        now = datetime.now()
        # Monday-Friday, 6:30 AM - 1:00 PM Pacific (9:30 AM - 4:00 PM Eastern)
        if now.weekday() >= 5:  # Weekend
            return False
        if now.hour < 6 or now.hour >= 13:
            return False
        if now.hour == 6 and now.minute < 30:
            return False
        return True
    
    def show_colosseum_status(self):
        """Display current colosseum status - models in each lifecycle state."""
        print("\n" + "="*80)
        print("COLOSSEUM STATUS")
        print("="*80)
        
        summary = self.lifecycle_manager.get_lifecycle_summary()
        
        print(f"\nModel Lifecycle States:")
        print(f"  Training:      {summary.get('training', 0)}")
        print(f"  Validation:    {summary.get('validation', 0)}")
        print(f"  Paper Trading: {summary.get('paper_trading', 0)} / {self.lifecycle_manager.MAX_PAPER_TRADING_MODELS} max")
        print(f"  Promoted:      {summary.get('promoted', 0)}")
        print(f"  Culled:        {summary.get('culled', 0)}")
        
        if summary.get('paper_trading_avg_return') is not None:
            print(f"\nPaper Trading Performance:")
            print(f"  Avg Return: {summary['paper_trading_avg_return']*100:.2f}%")
            print(f"  Avg Sharpe: {summary['paper_trading_avg_sharpe']:.2f}")
        
        # Show individual paper trading models
        paper_models = self.lifecycle_manager.get_paper_trading_models()
        if paper_models:
            print(f"\nActive Paper Trading Models:")
            for model in paper_models:
                perf = self.lifecycle_manager.get_model_performance(model['model_id'])
                days = self.lifecycle_manager.get_model_paper_trading_days(model['model_id'])
                if perf:
                    print(f"  {model['model_id']}: {perf['cumulative_return']*100:+.2f}% return, "
                          f"{perf['sharpe_ratio']:.2f} Sharpe, {days} days")
                else:
                    print(f"  {model['model_id']}: No performance data yet")
        
        print("="*80)
    
    def run_colosseum_mode(self):
        """
        Run in colosseum mode - continuous 24/7 operation:
        - Daily training (2AM) - 3 models per day, skips if 5+ eligible for paper trading
        - Daily paper trading (market hours)
        - Weekly culling (Saturday 6PM)
        """
        print("\n" + "="*80)
        print("COLOSSEUM MODE - CONTINUOUS AI TRADING ARENA")
        print("="*80)
        print("This mode runs continuously:")
        print("  - Daily training: 2AM (3 models/day, skips if 5+ eligible)")
        print("  - Daily paper trading: Market hours (9:30AM-4PM ET)")
        print("  - Weekly culling: Saturday 6PM")
        print("  - Max paper trading models: 5")
        print("  - Min paper trading days before culling: 10 (2 weeks)")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        last_training_day = -1
        last_culling_week = -1
        last_paper_trading_day = -1
        
        while self.running:
            try:
                now = datetime.now()
                current_week = now.isocalendar()[1]
                current_day = now.timetuple().tm_yday
                
                # Check if it's training time (daily at configured hour)
                training_hour = self.colosseum_config.get('training_hour', 2)
                
                if (now.hour == training_hour and 
                    current_day != last_training_day):
                    
                    print(f"\n[{now.strftime('%Y-%m-%d %H:%M')}] DAILY TRAINING TIME")
                    self._colosseum_training_cycle()
                    last_training_day = current_day
                
                # Check if it's culling time (Saturday 6PM)
                culling_day = self.colosseum_config.get('culling_day', 5)
                culling_hour = self.colosseum_config.get('culling_hour', 18)
                
                if (now.weekday() == culling_day and 
                    now.hour == culling_hour and 
                    current_week != last_culling_week):
                    
                    print(f"\n[{now.strftime('%Y-%m-%d %H:%M')}] CULLING TIME")
                    self._colosseum_culling_cycle()
                    last_culling_week = current_week
                
                # Check if market is open for paper trading
                if self._is_market_open():
                    # Run paper trading continuously during market hours
                    print(f"\n[{now.strftime('%Y-%m-%d %H:%M')}] PAPER TRADING - Market Open")
                    self._colosseum_paper_trading_session()
                    
                    # During market hours, check more frequently (every 5 min)
                    time.sleep(300)
                else:
                    # Outside market hours, check less frequently (every 15 min)
                    time.sleep(900)
                
            except KeyboardInterrupt:
                print("\n\nStopping colosseum mode...")
                break
            except Exception as e:
                print(f"\nError in colosseum loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # Wait a minute before retrying
        
        print("\nColosseum mode stopped.")
        self.show_colosseum_status()
    
    def _colosseum_training_cycle(self):
        """Run training cycle for colosseum mode with genetic evolution."""
        print("Starting daily training cycle...")
        reset_shutdown()  # Clear any previous shutdown flag
        
        # Check how many models are already eligible for paper trading (in VALIDATION state)
        eligible_count = self.lifecycle_manager.get_validation_count()
        max_eligible = self.colosseum_config.get('max_eligible_for_training', 5)
        print(f"Models eligible for paper trading (VALIDATION): {eligible_count}/{max_eligible}")
        
        if eligible_count >= max_eligible:
            print(f"Already {eligible_count}+ models eligible for paper trading - skipping training")
            return
        
        # Check available slots
        available_slots = self.lifecycle_manager.get_available_slots()
        print(f"Available paper trading slots: {available_slots}")
        
        if available_slots <= 0:
            print("No slots available - skipping training")
            return
        
        # Daily limit
        daily_limit = self.colosseum_config.get('daily_training_limit', 5)
        
        # GENETIC EVOLUTION: Check for elite parents to spawn offspring
        genetic_config = self.colosseum_config.get('genetic_training', {})
        genetic_enabled = genetic_config.get('enabled', True)
        offspring_ratio = genetic_config.get('offspring_ratio', 0.4)  # 40% offspring, 60% fresh
        
        offspring_results = []
        num_offspring = 0
        
        if genetic_enabled:
            elite_parents = self.lifecycle_manager.get_elite_parents_for_genetic_training(max_parents=3)
            
            if elite_parents:
                num_offspring = min(int(daily_limit * offspring_ratio), len(elite_parents))
                print(f"\n[GENETIC] Found {len(elite_parents)} elite parents, spawning {num_offspring} offspring")
                
                from src.agents.trainer import is_shutdown_requested
                for i, parent in enumerate(elite_parents[:num_offspring]):
                    # Check shutdown before each offspring
                    if is_shutdown_requested():
                        print("\n\nShutdown requested - stopping offspring training...")
                        break
                    
                    print(f"  Parent {i+1}: {parent['model_id']} "
                          f"(fitness={parent['fitness_score']:.3f}, "
                          f"paper={parent['paper_return']*100:+.1f}%, "
                          f"{parent['trading_days']} days)")
                    
                    # Detect parent's algorithm from metadata or model path
                    parent_algorithm = 'ppo'  # default fallback
                    parent_metadata_path = parent.get('metadata_path')
                    if parent_metadata_path and os.path.exists(parent_metadata_path):
                        try:
                            with open(parent_metadata_path, 'r') as f:
                                p_meta = yaml.safe_load(f)
                            parent_algorithm = p_meta.get('algorithm', 'ppo')
                        except Exception:
                            pass
                    if parent_algorithm == 'ppo':
                        # Try detecting from model filename as fallback
                        model_filename = os.path.basename(parent['model_path']).lower()
                        for algo in ['sac', 'a2c', 'ddpg']:
                            if f'_{algo}.' in model_filename or model_filename.endswith(f'_{algo}.zip'):
                                parent_algorithm = algo
                                break
                    
                    # Create offspring spec - inherits parent's algorithm and indicators
                    offspring_spec = {
                        'parent_model': parent['model_id'],
                        'parent_path': parent['model_path'],
                        'metadata_path': parent.get('metadata_path'),
                        'tickers': parent['tickers'],
                        'additional_timesteps': genetic_config.get('offspring_timesteps', 500000),
                        'learning_rate_multiplier': genetic_config.get('offspring_lr_multiplier', 0.3),
                        'generation': self._get_model_generation(parent['model_id']) + 1,
                        'params': {'algorithm': parent_algorithm, 'tickers': parent['tickers']}
                    }
                    
                    # Train offspring
                    result = self.trainer.continue_training_model(offspring_spec)
                    offspring_results.append(result)
                    
                    if result['success']:
                        print(f"  ✓ Offspring {result['model_id']} trained successfully")
                    else:
                        print(f"  ✗ Offspring training failed: {result.get('error')}")
            else:
                print("[GENETIC] No elite parents found yet - training fresh models only")
        
        # Train fresh models (remaining slots) - but check shutdown first
        from src.agents.trainer import is_shutdown_requested
        if is_shutdown_requested():
            print("\n\nShutdown requested - skipping fresh model training...")
            training_results = []
        else:
            num_fresh = daily_limit - num_offspring
            print(f"\nTraining {num_fresh} fresh models + {num_offspring} offspring = {daily_limit} total")
            
            parameter_sets = self.param_generator.generate_balanced_sets(num_fresh)
            training_results = self.trainer.train_all_models(parameter_sets)
        
        # Combine results
        all_results = offspring_results + training_results
        
        successful = [r for r in all_results if r['success']]
        print(f"Training complete: {len(successful)}/{len(all_results)} successful")
        
        if not successful:
            return
        
        # Register models in lifecycle (TRAINING state)
        for result in successful:
            model_id = result['model_id']
            self.lifecycle_manager.register_model(
                model_id=model_id,
                model_path=result['model_path'],
                metadata_path=result.get('metadata_path', ''),
                tickers=result['params'].get('tickers', [])
            )
        
        # Backtest successful models
        print("Backtesting models...")
        backtest_results = self.backtester.backtest_all_models(successful)
        
        # Backtest quality gate + Grokking gate
        # Models must pass BOTH backtest metrics AND grokking analysis to enter VALIDATION
        quality_gate = self.colosseum_config.get('backtest_quality_gate', {})
        min_sharpe = quality_gate.get('min_sharpe_ratio', 0.3)
        min_return = quality_gate.get('min_total_return', 0.0)
        max_drawdown = quality_gate.get('max_drawdown', 0.25)
        
        # Grokking gate config
        grokking_gate_enabled = self.colosseum_config.get('grokking_gate', {}).get('enabled', True)
        
        passed_models = []
        failed_models = []
        
        for result in backtest_results:
            if not result['success']:
                continue
                
            model_id = result['model_id']
            metrics = result['metrics']
            sharpe = metrics.get('sharpe_ratio', 0)
            total_return = metrics.get('total_return', 0)
            drawdown = abs(metrics.get('max_drawdown', 1))
            
            # Gate 1: Backtest quality check
            backtest_passed = (sharpe >= min_sharpe and total_return >= min_return and drawdown <= max_drawdown)
            
            if not backtest_passed:
                failed_models.append(result)
                self.lifecycle_manager.transition_state(model_id, ModelState.CULLED, 
                    f"Failed quality gate: Sharpe={sharpe:.2f}, Return={total_return*100:.1f}%, DD={drawdown*100:.1f}%")
                print(f"  ✗ {model_id}: FAILED backtest (Sharpe={sharpe:.2f}, Return={total_return*100:.1f}%, DD={drawdown*100:.1f}%)")
                continue
            
            # Gate 2: Grokking check - has this model genuinely learned?
            grokking_passed = True  # Default pass if gate disabled or check unavailable
            grokking_reason = ""
            
            if grokking_gate_enabled:
                # Check metadata for grokking analysis (saved by select_best_models)
                model_info = next((r for r in successful if r['model_id'] == model_id), None)
                metadata_path = model_info.get('metadata_path') if model_info else None
                
                if metadata_path and os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = yaml.safe_load(f)
                        
                        grokking_data = metadata.get('grokking_analysis', {})
                        has_grokked = metadata.get('has_grokked', None)
                        grokking_attempted = metadata.get('grokking_attempted', False)
                        grokking_error = metadata.get('grokking_error', None)
                        
                        if has_grokked is not None:
                            grokking_passed = has_grokked
                            grokking_reason = grokking_data.get('reason', '')
                        elif grokking_attempted and grokking_error:
                            # Grokking was attempted but failed (e.g., model couldn't load)
                            grokking_passed = False
                            grokking_reason = f"grokking analysis failed: {grokking_error}"
                        else:
                            # No grokking analysis was run (legacy model or error)
                            # Fall through - don't block on missing data
                            grokking_reason = "no grokking analysis available"
                    except Exception as e:
                        grokking_reason = f"error reading grokking data: {e}"
            
            if backtest_passed and grokking_passed:
                passed_models.append(result)
                self.lifecycle_manager.update_backtest_expectations(
                    model_id=model_id,
                    expected_return=total_return,
                    expected_sharpe=sharpe
                )
                self.lifecycle_manager.transition_state(model_id, ModelState.VALIDATION)
                grok_label = " [grokked]" if grokking_gate_enabled and grokking_reason else ""
                print(f"  ✓ {model_id}: PASSED (Sharpe={sharpe:.2f}, Return={total_return*100:.1f}%, DD={drawdown*100:.1f}%){grok_label}")
            elif backtest_passed and not grokking_passed:
                failed_models.append(result)
                self.lifecycle_manager.transition_state(model_id, ModelState.CULLED, 
                    f"Failed grokking gate: {grokking_reason}")
                print(f"  ✗ {model_id}: FAILED grokking check (Sharpe={sharpe:.2f} was good, but: {grokking_reason})")
        
        print(f"\nQuality + Grokking gate: {len(passed_models)} passed, {len(failed_models)} failed")
        
        # Promote top VALIDATION models to paper trading (ranked by composite score)
        if passed_models:
            promoted = self.lifecycle_manager.promote_to_paper_trading(max_to_promote=available_slots)
            print(f"Promoted {len(promoted)} models to paper trading: {promoted}")
    
    def _colosseum_culling_cycle(self):
        """Run culling cycle for colosseum mode."""
        print("Starting weekly culling cycle...")
        
        # Evaluate all paper trading models
        decisions = self.lifecycle_manager.evaluate_for_culling()
        
        print(f"\nCulling Decisions:")
        for d in decisions:
            print(f"  {d['model_id']}: {d['decision']} - {d['reason']}")
        
        # Execute decisions
        summary = self.lifecycle_manager.execute_culling_decisions(decisions)
        
        print(f"\nCulling Summary:")
        print(f"  Culled: {summary['culled']}")
        print(f"  Promoted: {summary['promoted']}")
        print(f"  Continued: {summary['continued']}")
        
        # If we have open slots, promote waiting models
        available_slots = self.lifecycle_manager.get_available_slots()
        if available_slots > 0:
            promoted = self.lifecycle_manager.promote_to_paper_trading(max_to_promote=available_slots)
            if promoted:
                print(f"Promoted {len(promoted)} new models to paper trading")
        
        # Run data cleanup after culling
        print("\nRunning data cleanup...")
        self._run_cleanup()
    
    def _colosseum_paper_trading_session(self):
        """Run daily paper trading session for all active models."""
        paper_models = self.lifecycle_manager.get_paper_trading_models()
        
        if not paper_models:
            print("No models currently paper trading")
            return
        
        print(f"Running paper trading for {len(paper_models)} models...")
        
        # Run paper trading for each model
        for model_info in paper_models:
            model_id = model_info['model_id']
            
            try:
                # Detect algorithm from model path (e.g. "..._ppo.zip" or "..._sac.zip")
                model_path = model_info['model_path']
                algorithm = 'ppo'  # default
                if model_path:
                    model_filename = os.path.basename(model_path).lower()
                    if '_sac.' in model_filename or model_filename.endswith('_sac.zip'):
                        algorithm = 'sac'
                    elif '_a2c.' in model_filename or model_filename.endswith('_a2c.zip'):
                        algorithm = 'a2c'
                    elif '_ddpg.' in model_filename or model_filename.endswith('_ddpg.zip'):
                        algorithm = 'ddpg'
                
                # Setup model for trading
                setup_result = self.orchestrator.setup_model_for_trading({
                    'model_id': model_id,
                    'model_path': model_path,
                    'metadata_path': model_info.get('metadata_path'),
                    'params': {
                        'algorithm': algorithm,
                        'tickers': model_info['tickers']
                    }
                })
                
                if not setup_result['success']:
                    print(f"  {model_id}: Failed to setup - {setup_result.get('error')}")
                    continue
                
                # Run trading session
                result = self.orchestrator.run_paper_trading_session(setup_result)
                
                if result['success']:
                    perf = result['performance']
                    
                    # Calculate actual cumulative return from portfolio value vs initial
                    initial_capital = self.config.get('paper_trading', {}).get('initial_capital', 100000)
                    cumulative_return = (perf['final_value'] - initial_capital) / initial_capital
                    
                    # Log daily performance
                    self.lifecycle_manager.log_daily_performance(
                        model_id=model_id,
                        portfolio_value=perf['final_value'],
                        daily_return=perf['total_return'],
                        cumulative_return=cumulative_return,
                        trades_executed=perf['trades_executed'],
                        positions=perf.get('positions', {}),
                        benchmark_return=perf.get('benchmark_return', 0.0)
                    )
                    
                    print(f"  {model_id}: {perf['total_return']*100:+.2f}% daily, "
                          f"{cumulative_return*100:+.2f}% cumulative, {perf['trades_executed']} trades")
                    
                    # CIRCUIT BREAKER: Don't wait for Saturday culling if a model is
                    # hemorrhaging money. Cull immediately if cumulative loss exceeds threshold.
                    circuit_breaker_loss = self.colosseum_config.get('circuit_breaker_loss', -0.10)
                    circuit_breaker_sharpe = self.colosseum_config.get('circuit_breaker_sharpe', -1.5)
                    
                    if cumulative_return < circuit_breaker_loss:
                        print(f"  !! CIRCUIT BREAKER: {model_id} cumulative loss "
                              f"{cumulative_return*100:.1f}% exceeds {circuit_breaker_loss*100:.0f}% threshold - CULLING")
                        self.lifecycle_manager.transition_state(
                            model_id, ModelState.CULLED,
                            f"Circuit breaker: {cumulative_return*100:.1f}% cumulative loss"
                        )
                    else:
                        # Also check Sharpe if we have enough data
                        perf_data = self.lifecycle_manager.get_model_performance(model_id)
                        if perf_data and perf_data['trading_days'] >= 5:
                            if perf_data['sharpe_ratio'] < circuit_breaker_sharpe:
                                print(f"  !! CIRCUIT BREAKER: {model_id} Sharpe "
                                      f"{perf_data['sharpe_ratio']:.2f} below {circuit_breaker_sharpe} - CULLING")
                                self.lifecycle_manager.transition_state(
                                    model_id, ModelState.CULLED,
                                    f"Circuit breaker: Sharpe {perf_data['sharpe_ratio']:.2f}"
                                )
                else:
                    print(f"  {model_id}: Trading failed - {result.get('error')}")
                    
            except Exception as e:
                print(f"  {model_id}: Error - {e}")
        
        print("Paper trading session complete")
    
    def _run_cleanup(self):
        """Run data cleanup to prevent disk bloat."""
        import shutil
        from pathlib import Path
        
        RLFI_ROOT = Path(__file__).parent.parent
        DATA_DIR = RLFI_ROOT / "data" / "raw"
        LOGS_DIR = RLFI_ROOT / "logs"
        
        # Retention policies (days)
        MARKET_DATA_RETENTION = 7
        TENSORBOARD_RETENTION = 14
        
        cleaned_count = 0
        
        # Clean old market data CSVs
        if DATA_DIR.exists():
            for csv_file in DATA_DIR.glob("*.csv"):
                file_age = (time.time() - csv_file.stat().st_mtime) / 86400
                if file_age > MARKET_DATA_RETENTION:
                    csv_file.unlink()
                    cleaned_count += 1
                    print(f"  Removed old data: {csv_file.name}")
        
        # Clean old TensorBoard logs
        if LOGS_DIR.exists():
            for log_dir in LOGS_DIR.iterdir():
                if log_dir.is_dir():
                    file_age = (time.time() - log_dir.stat().st_mtime) / 86400
                    if file_age > TENSORBOARD_RETENTION:
                        shutil.rmtree(log_dir)
                        cleaned_count += 1
                        print(f"  Removed old logs: {log_dir.name}")
        
        # Clean culled model files (30 days)
        try:
            cursor = self.lifecycle_manager.conn.cursor()
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute('''
                SELECT model_id, model_path FROM model_lifecycle
                WHERE current_state = 'culled' AND paper_trading_ended_at < ?
            ''', (cutoff,))
            
            for model_id, model_path in cursor.fetchall():
                if model_path and os.path.exists(model_path):
                    os.remove(model_path)
                    cleaned_count += 1
                    print(f"  Removed culled model: {model_id}")
        except Exception as e:
            print(f"  Warning: Could not clean culled models: {e}")
        
        print(f"Cleanup complete: {cleaned_count} items removed")


def main():
    parser = argparse.ArgumentParser(description='AUTOTEST - Automated AI Trading System')
    parser.add_argument('--force', action='store_true', 
                       help='Force run immediately without waiting for scheduled time')
    parser.add_argument('--train-only', action='store_true',
                       help='Run only the training phase')
    parser.add_argument('--backtest-only', action='store_true',
                       help='Run only the backtesting phase on existing models')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon, continuously checking for scheduled time')
    parser.add_argument('--colosseum', action='store_true',
                       help='Run in colosseum mode - continuous 24/7 training, paper trading, and culling')
    parser.add_argument('--status', action='store_true',
                       help='Show current colosseum status and exit')
    
    args = parser.parse_args()
    
    print("="*80)
    print("AUTOTEST - Automated AI Trading System")
    print("="*80)
    
    autotest = AutoTestSystem()
    
    if args.status:
        autotest.show_colosseum_status()
    elif args.train_only:
        autotest.run_training_only()
    elif args.backtest_only:
        autotest.run_backtest_only()
    elif args.colosseum:
        autotest.run_colosseum_mode()
    elif args.daemon:
        print("Running in daemon mode...")
        print("Will execute AUTOTEST during scheduled time windows")
        print("Press Ctrl+C to stop")
        
        while autotest.running:
            try:
                autotest.run_full_cycle(force=False)
                
                # After completion, wait until next scheduled window
                print("\nWaiting for next scheduled window...")
                time.sleep(3600)  # Check every hour
                
            except KeyboardInterrupt:
                print("\nStopping daemon...")
                break
    else:
        autotest.run_full_cycle(force=args.force)


if __name__ == '__main__':
    main()
