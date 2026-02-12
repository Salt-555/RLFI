import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.trainer import is_shutdown_requested

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.market_context_loader import MarketContextLoader
from src.data.walk_forward_validator import WalkForwardValidator
from src.environment.trading_env import StockTradingEnv
from src.agents.trainer import RLTrainer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.autotest.strategy_database import StrategyDatabase
import json


class AutomatedTrainer:
    """
    Automated training system for multiple AI models with different parameters
    """
    
    def __init__(self, base_config_path: str = 'config/default_config.yaml',
                 autotest_config_path: str = 'config/autotest_config.yaml'):
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        with open(autotest_config_path, 'r') as f:
            self.autotest_config = yaml.safe_load(f)['autotest']
        
        self.models_dir = self.autotest_config['storage']['models_dir']
        self.logs_dir = self.autotest_config['storage']['logs_dir']
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize strategy database for tracking
        db_path = self.autotest_config['storage'].get('strategy_db', 'autotest_strategies.db')
        self.strategy_db = StrategyDatabase(db_path)
        
        self.training_results = []

    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Load metadata safely, repairing legacy YAML with numpy tags if needed."""
        with open(metadata_path, 'r') as f:
            try:
                metadata = yaml.safe_load(f)
            except yaml.YAMLError:
                f.seek(0)
                metadata = yaml.unsafe_load(f)
                # Repair legacy metadata by re-saving sanitized content
                with open(metadata_path, 'w') as out_f:
                    yaml.safe_dump(self._to_builtin(metadata), out_f, default_flow_style=False)
        return metadata or {}

    def _to_builtin(self, obj: Any) -> Any:
        """Convert numpy/pandas types to plain Python types for YAML serialization."""
        if isinstance(obj, dict):
            return {k: self._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_builtin(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    def train_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a single model with given parameters
        
        Args:
            params: Parameter dictionary containing algorithm, tickers, etc.
        
        Returns:
            Dictionary with training results and model path
        """
        # Check shutdown FIRST before doing anything
        if is_shutdown_requested():
            return {
                'model_id': params['model_id'],
                'params': params,
                'success': False,
                'error': 'Shutdown requested',
                'model_path': None,
                'training_time': 0
            }
        
        model_id = params['model_id']
        algorithm = params['algorithm']
        tickers = params['tickers']
        
        print("\n" + "="*80)
        print(f"TRAINING MODEL: {model_id}")
        print("="*80)
        print(f"Algorithm: {algorithm}")
        print(f"Tickers: {tickers}")
        print(f"Tech Indicators: {len(params.get('tech_indicators', []))} selected")
        print(f"Learning Rate: {params['learning_rate']}")
        print(f"Network: {params.get('net_arch', [[256, 128], [256, 128]])}")
        print(f"Timesteps: {params['timesteps']:,}")
        print("="*80)
        
        result = {
            'model_id': model_id,
            'params': params,
            'success': False,
            'error': None,
            'model_path': None,
            'training_time': None
        }
        
        start_time = datetime.now()
        
        try:
            # Load and prepare data
            print(f"\n[{model_id}] Loading data...")
            data_loader = DataLoader(
                tickers=tickers,
                start_date=self.base_config['data']['start_date'],
                end_date=self.base_config['data']['end_date']
            )
            
            # Use shared data cache by ticker combination (not per-model)
            # This prevents duplicate data files for same ticker sets
            ticker_hash = '_'.join(sorted(tickers))
            data_path = f'data/raw/stock_data_{ticker_hash}.csv'
            
            # Check if cached data is fresh (less than 1 day old)
            use_cache = False
            if os.path.exists(data_path):
                file_age = time.time() - os.path.getmtime(data_path)
                if file_age < 86400:  # 24 hours
                    use_cache = True
            
            if use_cache:
                df = data_loader.load_data(data_path)
            else:
                df = data_loader.download_data()
                data_loader.save_data(df, data_path)
            
            # Load market context features (VIX, FOMC, earnings, put/call)
            print(f"[{model_id}] Loading market context data...")
            context_loader = MarketContextLoader(
                start_date=self.base_config['data']['start_date'],
                end_date=self.base_config['data']['end_date']
            )
            market_context_df, earnings_df = context_loader.load_all_context(tickers)
            
            # Feature engineering with model-specific indicators + market context
            print(f"[{model_id}] Engineering features...")
            tech_indicators = params.get('tech_indicators', self.base_config['features']['technical_indicators'])
            print(f"[{model_id}] Using {len(tech_indicators)} technical indicators + market context")
            feature_engineer = FeatureEngineer(
                tech_indicator_list=tech_indicators,
                use_turbulence=self.base_config['features']['use_turbulence'],
                market_context_df=market_context_df,
                earnings_df=earnings_df
            )
            df = feature_engineer.preprocess_data(df)
            
            # Log market context features
            context_cols = ['vix_close', 'vix_spread', 'days_to_fomc', 'days_to_earnings']
            available_context = [c for c in context_cols if c in df.columns]
            print(f"[{model_id}] Market context features: {available_context}")

            # Ensure tickers reflect actual data coverage (some tickers may be dropped)
            tickers_used = sorted(df['tic'].unique().tolist())
            if set(tickers_used) != set(tickers):
                print(f"[{model_id}] Adjusted tickers due to data coverage: {tickers_used}")
                tickers = tickers_used
                params['tickers'] = tickers_used
            
            # Split data using walk-forward validation (rolling windows from current date)
            print(f"[{model_id}] Splitting data with walk-forward validation...")
            
            # Initialize walk-forward validator
            validator = WalkForwardValidator(
                train_window_days=1260,      # 5 years training
                val_window_days=252,         # 1 year validation
                test_window_days=126,        # 6 months testing
                min_train_days=504,          # Minimum 2 years
                expanding_window=True,       # Grow training window over time
            )
            
            # Get optimal train/val/test split based on most recent data
            train_df, val_df, test_df = validator.get_optimal_train_val_split(df)
            
            # Holdout for final evaluation (last 90 days, never touched during training)
            all_dates = sorted(df['date'].unique())
            if len(all_dates) > 90:
                holdout_dates = all_dates[-90:]  # Last 90 days
                holdout_df = df[df['date'].isin(holdout_dates)].copy()
                print(f"[{model_id}] Holdout period reserved for final validation: {len(holdout_df)} rows (last 90 days)")
            else:
                holdout_df = pd.DataFrame()
            
            # Create training environment
            print(f"[{model_id}] Creating training environment...")
            # Use actual number of tickers in the data, not from params
            actual_stock_dim = len(train_df['tic'].unique())
            
            # Use fixed environment parameters (agent learns strategy, not hardcoded rules)
            train_env = StockTradingEnv(
                df=train_df,
                stock_dim=actual_stock_dim,
                initial_amount=self.base_config['environment']['initial_amount'],
                transaction_cost_pct=self.base_config['environment']['transaction_cost_pct'],
                reward_scaling=1.0,  # Not used with multi-component rewards
                tech_indicator_list=tech_indicators,
                turbulence_threshold=self.base_config['features']['turbulence_threshold'],
                max_position_pct=self.base_config['environment'].get('max_position_pct', 0.3)
            )
            
            val_env = StockTradingEnv(
                df=val_df,
                stock_dim=actual_stock_dim,
                initial_amount=self.base_config['environment']['initial_amount'],
                transaction_cost_pct=self.base_config['environment']['transaction_cost_pct'],
                reward_scaling=1.0,  # Not used with multi-component rewards
                tech_indicator_list=tech_indicators,
                turbulence_threshold=self.base_config['features']['turbulence_threshold'],
                max_position_pct=self.base_config['environment'].get('max_position_pct', 0.3)
            )
            
            # Modify config for this specific model with PPO parameters
            model_config = self.base_config.copy()
            if algorithm in model_config['training']:
                model_config['training'][algorithm]['learning_rate'] = params['learning_rate']
                # Apply PPO-specific hyperparameters
                if 'n_steps' in params:
                    model_config['training'][algorithm]['n_steps'] = params['n_steps']
                if 'batch_size' in params:
                    model_config['training'][algorithm]['batch_size'] = params['batch_size']
                if 'gamma' in params:
                    model_config['training'][algorithm]['gamma'] = params['gamma']
                if 'clip_range' in params:
                    model_config['training'][algorithm]['clip_range'] = params['clip_range']
                if 'ent_coef' in params:
                    model_config['training'][algorithm]['ent_coef'] = params['ent_coef']
                if 'weight_decay' in params:
                    model_config['training'][algorithm]['weight_decay'] = params['weight_decay']
                # Apply network architecture
                if 'net_arch' in params:
                    pi_arch, vf_arch = params['net_arch']
                    model_config['training'][algorithm]['policy_kwargs'] = {
                        'net_arch': {'pi': pi_arch, 'vf': vf_arch}
                    }
            
            # Train model with proper eval tracking for grokking detection
            print(f"[{model_id}] Training {algorithm.upper()} model...")
            trainer = RLTrainer(train_env, model_config, model_name=algorithm)
            
            from stable_baselines3.common.monitor import Monitor
            eval_env = DummyVecEnv([lambda: Monitor(val_env)])
            
            # Setup custom eval callback to capture rewards for grokking detection
            eval_rewards_history = []
            eval_timesteps_history = []
            eval_stds_history = []
            
            class GrokkingEvalCallback(EvalCallback):
                """Custom callback to capture eval data for proper grokking detection"""
                def __init__(self, eval_env, **kwargs):
                    super().__init__(eval_env, **kwargs)
                    self.eval_rewards = []
                    self.eval_timesteps = []
                    self.eval_stds = []
                
                def _on_step(self) -> bool:
                    result = super()._on_step()
                    # Capture eval results after each evaluation
                    if hasattr(self, 'last_mean_reward') and self.last_mean_reward is not None:
                        self.eval_rewards.append(float(self.last_mean_reward))
                        self.eval_timesteps.append(int(self.num_timesteps))
                        self.eval_stds.append(float(getattr(self, 'last_std_reward', 0.0)))
                    return result
            
            # Create custom eval callback
            custom_eval_callback = GrokkingEvalCallback(
                eval_env,
                best_model_save_path=f"./models/{algorithm}_best",
                log_path=f"./logs/{algorithm}_eval",
                eval_freq=params.get('eval_freq', 10000),
                n_eval_episodes=5,
                deterministic=True
            )
            
            model = trainer.train(
                total_timesteps=params['timesteps'], 
                eval_env=eval_env,
                eval_callback=custom_eval_callback
            )
            
            # Capture eval history from custom callback
            eval_rewards_history = custom_eval_callback.eval_rewards
            eval_timesteps_history = custom_eval_callback.eval_timesteps
            eval_stds_history = custom_eval_callback.eval_stds
            
            # Check if training was interrupted - don't save incomplete models
            if is_shutdown_requested():
                print(f"\n[{model_id}] Training interrupted - NOT saving incomplete model")
                result['error'] = 'Training interrupted by shutdown'
                return result
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{model_id}_{algorithm}.zip")
            trainer.save_model(model_path)
            
            # Save indicator stats for inference (CRITICAL for paper trading)
            indicator_stats = feature_engineer.get_indicator_stats(df)
            indicator_stats_path = os.path.join(self.models_dir, f"{model_id}_indicator_stats.json")
            feature_engineer.save_indicator_stats(indicator_stats, indicator_stats_path)
            
            # Load eval history from npz file for model selection
            eval_log_path = f"./logs/{algorithm}_eval/evaluations.npz"
            eval_rewards = []
            eval_timesteps = []
            eval_stds = []
            if os.path.exists(eval_log_path):
                try:
                    eval_data = np.load(eval_log_path)
                    eval_timesteps = eval_data['timesteps'].tolist()
                    eval_results = eval_data['results']
                    eval_rewards = [float(np.mean(r)) for r in eval_results]
                    eval_stds = [float(np.std(r)) for r in eval_results]
                except Exception as e:
                    print(f"[{model_id}] Warning: Could not load eval history: {e}")
            
            # Use captured eval data if available, otherwise fall back to file
            if eval_rewards_history:
                eval_rewards = eval_rewards_history
                eval_timesteps = eval_timesteps_history
                eval_stds = eval_stds_history
                print(f"[{model_id}] Captured {len(eval_rewards)} eval checkpoints from training")
            
            # Run proper grokking analysis
            print(f"[{model_id}] Running grokking analysis...")
            grokking_result = self._analyze_grokking(
                model_id=model_id,
                eval_timesteps=eval_timesteps,
                eval_rewards=eval_rewards,
                eval_stds=eval_stds,
                timesteps=params['timesteps']
            )
            
            # Save metadata with eval history and grokking analysis
            metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.yaml")
            with open(metadata_path, 'w') as f:
                metadata = {
                    'model_id': model_id,
                    'algorithm': algorithm,
                    'tickers': tickers,
                    'tech_indicators': tech_indicators,
                    'parameters': params,
                    'training_date': datetime.now().isoformat(),
                    'model_path': model_path,
                    'indicator_stats_path': indicator_stats_path,
                    'eval_rewards': eval_rewards,
                    'eval_timesteps': eval_timesteps,
                    'eval_stds': eval_stds,
                    'final_eval_reward': eval_rewards[-1] if eval_rewards else None,
                    'best_eval_reward': max(eval_rewards) if eval_rewards else None,
                    'grokking_analysis': grokking_result,
                    'has_grokked': grokking_result.get('has_grokked', False),
                    'state_dimension': train_env.state_dim,
                    'market_context_enabled': True
                }
                yaml.safe_dump(self._to_builtin(metadata), f, default_flow_style=False)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            result['success'] = True
            result['model_path'] = model_path
            result['metadata_path'] = metadata_path
            result['training_time'] = training_time
            
            # Save to strategy database with grokking metrics
            self.strategy_db.save_strategy({
                'model_id': model_id,
                'params': params,
                'training_time': training_time,
                'grokking_analysis': metadata.get('grokking_analysis', {})
            })
            
            print(f"\n[{model_id}] ✓ Training complete in {training_time:.1f}s")
            print(f"[{model_id}] Model saved to: {model_path}")
            print(f"[{model_id}] Strategy logged to database")
            
        except Exception as e:
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            result['error'] = str(e)
            result['training_time'] = training_time
            result['traceback'] = traceback.format_exc()
            
            print(f"\n[{model_id}] ✗ Training failed after {training_time:.1f}s")
            print(f"[{model_id}] Error: {e}")
            
            # Save error log
            error_log_path = os.path.join(self.logs_dir, f"{model_id}_error.log")
            with open(error_log_path, 'w') as f:
                f.write(f"Model ID: {model_id}\n")
                f.write(f"Parameters: {params}\n\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
        
        return result
    
    def train_all_models(self, parameter_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Train all models with different parameter sets
        
        Args:
            parameter_sets: List of parameter dictionaries
        
        Returns:
            List of training results
        """
        print("\n" + "="*80)
        print(f"AUTOMATED TRAINING SESSION")
        print(f"Training {len(parameter_sets)} models")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        results = []
        successful = 0
        failed = 0
        
        for i, params in enumerate(parameter_sets, 1):
            # Check if shutdown was requested
            if is_shutdown_requested():
                print("\n\nShutdown requested - stopping training loop...")
                break
            
            print(f"\n\nProgress: {i}/{len(parameter_sets)}")
            
            result = self.train_model(params)
            results.append(result)
            
            # Check again after training - if shutdown was requested during training, stop now
            if is_shutdown_requested():
                print("\n\nShutdown requested - stopping training loop...")
                break
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Save summary
        summary_path = os.path.join(self.logs_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        with open(summary_path, 'w') as f:
            yaml.dump({
                'total_models': len(parameter_sets),
                'successful': successful,
                'failed': failed,
                'results': results
            }, f)
        
        print("\n" + "="*80)
        print("AUTOMATED TRAINING COMPLETE")
        print("="*80)
        print(f"Total Models: {len(parameter_sets)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        # Get and display grokking statistics
        print("\n" + "-"*80)
        print("GROKKING STATISTICS")
        print("-"*80)
        try:
            grok_stats = self.get_grokking_statistics()
            if grok_stats['total_models'] > 0:
                print(f"Models Analyzed: {grok_stats['total_models']}")
                print(f"Grokking Rate: {grok_stats['grokking_percentage']:.1f}% ({grok_stats['grokked_count']}/{grok_stats['total_models']})")
                print(f"Avg Improvement: {grok_stats['avg_improvement_pct']:.1f}%")
                if grok_stats.get('phase_distribution'):
                    print("Phase Distribution:")
                    for phase, count in grok_stats['phase_distribution'].items():
                        print(f"  {phase}: {count}")
            else:
                print("No grokking data available yet")
        except Exception as e:
            print(f"Could not retrieve grokking stats: {e}")
        print("-"*80)
        
        print(f"\nSummary saved to: {summary_path}")
        print("="*80)
        
        self.training_results = results
        
        # Run model selection to identify best models
        selected_models = self.select_best_models(results)
        
        return results
    
    def select_best_models(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Select models based on eval performance, learning trend, and grokking analysis.
        Only models that show actual learning (not just memorization) are promoted.
        
        Grokking check: For each model, loads the trained weights and runs spectral
        analysis on weight matrices + eval curve phase detection to verify the model
        has genuinely internalized trading patterns vs memorizing training data.
        Models that fail the grokking check are blocked from backtesting.
        """
        from src.evaluation.model_selector import ModelSelector
        from stable_baselines3 import PPO, SAC, A2C, DDPG
        
        print("\n" + "="*60)
        print("MODEL SELECTION PHASE (with Grokking Analysis)")
        print("="*60)
        
        selector = ModelSelector(
            min_final_reward=0.0,      # Must be profitable
            min_trend_score=-0.1,      # Allow slight decline if converged
            max_variance_ratio=0.5,    # Must be reasonably consistent
            top_k=5                    # Keep top 5 models for testing
        )
        
        # Map algorithm names to SB3 classes for model loading
        algo_classes = {
            'ppo': PPO,
            'sac': SAC,
            'a2c': A2C,
            'ddpg': DDPG,
        }
        
        for result in results:
            if not result['success']:
                continue
            
            model_id = result['params']['model_id']
            model_path = result['model_path']
            
            # Load eval history from metadata
            metadata_path = result.get('metadata_path')
            if metadata_path and os.path.exists(metadata_path):
                metadata = self._load_metadata(metadata_path)
                
                eval_rewards = metadata.get('eval_rewards', [])
                eval_timesteps = metadata.get('eval_timesteps', [])
                eval_stds = metadata.get('eval_stds', [])
                
                if eval_rewards and len(eval_rewards) >= 3:
                    # Load the trained model for grokking analysis
                    loaded_model = None
                    load_error = None
                    
                    # Try loading with retry logic (file might still be writing)
                    algorithm = result['params'].get('algorithm', 'ppo').lower()
                    algo_class = algo_classes.get(algorithm, PPO)
                    
                    for attempt in range(3):
                        try:
                            loaded_model = algo_class.load(model_path, device='cpu')
                            print(f"[{model_id}] Loaded model for grokking analysis (attempt {attempt + 1})")
                            break
                        except Exception as e:
                            load_error = e
                            if attempt < 2:
                                print(f"[{model_id}] Model load attempt {attempt + 1} failed, retrying...")
                                import time
                                time.sleep(0.5)  # Wait for file to be fully written
                            else:
                                print(f"[{model_id}] ERROR: Could not load model for grokking analysis after 3 attempts: {e}")
                    
                    evaluation = selector.evaluate_model(
                        model_id=model_id,
                        model_path=model_path,
                        eval_rewards=eval_rewards,
                        eval_timesteps=eval_timesteps,
                        eval_stds=eval_stds,
                        loaded_model=loaded_model
                    )
                    
                    # ALWAYS store grokking results in metadata for downstream use
                    # This ensures colosseum mode can check if grokking analysis was performed
                    metadata['grokking_analysis'] = evaluation.grokking_details if evaluation.grokking_details else {}
                    metadata['has_grokked'] = evaluation.has_grokked
                    metadata['grokking_error'] = str(load_error) if load_error else None
                    metadata['grokking_attempted'] = True
                    
                    with open(metadata_path, 'w') as f:
                        yaml.safe_dump(self._to_builtin(metadata), f, default_flow_style=False)
                    
                    if loaded_model is None:
                        print(f"[{model_id}] WARNING: Model passed selection but grokking analysis was skipped due to load failure")
                    
                    # Free model memory
                    del loaded_model
                else:
                    print(f"[{model_id}] Skipped - insufficient eval history")
        
        # Print selection report
        print(selector.get_selection_report())
        
        # Save selection results
        selection_path = os.path.join(self.logs_dir, f"model_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        selector.save_results(selection_path)
        
        # Get selected model IDs
        selected = selector.select_models()
        selected_ids = [m.model_id for m in selected]
        
        # Mark selected models in their metadata
        for model_id in selected_ids:
            metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.yaml")
            if os.path.exists(metadata_path):
                metadata = self._load_metadata(metadata_path)
                metadata['selected_for_testing'] = True
                with open(metadata_path, 'w') as f:
                    yaml.safe_dump(self._to_builtin(metadata), f, default_flow_style=False)
        
        return selected_ids
    
    def get_successful_models(self) -> List[Dict[str, Any]]:
        """Get list of successfully trained models"""
        return [r for r in self.training_results if r['success']]
    
    def continue_training_model(self, model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Continue training an existing model
        
        Args:
            model_spec: Specification with parent_model, additional_timesteps, etc.
        
        Returns:
            Training result dictionary
        """
        from src.autotest.continuation_trainer import continue_training_model
        
        return continue_training_model(
            model_spec=model_spec,
            base_config=self.base_config,
            models_dir=self.models_dir,
            logs_dir=self.logs_dir
        )
    
    def _analyze_grokking(self, model_id: str, eval_timesteps: List[int], 
                         eval_rewards: List[float], eval_stds: List[float],
                         timesteps: int) -> Dict[str, Any]:
        """
        Analyze training run for genuine grokking - PHASE TRANSITION detection.
        
        Grokking is characterized by:
        1. Hockey-stick eval curve (flat early, sharp improvement late)
        2. Weight matrices developing structure (declining effective rank)
        3. Generalization gap closing
        
        This can happen at ANY timestep count, though research shows it's rare before 500K.
        """
        
        if len(eval_timesteps) < 3:
            return {
                'has_grokked': False,
                'phase': 'insufficient_checkpoints',
                'confidence': 0.0,
                'evidence': 'Need at least 3 eval checkpoints to detect phase transitions',
                'score': 0.0,
                'training_duration': timesteps,
                'relative_improvement_pct': 0.0,
                'early_mean': 0.0,
                'mid_mean': 0.0,
                'late_mean': 0.0,
                'recommendation': 'Train with more frequent evals'
            }
        
        # Analyze eval curve for phase transition
        rewards = np.array(eval_rewards, dtype=float)
        n = len(rewards)
        
        # Handle edge cases with few checkpoints
        if n == 3:
            early_rewards = rewards[:1]
            mid_rewards = rewards[1:2]
            late_rewards = rewards[2:]
        elif n == 4:
            early_rewards = rewards[:1]
            mid_rewards = rewards[1:3]
            late_rewards = rewards[3:]
        else:
            # Split into thirds for better phase detection
            third = n // 3
            early_rewards = rewards[:third] if third > 0 else rewards[:1]
            mid_rewards = rewards[third:2*third] if third > 0 else rewards[1:2]
            late_rewards = rewards[2*third:] if third > 0 else rewards[2:]
        
        early_mean = np.mean(early_rewards)
        mid_mean = np.mean(mid_rewards)
        late_mean = np.mean(late_rewards)
        
        # Compute normalized trends
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            y = np.array(values)
            # Linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            # Normalize by range
            value_range = np.max(y) - np.min(y)
            if value_range < 1e-6:
                value_range = abs(np.mean(y)) + 1e-6
            return m * len(values) / value_range
        
        early_trend = compute_trend(early_rewards)
        mid_trend = compute_trend(mid_rewards)
        late_trend = compute_trend(late_rewards)
        
        # Compute hockey-stick metrics
        early_to_mid = (mid_mean - early_mean) / (abs(early_mean) + 1e-8)
        mid_to_late = (late_mean - mid_mean) / (abs(mid_mean) + 1e-8)
        
        # GROKKING DETECTION LOGIC
        # Classic grokking: flat/declining early, sharp improvement late
        # Key: Must show SUDDEN transition, not steady improvement
        
        # Calculate relative changes
        early_flat = abs(early_to_mid) < 0.3  # Early phase relatively flat
        late_jump = mid_to_late > 1.0  # Late phase shows 100%+ improvement
        total_improvement = (late_mean - early_mean) / (abs(early_mean) + 1e-8)
        
        # STRICT hockey stick: must be flat early AND sudden jump late
        is_hockey_stick = early_flat and late_jump and total_improvement > 1.5
        
        # Grokking phase: sharp improvement but not full transition yet
        is_grokking_phase = (mid_to_late > 0.6 and late_mean > mid_mean * 1.3 and 
                            not early_flat and total_improvement > 0.8)
        
        # Declining/memorizing
        is_declining = (early_mean > late_mean * 1.2) or (mid_mean < early_mean * 0.7)
        
        # Steady learning (no phase transition)
        is_steady = abs(early_to_mid) > 0.1 and abs(mid_to_late) > 0.1 and abs(early_to_mid - mid_to_late) < 0.5
        
        # Determine phase
        if is_hockey_stick:
            phase = 'post_grok'
            confidence = min(1.0, mid_to_late / 2.0)
            evidence = f"Phase transition: flat early ({early_to_mid:.2f}), sudden jump ({mid_to_late:.2f}), {total_improvement:.1f}x total gain"
        elif is_grokking_phase and not is_steady:
            phase = 'grokking'
            confidence = min(1.0, mid_to_late)
            evidence = f"Transition occurring: {mid_to_late:.2f}x late improvement, {total_improvement:.1f}x total"
        elif is_declining:
            phase = 'memorizing'
            confidence = 0.8
            evidence = f"Declining: {early_mean:.1f} → {mid_mean:.1f} → {late_mean:.1f}"
        else:
            phase = 'learning'
            confidence = 0.4 + (0.15 if late_mean > early_mean else 0)
            evidence = f"Steady learning: {early_mean:.1f} → {mid_mean:.1f} → {late_mean:.1f}"
        
        # Score based on phase
        if phase == 'post_grok':
            base_score = 0.92
        elif phase == 'grokking':
            base_score = 0.72
        elif phase == 'learning':
            base_score = 0.32
        else:
            base_score = 0.08
        
        score = base_score * confidence
        
        # Adjust for final profitability
        final_mean = np.mean(eval_rewards[-3:]) if len(eval_rewards) >= 3 else np.mean(eval_rewards)
        if final_mean <= 0:
            score *= 0.3
            evidence += f". WARNING: Final performance negative ({final_mean:.2f})"
        else:
            evidence += f". Final performance: {final_mean:.2f}"
        
        # Adjust for timestep count (research shows grokking rare before 500K, but possible)
        timestep_factor = min(1.0, timesteps / 500000)
        if timesteps < 100000:
            score *= 0.7  # Unlikely but possible
            evidence += f". Note: Only {timesteps:,} steps (grokking typically 500K+)"
        
        # Final verdict
        has_grokked = (phase in ['post_grok', 'grokking']) and score > 0.45 and final_mean > 0
        
        # Calculate relative improvement percentage (late vs early)
        relative_improvement_pct = ((late_mean - early_mean) / (abs(early_mean) + 1e-8)) * 100
        
        return {
            'has_grokked': has_grokked,
            'phase': phase,
            'confidence': round(score, 3),
            'evidence': evidence,
            'score': round(score, 3),
            'training_duration': timesteps,
            'final_eval_mean': round(float(final_mean), 2),
            'num_checkpoints': len(eval_timesteps),
            'relative_improvement_pct': round(float(relative_improvement_pct), 2),
            'early_mean': round(float(early_mean), 2),
            'mid_mean': round(float(mid_mean), 2),
            'late_mean': round(float(late_mean), 2),
            'trends': {
                'early': round(float(early_trend), 3),
                'mid': round(float(mid_trend), 3) if not np.isnan(mid_trend) else 0.0,
                'late': round(float(late_trend), 3)
            },
            'phase_changes': {
                'early_to_mid': round(float(early_to_mid), 3),
                'mid_to_late': round(float(mid_to_late), 3)
            },
            'recommendation': 'Continue training' if phase in ['learning', 'grokking'] and timesteps < 500000 else 'Ready for validation' if has_grokked else 'Review hyperparameters'
        }
    
    def get_grokking_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on grokking patterns across all trained models.
        
        Returns:
            Dict with grokking percentages and metrics
        """
        grokking_data = []
        
        # Load metadata from all saved models
        import glob
        metadata_files = glob.glob(os.path.join(self.models_dir, '*_metadata.yaml'))
        
        for metadata_path in metadata_files:
            try:
                metadata = self._load_metadata(metadata_path)
                if 'grokking_analysis' in metadata:
                    grokking_data.append({
                        'model_id': metadata['model_id'],
                        'has_grokked': metadata.get('has_grokked', False),
                        'phase': metadata['grokking_analysis'].get('phase', 'unknown'),
                        'relative_improvement_pct': metadata['grokking_analysis'].get('relative_improvement_pct', 0),
                        'score': metadata['grokking_analysis'].get('score', 0),
                        'timesteps': metadata['grokking_analysis'].get('training_duration', 0)
                    })
            except Exception as e:
                print(f"Warning: Could not load metadata from {metadata_path}: {e}")
                continue
        
        if not grokking_data:
            return {
                'total_models': 0,
                'grokking_percentage': 0.0,
                'avg_improvement_pct': 0.0,
                'message': 'No grokking data available yet'
            }
        
        total_models = len(grokking_data)
        grokked_models = sum(1 for m in grokking_data if m['has_grokked'])
        grokking_pct = (grokked_models / total_models) * 100
        avg_improvement = np.mean([m['relative_improvement_pct'] for m in grokking_data])
        
        # Count by phase
        phase_counts = {}
        for m in grokking_data:
            phase = m['phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        return {
            'total_models': total_models,
            'grokking_percentage': round(grokking_pct, 2),
            'grokked_count': grokked_models,
            'avg_improvement_pct': round(float(avg_improvement), 2),
            'phase_distribution': phase_counts,
            'models': grokking_data
        }
