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
from src.environment.trading_env import StockTradingEnv
from src.agents.trainer import RLTrainer
from stable_baselines3.common.vec_env import DummyVecEnv
from src.autotest.strategy_database import StrategyDatabase


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
            
            # Feature engineering with model-specific indicators
            print(f"[{model_id}] Engineering features...")
            tech_indicators = params.get('tech_indicators', self.base_config['features']['technical_indicators'])
            print(f"[{model_id}] Using {len(tech_indicators)} indicators: {tech_indicators}")
            feature_engineer = FeatureEngineer(
                tech_indicator_list=tech_indicators,
                use_turbulence=self.base_config['features']['use_turbulence']
            )
            df = feature_engineer.preprocess_data(df)
            
            # Split data
            print(f"[{model_id}] Splitting data...")
            train_df, val_df, test_df = data_loader.split_data(
                df,
                train_ratio=self.base_config['data']['train_ratio'],
                val_ratio=self.base_config['data']['val_ratio'],
                test_ratio=self.base_config['data']['test_ratio']
            )
            
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
            
            # Train model
            print(f"[{model_id}] Training {algorithm.upper()} model...")
            trainer = RLTrainer(train_env, model_config, model_name=algorithm)
            
            from stable_baselines3.common.monitor import Monitor
            eval_env = DummyVecEnv([lambda: Monitor(val_env)])
            model = trainer.train(total_timesteps=params['timesteps'], eval_env=eval_env)
            
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
            
            # Save metadata with eval history
            metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump({
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
                    'best_eval_reward': max(eval_rewards) if eval_rewards else None
                }, f)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            result['success'] = True
            result['model_path'] = model_path
            result['metadata_path'] = metadata_path
            result['training_time'] = training_time
            
            # Save to strategy database
            self.strategy_db.save_strategy({
                'model_id': model_id,
                'params': params,
                'training_time': training_time
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
        print(f"Summary saved to: {summary_path}")
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
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
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
                        yaml.dump(metadata, f)
                    
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
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                metadata['selected_for_testing'] = True
                with open(metadata_path, 'w') as f:
                    yaml.dump(metadata, f)
        
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
