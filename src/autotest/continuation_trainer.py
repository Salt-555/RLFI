"""
Continuation Trainer - Handles loading and continuing training of existing models
"""
import os
import yaml
from typing import Dict, Any
from datetime import datetime

from src.agents.trainer import RLTrainer, is_shutdown_requested
from src.environment.trading_env import StockTradingEnv
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from stable_baselines3.common.vec_env import DummyVecEnv


def continue_training_model(model_spec: Dict[str, Any], base_config: Dict[str, Any],
                           models_dir: str, logs_dir: str) -> Dict[str, Any]:
    """
    Continue training an existing model
    
    Args:
        model_spec: Specification with parent_model, additional_timesteps, etc.
        base_config: Base configuration
        models_dir: Directory for saving models
        logs_dir: Directory for logs
    
    Returns:
        Training result dictionary
    """
    parent_id = model_spec['parent_model']
    parent_path = model_spec['parent_path']
    additional_timesteps = model_spec['additional_timesteps']
    lr_multiplier = model_spec.get('learning_rate_multiplier', 0.5)
    generation = model_spec.get('generation', 2)
    tickers = model_spec['tickers']
    params = model_spec.get('params', {})
    
    # Create new model ID
    new_model_id = f"{parent_id}_g{generation}"
    
    print(f"\n{'='*80}")
    print(f"CONTINUING MODEL: {parent_id} → {new_model_id}")
    print(f"{'='*80}")
    print(f"Parent: {parent_id}")
    print(f"Generation: {generation}")
    print(f"Additional timesteps: {additional_timesteps:,}")
    print(f"Learning rate multiplier: {lr_multiplier}")
    print(f"{'='*80}")
    
    result = {
        'model_id': new_model_id,
        'parent_model_id': parent_id,
        'generation': generation,
        'params': params,
        'success': False,
        'error': None,
        'model_path': None,
        'training_time': None
    }
    
    start_time = datetime.now()
    
    try:
        # Load data
        print(f"[{new_model_id}] Loading data...")
        data_loader = DataLoader(
            tickers=tickers,
            start_date=base_config['data']['start_date'],
            end_date=base_config['data']['end_date']
        )
        
        # Try to use cached data
        data_path = f'data/raw/stock_data_{new_model_id}.csv'
        if os.path.exists(data_path):
            df = data_loader.load_data(data_path)
        else:
            df = data_loader.download_data()
            data_loader.save_data(df, data_path)
        
        # Feature engineering
        print(f"[{new_model_id}] Engineering features...")
        feature_engineer = FeatureEngineer(
            tech_indicator_list=base_config['features']['technical_indicators'],
            use_turbulence=base_config['features']['use_turbulence']
        )
        df = feature_engineer.preprocess_data(df)
        
        # Split data
        train_df, val_df, test_df = data_loader.split_data(
            df,
            train_ratio=base_config['data']['train_ratio'],
            val_ratio=base_config['data']['val_ratio'],
            test_ratio=base_config['data']['test_ratio']
        )
        
        # Create environments
        print(f"[{new_model_id}] Creating environments...")
        # Use actual number of tickers in the data
        actual_stock_dim = len(train_df['tic'].unique())
        
        train_env = StockTradingEnv(
            df=train_df,
            stock_dim=actual_stock_dim,
            hmax=base_config['environment']['hmax'],
            initial_amount=params.get('initial_capital', 100000),
            transaction_cost_pct=base_config['environment']['transaction_cost_pct'],
            reward_scaling=params.get('reward_scaling', 1e-4),
            tech_indicator_list=base_config['features']['technical_indicators'],
            turbulence_threshold=base_config['features']['turbulence_threshold']
        )
        
        val_env = StockTradingEnv(
            df=val_df,
            stock_dim=actual_stock_dim,
            hmax=base_config['environment']['hmax'],
            initial_amount=params.get('initial_capital', 100000),
            transaction_cost_pct=base_config['environment']['transaction_cost_pct'],
            reward_scaling=params.get('reward_scaling', 1e-4),
            tech_indicator_list=base_config['features']['technical_indicators'],
            turbulence_threshold=base_config['features']['turbulence_threshold']
        )
        
        # Modify config with adjusted learning rate
        model_config = base_config.copy()
        algorithm = params.get('algorithm', 'ppo')
        
        # Ensure algorithm config section exists
        if algorithm not in model_config['training']:
            model_config['training'][algorithm] = {}
        
        original_lr = params.get('learning_rate', model_config['training'][algorithm].get('learning_rate', 3e-4))
        adjusted_lr = original_lr * lr_multiplier
        model_config['training'][algorithm]['learning_rate'] = adjusted_lr
        
        print(f"[{new_model_id}] Algorithm: {algorithm.upper()}")
        print(f"[{new_model_id}] Adjusted learning rate: {original_lr} → {adjusted_lr}")
        
        # Apply algorithm-specific parameters
        if algorithm == 'ppo':
            for param_key in ['n_steps', 'batch_size', 'gamma', 'clip_range', 'ent_coef']:
                if param_key in params:
                    model_config['training'][algorithm][param_key] = params[param_key]
        elif algorithm == 'sac':
            for param_key in ['buffer_size', 'batch_size', 'gamma', 'tau', 'train_freq', 'ent_coef']:
                if param_key in params:
                    model_config['training'][algorithm][param_key] = params[param_key]
        
        # Load parent model
        print(f"[{new_model_id}] Loading parent model from {parent_path}...")
        trainer = RLTrainer(train_env, model_config, model_name=algorithm)
        trainer.load_model(parent_path)
        
        # Continue training
        print(f"[{new_model_id}] Continuing training for {additional_timesteps:,} timesteps...")
        from stable_baselines3.common.monitor import Monitor
        eval_env = DummyVecEnv([lambda: Monitor(val_env)])
        trainer.train(total_timesteps=additional_timesteps, eval_env=eval_env)
        
        # Save continued model
        model_path = os.path.join(models_dir, f"{new_model_id}_{algorithm}.zip")
        trainer.save_model(model_path)
        
        # Save metadata
        metadata_path = os.path.join(models_dir, f"{new_model_id}_metadata.yaml")
        base_timesteps = model_spec.get('base_timesteps', 500000)
        total_timesteps = base_timesteps + additional_timesteps
        
        metadata = {
            'model_id': new_model_id,
            'parent_model_id': parent_id,
            'generation': generation,
            'algorithm': algorithm,
            'tickers': tickers,
            'training_type': 'continuation',
            'base_timesteps': base_timesteps,
            'additional_timesteps': additional_timesteps,
            'total_timesteps': total_timesteps,
            'learning_rate_multiplier': lr_multiplier,
            'training_date': datetime.now().isoformat(),
            'model_path': model_path,
            'parameters': params
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        result['success'] = True
        result['model_path'] = model_path
        result['metadata_path'] = metadata_path
        result['training_time'] = training_time
        result['params']['timesteps'] = total_timesteps
        
        print(f"\n[{new_model_id}] ✓ Continuation training complete in {training_time:.1f}s")
        print(f"[{new_model_id}] Total training: {total_timesteps:,} timesteps")
        print(f"[{new_model_id}] Model saved to: {model_path}")
        
    except Exception as e:
        import traceback
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        result['error'] = str(e)
        result['training_time'] = training_time
        result['traceback'] = traceback.format_exc()
        
        print(f"\n[{new_model_id}] ✗ Continuation training failed after {training_time:.1f}s")
        print(f"[{new_model_id}] Error: {e}")
        
        # Save error log
        error_log_path = os.path.join(logs_dir, f"{new_model_id}_error.log")
        with open(error_log_path, 'w') as f:
            f.write(f"Model ID: {new_model_id}\n")
            f.write(f"Parent: {parent_id}\n")
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())
    
    return result
