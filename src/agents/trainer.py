import os
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from typing import Dict, Any
import torch


class RLTrainer:
    def __init__(self, env, config: Dict[str, Any], model_name: str = 'ppo'):
        self.env = DummyVecEnv([lambda: env])
        self.config = config
        self.model_name = model_name.lower()
        self.model = None
        
    def create_model(self):
        algo_config = self.config['training'].get(self.model_name, {})
        
        policy_kwargs = algo_config.get('policy_kwargs', {})
        if 'net_arch' in policy_kwargs:
            net_arch = policy_kwargs['net_arch']
            if isinstance(net_arch, dict):
                policy_kwargs['net_arch'] = [
                    dict(pi=net_arch.get('pi', [256, 128]), 
                         vf=net_arch.get('vf', [256, 128]))
                ]
        
        common_params = {
            'policy': 'MlpPolicy',
            'env': self.env,
            'verbose': 1,
            'tensorboard_log': self.config['monitoring']['log_dir'],
            'device': 'cpu'
        }
        
        if self.model_name == 'ppo':
            self.model = PPO(
                **common_params,
                learning_rate=algo_config.get('learning_rate', 3e-4),
                n_steps=algo_config.get('n_steps', 2048),
                batch_size=algo_config.get('batch_size', 64),
                gamma=algo_config.get('gamma', 0.99),
                gae_lambda=algo_config.get('gae_lambda', 0.95),
                clip_range=algo_config.get('clip_range', 0.2),
                ent_coef=algo_config.get('ent_coef', 0.01),
                vf_coef=algo_config.get('vf_coef', 0.5),
                max_grad_norm=algo_config.get('max_grad_norm', 0.5),
                policy_kwargs=policy_kwargs
            )
        elif self.model_name == 'a2c':
            self.model = A2C(
                **common_params,
                learning_rate=algo_config.get('learning_rate', 7e-4),
                n_steps=algo_config.get('n_steps', 5),
                gamma=algo_config.get('gamma', 0.99),
                gae_lambda=algo_config.get('gae_lambda', 1.0),
                ent_coef=algo_config.get('ent_coef', 0.01),
                vf_coef=algo_config.get('vf_coef', 0.5),
                max_grad_norm=algo_config.get('max_grad_norm', 0.5),
                policy_kwargs=policy_kwargs
            )
        elif self.model_name == 'ddpg':
            self.model = DDPG(
                **common_params,
                learning_rate=algo_config.get('learning_rate', 1e-4),
                buffer_size=algo_config.get('buffer_size', 1000000),
                learning_starts=algo_config.get('learning_starts', 100),
                batch_size=algo_config.get('batch_size', 100),
                tau=algo_config.get('tau', 0.005),
                gamma=algo_config.get('gamma', 0.99),
                policy_kwargs={'net_arch': algo_config.get('policy_kwargs', {}).get('net_arch', [400, 300])}
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.model_name}")
        
        print(f"Created {self.model_name.upper()} model")
        return self.model
    
    def train(self, total_timesteps: int = None, eval_env=None):
        if self.model is None:
            self.create_model()
        
        if total_timesteps is None:
            total_timesteps = self.config['training']['total_timesteps']
        
        callbacks = []
        
        checkpoint_dir = self.config['monitoring']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training'].get('save_freq', 50000),
            save_path=f"{checkpoint_dir}/{self.model_name}",
            name_prefix=f"{self.model_name}_model"
        )
        callbacks.append(checkpoint_callback)
        
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"./models/{self.model_name}_best",
                log_path=f"./logs/{self.model_name}_eval",
                eval_freq=self.config['training'].get('eval_freq', 10000),
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        print(f"Training {self.model_name.upper()} for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        return self.model
    
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        if self.model_name == 'ppo':
            self.model = PPO.load(path, env=self.env)
        elif self.model_name == 'a2c':
            self.model = A2C.load(path, env=self.env)
        elif self.model_name == 'ddpg':
            self.model = DDPG.load(path, env=self.env)
        
        print(f"Model loaded from {path}")
        return self.model
    
    def predict(self, observation, deterministic=True):
        if self.model is None:
            raise ValueError("Model not created or loaded")
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
