import os
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from typing import Dict, Any
import torch

# Global shutdown flag - set by signal handlers
_shutdown_requested = False

def request_shutdown():
    """Call this to request graceful shutdown of training"""
    global _shutdown_requested
    _shutdown_requested = True

def reset_shutdown():
    """Reset shutdown flag for new training session"""
    global _shutdown_requested
    _shutdown_requested = False

def is_shutdown_requested():
    """Check if shutdown was requested - use this instead of importing the variable"""
    return _shutdown_requested


class ShutdownCallback(BaseCallback):
    """Callback that stops training when shutdown is requested"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        if _shutdown_requested:
            print("\nShutdown requested - stopping training gracefully...")
            return False  # Returning False stops training
        return True


class GrokkingMonitorCallback(BaseCallback):
    """
    Monitors weight matrix spectral properties during training to track
    whether the model is developing structured representations (grokking)
    vs staying random (memorizing).
    
    Logs to TensorBoard:
    - grokking/avg_effective_rank_ratio: Lower = more structured weights
    - grokking/avg_weight_norm: Weight magnitude (should stabilize)
    - grokking/spectral_norm_ratio: Top singular value dominance
    
    These metrics let you watch grokking happen in real-time on TensorBoard.
    A model that is grokking will show declining rank ratio over training.
    """
    
    def __init__(self, check_freq: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rank_history = []
        self.norm_history = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        
        try:
            policy = self.model.policy
            rank_ratios = []
            weight_norms = []
            spectral_norms = []
            
            for name, param in policy.named_parameters():
                if 'weight' not in name or len(param.shape) < 2:
                    continue
                
                weight = param.data.detach().cpu().float()
                if weight.shape[0] < 4 or weight.shape[1] < 4:
                    continue
                
                try:
                    S = torch.linalg.svdvals(weight)
                    
                    # Effective rank via participation ratio
                    S_sq = S ** 2
                    total = S_sq.sum()
                    if total > 0:
                        eff_rank = (total ** 2) / (S_sq ** 2).sum()
                        rank_ratio = eff_rank.item() / min(weight.shape)
                        rank_ratios.append(rank_ratio)
                    
                    weight_norms.append(torch.norm(weight, p='fro').item())
                    spectral_norms.append(S[0].item())
                    
                except Exception:
                    continue
            
            if rank_ratios:
                avg_rank = np.mean(rank_ratios)
                avg_norm = np.mean(weight_norms)
                
                self.rank_history.append(avg_rank)
                self.norm_history.append(avg_norm)
                
                # Log to TensorBoard
                self.logger.record("grokking/avg_effective_rank_ratio", avg_rank)
                self.logger.record("grokking/avg_weight_norm", avg_norm)
                
                if spectral_norms and avg_norm > 0:
                    # Spectral norm ratio: how dominant is the top singular value
                    avg_spectral = np.mean(spectral_norms)
                    self.logger.record("grokking/avg_spectral_norm", avg_spectral)
                
                # Track rank trend (is the model becoming more structured?)
                if len(self.rank_history) >= 3:
                    recent = np.mean(self.rank_history[-3:])
                    early = np.mean(self.rank_history[:3])
                    rank_trend = recent - early  # Negative = improving
                    self.logger.record("grokking/rank_trend", rank_trend)
        
        except Exception as e:
            if self.verbose > 0:
                print(f"GrokkingMonitor: Error computing spectral metrics: {e}")
        
        return True


class RLTrainer:
    def __init__(self, env, config: Dict[str, Any], model_name: str = 'ppo'):
        # Wrap environment with Monitor before DummyVecEnv
        # Note: We don't use VecNormalize here because normalization is handled in:
        # 1. Feature engineering (z-score normalization of indicators)
        # 2. Trading environment (_get_state normalizes all features)
        # This simplifies inference and avoids sync issues with eval env
        monitored_env = Monitor(env)
        self.env = DummyVecEnv([lambda: monitored_env])
        
        self.config = config
        self.model_name = model_name.lower()
        self.model = None
        
    def create_model(self):
        algo_config = self.config['training'].get(self.model_name, {})
        
        policy_kwargs = algo_config.get('policy_kwargs', {})
        if 'net_arch' in policy_kwargs:
            net_arch = policy_kwargs['net_arch']
            if isinstance(net_arch, dict):
                # SB3 v1.8.0+ format: dict(pi=..., vf=...) not [dict(...)]
                policy_kwargs['net_arch'] = dict(
                    pi=net_arch.get('pi', [256, 128]), 
                    vf=net_arch.get('vf', [256, 128])
                )
        
        # Add weight decay to optimizer if specified
        weight_decay = algo_config.get('weight_decay', 0.0)
        if weight_decay > 0:
            if 'optimizer_kwargs' not in policy_kwargs:
                policy_kwargs['optimizer_kwargs'] = {}
            policy_kwargs['optimizer_kwargs']['weight_decay'] = weight_decay
        
        common_params = {
            'policy': 'MlpPolicy',
            'env': self.env,
            'verbose': 0,
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
        elif self.model_name == 'sac':
            # SAC - Off-policy, sample efficient, good for continuous actions
            self.model = SAC(
                **common_params,
                learning_rate=algo_config.get('learning_rate', 3e-4),
                buffer_size=algo_config.get('buffer_size', 1000000),
                learning_starts=algo_config.get('learning_starts', 1000),
                batch_size=algo_config.get('batch_size', 256),
                tau=algo_config.get('tau', 0.005),
                gamma=algo_config.get('gamma', 0.99),
                ent_coef=algo_config.get('ent_coef', 'auto'),  # Auto-tune entropy
                target_entropy=algo_config.get('target_entropy', 'auto'),
                train_freq=algo_config.get('train_freq', 1),
                gradient_steps=algo_config.get('gradient_steps', 1),
                policy_kwargs={'net_arch': algo_config.get('policy_kwargs', {}).get('net_arch', [256, 256])}
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
        
        # Add shutdown callback first so it's checked on every step
        callbacks.append(ShutdownCallback())
        
        # Add grokking monitor to track weight structure during training
        # Logs spectral metrics to TensorBoard every 50k steps
        grokking_freq = self.config['training'].get('grokking_monitor_freq', 50000)
        callbacks.append(GrokkingMonitorCallback(check_freq=grokking_freq))
        
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
                n_eval_episodes=5,  # Run 5 episodes per eval for proper variance
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
            progress_bar=False
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
        elif self.model_name == 'sac':
            self.model = SAC.load(path, env=self.env)
        
        print(f"Model loaded from {path}")
        return self.model
    
    def predict(self, observation, deterministic=True):
        if self.model is None:
            raise ValueError("Model not created or loaded")
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
