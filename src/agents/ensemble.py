import numpy as np
import pandas as pd
from typing import List, Dict, Any
from stable_baselines3.common.vec_env import DummyVecEnv
from src.agents.trainer import RLTrainer
from src.evaluation.metrics import calculate_sharpe_ratio, calculate_max_drawdown


class EnsembleTrainer:
    def __init__(self, env_train, env_val, config: Dict[str, Any]):
        self.env_train = env_train
        self.env_val = env_val
        self.config = config
        self.models = {}
        self.trainers = {}
        self.performance_scores = {}
        
    def train_all_models(self, algorithms: List[str] = None):
        if algorithms is None:
            algorithms = self.config['training']['algorithms']
        
        print(f"Training ensemble with algorithms: {algorithms}")
        
        for algo in algorithms:
            print(f"\n{'='*60}")
            print(f"Training {algo.upper()}")
            print(f"{'='*60}")
            
            trainer = RLTrainer(self.env_train, self.config, model_name=algo)
            
            eval_env = DummyVecEnv([lambda: self.env_val])
            model = trainer.train(eval_env=eval_env)
            
            self.trainers[algo] = trainer
            self.models[algo] = model
            
            trainer.save_model(f"./models/{algo}_trained.zip")
            
            print(f"{algo.upper()} training complete")
        
        print(f"\n{'='*60}")
        print("All models trained successfully")
        print(f"{'='*60}")
        
        return self.models
    
    def evaluate_models(self, test_env):
        print("\nEvaluating all models on validation set...")
        
        for algo, model in self.models.items():
            print(f"\nEvaluating {algo.upper()}...")
            
            obs, _ = test_env.reset()
            done = False
            portfolio_values = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
                if 'portfolio_value' in info:
                    portfolio_values.append(info['portfolio_value'])
            
            history = test_env.get_portfolio_history()
            portfolio_values = history['portfolio_values']
            
            sharpe = calculate_sharpe_ratio(portfolio_values)
            max_dd = calculate_max_drawdown(portfolio_values)
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            
            self.performance_scores[algo] = {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_return': total_return,
                'final_value': portfolio_values[-1]
            }
            
            print(f"{algo.upper()} - Sharpe: {sharpe:.3f}, Max DD: {max_dd:.3f}, Return: {total_return:.3%}")
        
        return self.performance_scores
    
    def select_best_model(self, metric: str = 'sharpe_ratio'):
        if not self.performance_scores:
            raise ValueError("Models not evaluated yet. Call evaluate_models() first.")
        
        best_algo = max(self.performance_scores.items(), 
                       key=lambda x: x[1][metric])[0]
        
        print(f"\nBest model: {best_algo.upper()} (by {metric})")
        print(f"Performance: {self.performance_scores[best_algo]}")
        
        return best_algo, self.models[best_algo]
    
    def ensemble_predict(self, observation, method: str = 'average'):
        if not self.models:
            raise ValueError("No models trained")
        
        predictions = []
        for algo, model in self.models.items():
            action, _ = model.predict(observation, deterministic=True)
            predictions.append(action)
        
        if method == 'average':
            ensemble_action = np.mean(predictions, axis=0)
        elif method == 'weighted':
            weights = [self.performance_scores[algo]['sharpe_ratio'] 
                      for algo in self.models.keys()]
            weights = np.array(weights) / sum(weights)
            ensemble_action = np.average(predictions, axis=0, weights=weights)
        elif method == 'best':
            best_algo = self.select_best_model()[0]
            ensemble_action = predictions[list(self.models.keys()).index(best_algo)]
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_action
    
    def get_performance_summary(self) -> pd.DataFrame:
        if not self.performance_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_scores).T
        df = df.sort_values('sharpe_ratio', ascending=False)
        return df
