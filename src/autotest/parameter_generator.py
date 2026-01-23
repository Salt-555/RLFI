import itertools
import random
from typing import List, Dict, Any
import yaml


class ParameterGenerator:
    """
    Generates diverse parameter combinations for training multiple AI models
    """
    
    def __init__(self, config_path: str = 'config/autotest_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.autotest_config = self.config['autotest']
        self.training_config = self.autotest_config['training']
    
    def generate_parameter_sets(self, num_sets: int = None) -> List[Dict[str, Any]]:
        """
        Generate diverse parameter combinations for training
        
        Args:
            num_sets: Number of parameter sets to generate
        
        Returns:
            List of parameter dictionaries
        """
        if num_sets is None:
            num_sets = self.training_config['num_models_to_train']
        
        parameter_sets = []
        
        # Get all possible combinations
        algorithms = self.training_config['algorithms']
        ticker_combos = self.training_config['ticker_combinations']
        learning_rates = self.training_config['learning_rate_variations']
        reward_scalings = self.training_config['reward_scaling_variations']
        initial_capitals = self.training_config['initial_capital_variations']
        
        # Create a pool of all possible combinations
        all_combinations = []
        
        for algo in algorithms:
            for tickers in ticker_combos:
                for lr in learning_rates:
                    for reward_scale in reward_scalings:
                        for capital in initial_capitals:
                            all_combinations.append({
                                'algorithm': algo,
                                'tickers': tickers,
                                'learning_rate': lr,
                                'reward_scaling': reward_scale,
                                'initial_capital': capital,
                                'timesteps': self.training_config['base_timesteps']
                            })
        
        # Randomly sample from all combinations
        if len(all_combinations) > num_sets:
            parameter_sets = random.sample(all_combinations, num_sets)
        else:
            parameter_sets = all_combinations
        
        # Add unique identifiers
        for i, params in enumerate(parameter_sets):
            params['model_id'] = f"model_{i+1:03d}"
            params['name'] = f"{params['algorithm']}_{params['model_id']}"
        
        return parameter_sets
    
    def generate_balanced_sets(self, num_sets: int = None) -> List[Dict[str, Any]]:
        """
        Generate balanced parameter sets ensuring diversity across all dimensions.
        Uses round-robin basket selection for sector diversity.
        Supports both PPO and SAC algorithms with weighted selection.
        
        Args:
            num_sets: Number of parameter sets to generate
        
        Returns:
            List of parameter dictionaries
        """
        if num_sets is None:
            num_sets = self.training_config['num_models_to_train']
        
        parameter_sets = []
        
        # Get stock baskets (organized by sector) or fall back to old format
        stock_baskets = self.training_config.get('stock_baskets', {})
        if stock_baskets:
            basket_keys = list(stock_baskets.keys())
        else:
            # Fallback to old ticker_combinations format
            ticker_combos = self.training_config.get('ticker_combinations', [["SPY", "QQQ"]])
            basket_keys = None
        
        learning_rates = self.training_config['learning_rate_variations']
        
        # Algorithm selection with weights
        algorithms = self.training_config.get('algorithms', ['ppo'])
        algorithm_weights = self.training_config.get('algorithm_weights', {'ppo': 1.0})
        
        # PPO-specific hyperparameters
        n_steps_variations = self.training_config.get('n_steps_variations', [2048])
        batch_size_variations = self.training_config.get('batch_size_variations', [64])
        gamma_variations = self.training_config.get('gamma_variations', [0.99])
        clip_range_variations = self.training_config.get('clip_range_variations', [0.2])
        ent_coef_variations = self.training_config.get('ent_coef_variations', [0.01])
        
        # SAC-specific hyperparameters
        sac_buffer_size_variations = self.training_config.get('sac_buffer_size_variations', [1000000])
        sac_batch_size_variations = self.training_config.get('sac_batch_size_variations', [256])
        sac_tau_variations = self.training_config.get('sac_tau_variations', [0.005])
        sac_train_freq_variations = self.training_config.get('sac_train_freq_variations', [1])
        
        # Network architecture variations
        net_arch_variations = self.training_config.get('net_arch_variations', [[[256, 128], [256, 128]]])
        
        # Indicator pool for random selection
        indicator_pool = [
            'macd', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma',
            'adx', 'boll_ub', 'boll_lb', 'atr', 'willr', 'rsi_14', 'cci_14',
            'dx_14', 'close_20_sma', 'close_50_sma', 'adxr'
        ]
        
        # Generate unique model ID with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        for i in range(num_sets):
            # Round-robin basket selection for sector diversity
            if basket_keys:
                basket_key = basket_keys[i % len(basket_keys)]
                basket_info = stock_baskets[basket_key]
                tickers = basket_info['tickers']
                basket_name = basket_info['name']
            else:
                tickers = ticker_combos[i % len(ticker_combos)]
                basket_name = "legacy"
            
            # Randomly select 3-5 technical indicators for this model
            num_indicators = random.randint(3, 5)
            selected_indicators = random.sample(indicator_pool, num_indicators)
            
            # Select network architecture
            net_arch = random.choice(net_arch_variations)
            
            # Select algorithm based on weights
            algo = self._select_algorithm_weighted(algorithms, algorithm_weights)
            
            model_id = f"{timestamp}_{i+1:03d}"
            
            # Base parameters common to all algorithms
            params = {
                'model_id': model_id,
                'algorithm': algo,
                'tickers': tickers,
                'basket_name': basket_name,
                'tech_indicators': selected_indicators,
                'learning_rate': random.choice(learning_rates),
                'timesteps': self.training_config['base_timesteps'],
                'gamma': random.choice(gamma_variations),
                'net_arch': net_arch
            }
            
            # Algorithm-specific parameters
            if algo == 'ppo':
                params.update({
                    'n_steps': random.choice(n_steps_variations),
                    'batch_size': random.choice(batch_size_variations),
                    'clip_range': random.choice(clip_range_variations),
                    'ent_coef': random.choice(ent_coef_variations),
                })
            elif algo == 'sac':
                params.update({
                    'buffer_size': random.choice(sac_buffer_size_variations),
                    'batch_size': random.choice(sac_batch_size_variations),
                    'tau': random.choice(sac_tau_variations),
                    'train_freq': random.choice(sac_train_freq_variations),
                    'ent_coef': 'auto',  # SAC auto-tunes entropy
                })
            
            params['name'] = f"{algo}_{model_id}"
            parameter_sets.append(params)
        
        return parameter_sets
    
    def _select_algorithm_weighted(self, algorithms: List[str], weights: Dict[str, float]) -> str:
        """Select algorithm based on configured weights"""
        # Build weighted list
        algo_weights = []
        for algo in algorithms:
            weight = weights.get(algo, 1.0 / len(algorithms))
            algo_weights.append((algo, weight))
        
        # Normalize weights
        total_weight = sum(w for _, w in algo_weights)
        normalized = [(a, w / total_weight) for a, w in algo_weights]
        
        # Random selection based on weights
        r = random.random()
        cumulative = 0
        for algo, weight in normalized:
            cumulative += weight
            if r <= cumulative:
                return algo
        
        return algorithms[0]  # Fallback
    
    def save_parameter_sets(self, parameter_sets: List[Dict[str, Any]], filepath: str):
        """Save parameter sets to a YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump({'parameter_sets': parameter_sets}, f, default_flow_style=False)
        print(f"Saved {len(parameter_sets)} parameter sets to {filepath}")
    
    def load_parameter_sets(self, filepath: str) -> List[Dict[str, Any]]:
        """Load parameter sets from a YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('parameter_sets', [])
