"""
Model Manager - Handles model retention, continuation training, and lineage tracking
"""
import os
import yaml
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json

from .strategy_database import StrategyDatabase


class ModelManager:
    """
    Manages model lifecycle: retention, continuation, versioning, and lineage
    """
    
    def __init__(self, config_path: str = 'config/training_strategy.yaml',
                 autotest_config_path: str = 'config/autotest_config.yaml'):
        
        # Load training strategy config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.strategy_config = yaml.safe_load(f)['training_strategy']
        else:
            # Fallback to basic config
            self.strategy_config = self._get_default_strategy()
        
        # Load autotest config
        with open(autotest_config_path, 'r') as f:
            self.autotest_config = yaml.safe_load(f)['autotest']
        
        # Initialize database
        db_path = self.autotest_config['storage'].get('strategy_db', 'autotest_strategies.db')
        self.db = StrategyDatabase(db_path)
        
        # Setup directories
        self.models_dir = self.autotest_config['storage']['models_dir']
        self.archive_dir = self.strategy_config.get('retention', {}).get('archive_dir', 'champion_models')
        os.makedirs(self.archive_dir, exist_ok=True)
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default training strategy if config file doesn't exist"""
        return {
            'exploration': {'enabled': True, 'num_models': 15, 'timesteps': 1500000},
            'exploitation': {'enabled': True, 'num_models': 5, 'additional_timesteps': 2000000},
            'champions': {'enabled': True, 'num_models': 1, 'additional_timesteps': 3000000},
            'retention': {'enabled': True, 'keep_top_n_per_week': 5}
        }
    
    def get_training_plan(self) -> Dict[str, Any]:
        """
        Generate training plan for current week based on strategy
        
        Returns:
            Dictionary with exploration, exploitation, and champion model specs
        """
        plan = {
            'exploration': [],
            'exploitation': [],
            'champions': [],
            'total_models': 0,
            'total_timesteps': 0
        }
        
        # Phase 1: Exploration (new models)
        if self.strategy_config.get('exploration', {}).get('enabled', True):
            num_new = self.strategy_config['exploration']['num_models']
            timesteps = self.strategy_config['exploration']['timesteps']
            
            plan['exploration'] = [{
                'type': 'new',
                'timesteps': timesteps,
                'parent_model': None,
                'generation': 1
            } for _ in range(num_new)]
            
            plan['total_models'] += num_new
            plan['total_timesteps'] += num_new * timesteps
        
        # Phase 2: Exploitation (continue top models)
        if self.strategy_config.get('exploitation', {}).get('enabled', True):
            top_models = self._get_models_for_continuation()
            num_continue = min(
                len(top_models),
                self.strategy_config['exploitation']['num_models']
            )
            
            additional_timesteps = self.strategy_config['exploitation']['additional_timesteps']
            lr_multiplier = self.strategy_config['exploitation'].get('learning_rate_multiplier', 0.5)
            
            for model_info in top_models[:num_continue]:
                plan['exploitation'].append({
                    'type': 'continuation',
                    'parent_model': model_info['model_id'],
                    'parent_path': model_info['model_path'],
                    'base_timesteps': model_info['total_timesteps'],
                    'additional_timesteps': additional_timesteps,
                    'learning_rate_multiplier': lr_multiplier,
                    'generation': model_info.get('generation', 1) + 1,
                    'tickers': model_info['tickers'],
                    'params': model_info['params']
                })
            
            plan['total_models'] += num_continue
            plan['total_timesteps'] += num_continue * additional_timesteps
        
        # Phase 3: Champions (elite models)
        if self.strategy_config.get('champions', {}).get('enabled', True):
            champion = self._get_champion_model()
            
            if champion:
                additional_timesteps = self.strategy_config['champions']['additional_timesteps']
                lr_multiplier = self.strategy_config['champions'].get('learning_rate_multiplier', 0.3)
                
                plan['champions'].append({
                    'type': 'champion',
                    'parent_model': champion['model_id'],
                    'parent_path': champion['model_path'],
                    'base_timesteps': champion['total_timesteps'],
                    'additional_timesteps': additional_timesteps,
                    'learning_rate_multiplier': lr_multiplier,
                    'generation': champion.get('generation', 1) + 1,
                    'tickers': champion['tickers'],
                    'params': champion['params']
                })
                
                plan['total_models'] += 1
                plan['total_timesteps'] += additional_timesteps
        
        return plan
    
    def _get_models_for_continuation(self) -> List[Dict[str, Any]]:
        """
        Get top models from last week that qualify for continuation training
        
        Returns:
            List of model info dictionaries
        """
        # Get top models from database
        weeks_back = 2  # Look at last 2 weeks
        top_models = self.db.get_top_strategies(weeks_back=weeks_back, top_n=10)
        
        if top_models.empty:
            return []
        
        # Filter by continuation criteria
        criteria = self.strategy_config.get('continuation_criteria', {})
        min_sharpe = criteria.get('min_sharpe_ratio', 0.5)
        min_return = criteria.get('min_total_return', 0.05)
        max_dd = criteria.get('max_drawdown', 0.30)
        
        qualified_models = []
        
        for _, row in top_models.iterrows():
            # Check if model file exists
            model_path = self._find_model_path(row['model_id'])
            if not model_path:
                continue
            
            # Check performance criteria
            if (row.get('sharpe_ratio', 0) >= min_sharpe and
                row.get('total_return', 0) >= min_return and
                abs(row.get('max_drawdown', 1)) <= max_dd):
                
                qualified_models.append({
                    'model_id': row['model_id'],
                    'model_path': model_path,
                    'total_timesteps': row.get('training_timesteps', 500000),
                    'generation': self._get_generation(row['model_id']),
                    'tickers': json.loads(row['tickers']) if isinstance(row['tickers'], str) else row['tickers'],
                    'params': json.loads(row['parameters_json']) if 'parameters_json' in row else {},
                    'sharpe_ratio': row.get('sharpe_ratio', 0),
                    'total_return': row.get('total_return', 0)
                })
        
        return qualified_models
    
    def _get_champion_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the champion model (best across all time)
        
        Returns:
            Champion model info or None
        """
        # Get best model from database
        best_models = self.db.get_best_strategies_by_metric('sharpe_ratio', top_n=1)
        
        if best_models.empty:
            return None
        
        row = best_models.iloc[0]
        
        # Check champion criteria
        criteria = self.strategy_config.get('champion_criteria', {})
        min_sharpe = criteria.get('min_sharpe_ratio', 1.0)
        min_timesteps = criteria.get('min_total_timesteps', 2000000)
        
        if (row.get('sharpe_ratio', 0) < min_sharpe or
            row.get('training_timesteps', 0) < min_timesteps):
            return None
        
        model_path = self._find_model_path(row['model_id'])
        if not model_path:
            return None
        
        return {
            'model_id': row['model_id'],
            'model_path': model_path,
            'total_timesteps': row.get('training_timesteps', 500000),
            'generation': self._get_generation(row['model_id']),
            'tickers': json.loads(row['tickers']) if isinstance(row['tickers'], str) else row['tickers'],
            'params': json.loads(row['parameters_json']) if 'parameters_json' in row else {},
            'sharpe_ratio': row.get('sharpe_ratio', 0)
        }
    
    def _find_model_path(self, model_id: str) -> Optional[str]:
        """Find the path to a model file"""
        # Check regular models directory
        model_path = os.path.join(self.models_dir, f"{model_id}_ppo.zip")
        if os.path.exists(model_path):
            return model_path
        
        # Check archive directory
        archive_path = os.path.join(self.archive_dir, f"{model_id}_ppo.zip")
        if os.path.exists(archive_path):
            return archive_path
        
        return None
    
    def _get_generation(self, model_id: str) -> int:
        """Get generation number from model ID or metadata"""
        # Check if model_id has generation suffix (e.g., model_001_g2)
        if '_g' in model_id:
            try:
                return int(model_id.split('_g')[-1])
            except:
                pass
        
        # Check metadata file
        metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.yaml")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                return metadata.get('generation', 1)
        
        return 1
    
    def archive_top_models(self, week_number: int = None):
        """
        Archive top performing models from current week
        
        Args:
            week_number: Week number to archive (default: current week)
        """
        if not self.strategy_config.get('retention', {}).get('enabled', True):
            return
        
        if week_number is None:
            week_number = self.db.get_current_week_number()
        
        # Get top models for this week
        top_n = self.strategy_config['retention'].get('keep_top_n_per_week', 5)
        
        # Query database for top models from this week
        query = f'''
            SELECT s.model_id, b.ranking_score, b.rank_position
            FROM strategies s
            JOIN backtest_results b ON s.model_id = b.model_id
            WHERE s.week_number = ?
            ORDER BY b.ranking_score DESC
            LIMIT ?
        '''
        
        import sqlite3
        cursor = self.db.conn.cursor()
        cursor.execute(query, (week_number, top_n))
        top_models = cursor.fetchall()
        
        # Archive each model
        for model_id, score, rank in top_models:
            source_path = os.path.join(self.models_dir, f"{model_id}_ppo.zip")
            dest_path = os.path.join(self.archive_dir, f"{model_id}_ppo.zip")
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                print(f"Archived {model_id} (rank #{rank}, score: {score:.4f})")
            
            # Also copy metadata
            metadata_source = os.path.join(self.models_dir, f"{model_id}_metadata.yaml")
            metadata_dest = os.path.join(self.archive_dir, f"{model_id}_metadata.yaml")
            
            if os.path.exists(metadata_source):
                shutil.copy2(metadata_source, metadata_dest)
    
    def create_continuation_metadata(self, parent_model_id: str, new_model_id: str,
                                    additional_timesteps: int) -> Dict[str, Any]:
        """
        Create metadata for a continuation model
        
        Args:
            parent_model_id: ID of parent model
            new_model_id: ID of new continuation model
            additional_timesteps: Additional training timesteps
        
        Returns:
            Metadata dictionary
        """
        # Load parent metadata
        parent_metadata_path = os.path.join(self.models_dir, f"{parent_model_id}_metadata.yaml")
        if not os.path.exists(parent_metadata_path):
            parent_metadata_path = os.path.join(self.archive_dir, f"{parent_model_id}_metadata.yaml")
        
        parent_metadata = {}
        if os.path.exists(parent_metadata_path):
            with open(parent_metadata_path, 'r') as f:
                parent_metadata = yaml.safe_load(f)
        
        # Create new metadata
        generation = parent_metadata.get('generation', 1) + 1
        total_timesteps = parent_metadata.get('total_timesteps', 500000) + additional_timesteps
        
        metadata = {
            'model_id': new_model_id,
            'parent_model_id': parent_model_id,
            'generation': generation,
            'training_type': 'continuation',
            'base_timesteps': parent_metadata.get('total_timesteps', 500000),
            'additional_timesteps': additional_timesteps,
            'total_timesteps': total_timesteps,
            'training_date': datetime.now().isoformat(),
            'lineage': self._build_lineage(parent_model_id, parent_metadata)
        }
        
        # Copy relevant parameters from parent
        if 'parameters' in parent_metadata:
            metadata['parameters'] = parent_metadata['parameters'].copy()
        
        return metadata
    
    def _build_lineage(self, parent_id: str, parent_metadata: Dict[str, Any]) -> List[str]:
        """Build lineage chain for a model"""
        lineage = parent_metadata.get('lineage', [])
        lineage.append(parent_id)
        return lineage
    
    def cleanup_old_models(self, keep_days: int = None):
        """
        Clean up old models from models directory (keep archived models)
        
        Args:
            keep_days: Days to keep models (default: from config)
        """
        if keep_days is None:
            keep_days = self.autotest_config['storage'].get('keep_history_days', 90)
        
        cutoff_week = self.db.get_current_week_number() - (keep_days // 7)
        
        # Get models older than cutoff
        query = '''
            SELECT model_id FROM strategies
            WHERE week_number < ?
        '''
        
        cursor = self.db.conn.cursor()
        cursor.execute(query, (cutoff_week,))
        old_models = cursor.fetchall()
        
        # Delete old model files (but not archived ones)
        for (model_id,) in old_models:
            model_path = os.path.join(self.models_dir, f"{model_id}_ppo.zip")
            metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.yaml")
            
            # Only delete if not in archive
            archive_path = os.path.join(self.archive_dir, f"{model_id}_ppo.zip")
            if not os.path.exists(archive_path):
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
        
        print(f"Cleaned up {len(old_models)} old models (kept archived models)")
    
    def get_model_lineage(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get full lineage of a model (ancestors)
        
        Args:
            model_id: Model ID to trace
        
        Returns:
            List of ancestor model info
        """
        lineage = []
        current_id = model_id
        
        while current_id:
            metadata_path = os.path.join(self.models_dir, f"{current_id}_metadata.yaml")
            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(self.archive_dir, f"{current_id}_metadata.yaml")
            
            if not os.path.exists(metadata_path):
                break
            
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            lineage.append({
                'model_id': current_id,
                'generation': metadata.get('generation', 1),
                'total_timesteps': metadata.get('total_timesteps', 0),
                'training_date': metadata.get('training_date', 'unknown')
            })
            
            current_id = metadata.get('parent_model_id')
        
        return lineage
