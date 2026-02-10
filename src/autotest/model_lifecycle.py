"""
Model Lifecycle Manager - Manages the continuous lifecycle of trading models.

Lifecycle States:
1. TRAINING - Model is being trained
2. VALIDATION - Model passed training, undergoing backtest validation
3. PAPER_TRADING - Model is actively paper trading (minimum 2 weeks)
4. PROMOTED - Model has proven itself, eligible for live trading
5. CULLED - Model failed validation or paper trading, archived

Key Rules:
- Maximum 5 models in PAPER_TRADING at once
- Minimum 2 weeks (10 trading days) before culling decision
- Cull if significantly worse than backtest expectations
- Weekly training of new models
- Daily performance tracking
- Weekly culling decisions
"""
import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import yaml


class ModelState(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    PAPER_TRADING = "paper_trading"
    PROMOTED = "promoted"
    CULLED = "culled"


class ModelLifecycleManager:
    """
    Manages the continuous lifecycle of trading models in the colosseum.
    Uses connection-per-operation pattern to avoid database lock conflicts.
    """
    
    # Configuration
    MAX_PAPER_TRADING_MODELS = 10
    MIN_PAPER_TRADING_DAYS = 20  # 4 weeks of trading days
    CULL_THRESHOLD_VS_BACKTEST = 0.5  # Cull if return < 50% of backtest return
    
    def __init__(self, db_path: str = 'autotest_strategies.db'):
        self.db_path = db_path
        self._initialize_lifecycle_tables()
    
    def _get_connection(self):
        """Get a new database connection with proper settings for concurrency"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)  # 30 second timeout
        conn.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads during writes
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety and performance
        return conn
    
    def _execute_with_connection(self, operation):
        """Execute an operation with automatic connection management"""
        conn = None
        try:
            conn = self._get_connection()
            result = operation(conn)
            conn.commit()
            return result
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _initialize_lifecycle_tables(self):
        """Create lifecycle tracking tables if they don't exist."""
        def _init_tables(conn):
            cursor = conn.cursor()
            
            # Model lifecycle state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_lifecycle (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT UNIQUE NOT NULL,
                    current_state TEXT NOT NULL,
                    state_entered_at TIMESTAMP NOT NULL,
                    training_completed_at TIMESTAMP,
                    validation_completed_at TIMESTAMP,
                    paper_trading_started_at TIMESTAMP,
                    paper_trading_ended_at TIMESTAMP,
                    final_outcome TEXT,
                    backtest_expected_return REAL,
                    backtest_expected_sharpe REAL,
                    model_path TEXT,
                    metadata_path TEXT,
                    tickers TEXT,
                    algorithm TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add algorithm column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE model_lifecycle ADD COLUMN algorithm TEXT')
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Daily paper trading performance log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trading_daily_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    trading_date DATE NOT NULL,
                    portfolio_value REAL,
                    daily_return REAL,
                    cumulative_return REAL,
                    trades_executed INTEGER,
                    positions TEXT,
                    benchmark_return REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_id, trading_date),
                    FOREIGN KEY (model_id) REFERENCES model_lifecycle (model_id)
                )
            ''')
            
            # Culling decisions log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS culling_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    decision_date TIMESTAMP NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    paper_trading_days INTEGER,
                    actual_return REAL,
                    expected_return REAL,
                    return_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES model_lifecycle (model_id)
                )
            ''')
            
            # Create indexes for performance optimization
            # These are safe to run on existing databases (IF NOT EXISTS)
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_state ON model_lifecycle(current_state)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_lifecycle_model_id ON model_lifecycle(model_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_daily_log_model_date ON paper_trading_daily_log(model_id, trading_date)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_culling_model ON culling_decisions(model_id)
            ''')
        
        self._execute_with_connection(_init_tables)
    
    def register_model(self, model_id: str, model_path: str, metadata_path: str,
                       tickers: List[str], backtest_return: float = None,
                       backtest_sharpe: float = None, algorithm: str = None) -> bool:
        """
        Register a new model entering the lifecycle.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to saved model
            metadata_path: Path to model metadata
            tickers: List of tickers the model trades
            backtest_return: Expected return from backtesting
            backtest_sharpe: Expected Sharpe ratio from backtesting
            algorithm: RL algorithm used (ppo, sac, a2c, ddpg)
        
        Returns:
            True if registered successfully
        """
        def _register(conn):
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO model_lifecycle (
                        model_id, current_state, state_entered_at,
                        backtest_expected_return, backtest_expected_sharpe,
                        model_path, metadata_path, tickers, algorithm
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id,
                    ModelState.TRAINING.value,
                    datetime.now().isoformat(),
                    backtest_return,
                    backtest_sharpe,
                    model_path,
                    metadata_path,
                    json.dumps(tickers),
                    algorithm
                ))
                return True
            except sqlite3.IntegrityError:
                # Model already exists
                return False
        
        return self._execute_with_connection(_register)
    
    def transition_state(self, model_id: str, new_state: ModelState, 
                        reason: str = None) -> bool:
        """
        Transition a model to a new lifecycle state.
        
        Args:
            model_id: Model to transition
            new_state: New state to enter
            reason: Optional reason for transition
        
        Returns:
            True if transition successful
        """
        def _transition(conn):
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            # Update state
            cursor.execute('''
                UPDATE model_lifecycle
                SET current_state = ?,
                    state_entered_at = ?,
                    updated_at = ?
                WHERE model_id = ?
            ''', (new_state.value, now, now, model_id))
            
            # Update specific timestamp fields
            if new_state == ModelState.VALIDATION:
                cursor.execute('''
                    UPDATE model_lifecycle
                    SET training_completed_at = ?
                    WHERE model_id = ?
                ''', (now, model_id))
            elif new_state == ModelState.PAPER_TRADING:
                cursor.execute('''
                    UPDATE model_lifecycle
                    SET validation_completed_at = ?,
                        paper_trading_started_at = ?
                    WHERE model_id = ?
                ''', (now, now, model_id))
            elif new_state in [ModelState.PROMOTED, ModelState.CULLED]:
                cursor.execute('''
                    UPDATE model_lifecycle
                    SET paper_trading_ended_at = ?,
                        final_outcome = ?
                    WHERE model_id = ?
                ''', (now, new_state.value, model_id))
            
            return cursor.rowcount > 0
        
        return self._execute_with_connection(_transition)
    
    def update_backtest_expectations(self, model_id: str, expected_return: float,
                                     expected_sharpe: float):
        """Update backtest expectations for a model."""
        def _update(conn):
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE model_lifecycle
                SET backtest_expected_return = ?,
                    backtest_expected_sharpe = ?,
                    updated_at = ?
                WHERE model_id = ?
            ''', (expected_return, expected_sharpe, datetime.now().isoformat(), model_id))
        
        self._execute_with_connection(_update)
    
    def log_daily_performance(self, model_id: str, portfolio_value: float,
                              daily_return: float, cumulative_return: float,
                              trades_executed: int, positions: Dict[str, int],
                              benchmark_return: float = None):
        """
        Log daily paper trading performance for a model.
        
        Args:
            model_id: Model identifier
            portfolio_value: Current portfolio value
            daily_return: Return for the day
            cumulative_return: Cumulative return since paper trading started
            trades_executed: Number of trades executed today
            positions: Current positions {ticker: shares}
            benchmark_return: Benchmark (SPY) return for comparison
        """
        def _log(conn):
            cursor = conn.cursor()
            today = datetime.now().date().isoformat()
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO paper_trading_daily_log (
                        model_id, trading_date, portfolio_value, daily_return,
                        cumulative_return, trades_executed, positions, benchmark_return
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id, today, portfolio_value, daily_return,
                    cumulative_return, trades_executed, json.dumps(positions),
                    benchmark_return
                ))
            except Exception as e:
                print(f"Error logging daily performance for {model_id}: {e}")
        
        self._execute_with_connection(_log)
    
    def get_paper_trading_models(self) -> List[Dict[str, Any]]:
        """Get all models currently in paper trading state."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_id, model_path, metadata_path, tickers,
                       paper_trading_started_at, backtest_expected_return,
                       backtest_expected_sharpe, algorithm
                FROM model_lifecycle
                WHERE current_state = ?
            ''', (ModelState.PAPER_TRADING.value,))
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    'model_id': row[0],
                    'model_path': row[1],
                    'metadata_path': row[2],
                    'tickers': json.loads(row[3]) if row[3] else [],
                    'paper_trading_started_at': row[4],
                    'backtest_expected_return': row[5],
                    'backtest_expected_sharpe': row[6],
                    'algorithm': row[7],
                })
            return models
        
        return self._execute_with_connection(_get)
    
    def get_validation_models(self) -> List[Dict[str, Any]]:
        """Get all models currently in validation state."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_id, model_path, metadata_path, tickers,
                       backtest_expected_return, backtest_expected_sharpe, algorithm
                FROM model_lifecycle
                WHERE current_state = ?
            ''', (ModelState.VALIDATION.value,))
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    'model_id': row[0],
                    'model_path': row[1],
                    'metadata_path': row[2],
                    'tickers': json.loads(row[3]) if row[3] else [],
                    'backtest_expected_return': row[4],
                    'backtest_expected_sharpe': row[5],
                    'algorithm': row[6],
                })
            return models
        
        return self._execute_with_connection(_get)
    
    def get_eligible_models_for_paper_trading(self, max_models: int = 5) -> List[str]:
        """Get models in VALIDATION state eligible for paper trading."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_id, backtest_expected_sharpe
                FROM model_lifecycle
                WHERE current_state = ?
                ORDER BY backtest_expected_sharpe DESC
                LIMIT ?
            ''', (ModelState.VALIDATION.value, max_models))
            
            return [row[0] for row in cursor.fetchall()]
        
        return self._execute_with_connection(_get)
    
    def promote_to_paper_trading(self, max_to_promote: int = None) -> List[str]:
        """Promote top VALIDATION models to PAPER_TRADING."""
        if max_to_promote is None:
            max_to_promote = self.MAX_PAPER_TRADING_MODELS
        
        eligible = self.get_eligible_models_for_paper_trading(max_to_promote)
        promoted = []
        
        for model_id in eligible:
            if self.transition_state(model_id, ModelState.PAPER_TRADING):
                promoted.append(model_id)
        
        return promoted
    
    def get_model_performance(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get paper trading performance for a model."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT trading_date, portfolio_value, daily_return, cumulative_return,
                       trades_executed, benchmark_return
                FROM paper_trading_daily_log
                WHERE model_id = ?
                ORDER BY trading_date ASC
            ''', (model_id,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            return {
                'dates': [row[0] for row in rows],
                'portfolio_values': [row[1] for row in rows],
                'daily_returns': [row[2] for row in rows],
                'cumulative_returns': [row[3] for row in rows],
                'total_trades': sum(row[4] for row in rows if row[4]),
                'benchmark_returns': [row[5] for row in rows],
            }
        
        return self._execute_with_connection(_get)
    
    def get_model_paper_trading_days(self, model_id: str) -> int:
        """Get number of days a model has been paper trading."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(DISTINCT trading_date)
                FROM paper_trading_daily_log
                WHERE model_id = ?
            ''', (model_id,))
            
            result = cursor.fetchone()
            return result[0] if result else 0
        
        return self._execute_with_connection(_get)
    
    def get_validation_count(self) -> int:
        """Get count of models in VALIDATION state."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM model_lifecycle
                WHERE current_state IN (?, ?)
            ''', (ModelState.TRAINING.value, ModelState.VALIDATION.value))
            
            result = cursor.fetchone()
            return result[0] if result else 0
        
        return self._execute_with_connection(_get)
    
    def get_available_slots(self) -> int:
        """Get number of available paper trading slots."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM model_lifecycle
                WHERE current_state = ?
            ''', (ModelState.PAPER_TRADING.value,))
            
            result = cursor.fetchone()
            current_count = result[0] if result else 0
            return max(0, self.MAX_PAPER_TRADING_MODELS - current_count)
        
        return self._execute_with_connection(_get)
    
    def evaluate_for_culling(self) -> List[Dict[str, Any]]:
        """Evaluate paper trading models for culling decisions."""
        paper_models = self.get_paper_trading_models()
        decisions = []
        
        for model in paper_models:
            model_id = model['model_id']
            days_trading = self.get_model_paper_trading_days(model_id)
            
            if days_trading < self.MIN_PAPER_TRADING_DAYS:
                continue  # Not enough data yet
            
            performance = self.get_model_performance(model_id)
            if not performance:
                continue
            
            actual_return = performance['cumulative_returns'][-1] if performance['cumulative_returns'] else 0
            expected_return = model.get('backtest_expected_return', 0)
            
            # Calculate ratio of actual to expected
            if expected_return > 0:
                return_ratio = actual_return / expected_return
            else:
                return_ratio = 1.0 if actual_return >= 0 else 0.0
            
            # Make culling decision
            if return_ratio < self.CULL_THRESHOLD_VS_BACKTEST or actual_return < 0:
                decision = {
                    'model_id': model_id,
                    'decision': 'cull',
                    'reason': f'Underperforming: {return_ratio:.1%} of expected return',
                    'paper_trading_days': days_trading,
                    'actual_return': actual_return,
                    'expected_return': expected_return,
                    'return_ratio': return_ratio
                }
            else:
                decision = {
                    'model_id': model_id,
                    'decision': 'keep',
                    'reason': 'Meeting expectations',
                    'paper_trading_days': days_trading,
                    'actual_return': actual_return,
                    'expected_return': expected_return,
                    'return_ratio': return_ratio
                }
            
            decisions.append(decision)
        
        return decisions
    
    def execute_culling_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Execute culling decisions and log them."""
        culled = []
        kept = []
        
        for decision in decisions:
            model_id = decision['model_id']
            
            if decision['decision'] == 'cull':
                self.transition_state(model_id, ModelState.CULLED, decision['reason'])
                self._log_culling_decision(decision)
                culled.append(model_id)
            else:
                kept.append(model_id)
        
        return {'culled': culled, 'kept': kept}
    
    def _log_culling_decision(self, decision: Dict[str, Any]):
        """Log a culling decision to the database."""
        def _log(conn):
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO culling_decisions (
                    model_id, decision_date, decision, reason,
                    paper_trading_days, actual_return, expected_return, return_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision['model_id'],
                datetime.now().isoformat(),
                decision['decision'],
                decision['reason'],
                decision['paper_trading_days'],
                decision['actual_return'],
                decision['expected_return'],
                decision['return_ratio']
            ))
        
        self._execute_with_connection(_log)
    
    def get_lifecycle_summary(self) -> Dict[str, int]:
        """Get summary of models in each lifecycle state."""
        def _get(conn):
            cursor = conn.cursor()
            
            summary = {}
            for state in ModelState:
                cursor.execute('''
                    SELECT COUNT(*) FROM model_lifecycle
                    WHERE current_state = ?
                ''', (state.value,))
                result = cursor.fetchone()
                summary[state.value] = result[0] if result else 0
            
            return summary
        
        return self._execute_with_connection(_get)
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all models with their current state."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_id, current_state, state_entered_at, algorithm,
                       backtest_expected_return, backtest_expected_sharpe
                FROM model_lifecycle
                ORDER BY created_at DESC
            ''')
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    'model_id': row[0],
                    'state': row[1],
                    'state_entered_at': row[2],
                    'algorithm': row[3],
                    'backtest_expected_return': row[4],
                    'backtest_expected_sharpe': row[5],
                })
            return models
        
        return self._execute_with_connection(_get)
    
    def get_elite_parents_for_genetic_training(self, min_sharpe: float = 0.5,
                                                min_days: int = 15,
                                                max_parents: int = 3) -> List[Dict[str, Any]]:
        """Get elite performing models for genetic offspring training."""
        def _get(conn):
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ml.model_id, ml.algorithm, ml.model_path, ml.metadata_path,
                       ml.tickers, ml.backtest_expected_sharpe
                FROM model_lifecycle ml
                JOIN (
                    SELECT model_id, COUNT(*) as days_trading,
                           AVG(daily_return) as avg_daily_return
                    FROM paper_trading_daily_log
                    GROUP BY model_id
                    HAVING days_trading >= ?
                ) ptd ON ml.model_id = ptd.model_id
                WHERE ml.current_state IN (?, ?)
                  AND ml.backtest_expected_sharpe >= ?
                ORDER BY ml.backtest_expected_sharpe DESC
                LIMIT ?
            ''', (min_days, ModelState.PAPER_TRADING.value, ModelState.PROMOTED.value,
                  min_sharpe, max_parents))
            
            parents = []
            for row in cursor.fetchall():
                parents.append({
                    'model_id': row[0],
                    'algorithm': row[1],
                    'model_path': row[2],
                    'metadata_path': row[3],
                    'tickers': json.loads(row[4]) if row[4] else [],
                    'backtest_sharpe': row[5],
                })
            return parents
        
        return self._execute_with_connection(_get)
    
    def reset_stuck_models(self, max_hours: int = 6):
        """Reset models stuck in TRAINING state for too long."""
        def _reset(conn):
            cursor = conn.cursor()
            cutoff_time = (datetime.now() - timedelta(hours=max_hours)).isoformat()
            
            cursor.execute('''
                UPDATE model_lifecycle
                SET current_state = ?,
                    state_entered_at = ?,
                    updated_at = ?
                WHERE current_state = ?
                  AND state_entered_at < ?
            ''', (ModelState.CULLED.value, datetime.now().isoformat(),
                  datetime.now().isoformat(), ModelState.TRAINING.value, cutoff_time))
            
            return cursor.rowcount
        
        return self._execute_with_connection(_reset)
    
    def get_old_culled_models(self, days: int = 30) -> List[Dict[str, str]]:
        """Get culled models older than specified days for cleanup."""
        def _get(conn):
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute('''
                SELECT model_id, model_path FROM model_lifecycle
                WHERE current_state = 'culled' AND paper_trading_ended_at < ?
            ''', (cutoff,))
            
            return [{'model_id': row[0], 'model_path': row[1]} for row in cursor.fetchall()]
        
        return self._execute_with_connection(_get)
