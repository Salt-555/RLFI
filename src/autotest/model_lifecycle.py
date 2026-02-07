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
    """
    
    # Configuration
    MAX_PAPER_TRADING_MODELS = 10
    MIN_PAPER_TRADING_DAYS = 20  # 4 weeks of trading days
    CULL_THRESHOLD_VS_BACKTEST = 0.5  # Cull if return < 50% of backtest return
    
    def __init__(self, db_path: str = 'autotest_strategies.db'):
        self.db_path = db_path
        self.conn = None
        self._initialize_lifecycle_tables()
    
    def _initialize_lifecycle_tables(self):
        """Create lifecycle tracking tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads during writes
        cursor = self.conn.cursor()
        
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
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
        
        self.conn.commit()
    
    def register_model(self, model_id: str, model_path: str, metadata_path: str,
                       tickers: List[str], backtest_return: float = None,
                       backtest_sharpe: float = None) -> bool:
        """
        Register a new model entering the lifecycle.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to saved model
            metadata_path: Path to model metadata
            tickers: List of tickers the model trades
            backtest_return: Expected return from backtesting
            backtest_sharpe: Expected Sharpe ratio from backtesting
        
        Returns:
            True if registered successfully
        """
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO model_lifecycle (
                    model_id, current_state, state_entered_at,
                    backtest_expected_return, backtest_expected_sharpe,
                    model_path, metadata_path, tickers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                ModelState.TRAINING.value,
                datetime.now().isoformat(),
                backtest_return,
                backtest_sharpe,
                model_path,
                metadata_path,
                json.dumps(tickers)
            ))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Model already exists
            return False
    
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
        cursor = self.conn.cursor()
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
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def update_backtest_expectations(self, model_id: str, expected_return: float,
                                     expected_sharpe: float):
        """Update backtest expectations for a model."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE model_lifecycle
            SET backtest_expected_return = ?,
                backtest_expected_sharpe = ?,
                updated_at = ?
            WHERE model_id = ?
        ''', (expected_return, expected_sharpe, datetime.now().isoformat(), model_id))
        self.conn.commit()
    
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
        cursor = self.conn.cursor()
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
            self.conn.commit()
        except Exception as e:
            print(f"Error logging daily performance for {model_id}: {e}")
    
    def get_paper_trading_models(self) -> List[Dict[str, Any]]:
        """Get all models currently in paper trading state."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT model_id, model_path, metadata_path, tickers,
                   paper_trading_started_at, backtest_expected_return,
                   backtest_expected_sharpe
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
                'backtest_expected_sharpe': row[6]
            })
        return models
    
    def get_paper_trading_count(self) -> int:
        """Get count of models currently paper trading."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM model_lifecycle
            WHERE current_state = ?
        ''', (ModelState.PAPER_TRADING.value,))
        return cursor.fetchone()[0]
    
    def get_available_slots(self) -> int:
        """Get number of available paper trading slots."""
        return self.MAX_PAPER_TRADING_MODELS - self.get_paper_trading_count()
    
    def get_validation_count(self) -> int:
        """Get count of models in VALIDATION state (eligible for paper trading)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM model_lifecycle
            WHERE current_state = ?
        ''', (ModelState.VALIDATION.value,))
        return cursor.fetchone()[0]
    
    def get_model_paper_trading_days(self, model_id: str) -> int:
        """Get number of trading days a model has been paper trading."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(DISTINCT trading_date)
            FROM paper_trading_daily_log
            WHERE model_id = ?
        ''', (model_id,))
        return cursor.fetchone()[0]
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get paper trading performance summary for a model."""
        cursor = self.conn.cursor()
        
        # Get daily logs
        cursor.execute('''
            SELECT trading_date, portfolio_value, daily_return, 
                   cumulative_return, benchmark_return
            FROM paper_trading_daily_log
            WHERE model_id = ?
            ORDER BY trading_date
        ''', (model_id,))
        
        rows = cursor.fetchall()
        if not rows:
            return None
        
        daily_returns = [r[2] for r in rows if r[2] is not None]
        cumulative_returns = [r[3] for r in rows if r[3] is not None]
        benchmark_returns = [r[4] for r in rows if r[4] is not None]
        
        # Get backtest expectations
        cursor.execute('''
            SELECT backtest_expected_return, backtest_expected_sharpe,
                   paper_trading_started_at
            FROM model_lifecycle
            WHERE model_id = ?
        ''', (model_id,))
        lifecycle_row = cursor.fetchone()
        
        import numpy as np
        
        return {
            'model_id': model_id,
            'trading_days': len(rows),
            'current_value': rows[-1][1] if rows else None,
            'cumulative_return': cumulative_returns[-1] if cumulative_returns else 0,
            'avg_daily_return': np.mean(daily_returns) if daily_returns else 0,
            'return_std': np.std(daily_returns) if daily_returns else 0,
            'sharpe_ratio': (np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)) if daily_returns else 0,
            'benchmark_cumulative': sum(benchmark_returns) if benchmark_returns else 0,
            'vs_benchmark': (cumulative_returns[-1] if cumulative_returns else 0) - (sum(benchmark_returns) if benchmark_returns else 0),
            'backtest_expected_return': lifecycle_row[0] if lifecycle_row else None,
            'backtest_expected_sharpe': lifecycle_row[1] if lifecycle_row else None,
            'paper_trading_started_at': lifecycle_row[2] if lifecycle_row else None
        }
    
    def evaluate_for_culling(self) -> List[Dict[str, Any]]:
        """
        Evaluate all paper trading models for potential culling.
        
        Returns:
            List of culling decisions with reasons
        """
        decisions = []
        paper_trading_models = self.get_paper_trading_models()
        
        for model in paper_trading_models:
            model_id = model['model_id']
            trading_days = self.get_model_paper_trading_days(model_id)
            
            # Skip if not enough trading days
            if trading_days < self.MIN_PAPER_TRADING_DAYS:
                decisions.append({
                    'model_id': model_id,
                    'decision': 'CONTINUE',
                    'reason': f'Only {trading_days}/{self.MIN_PAPER_TRADING_DAYS} trading days completed',
                    'trading_days': trading_days
                })
                continue
            
            # Get performance
            perf = self.get_model_performance(model_id)
            if not perf:
                continue
            
            actual_return = perf['cumulative_return']
            expected_return = model['backtest_expected_return'] or 0
            
            # Calculate return ratio
            if expected_return > 0:
                return_ratio = actual_return / expected_return
            elif expected_return < 0:
                # If backtest was negative, any positive is good
                return_ratio = 2.0 if actual_return > 0 else 0.5
            else:
                return_ratio = 1.0 if actual_return >= 0 else 0.0
            
            # Decision logic
            if return_ratio < self.CULL_THRESHOLD_VS_BACKTEST:
                decision = 'CULL'
                reason = f'Return ratio {return_ratio:.2f} < threshold {self.CULL_THRESHOLD_VS_BACKTEST}'
            elif actual_return < -0.05:  # More than 5% loss
                decision = 'CULL'
                reason = f'Cumulative loss {actual_return*100:.1f}% exceeds -5% threshold'
            elif perf['sharpe_ratio'] < -0.5:
                decision = 'CULL'
                reason = f'Sharpe ratio {perf["sharpe_ratio"]:.2f} is very negative'
            elif trading_days >= 20 and actual_return > 0.05 and perf['sharpe_ratio'] > 0.5:
                decision = 'PROMOTE'
                reason = f'Strong performance: {actual_return*100:.1f}% return, {perf["sharpe_ratio"]:.2f} Sharpe over {trading_days} days'
            else:
                decision = 'CONTINUE'
                reason = f'Performance acceptable: {actual_return*100:.1f}% return, {perf["sharpe_ratio"]:.2f} Sharpe'
            
            decisions.append({
                'model_id': model_id,
                'decision': decision,
                'reason': reason,
                'trading_days': trading_days,
                'actual_return': actual_return,
                'expected_return': expected_return,
                'return_ratio': return_ratio,
                'sharpe_ratio': perf['sharpe_ratio']
            })
        
        return decisions
    
    def execute_culling_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Execute culling decisions and update model states.
        
        Args:
            decisions: List of culling decisions from evaluate_for_culling()
        
        Returns:
            Summary of actions taken
        """
        cursor = self.conn.cursor()
        summary = {'culled': 0, 'promoted': 0, 'continued': 0}
        
        for decision in decisions:
            model_id = decision['model_id']
            action = decision['decision']
            
            # Log the decision
            cursor.execute('''
                INSERT INTO culling_decisions (
                    model_id, decision_date, decision, reason,
                    paper_trading_days, actual_return, expected_return, return_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                datetime.now().isoformat(),
                action,
                decision['reason'],
                decision.get('trading_days'),
                decision.get('actual_return'),
                decision.get('expected_return'),
                decision.get('return_ratio')
            ))
            
            # Execute state transition
            if action == 'CULL':
                self.transition_state(model_id, ModelState.CULLED, decision['reason'])
                summary['culled'] += 1
            elif action == 'PROMOTE':
                self.transition_state(model_id, ModelState.PROMOTED, decision['reason'])
                summary['promoted'] += 1
            else:
                summary['continued'] += 1
        
        self.conn.commit()
        return summary
    
    def get_models_ready_for_paper_trading(self) -> List[Dict[str, Any]]:
        """
        Get models in VALIDATION state ready to enter paper trading.
        Ranked by composite score: 50% Sharpe + 50% Return (normalized).
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT model_id, model_path, metadata_path, tickers,
                   backtest_expected_return, backtest_expected_sharpe
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
                'backtest_expected_return': row[4] or 0,
                'backtest_expected_sharpe': row[5] or 0
            })
        
        if not models:
            return models
        
        # Calculate composite score for ranking
        # Normalize Sharpe (typically 0-2) and Return (typically -0.2 to 0.5)
        max_sharpe = max(m['backtest_expected_sharpe'] for m in models) or 1
        max_return = max(m['backtest_expected_return'] for m in models) or 1
        min_return = min(m['backtest_expected_return'] for m in models)
        
        for m in models:
            # Normalize to 0-1 range
            norm_sharpe = m['backtest_expected_sharpe'] / max_sharpe if max_sharpe > 0 else 0
            if max_return != min_return:
                norm_return = (m['backtest_expected_return'] - min_return) / (max_return - min_return)
            else:
                norm_return = 1 if m['backtest_expected_return'] > 0 else 0
            
            # Composite score: 50% Sharpe, 50% Return
            m['composite_score'] = 0.5 * norm_sharpe + 0.5 * norm_return
        
        # Sort by composite score descending
        models.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return models
    
    def promote_to_paper_trading(self, max_to_promote: int = None) -> List[str]:
        """
        Promote top validated models to paper trading.
        
        Args:
            max_to_promote: Maximum number to promote (defaults to available slots)
        
        Returns:
            List of model IDs promoted
        """
        available_slots = self.get_available_slots()
        if available_slots <= 0:
            return []
        
        if max_to_promote is not None:
            slots_to_fill = min(available_slots, max_to_promote)
        else:
            slots_to_fill = available_slots
        
        ready_models = self.get_models_ready_for_paper_trading()
        promoted = []
        
        for model in ready_models[:slots_to_fill]:
            self.transition_state(model['model_id'], ModelState.PAPER_TRADING)
            promoted.append(model['model_id'])
        
        return promoted
    
    def get_elite_parents_for_genetic_training(self, max_parents: int = 3) -> List[Dict[str, Any]]:
        """
        Get elite models to use as parents for genetic/continuation training.
        Selects models that performed well in BOTH backtest AND paper trading.
        
        Args:
            max_parents: Maximum number of parent models to return
        
        Returns:
            List of elite parent model info with paths and performance data
        """
        cursor = self.conn.cursor()
        
        # Get PROMOTED models (proven performers) and high-performing PAPER_TRADING models
        cursor.execute('''
            SELECT ml.model_id, ml.model_path, ml.metadata_path, ml.tickers,
                   ml.backtest_expected_return, ml.backtest_expected_sharpe,
                   ml.current_state
            FROM model_lifecycle ml
            WHERE ml.current_state IN (?, ?)
            AND ml.model_path IS NOT NULL
        ''', (ModelState.PROMOTED.value, ModelState.PAPER_TRADING.value))
        
        candidates = []
        for row in cursor.fetchall():
            model_id = row[0]
            
            # Get paper trading performance
            perf = self.get_model_performance(model_id)
            if not perf:
                continue
            
            # Must have minimum trading days for genetic parent eligibility
            min_days = 15  # 3 weeks minimum
            if perf['trading_days'] < min_days:
                continue
            
            backtest_return = row[4] or 0
            backtest_sharpe = row[5] or 0
            paper_return = perf['cumulative_return']
            paper_sharpe = perf['sharpe_ratio']
            
            # Calculate genetic fitness score
            # Weight: 40% backtest performance, 60% paper trading (real-world matters more)
            # Normalize each component to 0-1 range with proper capping
            capped_bt_sharpe = np.clip(backtest_sharpe / 2.0, 0, 1)  # Sharpe 0-2 -> 0-1
            capped_bt_return = np.clip(backtest_return * 5, 0, 1)     # Return 0-20% -> 0-1
            capped_pt_sharpe = np.clip(paper_sharpe / 2.0, 0, 1)
            capped_pt_return = np.clip(paper_return * 5, 0, 1)
            
            backtest_score = 0.5 * capped_bt_sharpe + 0.5 * capped_bt_return
            paper_score = 0.5 * capped_pt_sharpe + 0.5 * capped_pt_return
            
            fitness_score = 0.4 * backtest_score + 0.6 * paper_score
            
            # Only consider models with positive paper trading performance
            if paper_return > 0 and paper_sharpe > 0:
                candidates.append({
                    'model_id': model_id,
                    'model_path': row[1],
                    'metadata_path': row[2],
                    'tickers': json.loads(row[3]) if row[3] else [],
                    'backtest_return': backtest_return,
                    'backtest_sharpe': backtest_sharpe,
                    'paper_return': paper_return,
                    'paper_sharpe': paper_sharpe,
                    'trading_days': perf['trading_days'],
                    'fitness_score': fitness_score,
                    'state': row[6]
                })
        
        # Sort by fitness score descending
        candidates.sort(key=lambda x: x['fitness_score'], reverse=True)
        
        return candidates[:max_parents]
    
    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get summary of all models in each lifecycle state."""
        cursor = self.conn.cursor()
        
        summary = {}
        for state in ModelState:
            cursor.execute('''
                SELECT COUNT(*) FROM model_lifecycle
                WHERE current_state = ?
            ''', (state.value,))
            summary[state.value] = cursor.fetchone()[0]
        
        # Get paper trading performance summary
        paper_trading_models = self.get_paper_trading_models()
        if paper_trading_models:
            performances = []
            for model in paper_trading_models:
                perf = self.get_model_performance(model['model_id'])
                if perf:
                    performances.append(perf)
            
            if performances:
                import numpy as np
                summary['paper_trading_avg_return'] = np.mean([p['cumulative_return'] for p in performances])
                summary['paper_trading_avg_sharpe'] = np.mean([p['sharpe_ratio'] for p in performances])
        
        return summary
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
