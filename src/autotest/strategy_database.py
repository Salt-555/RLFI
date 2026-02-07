import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import os


class StrategyDatabase:
    """
    SQLite database for tracking and comparing AI trading strategies across weeks
    """
    
    def __init__(self, db_path: str = 'autotest_strategies.db'):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads during writes
        cursor = self.conn.cursor()
        
        # Strategies table - stores each trained model
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT UNIQUE NOT NULL,
                week_number INTEGER NOT NULL,
                run_date TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                tickers TEXT NOT NULL,
                learning_rate REAL,
                n_steps INTEGER,
                batch_size INTEGER,
                gamma REAL,
                clip_range REAL,
                ent_coef REAL,
                weight_decay REAL,
                reward_scaling REAL,
                initial_capital REAL,
                training_timesteps INTEGER,
                training_time_seconds REAL,
                parameters_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                week_number INTEGER NOT NULL,
                sharpe_ratio REAL,
                total_return REAL,
                max_drawdown REAL,
                win_rate REAL,
                sortino_ratio REAL,
                calmar_ratio REAL,
                volatility REAL,
                final_value REAL,
                ranking_score REAL,
                rank_position INTEGER,
                test_period_days INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES strategies (model_id)
            )
        ''')
        
        # Paper trading results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                week_number INTEGER NOT NULL,
                initial_capital REAL,
                final_value REAL,
                total_return REAL,
                trades_executed INTEGER,
                transaction_costs REAL,
                trading_duration_hours REAL,
                was_top_model BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES strategies (model_id)
            )
        ''')
        
        # Weekly summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_number INTEGER UNIQUE NOT NULL,
                run_date TEXT NOT NULL,
                models_trained INTEGER,
                models_backtested INTEGER,
                models_paper_traded INTEGER,
                best_model_id TEXT,
                best_sharpe_ratio REAL,
                best_total_return REAL,
                avg_sharpe_ratio REAL,
                avg_total_return REAL,
                execution_time_hours REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Strategy patterns table - for identifying what works
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_value TEXT NOT NULL,
                avg_sharpe_ratio REAL,
                avg_total_return REAL,
                occurrence_count INTEGER,
                success_rate REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def get_current_week_number(self) -> int:
        """Get week number since epoch for tracking"""
        return datetime.now().isocalendar()[1] + (datetime.now().year * 52)
    
    def save_strategy(self, strategy_data: Dict[str, Any]) -> int:
        """Save a trained strategy to database"""
        cursor = self.conn.cursor()
        
        week_number = self.get_current_week_number()
        params = strategy_data['params']
        
        cursor.execute('''
            INSERT OR REPLACE INTO strategies (
                model_id, week_number, run_date, algorithm, tickers,
                learning_rate, n_steps, batch_size, gamma, clip_range,
                ent_coef, weight_decay, reward_scaling, initial_capital, training_timesteps,
                training_time_seconds, parameters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_data['model_id'],
            week_number,
            datetime.now().isoformat(),
            params.get('algorithm', 'ppo'),
            json.dumps(params.get('tickers', [])),
            params.get('learning_rate'),
            params.get('n_steps'),
            params.get('batch_size'),
            params.get('gamma'),
            params.get('clip_range'),
            params.get('ent_coef'),
            params.get('weight_decay'),
            params.get('reward_scaling'),
            params.get('initial_capital'),
            params.get('timesteps'),
            strategy_data.get('training_time'),
            json.dumps(params)
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def save_backtest_result(self, model_id: str, metrics: Dict[str, Any], 
                            ranking_score: float, rank_position: int):
        """Save backtest results for a strategy"""
        cursor = self.conn.cursor()
        week_number = self.get_current_week_number()
        
        cursor.execute('''
            INSERT INTO backtest_results (
                model_id, week_number, sharpe_ratio, total_return,
                max_drawdown, win_rate, sortino_ratio, calmar_ratio,
                volatility, final_value, ranking_score, rank_position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id, week_number,
            metrics.get('sharpe_ratio'),
            metrics.get('total_return'),
            metrics.get('max_drawdown'),
            metrics.get('win_rate'),
            metrics.get('sortino_ratio'),
            metrics.get('calmar_ratio'),
            metrics.get('volatility'),
            metrics.get('final_value'),
            ranking_score,
            rank_position
        ))
        
        self.conn.commit()
    
    def save_paper_trade_result(self, model_id: str, performance: Dict[str, Any], 
                                was_top_model: bool):
        """Save paper trading results"""
        cursor = self.conn.cursor()
        week_number = self.get_current_week_number()
        
        cursor.execute('''
            INSERT INTO paper_trade_results (
                model_id, week_number, initial_capital, final_value,
                total_return, trades_executed, transaction_costs,
                was_top_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id, week_number,
            performance.get('initial_capital'),
            performance.get('final_value'),
            performance.get('total_return'),
            performance.get('trades_executed'),
            performance.get('transaction_costs'),
            was_top_model
        ))
        
        self.conn.commit()
    
    def save_weekly_summary(self, summary: Dict[str, Any]):
        """Save weekly execution summary"""
        cursor = self.conn.cursor()
        week_number = self.get_current_week_number()
        
        cursor.execute('''
            INSERT OR REPLACE INTO weekly_summaries (
                week_number, run_date, models_trained, models_backtested,
                models_paper_traded, best_model_id, best_sharpe_ratio,
                best_total_return, avg_sharpe_ratio, avg_total_return,
                execution_time_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            week_number,
            datetime.now().isoformat(),
            summary.get('models_trained'),
            summary.get('models_backtested'),
            summary.get('models_paper_traded'),
            summary.get('best_model_id'),
            summary.get('best_sharpe_ratio'),
            summary.get('best_total_return'),
            summary.get('avg_sharpe_ratio'),
            summary.get('avg_total_return'),
            summary.get('execution_time_hours')
        ))
        
        self.conn.commit()
    
    def get_historical_strategies(self, weeks_back: int = 12) -> pd.DataFrame:
        """Get strategies from past N weeks"""
        query = '''
            SELECT * FROM strategies
            WHERE week_number >= ?
            ORDER BY week_number DESC, created_at DESC
        '''
        current_week = self.get_current_week_number()
        return pd.read_sql_query(query, self.conn, params=(current_week - weeks_back,))
    
    def get_top_strategies(self, weeks_back: int = 12, top_n: int = 10) -> pd.DataFrame:
        """Get top performing strategies from past N weeks"""
        query = '''
            SELECT s.*, b.sharpe_ratio, b.total_return, b.ranking_score, b.rank_position
            FROM strategies s
            JOIN backtest_results b ON s.model_id = b.model_id
            WHERE s.week_number >= ?
            ORDER BY b.ranking_score DESC
            LIMIT ?
        '''
        current_week = self.get_current_week_number()
        return pd.read_sql_query(query, self.conn, params=(current_week - weeks_back, top_n))
    
    def get_weekly_summaries(self, weeks_back: int = 12) -> pd.DataFrame:
        """Get weekly summaries for comparison"""
        query = '''
            SELECT * FROM weekly_summaries
            WHERE week_number >= ?
            ORDER BY week_number DESC
        '''
        current_week = self.get_current_week_number()
        return pd.read_sql_query(query, self.conn, params=(current_week - weeks_back,))
    
    def analyze_parameter_patterns(self) -> pd.DataFrame:
        """Analyze which parameter combinations perform best"""
        cursor = self.conn.cursor()
        
        # Analyze learning rate patterns
        cursor.execute('''
            SELECT 
                'learning_rate' as pattern_type,
                CAST(s.learning_rate AS TEXT) as pattern_value,
                AVG(b.sharpe_ratio) as avg_sharpe_ratio,
                AVG(b.total_return) as avg_total_return,
                COUNT(*) as occurrence_count,
                SUM(CASE WHEN b.rank_position <= 5 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
            FROM strategies s
            JOIN backtest_results b ON s.model_id = b.model_id
            GROUP BY s.learning_rate
            HAVING COUNT(*) >= 3
        ''')
        lr_patterns = cursor.fetchall()
        
        # Analyze ticker combination patterns
        cursor.execute('''
            SELECT 
                'tickers' as pattern_type,
                s.tickers as pattern_value,
                AVG(b.sharpe_ratio) as avg_sharpe_ratio,
                AVG(b.total_return) as avg_total_return,
                COUNT(*) as occurrence_count,
                SUM(CASE WHEN b.rank_position <= 5 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
            FROM strategies s
            JOIN backtest_results b ON s.model_id = b.model_id
            GROUP BY s.tickers
            HAVING COUNT(*) >= 2
        ''')
        ticker_patterns = cursor.fetchall()
        
        # Combine all patterns
        all_patterns = lr_patterns + ticker_patterns
        
        df = pd.DataFrame(all_patterns, columns=[
            'pattern_type', 'pattern_value', 'avg_sharpe_ratio', 
            'avg_total_return', 'occurrence_count', 'success_rate'
        ])
        
        return df.sort_values('avg_sharpe_ratio', ascending=False)
    
    def get_best_strategies_by_metric(self, metric: str = 'sharpe_ratio', 
                                     top_n: int = 5) -> pd.DataFrame:
        """Get best strategies by specific metric"""
        query = f'''
            SELECT s.*, b.{metric}, b.total_return, b.sharpe_ratio, b.max_drawdown
            FROM strategies s
            JOIN backtest_results b ON s.model_id = b.model_id
            ORDER BY b.{metric} DESC
            LIMIT ?
        '''
        return pd.read_sql_query(query, self.conn, params=(top_n,))
    
    def compare_weeks(self, week1: int, week2: int) -> Dict[str, Any]:
        """Compare performance between two weeks"""
        query = '''
            SELECT * FROM weekly_summaries
            WHERE week_number IN (?, ?)
            ORDER BY week_number
        '''
        df = pd.read_sql_query(query, self.conn, params=(week1, week2))
        
        if len(df) < 2:
            return {'error': 'Not enough data for comparison'}
        
        return {
            'week1': df.iloc[0].to_dict(),
            'week2': df.iloc[1].to_dict(),
            'sharpe_improvement': df.iloc[1]['best_sharpe_ratio'] - df.iloc[0]['best_sharpe_ratio'],
            'return_improvement': df.iloc[1]['best_total_return'] - df.iloc[0]['best_total_return']
        }
    
    def export_to_csv(self, output_dir: str = 'autotest_results'):
        """Export all data to CSV files for analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export strategies
        df_strategies = pd.read_sql_query('SELECT * FROM strategies', self.conn)
        df_strategies.to_csv(f'{output_dir}/all_strategies.csv', index=False)
        
        # Export backtest results
        df_backtest = pd.read_sql_query('SELECT * FROM backtest_results', self.conn)
        df_backtest.to_csv(f'{output_dir}/all_backtest_results.csv', index=False)
        
        # Export paper trade results
        df_paper = pd.read_sql_query('SELECT * FROM paper_trade_results', self.conn)
        df_paper.to_csv(f'{output_dir}/all_paper_trade_results.csv', index=False)
        
        # Export weekly summaries
        df_weekly = pd.read_sql_query('SELECT * FROM weekly_summaries', self.conn)
        df_weekly.to_csv(f'{output_dir}/weekly_summaries.csv', index=False)
        
        print(f"Exported all data to {output_dir}/")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
