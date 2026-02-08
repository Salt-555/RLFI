"""
RLFI Colosseum TUI
Terminal-based status monitoring for the AI Trading Colosseum
Built with Textual framework
"""
import os
import sys
import subprocess
import sqlite3
import json
import yaml
from datetime import datetime, timedelta
from collections import defaultdict

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, DataTable, 
    TabbedContent, TabPane, Label, Rule, Tree, Collapsible
)
from textual.reactive import reactive
from textual.timer import Timer

# ============================================================================
# DATABASE FUNCTIONS (shared with app.py)
# ============================================================================

def get_daemon_status():
    """Check if RLFI daemon is running."""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'rlfi'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == 'active'
    except:
        return False


def get_lifecycle_data():
    """Load data from the lifecycle database."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_lifecycle'")
        if not cursor.fetchone():
            conn.close()
            return None

        states = {}
        for state in ['training', 'validation', 'paper_trading', 'promoted', 'culled']:
            cursor.execute('SELECT COUNT(*) FROM model_lifecycle WHERE current_state = ?', (state,))
            result = cursor.fetchone()
            states[state] = result[0] if result else 0

        cursor.execute('''
            SELECT model_id, tickers, paper_trading_started_at,
                   backtest_expected_return, backtest_expected_sharpe
            FROM model_lifecycle
            WHERE current_state = 'paper_trading'
            ORDER BY paper_trading_started_at DESC
        ''')
        paper_trading_models = cursor.fetchall()

        paper_models_data = []
        for model in paper_trading_models:
            model_id = model[0]
            cursor.execute('SELECT COUNT(*) FROM paper_trading_daily_log WHERE model_id = ?', (model_id,))
            trading_days = cursor.fetchone()[0]
            paper_models_data.append({
                'model_id': model_id,
                'tickers': json.loads(model[1]) if model[1] else [],
                'expected_return': model[3],
                'trading_days': trading_days,
            })

        cursor.execute('''
            SELECT s.model_id, s.algorithm, s.tickers, s.run_date
            FROM strategies s
            INNER JOIN model_lifecycle ml ON s.model_id = ml.model_id
            ORDER BY s.created_at DESC
            LIMIT 10
        ''')
        recent_models = cursor.fetchall()

        cursor.execute('''
            SELECT model_id, tickers
            FROM model_lifecycle
            WHERE current_state = 'promoted'
            LIMIT 5
        ''')
        promoted_models = cursor.fetchall()

        conn.close()

        return {
            'states': states,
            'paper_trading_models': paper_models_data,
            'promoted_models': promoted_models,
            'recent_models': recent_models
        }
    except Exception as e:
        return {'error': str(e)}


def get_all_models():
    """Get all models from the database with their current state."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                ml.model_id,
                ml.current_state,
                ml.state_entered_at,
                ml.training_completed_at,
                ml.paper_trading_started_at,
                ml.backtest_expected_return,
                ml.backtest_expected_sharpe,
                ml.tickers,
                ml.created_at,
                s.algorithm,
                s.training_timesteps,
                s.training_time_seconds
            FROM model_lifecycle ml
            LEFT JOIN strategies s ON ml.model_id = s.model_id
            ORDER BY ml.created_at DESC
        ''')
        models = cursor.fetchall()
        
        # Get paper trading performance
        perf_map = {}
        cursor.execute('''
            SELECT 
                model_id,
                COUNT(*) as trading_days,
                MAX(cumulative_return) as cumulative_return,
                AVG(daily_return) as avg_daily_return
            FROM paper_trading_daily_log
            GROUP BY model_id
        ''')
        for row in cursor.fetchall():
            perf_map[row[0]] = {
                'trading_days': row[1],
                'cumulative_return': row[2],
                'avg_daily_return': row[3]
            }
        
        conn.close()
        
        result = []
        for m in models:
            model_id = m[0]
            perf = perf_map.get(model_id, {})
            result.append({
                'model_id': model_id,
                'current_state': m[1],
                'backtest_expected_return': m[5],
                'backtest_expected_sharpe': m[6],
                'tickers': json.loads(m[7]) if m[7] else [],
                'algorithm': m[9],
                'training_timesteps': m[10],
                'trading_days': perf.get('trading_days', 0),
                'cumulative_return': perf.get('cumulative_return'),
            })
        
        return result
        
    except Exception as e:
        return []


def get_trading_data():
    """Get all paper trading data."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return [], []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                ptl.model_id,
                ptl.trading_date,
                ptl.portfolio_value,
                ptl.daily_return,
                ptl.cumulative_return,
                ptl.trades_executed,
                ml.current_state,
                s.algorithm
            FROM paper_trading_daily_log ptl
            LEFT JOIN model_lifecycle ml ON ptl.model_id = ml.model_id
            LEFT JOIN strategies s ON ptl.model_id = s.model_id
            ORDER BY ptl.trading_date DESC
        ''')
        daily_data = cursor.fetchall()
        
        cursor.execute('''
            SELECT 
                trading_date,
                COUNT(DISTINCT model_id) as active_models,
                SUM(trades_executed) as total_trades,
                AVG(daily_return) as avg_return
            FROM paper_trading_daily_log
            GROUP BY trading_date
            ORDER BY trading_date DESC
        ''')
        agg_data = cursor.fetchall()
        
        conn.close()
        return daily_data, agg_data
        
    except Exception as e:
        return [], []


def get_all_models_with_lineage():
    """Get all models and their parent relationships from database only."""
    models = {}
    
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return models
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ml.model_id, ml.current_state, ml.backtest_expected_sharpe, s.algorithm
            FROM model_lifecycle ml
            LEFT JOIN strategies s ON ml.model_id = s.model_id
        ''')
        
        for row in cursor.fetchall():
            model_id, state, sharpe, algorithm = row
            
            generation = 1
            parent_id = None
            if '_g' in model_id:
                parts = model_id.rsplit('_g', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    parent_id = parts[0]
                    generation = int(parts[1])
            
            models[model_id] = {
                'model_id': model_id,
                'parent_model_id': parent_id,
                'generation': generation,
                'state': state,
                'sharpe': sharpe,
                'algorithm': algorithm
            }
        
        conn.close()
        
        # Also try to load metadata from YAML files
        dirs = ['autotest_models', 'models', 'champion_models']
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            
            for filename in os.listdir(dir_path):
                if filename.endswith('_metadata.yaml'):
                    filepath = os.path.join(dir_path, filename)
                    try:
                        with open(filepath, 'r') as f:
                            meta = yaml.safe_load(f)
                        
                        if meta and 'model_id' in meta:
                            model_id = meta['model_id']
                            if model_id in models:
                                if meta.get('parent_model_id'):
                                    models[model_id]['parent_model_id'] = meta['parent_model_id']
                                if meta.get('generation'):
                                    models[model_id]['generation'] = meta['generation']
                                if meta.get('algorithm') and not models[model_id].get('algorithm'):
                                    models[model_id]['algorithm'] = meta['algorithm']
                    except:
                        continue
        
    except Exception as e:
        pass
    
    return models


def get_next_events():
    """Calculate next scheduled events."""
    now = datetime.now()
    
    next_training = now.replace(hour=2, minute=0, second=0, microsecond=0)
    if now.hour >= 2:
        next_training += timedelta(days=1)
    
    days_until_saturday = (5 - now.weekday()) % 7
    if days_until_saturday == 0 and now.hour >= 18:
        days_until_saturday = 7
    next_culling = now.replace(hour=18, minute=0, second=0) + timedelta(days=days_until_saturday)
    
    return next_training, next_culling


def get_model_details(model_id: str) -> dict:
    """Get detailed information about a specific model including grokking, backtest, and paper trading data."""
    details = {
        'model_id': model_id,
        'grokking': None,
        'backtest': None,
        'paper_trading': [],
        'metadata': None
    }
    
    # Try to load metadata from YAML files
    dirs = ['autotest_models', 'models', 'champion_models']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            continue
        
        metadata_file = os.path.join(dir_path, f"{model_id}_metadata.yaml")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    details['metadata'] = yaml.safe_load(f)
                break
            except:
                continue
    
    # Try to load grokking data from selection JSON files
    logs_dir = 'autotest_logs'
    if os.path.exists(logs_dir):
        try:
            for filename in os.listdir(logs_dir):
                if filename.startswith('model_selection_') and filename.endswith('.json'):
                    filepath = os.path.join(logs_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            selection_data = json.load(f)
                        
                        for eval_data in selection_data.get('evaluations', []):
                            if eval_data.get('model_id') == model_id:
                                if 'grokking_details' in eval_data:
                                    details['grokking'] = eval_data['grokking_details']
                                # Also grab eval rewards for the hockey stick chart
                                if 'eval_rewards' in eval_data:
                                    details['eval_rewards'] = eval_data['eval_rewards']
                                if 'eval_timesteps' in eval_data:
                                    details['eval_timesteps'] = eval_data['eval_timesteps']
                                break
                        
                        if details['grokking']:
                            break
                    except:
                        continue
        except:
            pass
    
    # Load from database
    db_path = 'autotest_strategies.db'
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get backtest results
            cursor.execute('''
                SELECT sharpe_ratio, total_return, max_drawdown, win_rate, 
                       sortino_ratio, calmar_ratio, volatility, final_value,
                       ranking_score, rank_position
                FROM backtest_results 
                WHERE model_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (model_id,))
            
            row = cursor.fetchone()
            if row:
                details['backtest'] = {
                    'sharpe_ratio': row[0],
                    'total_return': row[1],
                    'max_drawdown': row[2],
                    'win_rate': row[3],
                    'sortino_ratio': row[4],
                    'calmar_ratio': row[5],
                    'volatility': row[6],
                    'final_value': row[7],
                    'ranking_score': row[8],
                    'rank_position': row[9]
                }
            
            # Get paper trading daily log
            cursor.execute('''
                SELECT trading_date, portfolio_value, daily_return, trades_executed,
                       transaction_costs, cash, total_stock_value
                FROM paper_trading_daily_log
                WHERE model_id = ?
                ORDER BY trading_date ASC
            ''', (model_id,))
            
            details['paper_trading'] = []
            for row in cursor.fetchall():
                details['paper_trading'].append({
                    'date': row[0],
                    'portfolio_value': row[1],
                    'daily_return': row[2],
                    'trades_executed': row[3],
                    'transaction_costs': row[4],
                    'cash': row[5],
                    'stock_value': row[6]
                })
            
            conn.close()
        except Exception as e:
            pass
    
    return details


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================

class StatBox(Static):
    """A stat display box."""
    
    def __init__(self, label: str, value: str = "--", style_type: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.value = value
        self.style_type = style_type
    
    def compose(self) -> ComposeResult:
        yield Static(self.value, classes="stat-value")
        yield Static(self.label, classes="stat-label")
    
    def update_value(self, value: str):
        self.value = value
        self.query_one(".stat-value", Static).update(value)


class ModelCard(Static):
    """A model display card."""
    
    def __init__(self, model_id: str, info: str, right_info: str = "", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.info = info
        self.right_info = right_info
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="model-card-content"):
            yield Static(f"> {self.model_id}", classes="model-id")
            yield Static(self.info, classes="model-info")
            if self.right_info:
                yield Static(self.right_info, classes="model-right")


# ============================================================================
# TAB CONTENT CLASSES
# ============================================================================

class DashboardTab(Static):
    """Dashboard tab content."""
    
    def compose(self) -> ComposeResult:
        yield Static("", id="daemon-status", classes="daemon-status")
        yield Rule()
        
        with Horizontal(classes="stats-row"):
            yield StatBox("PAPER TRADING", "0", id="stat-paper")
            yield StatBox("PROMOTED", "0", id="stat-promoted")
            yield StatBox("VALIDATION", "0", id="stat-validation")
            yield StatBox("CULLED", "0", id="stat-culled")
            yield StatBox("TOTAL", "0", id="stat-total")
        
        yield Rule()
        
        with Horizontal(classes="main-content"):
            with Vertical(classes="left-panel"):
                yield Static("[ACTIVE] PAPER TRADING MODELS", classes="section-header")
                yield ScrollableContainer(id="paper-trading-list")
                yield Static("[LOG] RECENTLY TRAINED", classes="section-header")
                yield ScrollableContainer(id="recent-models-list")
            
            with Vertical(classes="right-panel"):
                yield Static("[CRON] SCHEDULE", classes="section-header")
                yield Static("", id="schedule-info")
                yield Static("[ELITE] CHAMPIONS", classes="section-header")
                yield ScrollableContainer(id="champions-list")
        
        yield Rule()
        
        with Horizontal(classes="button-row"):
            yield Button("[F5] REFRESH", id="btn-refresh", variant="primary")
            yield Button("[TOGGLE] DAEMON", id="btn-daemon")
            yield Button("[CAT] VIEW LOGS", id="btn-logs")
    
    def on_mount(self) -> None:
        self.refresh_data()
    
    def refresh_data(self) -> None:
        # Daemon status
        daemon_running = get_daemon_status()
        status_widget = self.query_one("#daemon-status", Static)
        if daemon_running:
            status_widget.update("[green]█ DAEMON RUNNING[/green]")
        else:
            status_widget.update("[red]█ DAEMON STOPPED[/red]")
        
        # Lifecycle data
        data = get_lifecycle_data()
        
        if data is None:
            return
        
        if 'error' in data:
            return
        
        states = data.get('states', {})
        
        # Update stats
        self.query_one("#stat-paper", StatBox).update_value(str(states.get('paper_trading', 0)))
        self.query_one("#stat-promoted", StatBox).update_value(str(states.get('promoted', 0)))
        self.query_one("#stat-validation", StatBox).update_value(str(states.get('validation', 0)))
        self.query_one("#stat-culled", StatBox).update_value(str(states.get('culled', 0)))
        self.query_one("#stat-total", StatBox).update_value(str(sum(states.values())))
        
        # Paper trading models
        paper_list = self.query_one("#paper-trading-list", ScrollableContainer)
        paper_list.remove_children()
        
        if data.get('paper_trading_models'):
            for model in data['paper_trading_models']:
                tickers_str = ', '.join(model['tickers'][:3]) if model['tickers'] else 'N/A'
                paper_list.mount(
                    Static(f"[yellow]> {model['model_id']}[/yellow] {tickers_str} [dim]DAY {model['trading_days']}/10[/dim]")
                )
        else:
            paper_list.mount(Static("[dim]No models currently paper trading[/dim]"))
        
        # Recent models
        recent_list = self.query_one("#recent-models-list", ScrollableContainer)
        recent_list.remove_children()
        
        if data.get('recent_models'):
            for model in data['recent_models'][:5]:
                tickers = json.loads(model[2]) if model[2] else []
                tickers_str = ', '.join(tickers[:3]) if tickers else 'N/A'
                recent_list.mount(
                    Static(f"[yellow]> {model[0]}[/yellow] [{model[1].upper()}] {tickers_str}")
                )
        else:
            recent_list.mount(Static("[dim]No models trained yet[/dim]"))
        
        # Schedule
        next_training, next_culling = get_next_events()
        now = datetime.now()
        training_delta = next_training - now
        culling_delta = next_culling - now
        
        schedule_widget = self.query_one("#schedule-info", Static)
        schedule_widget.update(
            f"[bold]Next Training:[/bold]\n"
            f"  {next_training.strftime('%A %I:%M %p')}\n"
            f"  [dim]in {training_delta.days}d {training_delta.seconds//3600}h[/dim]\n\n"
            f"[bold]Next Culling:[/bold]\n"
            f"  {next_culling.strftime('%A %I:%M %p')}\n"
            f"  [dim]in {culling_delta.days}d {culling_delta.seconds//3600}h[/dim]"
        )
        
        # Champions
        champions_list = self.query_one("#champions-list", ScrollableContainer)
        champions_list.remove_children()
        
        if data.get('promoted_models'):
            for model in data['promoted_models']:
                champions_list.mount(Static(f"[green]★ {model[0]}[/green]"))
        else:
            champions_list.mount(Static("[dim]No champions yet[/dim]"))


class ModelDetailsPanel(Static):
    """Collapsible panel showing detailed model information with visualizations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_model_id = None
    
    def compose(self) -> ComposeResult:
        yield Static("[MODEL DETAILS] Select a model from the table above", id="details-header", classes="section-header")
        
        with Collapsible(title="GROKKING ANALYSIS", collapsed=True, id="grokking-collapsible"):
            yield Static("No grokking data available", id="grokking-content")
        
        with Collapsible(title="BACKTEST RESULTS", collapsed=True, id="backtest-collapsible"):
            yield Static("No backtest data available", id="backtest-content")
        
        with Collapsible(title="PAPER TRADING", collapsed=True, id="paper-collapsible"):
            yield Static("No paper trading data available", id="paper-content")
    
    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        """Ensure only one collapsible is open at a time."""
        collapsible_id = event.collapsible.id
        
        # Close the other collapsibles
        for cid in ["grokking-collapsible", "backtest-collapsible", "paper-collapsible"]:
            if cid != collapsible_id:
                try:
                    other = self.query_one(f"#{cid}", Collapsible)
                    other.collapsed = True
                except:
                    pass
    
    def load_model_details(self, model_id: str):
        """Load and display detailed information for a model."""
        self.current_model_id = model_id
        
        # Update header
        header = self.query_one("#details-header", Static)
        header.update(f"[MODEL DETAILS] {model_id}")
        
        # Load data
        details = get_model_details(model_id)
        
        # Update Grokking Analysis
        grokking_widget = self.query_one("#grokking-content", Static)
        if details['grokking']:
            g = details['grokking']
            eval_rewards = details.get('eval_rewards', [])
            content = self._format_grokking_content(g, eval_rewards)
            grokking_widget.update(content)
        else:
            grokking_widget.update("[dim]No grokking analysis data available for this model[/dim]")
        
        # Update Backtest Results
        backtest_widget = self.query_one("#backtest-content", Static)
        if details['backtest']:
            b = details['backtest']
            content = self._format_backtest_content(b)
            backtest_widget.update(content)
        else:
            backtest_widget.update("[dim]No backtest results available for this model[/dim]")
        
        # Update Paper Trading
        paper_widget = self.query_one("#paper-content", Static)
        if details['paper_trading']:
            content = self._format_paper_trading_content(details['paper_trading'])
            paper_widget.update(content)
        else:
            paper_widget.update("[dim]No paper trading data available for this model[/dim]")
    
    def _format_grokking_content(self, g: dict, eval_rewards: list = None) -> str:
        """Format grokking analysis data as rich text with hockey stick chart."""
        lines = []
        
        # Status indicator
        has_grokked = g.get('has_grokked', False)
        score = g.get('grokking_score', 0)
        phase = g.get('phase', 'unknown')
        
        if has_grokked:
            lines.append(f"[green bold][OK] GROKKED[/green bold] (Score: {score:.3f})")
        else:
            lines.append(f"[red bold][NO] NOT GROKKED[/red bold] (Score: {score:.3f})")
        
        lines.append(f"[dim]Phase: {phase} (Confidence: {g.get('phase_confidence', 0):.2f})[/dim]")
        lines.append("")
        
        # HOCKEY STICK CHART - Visualize the grokking pattern
        if eval_rewards and len(eval_rewards) >= 5:
            lines.append("[bold cyan]HOCKEY STICK CHART (Eval Rewards Over Time)[/bold cyan]")
            lines.append(self._create_hockey_stick_chart(eval_rewards, phase))
            lines.append("")
        
        # Key metrics
        lines.append("[bold]Weight Matrix Analysis:[/bold]")
        rank_ratio = g.get('avg_effective_rank_ratio', 0)
        rank_status = "[green]OK[/green]" if rank_ratio < 0.65 else "[red]BAD[/red]"
        lines.append(f"  [{rank_status}] Effective Rank Ratio: {rank_ratio:.3f} (target < 0.65)")
        
        weight_norm = g.get('avg_weight_norm', 0)
        lines.append(f"  [-] Average Weight Norm: {weight_norm:.2f}")
        
        norm_trend = g.get('weight_norm_trend', 0)
        trend_icon = "DECR" if norm_trend < 0 else "INCR"
        lines.append(f"  [{trend_icon}] Weight Norm Trend: {norm_trend:+.3f} (negative = simplifying)")
        lines.append("")
        
        # Eval curve analysis
        lines.append("[bold]Evaluation Performance:[/bold]")
        stability = g.get('eval_stability', 0)
        stability_pct = stability * 100
        lines.append(f"  [-] Eval Stability: {stability_pct:.1f}%")
        
        improvement = g.get('eval_improvement_rate', 0)
        imp_icon = "UP" if improvement > 0 else "DOWN"
        lines.append(f"  [{imp_icon}] Late-Stage Improvement: {improvement:+.3f}")
        
        gap = g.get('generalization_gap', 0)
        lines.append(f"  • Generalization Gap: {gap:.3f} (train - eval)")
        lines.append("")
        
        # Reason
        if 'reason' in g:
            lines.append(f"[italic]{g['reason']}[/italic]")
        
        return "\n".join(lines)
    
    def _create_hockey_stick_chart(self, eval_rewards: list, phase: str) -> str:
        """Create an ASCII hockey stick chart showing the grokking pattern."""
        if not eval_rewards or len(eval_rewards) < 3:
            return "[dim]Insufficient data for chart[/dim]"
        
        lines = []
        rewards = list(eval_rewards)
        n = len(rewards)
        
        # Normalize to 0-1 range for display
        min_r = min(rewards)
        max_r = max(rewards)
        r_range = max_r - min_r if max_r != min_r else 1
        
        # Split into early, middle, late phases
        third = n // 3
        early_end = third
        middle_end = 2 * third
        
        # Chart dimensions
        chart_height = 8
        chart_width = min(50, n)  # Max 50 data points displayed
        
        # Sample data points if too many
        if n > chart_width:
            step = n // chart_width
            display_rewards = [rewards[i] for i in range(0, n, step)][:chart_width]
        else:
            display_rewards = rewards
        
        # Normalize display rewards
        normalized = [(r - min_r) / r_range for r in display_rewards]
        
        # Build chart
        lines.append("")
        lines.append(f"  [dim]Eval Performance (n={n} checkpoints)[/dim]")
        lines.append("")
        
        # Y-axis labels and chart
        for row in range(chart_height, -1, -1):
            threshold = row / chart_height
            
            # Y-axis label
            value_at_level = min_r + (r_range * row / chart_height)
            label = f"{value_at_level:6.0f} │"
            
            # Chart bars
            bar_line = ""
            for i, norm_val in enumerate(normalized):
                # Color code by phase
                pos = i / len(normalized)
                if pos < 0.33:
                    color = "dim"  # Early phase
                elif pos < 0.67:
                    color = "yellow"  # Middle phase  
                else:
                    color = "green" if phase in ['post_grok', 'grokking', 'improving'] else "red"
                
                if norm_val >= threshold:
                    bar_line += f"[{color}]█[/{color}]"
                else:
                    bar_line += " "
            
            lines.append(f"  {label}{bar_line}")
        
        # X-axis
        lines.append(f"       └{'─' * len(display_rewards)}")
        lines.append(f"        [dim]Early[/dim]{' ' * (len(display_rewards)//3 - 5)}[yellow]Middle[/yellow]{' ' * (len(display_rewards)//3 - 6)}[bold]Late[/bold]")
        
        # Phase indicator
        phase_emoji = {
            'post_grok': 'POST-GROK (Sharp Late Improvement)',
            'grokking': 'GROKKING (Active Improvement)', 
            'improving': 'IMPROVING (Steady Progress)',
            'pre_grok': 'PRE-GROK (Flat, Needs More Training)',
            'memorizing': 'MEMORIZING (Stuck/Declining)'
        }.get(phase, f'{phase.upper()}')
        
        lines.append("")
        lines.append(f"  Pattern: {phase_emoji}")
        
        return "\n".join(lines)
    
    def _format_backtest_content(self, b: dict) -> str:
        """Format backtest results as rich text."""
        lines = []
        
        # Key metrics
        lines.append("[bold]Performance Metrics:[/bold]")
        
        sharpe = b.get('sharpe_ratio', 0)
        sharpe_color = "green" if sharpe > 1.0 else "yellow" if sharpe > 0.5 else "red"
        lines.append(f"  [{sharpe_color}]• Sharpe Ratio: {sharpe:.3f}[/{sharpe_color}]")
        
        total_ret = b.get('total_return', 0)
        ret_color = "green" if total_ret > 0 else "red"
        lines.append(f"  [{ret_color}]• Total Return: {total_ret*100:.2f}%[/{ret_color}]")
        
        max_dd = b.get('max_drawdown', 0)
        dd_color = "green" if max_dd < 0.1 else "yellow" if max_dd < 0.2 else "red"
        lines.append(f"  [{dd_color}]• Max Drawdown: {max_dd*100:.2f}%[/{dd_color}]")
        
        win_rate = b.get('win_rate', 0)
        lines.append(f"  • Win Rate: {win_rate*100:.1f}%")
        lines.append("")
        
        # Additional metrics
        lines.append("[bold]Risk-Adjusted Metrics:[/bold]")
        lines.append(f"  • Sortino Ratio: {b.get('sortino_ratio', 0):.3f}")
        lines.append(f"  • Calmar Ratio: {b.get('calmar_ratio', 0):.3f}")
        lines.append(f"  • Volatility: {b.get('volatility', 0)*100:.2f}%")
        lines.append("")
        
        # Portfolio
        lines.append("[bold]Portfolio:[/bold]")
        lines.append(f"  • Final Value: ${b.get('final_value', 0):,.2f}")
        
        rank = b.get('rank_position')
        if rank:
            lines.append(f"  • Ranking: #{rank} (Score: {b.get('ranking_score', 0):.3f})")
        
        return "\n".join(lines)
    
    def _format_paper_trading_content(self, paper_data: list) -> str:
        """Format paper trading data as rich text with simple ASCII chart."""
        if not paper_data:
            return "[dim]No paper trading data available[/dim]"
        
        lines = []
        
        # Summary stats
        lines.append("[bold]Trading Summary:[/bold]")
        days_traded = len(paper_data)
        lines.append(f"  • Days Traded: {days_traded}")
        
        if days_traded > 0:
            start_value = paper_data[0].get('portfolio_value', 0)
            end_value = paper_data[-1].get('portfolio_value', 0)
            total_return = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
            
            ret_color = "green" if total_return > 0 else "red"
            lines.append(f"  [{ret_color}]• Total Return: {total_return:+.2f}%[/{ret_color}]")
            lines.append(f"  • Start Value: ${start_value:,.2f}")
            lines.append(f"  • Current Value: ${end_value:,.2f}")
            
            total_trades = sum(d.get('trades_executed', 0) for d in paper_data)
            lines.append(f"  • Total Trades: {total_trades}")
        
        lines.append("")
        
        # Simple ASCII chart of portfolio value over time
        if days_traded >= 5:
            lines.append("[bold]Portfolio Value Trend:[/bold]")
            values = [d.get('portfolio_value', 0) for d in paper_data]
            min_val = min(values)
            max_val = max(values)
            val_range = max_val - min_val if max_val != min_val else 1
            
            # Show last 20 data points max
            display_data = values[-20:] if len(values) > 20 else values
            
            chart_height = 10
            for row in range(chart_height, -1, -1):
                threshold = min_val + (val_range * row / chart_height)
                line = "  "
                for val in display_data:
                    if val >= threshold:
                        line += "█"
                    else:
                        line += " "
                lines.append(line)
            
            lines.append(f"  ${min_val:,.0f}{' ' * (len(display_data) - len(f'{min_val:,.0f}') - len(f'{max_val:,.0f}'))}${max_val:,.0f}")
        
        # Recent trades
        lines.append("")
        lines.append("[bold]Recent Activity (Last 5 Days):[/bold]")
        recent = paper_data[-5:] if len(paper_data) > 5 else paper_data
        for day in reversed(recent):
            date = day.get('date', 'Unknown')
            ret = day.get('daily_return', 0) * 100
            ret_str = f"{ret:+.2f}%"
            ret_color = "green" if ret > 0 else "red" if ret < 0 else "white"
            value = day.get('portfolio_value', 0)
            trades = day.get('trades_executed', 0)
            
            lines.append(f"  {date}: [{ret_color}]{ret_str}[/{ret_color}] | Value: ${value:,.0f} | Trades: {trades}")
        
        return "\n".join(lines)


class ModelsTab(Static):
    """Models browser tab content."""
    
    def compose(self) -> ComposeResult:
        yield Static("[MODELS] ALL MODELS", classes="section-header")
        yield Static("", id="models-summary")
        yield Rule()
        with ScrollableContainer(id="models-table-container", classes="scrollable-table"):
            yield DataTable(id="models-table")
        yield Rule()
        yield ModelDetailsPanel(id="model-details")
    
    def on_mount(self) -> None:
        self.refresh_data()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection to show model details."""
        table = self.query_one("#models-table", DataTable)
        row_key = event.row_key
        
        # Get the model_id from the first column of the selected row
        if row_key:
            row_data = table.get_row(row_key)
            if row_data:
                model_id = str(row_data[0])  # First column is model_id
                details_panel = self.query_one("#model-details", ModelDetailsPanel)
                details_panel.load_model_details(model_id)
    
    def refresh_data(self) -> None:
        models = get_all_models()
        
        # Summary
        total = len(models)
        paper = sum(1 for m in models if m['current_state'] == 'paper_trading')
        promoted = sum(1 for m in models if m['current_state'] == 'promoted')
        culled = sum(1 for m in models if m['current_state'] == 'culled')
        
        summary = self.query_one("#models-summary", Static)
        summary.update(
            f"Total: [bold]{total}[/bold] | "
            f"Paper Trading: [yellow]{paper}[/yellow] | "
            f"Promoted: [green]{promoted}[/green] | "
            f"Culled: [red]{culled}[/red]"
        )
        
        # Table
        table = self.query_one("#models-table", DataTable)
        table.clear(columns=True)
        
        table.add_columns("Model ID", "State", "Algorithm", "Sharpe", "Return", "Days", "Tickers")
        
        for m in models:
            state = m['current_state'] or 'unknown'
            if state == 'promoted':
                state_str = f"[green]{state.upper()}[/green]"
            elif state == 'culled':
                state_str = f"[red]{state.upper()}[/red]"
            elif state == 'paper_trading':
                state_str = f"[yellow]{state.upper()}[/yellow]"
            else:
                state_str = state.upper()
            
            sharpe = f"{m['backtest_expected_sharpe']:.2f}" if m['backtest_expected_sharpe'] else "--"
            ret = f"{m['backtest_expected_return']:.1f}%" if m['backtest_expected_return'] else "--"
            days = str(m['trading_days']) if m['trading_days'] else "--"
            tickers = ', '.join(m['tickers'][:3]) if m['tickers'] else "--"
            algo = (m['algorithm'] or 'unknown').upper()
            
            table.add_row(
                m['model_id'],
                state_str,
                algo,
                sharpe,
                ret,
                days,
                tickers
            )


class LineageTab(Static):
    """Lineage tracker tab content."""
    
    def compose(self) -> ComposeResult:
        yield Static("[LINEAGE] EVOLUTION TRACKER", classes="section-header")
        
        with Horizontal(classes="stats-row"):
            yield StatBox("TOTAL MODELS", "0", id="lineage-total")
            yield StatBox("ORIGINAL (G1)", "0", id="lineage-g1")
            yield StatBox("OFFSPRING", "0", id="lineage-offspring")
            yield StatBox("MAX GENERATION", "0", id="lineage-maxgen")
        
        yield Rule()
        yield Static("[dim]Legend: [green][+] Promoted[/green] | [red][X] Culled[/red] | [yellow][>] Active[/yellow] | [ ] Other[/dim]")
        yield Rule()
        yield Static("[FAMILY TREES]", classes="section-header")
        yield ScrollableContainer(Tree("Models", id="lineage-tree"), id="lineage-container")
    
    def on_mount(self) -> None:
        self.refresh_data()
    
    def refresh_data(self) -> None:
        models = get_all_models_with_lineage()
        
        if not models:
            return
        
        # Stats
        total_models = len(models)
        gen1_models = sum(1 for m in models.values() if m.get('generation', 1) == 1)
        offspring_models = total_models - gen1_models
        max_gen = max((m.get('generation', 1) for m in models.values()), default=1)
        
        self.query_one("#lineage-total", StatBox).update_value(str(total_models))
        self.query_one("#lineage-g1", StatBox).update_value(str(gen1_models))
        self.query_one("#lineage-offspring", StatBox).update_value(str(offspring_models))
        self.query_one("#lineage-maxgen", StatBox).update_value(str(max_gen))
        
        # Build family trees
        roots = []
        children_map = defaultdict(list)
        
        for model_id, data in models.items():
            parent = data.get('parent_model_id')
            if parent and parent in models:
                children_map[parent].append(model_id)
            elif not parent:
                roots.append(model_id)
        
        for model_id, data in models.items():
            parent = data.get('parent_model_id')
            if parent and parent not in models and model_id not in roots:
                roots.append(model_id)
        
        # Build tree widget
        tree = self.query_one("#lineage-tree", Tree)
        tree.clear()
        tree.root.expand()
        
        def add_node(parent_node, model_id):
            data = models.get(model_id, {})
            state = data.get('state', 'unknown')
            gen = data.get('generation', 1)
            sharpe = data.get('sharpe')
            algo = data.get('algorithm', '?')
            
            if state == 'promoted':
                icon = "[green][+][/green]"
            elif state == 'culled':
                icon = "[red][X][/red]"
            elif state == 'paper_trading':
                icon = "[yellow][>][/yellow]"
            else:
                icon = "[ ]"
            
            sharpe_str = f"S:{sharpe:.2f}" if sharpe else ""
            algo_str = algo.upper()[:3] if algo else ""
            
            label = f"{icon} {model_id} [dim]G{gen} {algo_str} {sharpe_str}[/dim]"
            node = parent_node.add(label)
            
            for child in children_map.get(model_id, []):
                add_node(node, child)
            
            return node
        
        for root in roots:
            node = add_node(tree.root, root)
            if children_map.get(root):
                node.expand()


class TradingTab(Static):
    """Trading history tab content."""
    
    def compose(self) -> ComposeResult:
        yield Static("[TRADING] PAPER TRADING HISTORY", classes="section-header")
        
        with Horizontal(classes="stats-row"):
            yield StatBox("TRADING DAYS", "0", id="trading-days")
            yield StatBox("MODELS ACTIVE", "0", id="trading-models")
            yield StatBox("TOTAL TRADES", "0", id="trading-trades")
            yield StatBox("AVG DAILY RETURN", "0%", id="trading-return")
        
        yield Rule()
        yield Static("[DAILY SUMMARY]", classes="section-header")
        yield DataTable(id="trading-table")
    
    def on_mount(self) -> None:
        self.refresh_data()
    
    def refresh_data(self) -> None:
        daily_data, agg_data = get_trading_data()
        
        if not daily_data and not agg_data:
            return
        
        # Stats from aggregated data
        total_days = len(agg_data)
        total_models = len(set(d[0] for d in daily_data)) if daily_data else 0
        total_trades = sum(d[2] or 0 for d in agg_data) if agg_data else 0
        avg_return = sum(d[3] or 0 for d in agg_data) / len(agg_data) if agg_data else 0
        
        self.query_one("#trading-days", StatBox).update_value(str(total_days))
        self.query_one("#trading-models", StatBox).update_value(str(total_models))
        self.query_one("#trading-trades", StatBox).update_value(f"{int(total_trades):,}")
        
        ret_color = "green" if avg_return >= 0 else "red"
        self.query_one("#trading-return", StatBox).update_value(f"[{ret_color}]{avg_return:.3f}%[/{ret_color}]")
        
        # Table
        table = self.query_one("#trading-table", DataTable)
        table.clear(columns=True)
        
        table.add_columns("Date", "Active Models", "Trades", "Avg Return")
        
        for row in agg_data[:30]:  # Last 30 days
            date_str = row[0]
            active = str(row[1])
            trades = str(int(row[2])) if row[2] else "0"
            ret = row[3]
            if ret is not None:
                ret_color = "green" if ret >= 0 else "red"
                ret_str = f"[{ret_color}]{ret:.3f}%[/{ret_color}]"
            else:
                ret_str = "--"
            
            table.add_row(date_str, active, trades, ret_str)


# ============================================================================
# MAIN APP
# ============================================================================

class RLFIColosseumTUI(App):
    """RLFI Colosseum TUI Application."""
    
    CSS = """
    Screen {
        background: #141414;
    }
    
    Header {
        background: #1a1a1a;
        color: #ffaa00;
    }
    
    Footer {
        background: #1a1a1a;
        color: #996600;
    }
    
    TabbedContent {
        background: #141414;
    }
    
    TabPane {
        background: #141414;
        padding: 1;
    }
    
    Tab {
        background: #1a1a1a;
        color: #996600;
    }
    
    Tab.-active {
        background: #1a1a1a;
        color: #ffaa00;
        text-style: bold;
    }
    
    .daemon-status {
        text-align: center;
        padding: 1;
        background: #1a1a1a;
        border: solid #996600;
    }
    
    .section-header {
        color: #ffaa00;
        text-style: bold;
        padding: 1 0;
    }
    
    .stats-row {
        height: auto;
        padding: 1 0;
    }
    
    StatBox {
        width: 1fr;
        height: auto;
        background: #1a1a1a;
        border: solid #996600;
        padding: 1;
        margin: 0 1;
    }
    
    .stat-value {
        text-align: center;
        color: #ffcc00;
        text-style: bold;
    }
    
    .stat-label {
        text-align: center;
        color: #996600;
    }
    
    .main-content {
        height: 1fr;
    }
    
    .left-panel {
        width: 2fr;
        padding-right: 2;
    }
    
    .right-panel {
        width: 1fr;
        border-left: solid #996600;
        padding-left: 2;
    }
    
    ScrollableContainer {
        height: auto;
        max-height: 10;
        background: #1a1a1a;
        border: solid #333333;
        padding: 1;
    }
    
    #lineage-container {
        height: 1fr;
        max-height: 100%;
    }
    
    .button-row {
        height: auto;
        padding: 1 0;
        align: center middle;
    }
    
    Button {
        margin: 0 1;
        background: #1a1a1a;
        color: #cc8800;
        border: solid #cc8800;
    }
    
    Button:hover {
        background: #1a1100;
        color: #ffcc00;
        border: solid #ffcc00;
    }
    
    Button.-primary {
        background: #1a1a1a;
        color: #ffaa00;
        border: solid #ffaa00;
    }
    
    #models-table-container {
        height: auto;
        max-height: 15;
        background: #1a1a1a;
        border: solid #996600;
    }
    
    DataTable {
        height: auto;
        background: #1a1a1a;
    }
    
    DataTable > .datatable--header {
        background: #1a1a1a;
        color: #ffaa00;
        text-style: bold;
    }
    
    DataTable > .datatable--cursor {
        background: #332200;
    }
    
    Tree {
        background: #1a1a1a;
    }
    
    Tree > .tree--cursor {
        background: #332200;
    }
    
    Rule {
        color: #996600;
    }
    
    Static {
        color: #cc8800;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("d", "toggle_daemon", "Toggle Daemon"),
        ("1", "tab_dashboard", "Dashboard"),
        ("2", "tab_models", "Models"),
        ("3", "tab_lineage", "Lineage"),
        ("4", "tab_trading", "Trading"),
    ]
    
    TITLE = "RLFI COLOSSEUM"
    SUB_TITLE = "Reinforcement Learning Financial Intelligence // Trading Arena v1.0"
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with TabbedContent():
            with TabPane("DASHBOARD", id="tab-dashboard"):
                yield DashboardTab()
            with TabPane("MODELS", id="tab-models"):
                yield ModelsTab()
            with TabPane("LINEAGE", id="tab-lineage"):
                yield LineageTab()
            with TabPane("TRADING", id="tab-trading"):
                yield TradingTab()
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh":
            self.action_refresh()
        elif event.button.id == "btn-daemon":
            self.action_toggle_daemon()
        elif event.button.id == "btn-logs":
            self.show_logs()
    
    def action_refresh(self) -> None:
        """Refresh all data."""
        try:
            self.query_one(DashboardTab).refresh_data()
        except:
            pass
        try:
            self.query_one(ModelsTab).refresh_data()
        except:
            pass
        try:
            self.query_one(LineageTab).refresh_data()
        except:
            pass
        try:
            self.query_one(TradingTab).refresh_data()
        except:
            pass
        self.notify("Data refreshed")
    
    def action_toggle_daemon(self) -> None:
        """Toggle the RLFI daemon."""
        daemon_running = get_daemon_status()
        if daemon_running:
            os.system("sudo systemctl stop rlfi")
            self.notify("Stopping daemon...")
        else:
            os.system("sudo systemctl start rlfi")
            self.notify("Starting daemon...")
        self.action_refresh()
    
    def action_tab_dashboard(self) -> None:
        self.query_one(TabbedContent).active = "tab-dashboard"
    
    def action_tab_models(self) -> None:
        self.query_one(TabbedContent).active = "tab-models"
    
    def action_tab_lineage(self) -> None:
        self.query_one(TabbedContent).active = "tab-lineage"
    
    def action_tab_trading(self) -> None:
        self.query_one(TabbedContent).active = "tab-trading"
    
    def show_logs(self) -> None:
        """Show recent logs."""
        log_path = 'logs/rlfi.log'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = f.readlines()[-20:]
            self.notify('\n'.join(logs), title="Recent Logs", timeout=10)
        else:
            self.notify("No logs found yet", title="Logs")


def main():
    app = RLFIColosseumTUI()
    app.run()


if __name__ == "__main__":
    main()
