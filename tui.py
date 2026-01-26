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
    TabbedContent, TabPane, Label, Rule, Tree
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
    """Fetch all models with their lifecycle data and metrics."""
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


class ModelsTab(Static):
    """Models browser tab content."""
    
    def compose(self) -> ComposeResult:
        yield Static("[MODELS] ALL MODELS", classes="section-header")
        yield Static("", id="models-summary")
        yield Rule()
        yield DataTable(id="models-table")
    
    def on_mount(self) -> None:
        self.refresh_data()
    
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
    
    DataTable {
        height: 1fr;
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
