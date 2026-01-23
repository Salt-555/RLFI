"""
RLFI Colosseum Dashboard
Real-time status monitoring for the AI Trading Colosseum
"""
import streamlit as st
import pandas as pd
import os
import sys
import subprocess
from datetime import datetime, timedelta
import sqlite3
import json
import yaml
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config - hide sidebar completely
st.set_page_config(
    page_title="RLFI Colosseum",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - amber terminal theme with hidden sidebar
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');
    
    *:not(svg):not(path) {
        font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    }
    
    /* Fix expander icons - hide broken icon and use CSS arrow */
    [data-testid="stExpander"] svg {
        display: none !important;
    }
    [data-testid="stExpander"] summary > span:first-child::before {
        content: "[+] " !important;
        color: #ffaa00;
        font-family: 'Share Tech Mono', monospace !important;
    }
    [data-testid="stExpander"][open] summary > span:first-child::before {
        content: "[-] " !important;
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    
    .stApp {
        background-color: #141414;
        background-image: 
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(20, 10, 0, 0.3) 2px,
                rgba(20, 10, 0, 0.3) 4px
            );
    }
    
    .main-header {
        font-family: 'VT323', monospace !important;
        font-size: 2.5rem;
        text-align: center;
        padding: 0.75rem;
        background: #141414;
        color: #ffaa00;
        border: 1px solid #ffaa00;
        margin-bottom: 0.5rem;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    .main-header::before {
        content: "[ ";
        color: #996600;
    }
    .main-header::after {
        content: " ]";
        color: #996600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #1a1a1a;
        border: 1px solid #996600;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: #141414;
        color: #996600;
        border: none;
        border-right: 1px solid #996600;
        padding: 0.75rem 1.5rem;
        font-family: 'Share Tech Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #1a1a1a;
        color: #ffaa00;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1a1a !important;
        color: #ffaa00 !important;
        border-bottom: 2px solid #ffaa00 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: #ffaa00 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    .status-running {
        background: #141414;
        color: #ffcc00;
        padding: 0.25rem 0.75rem;
        border: 1px solid #ffcc00;
        font-size: 0.85rem;
    }
    .status-running::before {
        content: "[ACTIVE] ";
    }
    
    .status-stopped {
        background: #141414;
        color: #cc3333;
        padding: 0.25rem 0.75rem;
        border: 1px solid #cc3333;
        font-size: 0.85rem;
    }
    .status-stopped::before {
        content: "[OFFLINE] ";
    }
    
    .model-card {
        background: #1a1a1a;
        border: 1px solid #996600;
        border-left: 3px solid #ffaa00;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #cc8800;
    }
    .model-card::before {
        content: "> ";
        color: #996600;
    }
    
    .stat-box {
        background: #1a1a1a;
        border: 1px solid #cc8800;
        color: #cc8800;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-number {
        font-family: 'VT323', monospace !important;
        font-size: 2.5rem;
        font-weight: 400;
        color: #ffcc00;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #996600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-box-warn {
        background: #1a1a1a;
        border: 1px solid #cccc33;
        color: #cccc33;
        padding: 1rem;
        text-align: center;
    }
    .stat-box-warn .stat-number { color: #cccc33; }
    .stat-box-warn .stat-label { color: #999933; }
    
    .stat-box-danger {
        background: #1a1a1a;
        border: 1px solid #cc3333;
        color: #cc3333;
        padding: 1rem;
        text-align: center;
    }
    .stat-box-danger .stat-number { color: #cc3333; }
    .stat-box-danger .stat-label { color: #992222; }
    
    .stat-box-success {
        background: #1a1a1a;
        border: 1px solid #44bb44;
        color: #44bb44;
        padding: 1rem;
        text-align: center;
    }
    .stat-box-success .stat-number { color: #44ff44; }
    .stat-box-success .stat-label { color: #44bb44; }
    
    .stat-box-muted {
        background: #1a1a1a;
        border: 1px solid #555555;
        color: #888888;
        padding: 1rem;
        text-align: center;
    }
    .stat-box-muted .stat-number { color: #888888; }
    .stat-box-muted .stat-label { color: #555555; }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffaa00 !important;
        font-family: 'VT323', monospace !important;
        letter-spacing: 2px;
    }
    
    .stMarkdown, p, span, div {
        color: #cc8800;
    }
    
    hr {
        border-color: #996600 !important;
        border-style: dashed !important;
    }
    
    .stButton > button {
        background: #141414 !important;
        color: #cc8800 !important;
        border: 1px solid #cc8800 !important;
        border-radius: 0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background: #1a1100 !important;
        color: #ffcc00 !important;
        border-color: #ffcc00 !important;
    }
    
    .stSelectbox label, .stMultiSelect label { color: #cc8800 !important; }
    
    .stAlert {
        background: #1a1a1a !important;
        border: 1px solid #996600 !important;
        color: #cc8800 !important;
    }
    
    code {
        background: #1a1a1a !important;
        color: #ffcc00 !important;
        border: 1px solid #996600 !important;
    }
    
    .info-box {
        background: #1a1a1a;
        border: 1px solid #996600;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-good { color: #44ff44; }
    .metric-bad { color: #cc3333; }
    .metric-neutral { color: #cc8800; }
    
    .state-training { color: #cccc33; }
    .state-validation { color: #cccc33; }
    .state-paper_trading { color: #ffaa00; }
    .state-promoted { color: #44ff44; }
    .state-culled { color: #cc3333; }
    
    .tree-node {
        background: #1a1a1a;
        border: 1px solid #996600;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        display: inline-block;
    }
    .tree-node-promoted { border-color: #44ff44; }
    .tree-node-culled { border-color: #cc3333; }
    .tree-node-active { border-color: #ffaa00; border-width: 2px; }
    
    .family-tree {
        background: #1a1a1a;
        border: 1px solid #996600;
        padding: 1rem;
        margin: 1rem 0;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATABASE FUNCTIONS
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
            SELECT model_id, algorithm, tickers, run_date
            FROM strategies
            ORDER BY created_at DESC
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
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        
        query = '''
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
        '''
        
        df = pd.read_sql_query(query, conn)
        
        perf_query = '''
            SELECT 
                model_id,
                COUNT(*) as trading_days,
                MAX(cumulative_return) as cumulative_return,
                AVG(daily_return) as avg_daily_return
            FROM paper_trading_daily_log
            GROUP BY model_id
        '''
        perf_df = pd.read_sql_query(perf_query, conn)
        
        if not perf_df.empty:
            df = df.merge(perf_df, on='model_id', how='left')
        
        conn.close()
        return df
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


def get_model_details(model_id):
    """Get comprehensive model details."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        
        lifecycle_df = pd.read_sql_query(
            'SELECT * FROM model_lifecycle WHERE model_id = ?', 
            conn, params=[model_id]
        )
        strategy_df = pd.read_sql_query(
            'SELECT * FROM strategies WHERE model_id = ?', 
            conn, params=[model_id]
        )
        backtest_df = pd.read_sql_query(
            'SELECT * FROM backtest_results WHERE model_id = ?', 
            conn, params=[model_id]
        )
        trading_df = pd.read_sql_query(
            'SELECT * FROM paper_trading_daily_log WHERE model_id = ? ORDER BY trading_date ASC',
            conn, params=[model_id]
        )
        culling_df = pd.read_sql_query(
            'SELECT * FROM culling_decisions WHERE model_id = ?',
            conn, params=[model_id]
        )
        
        conn.close()
        
        return {
            'lifecycle': lifecycle_df.iloc[0].to_dict() if not lifecycle_df.empty else None,
            'strategy': strategy_df.iloc[0].to_dict() if not strategy_df.empty else None,
            'backtest': backtest_df.iloc[0].to_dict() if not backtest_df.empty else None,
            'trading_history': trading_df,
            'culling': culling_df.iloc[0].to_dict() if not culling_df.empty else None
        }
        
    except Exception as e:
        return None


def get_trading_data():
    """Get all paper trading data."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return None, None
    
    try:
        conn = sqlite3.connect(db_path)
        
        daily_query = '''
            SELECT 
                ptl.model_id,
                ptl.trading_date,
                ptl.portfolio_value,
                ptl.daily_return,
                ptl.cumulative_return,
                ptl.trades_executed,
                ptl.positions,
                ml.current_state,
                s.algorithm
            FROM paper_trading_daily_log ptl
            LEFT JOIN model_lifecycle ml ON ptl.model_id = ml.model_id
            LEFT JOIN strategies s ON ptl.model_id = s.model_id
            ORDER BY ptl.trading_date DESC
        '''
        daily_df = pd.read_sql_query(daily_query, conn)
        
        if not daily_df.empty:
            daily_df['trading_date'] = pd.to_datetime(daily_df['trading_date'])
            
            agg_query = '''
                SELECT 
                    trading_date,
                    COUNT(DISTINCT model_id) as active_models,
                    SUM(trades_executed) as total_trades,
                    AVG(daily_return) as avg_return
                FROM paper_trading_daily_log
                GROUP BY trading_date
                ORDER BY trading_date DESC
            '''
            agg_df = pd.read_sql_query(agg_query, conn)
            agg_df['trading_date'] = pd.to_datetime(agg_df['trading_date'])
        else:
            agg_df = pd.DataFrame()
        
        conn.close()
        return daily_df, agg_df
        
    except Exception as e:
        return None, None


def get_all_models_with_lineage():
    """Get all models and their parent relationships from database only."""
    models = {}
    
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return models
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get models from model_lifecycle table (these are the real models)
        cursor.execute('''
            SELECT ml.model_id, ml.current_state, ml.backtest_expected_sharpe, s.algorithm
            FROM model_lifecycle ml
            LEFT JOIN strategies s ON ml.model_id = s.model_id
        ''')
        
        for row in cursor.fetchall():
            model_id, state, sharpe, algorithm = row
            
            # Parse generation from model_id if it contains _g pattern
            generation = 1
            parent_id = None
            if '_g' in model_id:
                # Model ID format: parent_id_gN where N is generation
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
        
        # Also try to load metadata from YAML files for additional lineage info
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
                            # Only update if model exists in database
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
# HELPER FUNCTIONS
# ============================================================================

def format_state(state):
    """Format state with color coding."""
    state_colors = {
        'training': ('TRAINING', 'state-training'),
        'validation': ('VALIDATION', 'state-validation'),
        'paper_trading': ('PAPER TRADING', 'state-paper_trading'),
        'promoted': ('PROMOTED', 'state-promoted'),
        'culled': ('CULLED', 'state-culled')
    }
    label, css_class = state_colors.get(state, (state.upper(), 'metric-neutral'))
    return f'<span class="{css_class}">[{label}]</span>'


def format_metric(value, threshold_good=0, threshold_bad=None, suffix='', precision=2):
    """Format metric with color based on thresholds."""
    if value is None:
        return '<span class="metric-neutral">--</span>'
    
    if threshold_bad is not None and value < threshold_bad:
        css_class = 'metric-bad'
    elif value >= threshold_good:
        css_class = 'metric-good'
    else:
        css_class = 'metric-neutral'
    
    return f'<span class="{css_class}">{value:.{precision}f}{suffix}</span>'


def render_tree_node(model_id, models, level=0):
    """Render a single tree node with ASCII art."""
    data = models.get(model_id, {})
    state = data.get('state', 'unknown')
    gen = data.get('generation', 1)
    sharpe = data.get('sharpe')
    algo = data.get('algorithm', '?')
    
    state_class = ''
    if state == 'promoted':
        state_class = 'tree-node-promoted'
        state_icon = '[+]'
    elif state == 'culled':
        state_class = 'tree-node-culled'
        state_icon = '[X]'
    elif state == 'paper_trading':
        state_class = 'tree-node-active'
        state_icon = '[>]'
    else:
        state_icon = '[ ]'
    
    sharpe_str = f"S:{sharpe:.2f}" if sharpe else ""
    algo_str = algo.upper()[:3] if algo else ""
    
    indent = "│   " * level
    branch = "├── " if level > 0 else ""
    
    return f'{indent}{branch}<span class="tree-node {state_class}">{state_icon} {model_id} <span style="color:#996600">G{gen} {algo_str} {sharpe_str}</span></span>'


# ============================================================================
# PAGE CONTENT FUNCTIONS
# ============================================================================

def render_dashboard():
    """Render the main dashboard page."""
    daemon_running = get_daemon_status()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if daemon_running:
            st.markdown('<div style="text-align: center;"><span class="status-running">DAEMON RUNNING</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center;"><span class="status-stopped">DAEMON STOPPED</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    data = get_lifecycle_data()
    
    if data is None:
        st.warning("Waiting for first training cycle...")
        st.info("Training runs daily at 2AM.")
        next_training, _ = get_next_events()
        st.markdown(f"**Next Training:** {next_training.strftime('%A %B %d at %I:%M %p')}")
        return
    
    if 'error' in data:
        st.error(f"Database error: {data['error']}")
        return
    
    states = data.get('states', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{states.get('paper_trading', 0)}</div>
            <div class="stat-label">> Paper Trading</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div class="stat-box-success">
            <div class="stat-number">{states.get('promoted', 0)}</div>
            <div class="stat-label">> Promoted</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="stat-box-warn">
            <div class="stat-number">{states.get('validation', 0)}</div>
            <div class="stat-label">> Validation</div>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        st.markdown(f'''
        <div class="stat-box-danger">
            <div class="stat-number">{states.get('culled', 0)}</div>
            <div class="stat-label">> Culled</div>
        </div>
        ''', unsafe_allow_html=True)

    with col5:
        total = sum(states.values())
        st.markdown(f'''
        <div class="stat-box-muted">
            <div class="stat-number">{total}</div>
            <div class="stat-label">> Total Models</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### [ACTIVE] PAPER TRADING MODELS")
        
        if data.get('paper_trading_models'):
            for model in data['paper_trading_models']:
                tickers_str = ', '.join(model['tickers'][:3]) if model['tickers'] else 'N/A'
                st.markdown(f'''
                <div class="model-card">
                    <strong style="color: #ffcc00;">{model['model_id']}</strong>
                    <span style="color: #996600; margin-left: 1rem;">{tickers_str}</span>
                    <span style="float: right; color: #996600;">DAY {model['trading_days']}/10</span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No models currently paper trading")
        
        st.markdown("### [LOG] RECENTLY TRAINED")
        
        if data.get('recent_models'):
            for model in data['recent_models'][:5]:
                tickers = json.loads(model[2]) if model[2] else []
                tickers_str = ', '.join(tickers[:3]) if tickers else 'N/A'
                st.markdown(f'''
                <div class="model-card">
                    <strong style="color: #ffaa00;">{model[0]}</strong>
                    <span style="color: #996600; margin-left: 1rem;">{model[1].upper()}</span>
                    <span style="color: #996600; margin-left: 1rem;">{tickers_str}</span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No models trained yet")
    
    with col_right:
        st.markdown("### [CRON] SCHEDULE")
        
        next_training, next_culling = get_next_events()
        now = datetime.now()
        
        training_delta = next_training - now
        culling_delta = next_culling - now
        
        st.markdown(f"""
        **Next Training:**  
        {next_training.strftime('%A %I:%M %p')}  
        *in {training_delta.days}d {training_delta.seconds//3600}h*

        **Next Culling:**  
        {next_culling.strftime('%A %I:%M %p')}  
        *in {culling_delta.days}d {culling_delta.seconds//3600}h*
        """)
        
        if data.get('promoted_models'):
            st.markdown("---")
            st.markdown("### [ELITE] CHAMPIONS")
            for model in data['promoted_models']:
                st.success(f"**{model[0]}**")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("[F5] REFRESH", use_container_width=True):
            st.rerun()
    
    with col2:
        daemon_running = get_daemon_status()
        if daemon_running:
            if st.button("[STOP] HALT DAEMON", use_container_width=True):
                os.system("sudo systemctl stop rlfi")
                st.rerun()
        else:
            if st.button("[START] INIT DAEMON", use_container_width=True):
                os.system("sudo systemctl start rlfi")
                st.rerun()
    
    with col3:
        if st.button("[CAT] VIEW LOGS", use_container_width=True):
            log_path = 'logs/rlfi.log'
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = f.readlines()[-30:]
                st.code(''.join(logs), language="text")
            else:
                st.warning("No logs found yet")


def render_models():
    """Render the models browser page."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state_filter = st.multiselect(
            "Filter by State",
            options=['training', 'validation', 'paper_trading', 'promoted', 'culled'],
            default=[]
        )
    
    with col2:
        algo_filter = st.multiselect(
            "Filter by Algorithm",
            options=['ppo', 'sac'],
            default=[]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=['Created (Newest)', 'Created (Oldest)', 'Sharpe (High)', 'Return (High)', 'Trading Days']
        )
    
    st.markdown("---")
    
    df = get_all_models()
    
    if df is None or df.empty:
        st.warning("No models found in database. Run training first.")
        return
    
    if state_filter:
        df = df[df['current_state'].isin(state_filter)]
    if algo_filter:
        df = df[df['algorithm'].isin(algo_filter)]
    
    if sort_by == 'Created (Newest)':
        df = df.sort_values('created_at', ascending=False)
    elif sort_by == 'Created (Oldest)':
        df = df.sort_values('created_at', ascending=True)
    elif sort_by == 'Sharpe (High)':
        df = df.sort_values('backtest_expected_sharpe', ascending=False)
    elif sort_by == 'Return (High)':
        df = df.sort_values('backtest_expected_return', ascending=False)
    elif sort_by == 'Trading Days':
        df = df.sort_values('trading_days', ascending=False)
    
    st.markdown(f"### Showing {len(df)} models")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total", len(df))
    with col2:
        st.metric("Paper Trading", len(df[df['current_state'] == 'paper_trading']))
    with col3:
        st.metric("Promoted", len(df[df['current_state'] == 'promoted']))
    with col4:
        st.metric("Culled", len(df[df['current_state'] == 'culled']))
    with col5:
        avg_sharpe = df['backtest_expected_sharpe'].mean()
        st.metric("Avg Sharpe", f"{avg_sharpe:.2f}" if pd.notna(avg_sharpe) else "--")
    
    st.markdown("---")
    
    for _, row in df.iterrows():
        model_id = row['model_id']
        state = row['current_state']
        algo = row['algorithm'] or 'unknown'
        tickers = json.loads(row['tickers']) if row['tickers'] else []
        tickers_str = ', '.join(tickers[:4]) + ('...' if len(tickers) > 4 else '')
        
        sharpe = row.get('backtest_expected_sharpe')
        ret = row.get('backtest_expected_return')
        trading_days = row.get('trading_days', 0) or 0
        cum_return = row.get('cumulative_return')
        
        state_html = format_state(state)
        sharpe_html = format_metric(sharpe, threshold_good=0.5, threshold_bad=0)
        ret_html = format_metric(ret, threshold_good=0, threshold_bad=-0.1, suffix='%', precision=1)
        
        paper_info = ""
        if trading_days > 0:
            cum_ret_html = format_metric(cum_return, threshold_good=0, threshold_bad=-0.05, suffix='%', precision=2)
            paper_info = f' | Paper: {int(trading_days)}d @ {cum_ret_html}'
        
        st.markdown(f'''
        <div class="model-card">
            <strong style="color: #ffcc00; font-size: 1.1rem;">{model_id}</strong>
            {state_html}
            <span style="color: #996600; margin-left: 1rem;">[{algo.upper()}]</span>
            <span style="color: #996600; margin-left: 0.5rem;">{tickers_str}</span>
            <span style="float: right;">
                Sharpe: {sharpe_html} | Return: {ret_html}{paper_info}
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
        with st.expander(f"Details: {model_id}"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Training Info**")
                st.text(f"Algorithm: {algo.upper()}")
                st.text(f"Timesteps: {row.get('training_timesteps', 'N/A'):,}" if row.get('training_timesteps') else "Timesteps: N/A")
                train_time = row.get('training_time_seconds')
                if train_time:
                    st.text(f"Training Time: {train_time/60:.1f} min")
                st.text(f"Tickers: {', '.join(tickers)}")
            
            with detail_col2:
                st.markdown("**Performance**")
                st.text(f"Backtest Sharpe: {sharpe:.3f}" if sharpe else "Backtest Sharpe: N/A")
                st.text(f"Backtest Return: {ret:.2f}%" if ret else "Backtest Return: N/A")
                if trading_days > 0:
                    st.text(f"Paper Trading Days: {int(trading_days)}")
                    st.text(f"Paper Return: {cum_return:.2f}%" if cum_return else "Paper Return: N/A")


def render_lineage():
    """Render the lineage tracker page."""
    models = get_all_models_with_lineage()
    
    if not models:
        st.warning("No models with lineage data found.")
        st.info("Lineage tracking requires model metadata files with parent_model_id fields.")
        return
    
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
    
    # Stats
    st.markdown("### Evolution Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_models = len(models)
    gen1_models = sum(1 for m in models.values() if m.get('generation', 1) == 1)
    offspring_models = total_models - gen1_models
    max_gen = max((m.get('generation', 1) for m in models.values()), default=1)
    
    with col1:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{total_models}</div>
            <div class="stat-label">Total Models</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{gen1_models}</div>
            <div class="stat-label">Original (G1)</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{offspring_models}</div>
            <div class="stat-label">Offspring</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{max_gen}</div>
            <div class="stat-label">Max Generation</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Family Trees")
    st.markdown("""
    <p style="color: #996600; font-size: 0.85rem;">
    Legend: <span style="color:#44ff44">[+] Promoted</span> | 
    <span style="color:#cc3333">[X] Culled</span> | 
    <span style="color:#ffaa00">[>] Active</span> | 
    [ ] Other | G# = Generation | S: = Sharpe
    </p>
    """, unsafe_allow_html=True)
    
    families_with_offspring = [(r, children_map.get(r, [])) for r in roots if children_map.get(r)]
    families_without_offspring = [r for r in roots if not children_map.get(r)]
    
    if families_with_offspring:
        st.markdown("#### Families with Offspring")
        
        for root, children in families_with_offspring:
            def render_family(model_id, level=0):
                lines = [render_tree_node(model_id, models, level)]
                for child in children_map.get(model_id, []):
                    lines.extend(render_family(child, level + 1))
                return lines
            
            tree_lines = render_family(root)
            tree_html = '<br>'.join(tree_lines)
            
            with st.expander(f"Family: {root} ({len(children)} direct offspring)", expanded=True):
                st.markdown(f'<div class="family-tree">{tree_html}</div>', unsafe_allow_html=True)
    
    if families_without_offspring:
        st.markdown("#### Standalone Models (No Offspring)")
        
        by_gen = defaultdict(list)
        for model_id in families_without_offspring:
            gen = models[model_id].get('generation', 1)
            by_gen[gen].append(model_id)
        
        for gen in sorted(by_gen.keys()):
            with st.expander(f"Generation {gen} ({len(by_gen[gen])} models)"):
                for model_id in by_gen[gen]:
                    node_html = render_tree_node(model_id, models, 0)
                    st.markdown(node_html, unsafe_allow_html=True)


def render_trading():
    """Render the trading history page."""
    daily_df, agg_df = get_trading_data()
    
    if daily_df is None or daily_df.empty:
        st.warning("No trading history found.")
        st.info("Paper trading data will appear here once models start trading.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.selectbox(
            "Date Range",
            options=['Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'All Time']
        )
    
    with col2:
        view_mode = st.selectbox(
            "View Mode",
            options=['Aggregated by Day', 'By Model', 'Raw Data']
        )
    
    now = datetime.now()
    if date_range == 'Last 7 Days':
        cutoff = now - timedelta(days=7)
    elif date_range == 'Last 30 Days':
        cutoff = now - timedelta(days=30)
    elif date_range == 'Last 90 Days':
        cutoff = now - timedelta(days=90)
    else:
        cutoff = datetime.min
    
    filtered_daily = daily_df[daily_df['trading_date'] >= cutoff]
    filtered_agg = agg_df[agg_df['trading_date'] >= cutoff] if not agg_df.empty else agg_df
    
    st.markdown("---")
    
    # Summary Stats
    total_trading_days = filtered_daily['trading_date'].nunique()
    total_models = filtered_daily['model_id'].nunique()
    total_trades = filtered_daily['trades_executed'].sum() if 'trades_executed' in filtered_daily.columns else 0
    avg_daily_return = filtered_daily['daily_return'].mean() if 'daily_return' in filtered_daily.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{total_trading_days}</div>
            <div class="stat-label">Trading Days</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{total_models}</div>
            <div class="stat-label">Models Active</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{int(total_trades):,}</div>
            <div class="stat-label">Total Trades</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        return_color = '#44ff44' if avg_daily_return >= 0 else '#cc3333'
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number" style="color: {return_color};">{avg_daily_return:.3f}%</div>
            <div class="stat-label">Avg Daily Return</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if view_mode == 'Aggregated by Day':
        if not filtered_agg.empty:
            chart_data = filtered_agg[['trading_date', 'avg_return']].copy()
            chart_data = chart_data.set_index('trading_date').sort_index()
            
            st.markdown("#### Average Daily Return Across All Models")
            st.bar_chart(chart_data['avg_return'], color='#ffaa00')
            
            st.markdown("#### Daily Summary")
            display_df = filtered_agg[['trading_date', 'active_models', 'total_trades', 'avg_return']].copy()
            display_df['trading_date'] = display_df['trading_date'].dt.strftime('%Y-%m-%d')
            display_df['avg_return'] = display_df['avg_return'].apply(lambda x: f"{x:.3f}%")
            display_df.columns = ['Date', 'Active Models', 'Trades', 'Avg Return']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    elif view_mode == 'By Model':
        model_summary = filtered_daily.groupby('model_id').agg({
            'trading_date': 'count',
            'daily_return': 'mean',
            'cumulative_return': 'last',
            'trades_executed': 'sum'
        }).reset_index()
        
        model_summary.columns = ['Model', 'Trading Days', 'Avg Daily Return', 'Cumulative Return', 'Total Trades']
        model_summary = model_summary.sort_values('Cumulative Return', ascending=False)
        
        for _, row in model_summary.iterrows():
            model_id = row['Model']
            days = row['Trading Days']
            cum_ret = row['Cumulative Return']
            trades = row['Total Trades']
            
            ret_color = '#44ff44' if cum_ret and cum_ret >= 0 else '#cc3333'
            border_color = '#44ff44' if cum_ret and cum_ret >= 0 else '#cc3333'
            
            st.markdown(f'''
            <div class="model-card" style="border-left-color: {border_color};">
                <strong style="color: #ffcc00;">{model_id}</strong>
                <span style="color: #996600; margin-left: 1rem;">{int(days)} days</span>
                <span style="color: #996600; margin-left: 1rem;">{int(trades) if trades else 0} trades</span>
                <span style="float: right;">
                    <span style="color: {ret_color};">{cum_ret:.2f}%</span> cumulative
                </span>
            </div>
            ''', unsafe_allow_html=True)
    
    else:
        display_cols = ['trading_date', 'model_id', 'portfolio_value', 'daily_return', 'cumulative_return', 'trades_executed']
        available_cols = [c for c in display_cols if c in filtered_daily.columns]
        
        display_df = filtered_daily[available_cols].copy()
        display_df['trading_date'] = display_df['trading_date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<div class="main-header">RLFI COLOSSEUM</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #996600; font-size: 0.8rem; margin-top: -0.5rem;">REINFORCEMENT LEARNING FINANCIAL INTELLIGENCE // TRADING ARENA v1.0</p>', unsafe_allow_html=True)

# Top navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["DASHBOARD", "MODELS", "LINEAGE", "TRADING"])

with tab1:
    render_dashboard()

with tab2:
    render_models()

with tab3:
    render_lineage()

with tab4:
    render_trading()
