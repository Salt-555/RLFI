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
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
 
# Page config
st.set_page_config(
    page_title="RLFI Colosseum",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');
    
    * {
        font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    }
    
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
        font-size: 3rem;
        text-align: center;
        padding: 1rem;
        background: #141414;
        color: #ffaa00;
        border: 1px solid #ffaa00;
        margin-bottom: 1.5rem;
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
    
    .status-running {
        background: #141414;
        color: #ffcc00;
        padding: 0.5rem 1rem;
        border: 1px solid #ffcc00;
        font-weight: 400;
        font-family: 'Share Tech Mono', monospace !important;
    }
    .status-running::before {
        content: "[ACTIVE] ";
    }
    
    .status-stopped {
        background: #141414;
        color: #cc3333;
        padding: 0.5rem 1rem;
        border: 1px solid #cc3333;
        font-weight: 400;
        font-family: 'Share Tech Mono', monospace !important;
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
        font-family: 'Share Tech Mono', monospace !important;
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
        font-family: 'Share Tech Mono', monospace !important;
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
        border: 1px solid #ff9900;
        color: #ff9900;
        padding: 1rem;
        text-align: center;
    }
    .stat-box-success .stat-number { color: #ff9900; }
    .stat-box-success .stat-label { color: #ff9900; }
    
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
        background: #0a0a0a !important;
        color: #cc8800 !important;
        border: 1px solid #cc8800 !important;
        border-radius: 0 !important;
        font-family: 'Share Tech Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background: #1a1100 !important;
        color: #ffcc00 !important;
        border-color: #ffcc00 !important;
    }
    
    .stAlert {
        background: #0d0d0d !important;
        border: 1px solid #996600 !important;
        color: #cc8800 !important;
    }
    
    code {
        background: #0d0d0d !important;
        color: #ffcc00 !important;
        border: 1px solid #996600 !important;
    }
    
    .terminal-line {
        font-family: 'Share Tech Mono', monospace;
        color: #cc8800;
        padding: 2px 0;
    }
    .terminal-line::before {
        content: "$ ";
        color: #996600;
    }
</style>
""", unsafe_allow_html=True)
 
 
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
 
        # Check if model_lifecycle table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_lifecycle'")
        if not cursor.fetchone():
            conn.close()
            return None
 
        # Get lifecycle counts
        states = {}
        for state in ['training', 'validation', 'paper_trading', 'promoted', 'culled']:
            cursor.execute('SELECT COUNT(*) FROM model_lifecycle WHERE current_state = ?', (state,))
            result = cursor.fetchone()
            states[state] = result[0] if result else 0
 
        # Get paper trading models
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
 
        # Get recent trained models
        cursor.execute('''
            SELECT model_id, algorithm, tickers, run_date
            FROM strategies
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        recent_models = cursor.fetchall()
 
        # Get promoted models
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
 
 
def get_next_events():
    """Calculate next scheduled events."""
    now = datetime.now()

    # Next training (daily at 2AM)
    next_training = now.replace(hour=2, minute=0, second=0, microsecond=0)
    if now.hour >= 2:
        next_training += timedelta(days=1)

    # Next culling (Saturday 6PM)
    days_until_saturday = (5 - now.weekday()) % 7
    if days_until_saturday == 0 and now.hour >= 18:
        days_until_saturday = 7
    next_culling = now.replace(hour=18, minute=0, second=0) + timedelta(days=days_until_saturday)

    return next_training, next_culling
 
 
# Header
st.markdown('<div class="main-header">RLFI COLOSSEUM</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #996600; font-size: 0.8rem; margin-top: -1rem;">REINFORCEMENT LEARNING FINANCIAL INTELLIGENCE // TRADING ARENA v1.0</p>', unsafe_allow_html=True)
 
# Daemon Status
daemon_running = get_daemon_status()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if daemon_running:
        st.markdown('<div style="text-align: center;"><span class="status-running">DAEMON RUNNING</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center;"><span class="status-stopped">DAEMON STOPPED</span></div>', unsafe_allow_html=True)
 
st.markdown("---")
 
# Load data
data = get_lifecycle_data()
 
if data is None:
    st.warning("⏳ Waiting for first training cycle...")
    st.info("The daemon is running but hasn't completed a training cycle yet. Training runs every Sunday at 2AM.")
 
    next_training, next_culling = get_next_events()
    st.markdown(f"**Next Training:** {next_training.strftime('%A %B %d at %I:%M %p')}")
 
elif 'error' in data:
    st.error(f"Database error: {data['error']}")
 
else:
    # Stats Row
    col1, col2, col3, col4, col5 = st.columns(5)
 
    states = data.get('states', {})
 
    with col1:
        st.markdown(f'''
        <div class="stat-box">
            <div class="stat-number">{states.get('paper_trading', 0)}</div>
            <div class="stat-label">&gt; Paper Trading</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div class="stat-box-success">
            <div class="stat-number">{states.get('promoted', 0)}</div>
            <div class="stat-label">&gt; Promoted</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="stat-box-warn">
            <div class="stat-number">{states.get('validation', 0)}</div>
            <div class="stat-label">&gt; Validation</div>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        st.markdown(f'''
        <div class="stat-box-danger">
            <div class="stat-number">{states.get('culled', 0)}</div>
            <div class="stat-label">&gt; Culled</div>
        </div>
        ''', unsafe_allow_html=True)

    with col5:
        total = sum(states.values())
        st.markdown(f'''
        <div class="stat-box-muted">
            <div class="stat-number">{total}</div>
            <div class="stat-label">&gt; Total Models</div>
        </div>
        ''', unsafe_allow_html=True)
 
    st.markdown("---")
 
    # Main Content
    col_left, col_right = st.columns([2, 1])
 
    with col_left:
        st.markdown("### [ACTIVE] PAPER TRADING MODELS")
 
        if data.get('paper_trading_models'):
            for model in data['paper_trading_models']:
                tickers_str = ', '.join(model['tickers'][:3]) if model['tickers'] else 'N/A'
                st.markdown(f'''
                <div class="model-card">
                    <strong style="font-size: 1.1rem; color: #ffcc00;">{model['model_id']}</strong>
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
 
# Footer with controls
st.markdown("---")
col1, col2, col3 = st.columns(3)
 
with col1:
    if st.button("[F5] REFRESH", use_container_width=True):
        st.rerun()
 
with col2:
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
        try:
            log_path = 'logs/rlfi.log'
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = f.readlines()[-30:]
                st.code(''.join(logs), language="text")
            else:
                st.warning("No logs found yet")
        except Exception as e:
            st.warning(f"Could not read logs: {e}")
 