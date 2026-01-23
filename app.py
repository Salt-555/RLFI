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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e94560;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        letter-spacing: 2px;
    }
    .status-running {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .status-stopped {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .model-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
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
 
    # Next training (Sunday 2AM)
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour >= 2:
        days_until_sunday = 7
    next_training = now.replace(hour=2, minute=0, second=0) + timedelta(days=days_until_sunday)
 
    # Next culling (Saturday 6PM)
    days_until_saturday = (5 - now.weekday()) % 7
    if days_until_saturday == 0 and now.hour >= 18:
        days_until_saturday = 7
    next_culling = now.replace(hour=18, minute=0, second=0) + timedelta(days=days_until_saturday)
 
    return next_training, next_culling
 
 
# Header
st.markdown('<div class="main-header">🏛️ RLFI COLOSSEUM</div>', unsafe_allow_html=True)
 
# Daemon Status
daemon_running = get_daemon_status()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if daemon_running:
        st.markdown('<div style="text-align: center;"><span class="status-running">● DAEMON RUNNING</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center;"><span class="status-stopped">● DAEMON STOPPED</span></div>', unsafe_allow_html=True)
 
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
            <div class="stat-label">Paper Trading</div>
        </div>
        ''', unsafe_allow_html=True)
 
    with col2:
        st.markdown(f'''
        <div class="stat-box" style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);">
            <div class="stat-number">{states.get('promoted', 0)}</div>
            <div class="stat-label">Promoted</div>
        </div>
        ''', unsafe_allow_html=True)
 
    with col3:
        st.markdown(f'''
        <div class="stat-box" style="background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);">
            <div class="stat-number">{states.get('validation', 0)}</div>
            <div class="stat-label">In Validation</div>
        </div>
        ''', unsafe_allow_html=True)
 
    with col4:
        st.markdown(f'''
        <div class="stat-box" style="background: linear-gradient(135deg, #e17055 0%, #d63031 100%);">
            <div class="stat-number">{states.get('culled', 0)}</div>
            <div class="stat-label">Culled</div>
        </div>
        ''', unsafe_allow_html=True)
 
    with col5:
        total = sum(states.values())
        st.markdown(f'''
        <div class="stat-box" style="background: linear-gradient(135deg, #636e72 0%, #2d3436 100%);">
            <div class="stat-number">{total}</div>
            <div class="stat-label">Total Models</div>
        </div>
        ''', unsafe_allow_html=True)
 
    st.markdown("---")
 
    # Main Content
    col_left, col_right = st.columns([2, 1])
 
    with col_left:
        st.markdown("### 📊 Active Paper Trading Models")
 
        if data.get('paper_trading_models'):
            for model in data['paper_trading_models']:
                tickers_str = ', '.join(model['tickers'][:3]) if model['tickers'] else 'N/A'
                st.markdown(f'''
                <div class="model-card">
                    <strong style="font-size: 1.1rem;">{model['model_id']}</strong>
                    <span style="color: #666; margin-left: 1rem;">{tickers_str}</span>
                    <span style="float: right; color: #666;">Day {model['trading_days']}/10</span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No models currently paper trading")
 
        st.markdown("### 📜 Recently Trained Models")
 
        if data.get('recent_models'):
            for model in data['recent_models'][:5]:
                tickers = json.loads(model[2]) if model[2] else []
                tickers_str = ', '.join(tickers[:3]) if tickers else 'N/A'
                st.markdown(f'''
                <div class="model-card">
                    <strong>{model[0]}</strong>
                    <span style="color: #666; margin-left: 1rem;">{model[1].upper()}</span>
                    <span style="color: #888; margin-left: 1rem;">{tickers_str}</span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No models trained yet")
 
    with col_right:
        st.markdown("### ⏰ Schedule")
 
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
            st.markdown("### 🏆 Champions")
            for model in data['promoted_models']:
                st.success(f"**{model[0]}**")
 
# Footer with controls
st.markdown("---")
col1, col2, col3 = st.columns(3)
 
with col1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
 
with col2:
    if daemon_running:
        if st.button("⏹️ Stop Daemon", use_container_width=True):
            os.system("sudo systemctl stop rlfi")
            st.rerun()
    else:
        if st.button("▶️ Start Daemon", use_container_width=True):
            os.system("sudo systemctl start rlfi")
            st.rerun()
 
with col3:
    if st.button("📜 View Logs", use_container_width=True):
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
 