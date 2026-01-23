"""
Model Detail - Deep dive on a single model's performance and trading history
"""
import streamlit as st
import pandas as pd
import sqlite3
import json
import os
import yaml
from datetime import datetime

st.set_page_config(
    page_title="Model Detail | RLFI",
    page_icon="📊",
    layout="wide"
)

# Amber terminal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');
    
    *:not([data-testid="collapsedControl"]):not([class*="icon"]):not(svg):not(path) { font-family: 'Share Tech Mono', 'Courier New', monospace !important; }
    
    .stApp {
        background-color: #141414;
        background-image: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(20, 10, 0, 0.3) 2px, rgba(20, 10, 0, 0.3) 4px);
    }
    
    .page-header {
        font-family: 'VT323', monospace !important;
        font-size: 2.5rem;
        color: #ffaa00;
        border-bottom: 1px dashed #996600;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: #1a1a1a;
        border: 1px solid #996600;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-label { color: #996600; font-size: 0.85rem; }
    .metric-value { color: #ffcc00; font-size: 1.5rem; font-family: 'VT323', monospace !important; }
    .metric-good { color: #44ff44; }
    .metric-bad { color: #cc3333; }
    
    h1, h2, h3 { color: #ffaa00 !important; font-family: 'VT323', monospace !important; }
    .stMarkdown, p, span, div { color: #cc8800; }
    hr { border-color: #996600 !important; border-style: dashed !important; }
    
    .stSelectbox label { color: #cc8800 !important; }
    
    .trade-row {
        background: #1a1a1a;
        border-left: 2px solid #ffaa00;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .trade-positive { border-left-color: #44ff44; }
    .trade-negative { border-left-color: #cc3333; }
</style>
""", unsafe_allow_html=True)


def get_model_list():
    """Get list of all model IDs."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT model_id FROM model_lifecycle ORDER BY created_at DESC')
        models = [row[0] for row in cursor.fetchall()]
        conn.close()
        return models
    except:
        return []


def get_model_details(model_id):
    """Get comprehensive model details."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Lifecycle data
        lifecycle_query = '''
            SELECT * FROM model_lifecycle WHERE model_id = ?
        '''
        lifecycle_df = pd.read_sql_query(lifecycle_query, conn, params=[model_id])
        
        # Strategy data
        strategy_query = '''
            SELECT * FROM strategies WHERE model_id = ?
        '''
        strategy_df = pd.read_sql_query(strategy_query, conn, params=[model_id])
        
        # Backtest results
        backtest_query = '''
            SELECT * FROM backtest_results WHERE model_id = ?
        '''
        backtest_df = pd.read_sql_query(backtest_query, conn, params=[model_id])
        
        # Paper trading daily log
        trading_query = '''
            SELECT * FROM paper_trading_daily_log 
            WHERE model_id = ? 
            ORDER BY trading_date ASC
        '''
        trading_df = pd.read_sql_query(trading_query, conn, params=[model_id])
        
        # Culling decision if any
        culling_query = '''
            SELECT * FROM culling_decisions WHERE model_id = ?
        '''
        culling_df = pd.read_sql_query(culling_query, conn, params=[model_id])
        
        conn.close()
        
        return {
            'lifecycle': lifecycle_df.iloc[0].to_dict() if not lifecycle_df.empty else None,
            'strategy': strategy_df.iloc[0].to_dict() if not strategy_df.empty else None,
            'backtest': backtest_df.iloc[0].to_dict() if not backtest_df.empty else None,
            'trading_history': trading_df,
            'culling': culling_df.iloc[0].to_dict() if not culling_df.empty else None
        }
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


def get_model_metadata(model_id):
    """Load model metadata from YAML file."""
    paths = [
        f'autotest_models/{model_id}_metadata.yaml',
        f'models/{model_id}_metadata.yaml',
        f'champion_models/{model_id}_metadata.yaml'
    ]
    
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    return None


# Page Header
st.markdown('<div class="page-header">[INSPECT] MODEL DETAIL</div>', unsafe_allow_html=True)

# Model selector
models = get_model_list()

if not models:
    st.warning("No models found in database.")
    st.stop()

# Check for URL parameter
query_params = st.query_params
selected_model = query_params.get('model', models[0] if models else None)

selected_model = st.selectbox(
    "Select Model",
    options=models,
    index=models.index(selected_model) if selected_model in models else 0
)

if not selected_model:
    st.stop()

st.markdown("---")

# Load data
data = get_model_details(selected_model)
metadata = get_model_metadata(selected_model)

if not data or not data['lifecycle']:
    st.error(f"No data found for model: {selected_model}")
    st.stop()

lifecycle = data['lifecycle']
strategy = data['strategy'] or {}
backtest = data['backtest'] or {}
trading_history = data['trading_history']
culling = data['culling']

# Header with state
state = lifecycle.get('current_state', 'unknown')
state_colors = {
    'training': '#cccc33',
    'validation': '#cccc33', 
    'paper_trading': '#ffaa00',
    'promoted': '#44ff44',
    'culled': '#cc3333'
}
state_color = state_colors.get(state, '#cc8800')

st.markdown(f'''
<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
    <span style="font-family: VT323, monospace; font-size: 2rem; color: #ffcc00;">{selected_model}</span>
    <span style="background: #1a1a1a; border: 1px solid {state_color}; color: {state_color}; padding: 0.25rem 0.75rem;">
        [{state.upper()}]
    </span>
</div>
''', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trading History", "Backtest Results", "Lineage"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Configuration")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        
        algo = strategy.get('algorithm', metadata.get('algorithm', 'N/A') if metadata else 'N/A')
        st.markdown(f"**Algorithm:** `{algo.upper() if algo else 'N/A'}`")
        
        tickers = json.loads(lifecycle.get('tickers', '[]')) if lifecycle.get('tickers') else []
        st.markdown(f"**Tickers:** `{', '.join(tickers)}`")
        
        timesteps = strategy.get('training_timesteps') or (metadata.get('total_timesteps') if metadata else None)
        st.markdown(f"**Timesteps:** `{timesteps:,}`" if timesteps else "**Timesteps:** N/A")
        
        train_time = strategy.get('training_time_seconds')
        if train_time:
            st.markdown(f"**Training Time:** `{train_time/60:.1f} minutes`")
        
        created = lifecycle.get('created_at', 'N/A')
        st.markdown(f"**Created:** `{created}`")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Hyperparameters
        if strategy:
            st.markdown("### Hyperparameters")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            params = ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'clip_range', 'ent_coef']
            for param in params:
                val = strategy.get(param)
                if val is not None:
                    st.markdown(f"**{param}:** `{val}`")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Performance Metrics")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        
        # Backtest metrics
        sharpe = lifecycle.get('backtest_expected_sharpe') or backtest.get('sharpe_ratio')
        ret = lifecycle.get('backtest_expected_return') or backtest.get('total_return')
        
        sharpe_color = '#44ff44' if sharpe and sharpe > 0.5 else ('#cc3333' if sharpe and sharpe < 0 else '#ffcc00')
        ret_color = '#44ff44' if ret and ret > 0 else ('#cc3333' if ret and ret < 0 else '#ffcc00')
        
        st.markdown(f'<div class="metric-label">BACKTEST SHARPE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: {sharpe_color};">{sharpe:.3f}</div>' if sharpe else '<div class="metric-value">--</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-label">BACKTEST RETURN</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: {ret_color};">{ret:.2f}%</div>' if ret else '<div class="metric-value">--</div>', unsafe_allow_html=True)
        
        # Paper trading metrics
        if not trading_history.empty:
            trading_days = len(trading_history)
            cum_return = trading_history['cumulative_return'].iloc[-1] if 'cumulative_return' in trading_history.columns else None
            
            cum_color = '#44ff44' if cum_return and cum_return > 0 else ('#cc3333' if cum_return and cum_return < 0 else '#ffcc00')
            
            st.markdown("---")
            st.markdown(f'<div class="metric-label">PAPER TRADING DAYS</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{trading_days}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-label">PAPER RETURN</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color: {cum_color};">{cum_return:.2f}%</div>' if cum_return else '<div class="metric-value">--</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Culling info if applicable
        if culling:
            st.markdown("### Culling Decision")
            st.markdown('<div class="info-box" style="border-color: #cc3333;">', unsafe_allow_html=True)
            st.markdown(f"**Decision:** `{culling.get('decision', 'N/A')}`")
            st.markdown(f"**Reason:** {culling.get('reason', 'N/A')}")
            st.markdown(f"**Date:** `{culling.get('decision_date', 'N/A')}`")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Paper Trading History")
    
    if trading_history.empty:
        st.info("No paper trading history for this model.")
    else:
        # Chart
        if 'cumulative_return' in trading_history.columns:
            chart_data = trading_history[['trading_date', 'cumulative_return']].copy()
            chart_data['trading_date'] = pd.to_datetime(chart_data['trading_date'])
            chart_data = chart_data.set_index('trading_date')
            
            st.markdown("#### Cumulative Return")
            st.line_chart(chart_data['cumulative_return'], color='#ffaa00')
        
        # Daily returns chart
        if 'daily_return' in trading_history.columns:
            st.markdown("#### Daily Returns")
            daily_data = trading_history[['trading_date', 'daily_return']].copy()
            daily_data['trading_date'] = pd.to_datetime(daily_data['trading_date'])
            daily_data = daily_data.set_index('trading_date')
            st.bar_chart(daily_data['daily_return'], color='#ffaa00')
        
        # Table
        st.markdown("#### Daily Log")
        display_cols = ['trading_date', 'portfolio_value', 'daily_return', 'cumulative_return', 'trades_executed']
        available_cols = [c for c in display_cols if c in trading_history.columns]
        st.dataframe(
            trading_history[available_cols].sort_values('trading_date', ascending=False),
            use_container_width=True,
            hide_index=True
        )

with tab3:
    st.markdown("### Backtest Results")
    
    if not backtest:
        st.info("No backtest results for this model.")
    else:
        col1, col2, col3 = st.columns(3)
        
        metrics = [
            ('Sharpe Ratio', backtest.get('sharpe_ratio'), 0.5),
            ('Sortino Ratio', backtest.get('sortino_ratio'), 0.5),
            ('Calmar Ratio', backtest.get('calmar_ratio'), 0.5),
            ('Total Return', backtest.get('total_return'), 0, '%'),
            ('Max Drawdown', backtest.get('max_drawdown'), -0.25, '%'),
            ('Win Rate', backtest.get('win_rate'), 0.5, '%'),
            ('Volatility', backtest.get('volatility'), None, '%'),
            ('Final Value', backtest.get('final_value'), None, '$'),
        ]
        
        for i, (name, value, threshold, *suffix) in enumerate(metrics):
            col = [col1, col2, col3][i % 3]
            with col:
                if value is not None:
                    suffix_str = suffix[0] if suffix else ''
                    if threshold is not None:
                        color = '#44ff44' if value >= threshold else '#cc3333'
                    else:
                        color = '#ffcc00'
                    st.markdown(f'''
                    <div class="info-box">
                        <div class="metric-label">{name.upper()}</div>
                        <div class="metric-value" style="color: {color};">{value:.2f}{suffix_str}</div>
                    </div>
                    ''', unsafe_allow_html=True)

with tab4:
    st.markdown("### Model Lineage")
    
    if metadata:
        parent = metadata.get('parent_model_id')
        generation = metadata.get('generation', 1)
        lineage = metadata.get('lineage', [])
        
        st.markdown(f"**Generation:** `{generation}`")
        
        if parent:
            st.markdown(f"**Parent Model:** `{parent}`")
            st.markdown(f"[View Parent →](Model_Detail?model={parent})")
        else:
            st.markdown("**Parent Model:** None (Original)")
        
        if lineage:
            st.markdown("#### Ancestry Chain")
            for i, ancestor in enumerate(lineage):
                indent = "  " * i
                st.markdown(f"`{indent}└─ {ancestor}`")
    else:
        st.info("No lineage metadata available for this model.")
        st.markdown("*Lineage tracking requires model metadata files.*")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #996600; font-size: 0.8rem;">MODEL INSPECTOR // RLFI COLOSSEUM</p>', unsafe_allow_html=True)
