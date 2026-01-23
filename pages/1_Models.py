"""
Models Browser - View all models, their status, and performance metrics
"""
import streamlit as st
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Models | RLFI",
    page_icon="🤖",
    layout="wide"
)

# Reuse the amber terminal CSS
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
    
    .model-row {
        background: #1a1a1a;
        border: 1px solid #996600;
        border-left: 3px solid #ffaa00;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #cc8800;
    }
    
    .model-row:hover {
        border-color: #ffaa00;
        background: #1f1a14;
    }
    
    .state-training { color: #cccc33; }
    .state-validation { color: #cccc33; }
    .state-paper_trading { color: #ffaa00; }
    .state-promoted { color: #44ff44; }
    .state-culled { color: #cc3333; }
    
    .metric-good { color: #44ff44; }
    .metric-bad { color: #cc3333; }
    .metric-neutral { color: #cc8800; }
    
    h1, h2, h3 { color: #ffaa00 !important; font-family: 'VT323', monospace !important; }
    .stMarkdown, p, span, div { color: #cc8800; }
    
    .stSelectbox label, .stMultiSelect label { color: #cc8800 !important; }
    
    .stDataFrame { background: #1a1a1a !important; }
    
    hr { border-color: #996600 !important; border-style: dashed !important; }
    
    .stButton > button {
        background: #141414 !important;
        color: #cc8800 !important;
        border: 1px solid #cc8800 !important;
        border-radius: 0 !important;
    }
    .stButton > button:hover {
        background: #1a1100 !important;
        color: #ffcc00 !important;
        border-color: #ffcc00 !important;
    }
</style>
""", unsafe_allow_html=True)


def get_all_models():
    """Fetch all models with their lifecycle data and metrics."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get models from lifecycle table with strategy info
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
        
        # Get paper trading performance for each model
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
        
        # Merge performance data
        if not perf_df.empty:
            df = df.merge(perf_df, on='model_id', how='left')
        
        conn.close()
        return df
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


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


# Page Header
st.markdown('<div class="page-header">[DATABASE] MODEL REGISTRY</div>', unsafe_allow_html=True)

# Filters
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

# Load data
df = get_all_models()

if df is None or df.empty:
    st.warning("No models found in database. Run training first.")
else:
    # Apply filters
    if state_filter:
        df = df[df['current_state'].isin(state_filter)]
    if algo_filter:
        df = df[df['algorithm'].isin(algo_filter)]
    
    # Apply sorting
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
    
    # Summary stats
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
    
    # Model list
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
        
        # Build model card HTML
        state_html = format_state(state)
        sharpe_html = format_metric(sharpe, threshold_good=0.5, threshold_bad=0)
        ret_html = format_metric(ret, threshold_good=0, threshold_bad=-0.1, suffix='%', precision=1)
        
        paper_info = ""
        if trading_days > 0:
            cum_ret_html = format_metric(cum_return, threshold_good=0, threshold_bad=-0.05, suffix='%', precision=2)
            paper_info = f' | Paper: {int(trading_days)}d @ {cum_ret_html}'
        
        st.markdown(f'''
        <div class="model-row">
            <strong style="color: #ffcc00; font-size: 1.1rem;">{model_id}</strong>
            {state_html}
            <span style="color: #996600; margin-left: 1rem;">[{algo.upper()}]</span>
            <span style="color: #996600; margin-left: 0.5rem;">{tickers_str}</span>
            <span style="float: right;">
                Sharpe: {sharpe_html} | Return: {ret_html}{paper_info}
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Expandable details
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
            
            # Link to detail page
            st.markdown(f"[View Full Details →](Model_Detail?model={model_id})")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #996600; font-size: 0.8rem;">MODEL REGISTRY // RLFI COLOSSEUM</p>', unsafe_allow_html=True)
