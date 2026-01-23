"""
Trading History - Aggregated view of all paper trading activity
"""
import streamlit as st
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Trading History | RLFI",
    page_icon="📈",
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
    
    .stat-box {
        background: #1a1a1a;
        border: 1px solid #cc8800;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-family: 'VT323', monospace !important;
        font-size: 2rem;
        color: #ffcc00;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #996600;
        text-transform: uppercase;
    }
    
    .stat-positive .stat-number { color: #44ff44; }
    .stat-negative .stat-number { color: #cc3333; }
    
    .day-row {
        background: #1a1a1a;
        border: 1px solid #996600;
        border-left: 3px solid #ffaa00;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
    }
    .day-row-positive { border-left-color: #44ff44; }
    .day-row-negative { border-left-color: #cc3333; }
    
    h1, h2, h3 { color: #ffaa00 !important; font-family: 'VT323', monospace !important; }
    .stMarkdown, p, span, div { color: #cc8800; }
    hr { border-color: #996600 !important; border-style: dashed !important; }
    
    .stSelectbox label, .stDateInput label { color: #cc8800 !important; }
</style>
""", unsafe_allow_html=True)


def get_trading_data():
    """Get all paper trading data."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return None, None
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Daily trading log
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
        
        # Aggregate by date
        if not daily_df.empty:
            daily_df['trading_date'] = pd.to_datetime(daily_df['trading_date'])
            
            agg_query = '''
                SELECT 
                    trading_date,
                    COUNT(DISTINCT model_id) as active_models,
                    SUM(trades_executed) as total_trades,
                    AVG(daily_return) as avg_return,
                    SUM(portfolio_value) as total_portfolio_value
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
        st.error(f"Database error: {e}")
        return None, None


def get_model_list():
    """Get list of models that have trading history."""
    db_path = 'autotest_strategies.db'
    if not os.path.exists(db_path):
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT model_id FROM paper_trading_daily_log ORDER BY model_id')
        models = [row[0] for row in cursor.fetchall()]
        conn.close()
        return models
    except:
        return []


# Page Header
st.markdown('<div class="page-header">[TRADES] TRADING HISTORY</div>', unsafe_allow_html=True)

# Load data
daily_df, agg_df = get_trading_data()

if daily_df is None or daily_df.empty:
    st.warning("No trading history found.")
    st.info("Paper trading data will appear here once models start trading.")
    st.stop()

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    model_list = ['All Models'] + get_model_list()
    selected_model = st.selectbox("Filter by Model", options=model_list)

with col2:
    date_range = st.selectbox(
        "Date Range",
        options=['Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'All Time']
    )

with col3:
    view_mode = st.selectbox(
        "View Mode",
        options=['Aggregated by Day', 'By Model', 'Raw Data']
    )

# Apply date filter
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

# Apply model filter
if selected_model != 'All Models':
    filtered_daily = filtered_daily[filtered_daily['model_id'] == selected_model]

st.markdown("---")

# Summary Stats
st.markdown("### Summary")

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
    return_class = 'stat-positive' if avg_daily_return >= 0 else 'stat-negative'
    st.markdown(f'''
    <div class="stat-box {return_class}">
        <div class="stat-number">{avg_daily_return:.3f}%</div>
        <div class="stat-label">Avg Daily Return</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Main content based on view mode
if view_mode == 'Aggregated by Day':
    st.markdown("### Daily Aggregate Performance")
    
    if not filtered_agg.empty:
        # Chart
        chart_data = filtered_agg[['trading_date', 'avg_return']].copy()
        chart_data = chart_data.set_index('trading_date').sort_index()
        
        st.markdown("#### Average Daily Return Across All Models")
        st.bar_chart(chart_data['avg_return'], color='#ffaa00')
        
        # Table
        st.markdown("#### Daily Summary")
        display_df = filtered_agg[['trading_date', 'active_models', 'total_trades', 'avg_return']].copy()
        display_df['trading_date'] = display_df['trading_date'].dt.strftime('%Y-%m-%d')
        display_df['avg_return'] = display_df['avg_return'].apply(lambda x: f"{x:.3f}%")
        display_df.columns = ['Date', 'Active Models', 'Trades', 'Avg Return']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

elif view_mode == 'By Model':
    st.markdown("### Performance by Model")
    
    # Group by model
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
        avg_ret = row['Avg Daily Return']
        cum_ret = row['Cumulative Return']
        trades = row['Total Trades']
        
        row_class = 'day-row-positive' if cum_ret and cum_ret >= 0 else 'day-row-negative'
        ret_color = '#44ff44' if cum_ret and cum_ret >= 0 else '#cc3333'
        
        st.markdown(f'''
        <div class="day-row {row_class}">
            <strong style="color: #ffcc00;">{model_id}</strong>
            <span style="color: #996600; margin-left: 1rem;">{int(days)} days</span>
            <span style="color: #996600; margin-left: 1rem;">{int(trades) if trades else 0} trades</span>
            <span style="float: right;">
                <span style="color: {ret_color};">{cum_ret:.2f}%</span> cumulative
            </span>
        </div>
        ''', unsafe_allow_html=True)

else:  # Raw Data
    st.markdown("### Raw Trading Data")
    
    display_cols = ['trading_date', 'model_id', 'portfolio_value', 'daily_return', 'cumulative_return', 'trades_executed']
    available_cols = [c for c in display_cols if c in filtered_daily.columns]
    
    display_df = filtered_daily[available_cols].copy()
    display_df['trading_date'] = display_df['trading_date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Positions breakdown if available
if 'positions' in filtered_daily.columns and view_mode != 'Raw Data':
    st.markdown("---")
    st.markdown("### Position Analysis")
    
    # Parse positions and count occurrences
    all_positions = []
    for pos_json in filtered_daily['positions'].dropna():
        try:
            positions = json.loads(pos_json) if isinstance(pos_json, str) else pos_json
            if isinstance(positions, dict):
                all_positions.extend(positions.keys())
            elif isinstance(positions, list):
                all_positions.extend(positions)
        except:
            continue
    
    if all_positions:
        from collections import Counter
        pos_counts = Counter(all_positions)
        
        st.markdown("#### Most Traded Symbols")
        pos_df = pd.DataFrame(pos_counts.most_common(10), columns=['Symbol', 'Occurrences'])
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #996600; font-size: 0.8rem;">TRADING HISTORY // RLFI COLOSSEUM</p>', unsafe_allow_html=True)
