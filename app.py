"""
RL Trading Bot - Web UI
Streamlit application for training, loading, and paper trading models
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.callbacks import BaseCallback
import threading

# Page config
st.set_page_config(
    page_title="RL Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sidebar-logo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_running' not in st.session_state:
    st.session_state.training_running = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'loaded_model' not in st.session_state:
    st.session_state.loaded_model = None
if 'paper_trading_active' not in st.session_state:
    st.session_state.paper_trading_active = False
if 'paper_trading_data' not in st.session_state:
    st.session_state.paper_trading_data = []

# Header
st.markdown('<h1 class="main-header">RL Trading Bot Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-logo">RL TRADING BOT</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "Train Model", "Load Model", "Paper Trade", "Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    
    # Count available models
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        st.metric("Available Models", len(model_files))
    else:
        st.metric("Available Models", 0)
    
    if st.session_state.loaded_model:
        st.success("Model Loaded")
    else:
        st.info("No Model Loaded")
    
    if st.session_state.paper_trading_active:
        st.success("Paper Trading Active")
    else:
        st.info("Paper Trading Inactive")

# Home Page
if page == "Home":
    st.markdown("## Welcome to RL Trading Bot Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Train Model</h3>
            <p>Train new RL models with custom parameters, multiple algorithms, and real-time monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Load Model</h3>
            <p>Load and manage trained models, view performance metrics, and prepare for trading.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Paper Trade</h3>
            <p>Test models with paper trading, monitor performance in real-time, and analyze results.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Quick Start Guide")
    
    st.markdown("""
    1. **Train a Model**: Go to the Train Model page and configure your training parameters
    2. **Load Your Model**: Once trained, load it from the Load Model page
    3. **Paper Trade**: Test your model with paper trading before going live
    4. **Analyze**: Review performance metrics and refine your strategy
    """)
    
    st.markdown("---")
    st.markdown("### System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Python", "3.10", delta="Active")
    with col2:
        st.metric("FinRL", "0.3.8", delta="Installed")
    with col3:
        st.metric("PyTorch", "2.5.1", delta="ROCm")
    with col4:
        st.metric("Device", "CPU", delta="Ready")

# Train Model Page
elif page == "Train Model":
    st.markdown("## Train New Model")
    
    tab1, tab2, tab3 = st.tabs(["Data Configuration", "Training Parameters", "Train & Monitor"])
    
    with tab1:
        st.markdown("### Data Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tickers_input = st.text_input(
                "Stock Tickers (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN",
                help="Enter stock tickers separated by commas"
            )
            tickers = [t.strip().upper() for t in tickers_input.split(',')]
            
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=730),
                help="Training data start date"
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Training data end date"
            )
        
        with col2:
            tech_indicators = st.multiselect(
                "Technical Indicators",
                ["macd", "rsi_30", "cci_30", "dx_30", "boll_ub", "boll_lb", "close_30_sma"],
                default=["macd", "rsi_30", "cci_30", "dx_30"],
                help="Select technical indicators to use"
            )
            
            train_ratio = st.slider("Training Data Ratio", 0.5, 0.9, 0.7, 0.05)
            val_ratio = st.slider("Validation Data Ratio", 0.05, 0.3, 0.15, 0.05)
            test_ratio = 1.0 - train_ratio - val_ratio
            
            st.info(f"Test Ratio: {test_ratio:.2f}")
        
        if st.button("Download & Preview Data", use_container_width=True):
            with st.spinner("Downloading data..."):
                try:
                    downloader = YahooDownloader(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        ticker_list=tickers
                    )
                    df = downloader.fetch_data()
                    
                    st.success(f"Downloaded {len(df)} rows for {len(tickers)} tickers")
                    
                    # Preview
                    st.markdown("### Data Preview")
                    st.dataframe(df.head(20), use_container_width=True)
                    
                    # Store in session state
                    st.session_state.training_data = df
                    st.session_state.tickers = tickers
                    st.session_state.tech_indicators = tech_indicators
                    
                except Exception as e:
                    st.error(f"Error downloading data: {str(e)}")
    
    with tab2:
        st.markdown("### Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            algorithm = st.selectbox(
                "Algorithm",
                ["PPO", "A2C", "DDPG"],
                help="Select RL algorithm"
            )
            
            total_timesteps = st.number_input(
                "Total Timesteps",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000,
                help="Total training timesteps"
            )
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.00001,
                max_value=0.01,
                value=0.0003,
                format="%.5f",
                help="Learning rate for optimizer"
            )
        
        with col2:
            initial_amount = st.number_input(
                "Initial Amount ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000
            )
            
            transaction_cost = st.number_input(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.2f"
            ) / 100
            
            hmax = st.number_input(
                "Max Shares per Trade",
                min_value=1,
                max_value=1000,
                value=100,
                step=10
            )
        
        model_name = st.text_input(
            "Model Name",
            value=f"{algorithm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for saving the model"
        )
        
        st.session_state.training_config = {
            'algorithm': algorithm,
            'total_timesteps': total_timesteps,
            'learning_rate': learning_rate,
            'initial_amount': initial_amount,
            'transaction_cost': transaction_cost,
            'hmax': hmax,
            'model_name': model_name,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio
        }
    
    with tab3:
        st.markdown("### Train & Monitor")
        
        if 'training_data' not in st.session_state:
            st.warning("Please download data first in the Data Configuration tab")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("#### Training Configuration Summary")
                config = st.session_state.training_config
                st.json(config)
            
            with col2:
                if not st.session_state.training_running:
                    if st.button("Start Training", use_container_width=True, type="primary"):
                        st.session_state.training_running = True
                        st.rerun()
                else:
                    if st.button("Stop Training", use_container_width=True, type="secondary"):
                        st.session_state.training_running = False
                        st.rerun()
            
            if st.session_state.training_running:
                st.markdown("---")
                st.markdown("### Training Progress")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()
                
                # Simulate training (in real implementation, this would be async)
                with st.spinner("Training in progress..."):
                    try:
                        df = st.session_state.training_data
                        config = st.session_state.training_config
                        
                        # Feature engineering
                        status_text.text("Engineering features...")
                        fe = FeatureEngineer(
                            use_technical_indicator=True,
                            tech_indicator_list=st.session_state.tech_indicators,
                            use_vix=False,
                            use_turbulence=False,
                            user_defined_feature=False
                        )
                        processed = fe.preprocess_data(df)
                        progress_bar.progress(20)
                        
                        # Split data
                        status_text.text("Splitting data...")
                        unique_dates = processed['date'].unique()
                        train_end = int(len(unique_dates) * config['train_ratio'])
                        val_end = int(len(unique_dates) * (config['train_ratio'] + config['val_ratio']))
                        
                        train_dates = unique_dates[:train_end]
                        train = processed[processed['date'].isin(train_dates)].reset_index(drop=True)
                        train.index = train['date'].factorize()[0]
                        progress_bar.progress(30)
                        
                        # Create environment
                        status_text.text("Creating environment...")
                        stock_dim = len(st.session_state.tickers)
                        state_space = 1 + 2*stock_dim + len(st.session_state.tech_indicators)*stock_dim
                        
                        env = StockTradingEnv(
                            df=train,
                            stock_dim=stock_dim,
                            hmax=config['hmax'],
                            initial_amount=config['initial_amount'],
                            num_stock_shares=[0] * stock_dim,
                            buy_cost_pct=[config['transaction_cost']] * stock_dim,
                            sell_cost_pct=[config['transaction_cost']] * stock_dim,
                            reward_scaling=1e-4,
                            state_space=state_space,
                            action_space=stock_dim,
                            tech_indicator_list=st.session_state.tech_indicators
                        )
                        progress_bar.progress(40)
                        
                        # Train model
                        status_text.text(f"Training {config['algorithm']} model...")
                        
                        if config['algorithm'] == 'PPO':
                            model = PPO("MlpPolicy", env, learning_rate=config['learning_rate'], 
                                      verbose=1, device='cpu')
                        elif config['algorithm'] == 'A2C':
                            model = A2C("MlpPolicy", env, learning_rate=config['learning_rate'],
                                      verbose=1, device='cpu')
                        else:  # DDPG
                            model = DDPG("MlpPolicy", env, learning_rate=config['learning_rate'],
                                       verbose=1, device='cpu')
                        
                        model.learn(total_timesteps=config['total_timesteps'])
                        progress_bar.progress(90)
                        
                        # Save model
                        status_text.text("Saving model...")
                        os.makedirs('models', exist_ok=True)
                        model_path = f"models/{config['model_name']}.zip"
                        model.save(model_path)
                        progress_bar.progress(100)
                        
                        st.session_state.training_running = False
                        st.success(f"Training complete! Model saved to {model_path}")
                        
                        # Save metadata
                        metadata = {
                            'model_name': config['model_name'],
                            'algorithm': config['algorithm'],
                            'tickers': st.session_state.tickers,
                            'training_date': datetime.now().isoformat(),
                            'total_timesteps': config['total_timesteps'],
                            'initial_amount': config['initial_amount']
                        }
                        
                        with open(f"models/{config['model_name']}_metadata.json", 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
                        st.session_state.training_running = False

# Load Model Page
elif page == "Load Model":
    st.markdown("## Load & Manage Models")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        st.warning("No models directory found. Train a model first!")
    else:
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        
        if not model_files:
            st.info("No trained models found. Train a model first!")
        else:
            st.markdown(f"### Available Models ({len(model_files)})")
            
            # Display models in cards
            cols = st.columns(3)
            for idx, model_file in enumerate(model_files):
                with cols[idx % 3]:
                    model_name = model_file.replace('.zip', '')
                    
                    # Load metadata if exists
                    metadata_file = f"{models_dir}/{model_name}_metadata.json"
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {'algorithm': 'Unknown', 'tickers': [], 'training_date': 'Unknown'}
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{model_name}</h4>
                        <p><strong>Algorithm:</strong> {metadata.get('algorithm', 'Unknown')}</p>
                        <p><strong>Tickers:</strong> {', '.join(metadata.get('tickers', []))}</p>
                        <p><strong>Trained:</strong> {metadata.get('training_date', 'Unknown')[:10]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Load {model_name}", key=f"load_{idx}", use_container_width=True):
                        try:
                            # Load model based on algorithm
                            algo = metadata.get('algorithm', 'PPO')
                            if algo == 'PPO':
                                model = PPO.load(f"{models_dir}/{model_file}", device='cpu')
                            elif algo == 'A2C':
                                model = A2C.load(f"{models_dir}/{model_file}", device='cpu')
                            else:
                                model = DDPG.load(f"{models_dir}/{model_file}", device='cpu')
                            
                            st.session_state.loaded_model = model
                            st.session_state.loaded_model_name = model_name
                            st.session_state.loaded_model_metadata = metadata
                            st.success(f"Loaded {model_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading model: {str(e)}")
            
            if st.session_state.loaded_model:
                st.markdown("---")
                st.markdown("### Currently Loaded Model")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Model:** {st.session_state.loaded_model_name}")
                    metadata = st.session_state.loaded_model_metadata
                    st.json(metadata)
                
                with col2:
                    if st.button("Unload Model", use_container_width=True):
                        st.session_state.loaded_model = None
                        st.session_state.loaded_model_name = None
                        st.session_state.loaded_model_metadata = None
                        st.success("Model unloaded")
                        st.rerun()

# Paper Trade Page
elif page == "Paper Trade":
    st.markdown("## Paper Trading")
    
    if not st.session_state.loaded_model:
        st.warning("Please load a model first from the Load Model page")
    else:
        st.success(f"Using model: {st.session_state.loaded_model_name}")
        
        tab1, tab2 = st.tabs(["Configuration", "Live Trading"])
        
        with tab1:
            st.markdown("### Paper Trading Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metadata = st.session_state.loaded_model_metadata
                tickers = st.multiselect(
                    "Tickers to Trade",
                    metadata.get('tickers', ['AAPL', 'MSFT']),
                    default=metadata.get('tickers', ['AAPL', 'MSFT'])
                )
                
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=10000
                )
            
            with col2:
                trading_days = st.number_input(
                    "Trading Days",
                    min_value=1,
                    max_value=365,
                    value=30,
                    help="Number of days to simulate"
                )
                
                update_frequency = st.selectbox(
                    "Update Frequency",
                    ["Daily", "Hourly", "Every 15 min"],
                    help="How often to update positions"
                )
            
            st.session_state.paper_trade_config = {
                'tickers': tickers,
                'initial_capital': initial_capital,
                'trading_days': trading_days,
                'update_frequency': update_frequency
            }
        
        with tab2:
            st.markdown("### Live Paper Trading")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if not st.session_state.paper_trading_active:
                    if st.button("Start Paper Trading", use_container_width=True, type="primary"):
                        st.session_state.paper_trading_active = True
                        st.rerun()
                else:
                    if st.button("Stop Paper Trading", use_container_width=True, type="secondary"):
                        st.session_state.paper_trading_active = False
                        st.rerun()
            
            if st.session_state.paper_trading_active:
                st.markdown("---")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Portfolio Value", "$100,000", "+$0")
                with col2:
                    st.metric("Daily Return", "0.00%", "0.00%")
                with col3:
                    st.metric("Total Return", "0.00%", "0.00%")
                with col4:
                    st.metric("Sharpe Ratio", "0.00", "0.00")
                
                # Chart placeholder
                st.markdown("### Portfolio Performance")
                chart_placeholder = st.empty()
                
                # Positions table
                st.markdown("### Current Positions")
                positions_placeholder = st.empty()
                
                # Trading log
                st.markdown("### Trading Log")
                log_placeholder = st.empty()
                
                st.info("Paper trading simulation would run here in real-time")

# Analytics Page
elif page == "Analytics":
    st.markdown("## Analytics & Performance")
    
    if not os.path.exists("models") or not os.listdir("models"):
        st.info("No models available for analysis. Train a model first!")
    else:
        st.markdown("### Model Performance Comparison")
        
        # Placeholder for analytics
        st.info("Analytics dashboard coming soon!")
        
        st.markdown("""
        **Planned Features:**
        - Model performance comparison
        - Backtest results visualization
        - Risk metrics analysis
        - Strategy optimization recommendations
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>RL Trading Bot v1.0 | Built with FinRL & Streamlit | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
