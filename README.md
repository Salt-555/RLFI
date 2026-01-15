# RL Trading Bot - Reinforcement Learning Stock Trading System

A production-ready reinforcement learning trading bot built with the FinRL framework, featuring ensemble learning, comprehensive backtesting, and paper trading capabilities.

## Features

- **Multiple RL Algorithms**: PPO, A2C, DDPG with optimized hyperparameters
- **Ensemble Learning**: Train multiple agents and select the best performer
- **Advanced Features**: Technical indicators, turbulence detection, risk management
- **Comprehensive Backtesting**: Sharpe ratio, max drawdown, win rate, and more
- **Paper Trading**: Live testing with Alpaca's paper trading API
- **Modular Architecture**: Easy to extend and customize

## Realistic Expectations

Based on verified research and FinRL contests:
- **Target Annual Returns**: 12-25%
- **Sharpe Ratio**: 1.0-1.5
- **Max Drawdown**: <20%
- **Win Rate**: 55-65%

⚠️ **Warning**: Claims of 80%+ returns are rare and typically curve-fitted. Focus on risk-adjusted returns.

## Quick Start

### 1. Setup

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv, installs dependencies)
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Train a Single Agent

```bash
python scripts/train_single.py
```

This will:
- Download stock data (AAPL, MSFT, GOOGL, AMZN, NVDA)
- Calculate technical indicators
- Train a PPO/A2C/DDPG agent
- Save the trained model

### 3. Train Ensemble (Recommended)

```bash
python scripts/train_ensemble.py
```

Trains multiple algorithms and selects the best performer based on Sharpe ratio.

### 4. Backtest

```bash
python scripts/backtest.py
```

Evaluates the trained model on unseen test data and generates:
- Performance metrics
- Visualization plots
- Detailed report

### 5. Paper Trading (Optional)

```bash
# First, set up Alpaca API keys in .env
python scripts/paper_trade.py
```

## Project Structure

```
RLFI/
├── config/
│   └── default_config.yaml      # Configuration settings
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Yahoo Finance data download
│   │   └── feature_engineer.py  # Technical indicators & turbulence
│   ├── environment/
│   │   └── trading_env.py       # Gymnasium trading environment
│   ├── agents/
│   │   ├── trainer.py           # Single agent training
│   │   └── ensemble.py          # Ensemble training & selection
│   ├── evaluation/
│   │   ├── metrics.py           # Performance metrics
│   │   └── backtester.py        # Backtesting framework
│   └── trading/
│       └── paper_trading.py     # Alpaca paper trading
├── scripts/
│   ├── train_single.py          # Train single agent
│   ├── train_ensemble.py        # Train ensemble
│   ├── backtest.py              # Run backtest
│   └── paper_trade.py           # Paper trading
├── requirements.txt             # Python dependencies
├── setup.sh                     # Setup script
└── README.md                    # This file
```

## Configuration

Edit `config/default_config.yaml` to customize:

### Data Settings
```yaml
data:
  tickers: [AAPL, MSFT, GOOGL, AMZN, NVDA]
  start_date: "2018-01-01"
  end_date: "2024-12-31"
```

### Environment Settings
```yaml
environment:
  initial_amount: 100000
  transaction_cost_pct: 0.001
  hmax: 100  # Max shares per trade
```

### Training Settings
```yaml
training:
  algorithms: [ppo, a2c, ddpg]
  total_timesteps: 1000000
```

## Technical Indicators

The system uses 10 technical indicators:
- **MACD**: Momentum
- **RSI**: Relative Strength Index
- **CCI**: Commodity Channel Index
- **DX**: Directional Movement Index
- **SMA**: Simple Moving Averages (30, 60 day)
- **ADX**: Average Directional Index
- **Bollinger Bands**: Upper/Lower bands
- **ATR**: Average True Range
- **Turbulence Index**: Market crash detection

## Environment Design

### State Space (~180 dimensions)
- Cash balance (normalized)
- Stock prices (normalized)
- Shares owned
- Technical indicators (10 per stock)
- Turbulence index

### Action Space
- Continuous: [-1, 1] per stock
  - -1 = Sell all shares
  - 0 = Hold
  - +1 = Buy maximum shares

### Reward Function
```python
reward = (portfolio_value_t+1 - portfolio_value_t) * reward_scaling
```

### Risk Controls
- **Turbulence Threshold**: Liquidate positions when turbulence > 120
- **Position Limits**: Max 100 shares per trade
- **Transaction Costs**: 0.1% per trade

## Training Details

### PPO (Recommended)
- Learning rate: 3e-4
- Batch size: 64
- Network: [256, 128] hidden layers
- Training time: 2-10 hours on CPU

### A2C
- Learning rate: 7e-4
- Faster training than PPO
- Good for multi-asset portfolios

### DDPG
- Learning rate: 1e-4
- Continuous action space
- More sample efficient

## Backtesting Metrics

- **Sharpe Ratio**: Risk-adjusted returns (target: >1.0)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline (target: <20%)
- **Cumulative Return**: Total return over period
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return / Max Drawdown

## Paper Trading Setup

1. Create free Alpaca account: https://alpaca.markets/
2. Get paper trading API keys
3. Update `.env` file:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Best Practices

### Before Live Trading
✅ 6+ months successful paper trading  
✅ Sharpe ratio >1.0 out-of-sample  
✅ Max drawdown <20%  
✅ Start with <10% of intended capital  

### Monitoring
- Track daily portfolio value vs. benchmark
- Monitor transaction costs
- Set up alerts for unusual losses
- Retrain quarterly with new data

### Risk Management
- 2% max loss per trade
- 10% portfolio stop-loss
- Diversify across multiple stocks
- Never risk more than you can afford to lose

## Common Issues

### Installation Errors
```bash
# If torch installation fails on ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# If stockstats fails
pip install stockstats==0.6.2
```

### Data Download Issues
- Yahoo Finance may rate-limit: Add delays between requests
- Use cached data when available
- Consider Alpha Vantage as backup

### Training Instability
- Normalize inputs with reward_scaling
- Reduce learning rate
- Increase batch size
- Use VecNormalize wrapper

## Advanced Features (Optional)

### LLM-Augmented Signals
Fine-tune language models for sentiment analysis:
```python
# Add sentiment scores to state space
# Requires transformers library
```

### Multi-Agent Strategies
Train specialized agents per market regime:
- Bull market agent (momentum-focused)
- Bear market agent (defensive)
- High volatility agent (scalping)

## Resources

- **FinRL Framework**: https://github.com/AI4Finance-Foundation/FinRL
- **FinRL Contest 2025**: https://github.com/Open-Finance-Lab/FinRL_Contest_2025
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Alpaca API**: https://alpaca.markets/docs/

## Performance Benchmarks

Based on FinRL Contest 2024 winners:
- **Best Single Agent**: 72% annual return (PPO)
- **Best Ensemble**: 134% annual return (PPO + LLM signals)
- **Average**: 25-40% annual return

## License

MIT License - Feel free to use and modify for your trading strategies.

## Disclaimer

⚠️ **IMPORTANT**: This software is for educational purposes only. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose.

## Support

For issues and questions:
1. Check the documentation
2. Review FinRL tutorials
3. Consult Stable-Baselines3 docs
4. Open an issue on GitHub

---

**Happy Trading! 🚀📈**
