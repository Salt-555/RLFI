# RLFI - Reinforcement Learning Financial Intelligence

> **⚠️ Project Status**: I do not plan to maintain this long term at this point. The best route for interested tinkerers is to fork the main repo and work on your own iteration rather than branching and attempting to push changes to this repo.

A continuous AI trading system that trains, validates, and paper trades reinforcement learning models using a "Colosseum" approach - where models compete, evolve, and only the best survive.

## Features

- **Colosseum System**: Continuous model lifecycle management (train → validate → paper trade → promote/cull)
- **Genetic Evolution**: Top performers spawn offspring models for accelerated learning
- **Multi-Algorithm Support**: PPO (stable, diverse) and SAC (sample-efficient, concentrated)
- **Automated Backtesting**: Quality gates ensure only profitable models reach paper trading
- **Live Paper Trading**: Real-time trading via Alpaca API with rate limiting
- **Sector-Based Baskets**: Diversified training across tech, financials, healthcare, ETFs, and hedging strategies
- **Streamlit Dashboard**: Monitor model performance, positions, and system status

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/RLFI.git
cd RLFI
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Configure API keys (required for paper trading)
nano .env  # Add your Alpaca API keys

# Start the Colosseum
python scripts/autotest.py --mode colosseum
```

## Requirements

- Python 3.9+
- ~2GB disk space (CPU-only PyTorch)
- Alpaca account for paper trading (free)

## Installation Options

### Standard (CPU-only, recommended)
```bash
./setup.sh
```

### AMD ROCm GPU (Strix Halo / gfx1151)
```bash
# After running setup.sh, replace PyTorch:
./venv/bin/pip uninstall torch -y
./venv/bin/pip install --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/ torch
```

### Manual Installation
```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt
```

## Configuration

All settings are in `config/autotest_config.yaml`:

```yaml
autotest:
  training:
    algorithms:
      - ppo  # 60% - Stable, diverse trading
      - sac  # 40% - Sample efficient, concentrated
    base_timesteps: 1500000
    
  colosseum:
    max_paper_trading_models: 10
    min_paper_trading_days: 20
    promote_min_days: 40
    genetic_training:
      enabled: true
      offspring_ratio: 0.4
```

### Stock Baskets

Models train on sector-based baskets for diversity:
- **Tech Megacaps**: AAPL, MSFT, GOOGL, AMZN, META
- **Semiconductors**: NVDA, AMD, INTC, AVGO, QCOM
- **Financials**: JPM, BAC, WFC, GS, MS
- **ETFs**: SPY, QQQ, DIA, IWM, VTI
- **Hedging**: GLD, TLT, VXX (counter-correlated)

## Usage

### Commands

```bash
# Continuous Colosseum (trains daily at 2 AM, trades during market hours)
python scripts/autotest.py --mode colosseum

# Train models only
python scripts/autotest.py --mode train

# Backtest existing models
python scripts/autotest.py --mode backtest

# Launch dashboard
streamlit run app.py
```

### Systemd Service (Linux)

Run RLFI as a background service:

```bash
# Install service
sudo cp rlfi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rlfi.service

# Check status
sudo systemctl status rlfi.service

# View logs
sudo journalctl -u rlfi.service -f
```

## Model Lifecycle

```
┌─────────────┐
│  TRAINING   │  Daily at 2 AM (5 models)
└──────┬──────┘
       ▼
┌─────────────┐
│ VALIDATION  │  Backtest quality gate
└──────┬──────┘  (Sharpe > 0.3, Return > 0%, Drawdown < 25%)
       ▼
┌─────────────┐
│PAPER_TRADING│  20-40 trading days
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌────────┐
│CULLED│ │PROMOTED│  → Genetic parent for offspring
└──────┘ └────────┘
```

## Algorithms

### PPO (Proximal Policy Optimization)
- **Pros**: Stable training, diverse trading behavior, good for exploration
- **Cons**: Lower sample efficiency
- **Best for**: General-purpose trading, sector rotation

### SAC (Soft Actor-Critic)
- **Pros**: High sample efficiency, auto-tuned entropy, learns faster
- **Cons**: Can concentrate positions
- **Best for**: Momentum strategies, quick adaptation

## Project Structure

```
RLFI/
├── config/
│   └── autotest_config.yaml    # Main configuration
├── src/
│   ├── agents/
│   │   └── trainer.py          # PPO/SAC model training
│   ├── autotest/
│   │   ├── parameter_generator.py
│   │   └── continuation_trainer.py
│   ├── data/
│   │   ├── data_loader.py      # yfinance data fetching
│   │   └── feature_engineer.py # Technical indicators
│   ├── environment/
│   │   └── trading_env.py      # Gymnasium trading environment
│   └── trading/
│       └── live_paper_trading.py
├── scripts/
│   └── autotest.py             # Main entry point
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── setup.sh
└── rlfi.service                # Systemd service file
```

## Environment Variables

Create a `.env` file:

```bash
# Required for paper trading
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get free API keys at [alpaca.markets](https://alpaca.markets/)

## Performance Notes

- **CPU Training**: ~30-60 min per model (1.5M timesteps)
- **Memory**: ~2-4GB RAM per training model
- **Disk**: ~50MB per saved model

The system is designed to run on CPU. GPU acceleration provides minimal benefit for RL trading due to environment stepping being the bottleneck, not neural network computation.

## License

O'Saasy License - See [LICENSE](LICENSE)

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always paper trade before using real money.
