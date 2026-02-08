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
- **Terminal UI**: Monitor model performance, positions, and system status via text-based interface
- **Always-On Daemon**: Runs continuously via systemd, trains daily at 2 AM, manages lifecycle automatically

## Requirements

- Python 3.9+
- ~2GB disk space (CPU-only PyTorch)
- Alpaca account for paper trading (free)
- Linux system with systemd (for daemon mode)

## Quick Start (Daemon Mode - Recommended)

```bash
# Clone and setup
git clone https://github.com/yourusername/RLFI.git
cd RLFI
chmod +x setup.sh
./setup.sh

# Configure API keys (required for paper trading)
cp .env.example .env
nano .env  # Add your Alpaca API keys

# Install and start the daemon
sudo cp rlfi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rlfi.service

# Monitor the system
python tui.py  # Launch terminal UI
```

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

## Daemon Management

The daemon runs continuously and handles all operations automatically:

```bash
# Check if daemon is running
sudo systemctl status rlfi.service

# View real-time logs
sudo journalctl -u rlfi.service -f

# Start/stop/restart the daemon
sudo systemctl start rlfi.service
sudo systemctl stop rlfi.service
sudo systemctl restart rlfi.service

# View recent logs (last 50 lines)
sudo journalctl -u rlfi.service -n 50
```

### What the Daemon Does

- **Training**: Trains 5 new models daily at 2 AM
- **Validation**: Backtests models against quality gates (Sharpe > 0.3, Return > 0%, Drawdown < 25%)
- **Paper Trading**: Promotes passing models to 20-40 day paper trading
- **Weekly Culling**: Every Saturday at 6 PM, evaluates and promotes/culls models
- **Genetic Evolution**: Champions spawn offspring with hyperparameter mutations
- **Live Trading**: Manages real-time paper trading during market hours

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

## Monitoring with TUI

Launch the Terminal UI to monitor the daemon in real-time:

```bash
# Launch TUI
python tui.py

# Or use the launcher script
./run_tui.sh
```

**TUI Features:**
- **Dashboard**: Live daemon status, model counts, paper trading overview
<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/d5080bfa-8043-4836-b91b-dff9c2bc868f" />

- **Models**: Browse all models with state, algorithm, Sharpe, and returns
<img width="2547" height="1385" alt="image" src="https://github.com/user-attachments/assets/70c81e2e-663e-43a9-bde4-eb4b7bbc2750" />

- **Lineage**: View family trees and genetic evolution
<img width="2547" height="1385" alt="image" src="https://github.com/user-attachments/assets/a5bbf558-9258-42e0-a8e4-f296b67240bd" />

- **Trading**: Paper trading history and performance
<img width="2547" height="1385" alt="image" src="https://github.com/user-attachments/assets/622a060e-ca4c-4289-9de2-8249552b7f62" />


**TUI Key Bindings:**
- `1-4`: Switch between tabs (Dashboard, Models, Lineage, Trading)
- `r`: Refresh data
- `d`: Toggle daemon on/off
- `q`: Quit

## Manual Mode (Without Daemon)

If you prefer not to use the systemd daemon, you can run modes manually:

```bash
# Train models only (one-time)
python scripts/autotest.py --mode train

# Backtest existing models
python scripts/autotest.py --mode backtest

# Run full colosseum cycle (continuous until Ctrl+C)
python scripts/autotest.py --mode colosseum
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
│   ├── evaluation/
│   │   └── backtester.py       # Backtesting engine
│   └── trading/
│       └── live_paper_trading.py
├── scripts/
│   └── autotest.py             # Main entry point
├── tui.py                      # Terminal UI dashboard
├── run_tui.sh                  # Script to launch TUI
├── rlfi.service                # Systemd service file
├── requirements.txt
├── setup.sh
└── LICENSE                     # MIT License
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
- **Daemon Resource Usage**: ~500MB RAM when idle, 2-4GB during training

The system is designed to run on CPU. GPU acceleration provides minimal benefit for RL trading due to environment stepping being the bottleneck, not neural network computation.

## Troubleshooting

### Daemon won't start
```bash
# Check for Python path issues
sudo systemctl status rlfi.service
# View detailed logs
sudo journalctl -u rlfi.service -n 100 --no-pager
```

### Database locked
```bash
# If the database gets corrupted, backup and reset
mv autotest_strategies.db autotest_strategies.db.bak
# Restart daemon - it will create a fresh database
sudo systemctl restart rlfi.service
```

### Out of disk space
```bash
# Clean up old models
python scripts/cleanup.py --dry-run  # See what would be deleted
python scripts/cleanup.py             # Actually delete
```

## License

MIT License - See [LICENSE](LICENSE)

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always paper trade before using real money.
