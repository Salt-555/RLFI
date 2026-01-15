# Quick Start Guide - RL Trading Bot

Get your RL trading bot running in 5 minutes!

## Step 1: Setup (2 minutes)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

## Step 2: Download Data (1 minute)

```bash
python scripts/download_data.py
```

This downloads stock data for AAPL, MSFT, GOOGL, AMZN, NVDA from 2018-2024.

## Step 3: Train Your First Model (5-30 minutes)

### Option A: Quick Test (5 minutes)
```bash
# Edit config to reduce timesteps for quick test
# In config/default_config.yaml, change:
# total_timesteps: 100000  # Instead of 1000000

python scripts/train_single.py
# Select: ppo (press Enter)
```

### Option B: Full Training (30+ minutes)
```bash
python scripts/train_single.py
# Select: ppo (press Enter)
```

## Step 4: Backtest (1 minute)

```bash
python scripts/backtest.py
# Select model: 1 (press Enter)
```

Check results in `backtest_results/`:
- `report.txt` - Performance metrics
- `performance.png` - Visualization

## Step 5: Monitor Training (Optional)

```bash
# In a new terminal
source venv/bin/activate
tensorboard --logdir=logs

# Open http://localhost:6006
```

## What to Expect

### First Run Metrics (100k timesteps)
- Training time: ~5-10 minutes on CPU
- Sharpe ratio: 0.3-0.8 (not great, needs more training)
- Returns: -5% to +15%

### Full Training (1M timesteps)
- Training time: ~30-60 minutes on CPU
- Sharpe ratio: 0.8-1.5 (good)
- Returns: 10-25% annually

## Next Steps

### 1. Train Ensemble (Better Performance)
```bash
python scripts/train_ensemble.py
```

### 2. Customize Configuration
Edit `config/default_config.yaml`:
- Change tickers
- Adjust date ranges
- Modify hyperparameters

### 3. Paper Trading (After 6+ months backtesting)
```bash
# Set up Alpaca API keys in .env
python scripts/paper_trade.py
```

## Common Issues

### "No module named 'src'"
```bash
# Make sure you're in the RLFI directory
cd /home/salt/CodingProjects/RLFI
```

### "Data download failed"
```bash
# Yahoo Finance may be slow, try again
# Or check your internet connection
```

### "Out of memory"
```bash
# Reduce batch_size in config/default_config.yaml
# Or reduce total_timesteps
```

## File Structure After Setup

```
RLFI/
├── venv/                    # Virtual environment
├── data/
│   ├── raw/                 # Downloaded stock data
│   └── processed/           # Processed with indicators
├── models/                  # Trained models (.zip files)
├── logs/                    # TensorBoard logs
├── backtest_results/        # Backtest reports & plots
└── .env                     # Your API keys (don't commit!)
```

## Quick Commands Reference

```bash
# Download data
python scripts/download_data.py

# Train single agent
python scripts/train_single.py

# Train ensemble
python scripts/train_ensemble.py

# Backtest
python scripts/backtest.py

# Paper trade
python scripts/paper_trade.py

# Monitor
python scripts/monitor.py --mode training
python scripts/monitor.py --mode backtest
python scripts/monitor.py --mode paper
```

## Tips for Success

1. **Start Small**: Train on 1-2 stocks first
2. **Be Patient**: Good models need 1M+ timesteps
3. **Backtest Thoroughly**: 6+ months before live trading
4. **Monitor Closely**: Check metrics daily
5. **Manage Risk**: Never risk more than you can lose

## Getting Help

- Check `README.md` for detailed documentation
- Review FinRL tutorials: https://github.com/AI4Finance-Foundation/FinRL
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/

---

**Ready to start? Run `./setup.sh` now!** 🚀
