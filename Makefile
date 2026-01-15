.PHONY: setup install clean data train-single train-ensemble backtest paper-trade monitor help

help:
	@echo "RL Trading Bot - Makefile Commands"
	@echo "=================================="
	@echo "setup          - Create venv and install dependencies"
	@echo "install        - Install dependencies only"
	@echo "clean          - Remove generated files and cache"
	@echo "data           - Download and process stock data"
	@echo "train-single   - Train single RL agent"
	@echo "train-ensemble - Train ensemble of agents"
	@echo "backtest       - Run backtesting"
	@echo "paper-trade    - Start paper trading"
	@echo "monitor        - View training progress"
	@echo "tensorboard    - Launch TensorBoard"

setup:
	@echo "Setting up virtual environment..."
	python -m venv venv
	@echo "Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Creating directories..."
	mkdir -p data/raw data/processed models logs checkpoints backtest_results paper_trading_logs
	@echo "Setup complete! Activate with: source venv/bin/activate"

install:
	pip install -r requirements.txt

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__
	rm -rf *.pyc src/*.pyc src/*/*.pyc
	rm -rf .pytest_cache
	rm -rf build dist *.egg-info
	@echo "Clean complete!"

data:
	python scripts/download_data.py

train-single:
	python scripts/train_single.py

train-ensemble:
	python scripts/train_ensemble.py

backtest:
	python scripts/backtest.py

paper-trade:
	python scripts/paper_trade.py

monitor:
	python scripts/monitor.py --mode training

tensorboard:
	tensorboard --logdir=logs
