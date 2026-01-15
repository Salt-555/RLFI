#!/bin/bash

echo "=========================================="
echo "RL Trading Bot - Setup Script"
echo "=========================================="

echo -e "\n1. Creating virtual environment..."
python -m venv venv

echo -e "\n2. Activating virtual environment..."
source venv/bin/activate

echo -e "\n3. Upgrading pip..."
pip install --upgrade pip

echo -e "\n4. Installing dependencies..."
pip install -r requirements.txt

echo -e "\n5. Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p checkpoints
mkdir -p backtest_results
mkdir -p paper_trading_logs

echo -e "\n6. Setting up environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please update with your API keys."
else
    echo ".env file already exists."
fi

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo -e "\nNext steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Update .env file with your API keys (optional for paper trading)"
echo "  3. Train a model: python scripts/train_single.py"
echo "  4. Or train ensemble: python scripts/train_ensemble.py"
echo -e "\nFor more information, see README.md"
