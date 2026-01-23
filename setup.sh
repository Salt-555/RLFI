#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "RLFI - Reinforcement Learning Financial Intelligence"
echo "Colosseum Setup Script"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "\nDetected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.9" ]]; then
    echo "Error: Python 3.9+ required. Found $PYTHON_VERSION"
    exit 1
fi

echo -e "\n[1/7] Creating virtual environment..."
python3 -m venv venv

echo -e "\n[2/7] Upgrading pip..."
./venv/bin/pip install --upgrade pip

echo -e "\n[3/7] Installing PyTorch (CPU-only for smaller footprint)..."
echo "       For ROCm GPU support, see README.md"
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

echo -e "\n[4/7] Installing dependencies..."
./venv/bin/pip install -r requirements.txt

echo -e "\n[5/7] Creating directory structure..."
mkdir -p data/raw data/processed
mkdir -p models checkpoints
mkdir -p logs paper_trading_logs
mkdir -p autotest_models autotest_logs autotest_results
mkdir -p champion_models

echo -e "\n[6/7] Setting up environment file..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        cat > .env << 'EOF'
# Alpaca API Keys (required for paper trading)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional: For notifications
# SLACK_WEBHOOK_URL=
# EMAIL_ADDRESS=
EOF
        echo "Created .env template"
    fi
    echo "Please update .env with your Alpaca API keys for paper trading."
else
    echo ".env file already exists."
fi

echo -e "\n[7/7] Verifying installation..."
./venv/bin/python -c "
import torch
import stable_baselines3
import gymnasium
import pandas
import yfinance
print('✓ All core packages installed successfully')
print(f'  PyTorch: {torch.__version__}')
print(f'  Stable-Baselines3: {stable_baselines3.__version__}')
"

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo -e "\nQuick Start:"
echo "  source venv/bin/activate"
echo ""
echo "Commands:"
echo "  python scripts/autotest.py --mode colosseum  # Start Colosseum (continuous)"
echo "  python scripts/autotest.py --mode train      # Train models only"
echo "  python scripts/autotest.py --mode backtest   # Backtest existing models"
echo "  streamlit run app.py                         # Launch dashboard"
echo ""
echo "Systemd Service (optional):"
echo "  sudo cp rlfi.service /etc/systemd/system/"
echo "  sudo systemctl enable --now rlfi.service"
echo ""
echo "For more information, see README.md"
