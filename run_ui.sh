#!/bin/bash

echo "=========================================="
echo "Starting RL Trading Bot Web UI"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost

echo "UI stopped."
