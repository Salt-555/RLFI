# RL Trading Bot - Web UI Guide

## 🚀 Quick Start

### Launch the UI

```bash
# Option 1: Using the script
./run_ui.sh

# Option 2: Direct command
source venv/bin/activate
streamlit run app.py
```

The UI will open at: **http://localhost:8501**

## 📋 Features

### 🏠 Home Page
- Overview of the system
- Quick stats dashboard
- System status monitoring
- Navigation to all features

### 🎯 Train Model
Complete model training workflow with three tabs:

**1. Data Configuration**
- Select stock tickers (comma-separated)
- Set date ranges for training data
- Choose technical indicators (MACD, RSI, CCI, DX, etc.)
- Configure train/validation/test split ratios
- Download and preview data before training

**2. Training Parameters**
- Algorithm selection: PPO, A2C, or DDPG
- Total timesteps configuration
- Learning rate tuning
- Initial capital amount
- Transaction cost percentage
- Max shares per trade
- Custom model naming

**3. Train & Monitor**
- Review configuration summary
- Start/stop training
- Real-time progress monitoring
- Training metrics display
- Automatic model saving with metadata

### 📦 Load Model
- Browse all trained models
- View model metadata (algorithm, tickers, training date)
- Load models for paper trading
- Model management (load/unload)
- Visual model cards with key information

### 📊 Paper Trade
- Configure paper trading parameters
- Select tickers from trained model
- Set initial capital and trading days
- Real-time portfolio monitoring
- Performance metrics dashboard
- Trading log and position tracking

### 📈 Analytics
- Model performance comparison (coming soon)
- Backtest visualization
- Risk metrics analysis
- Strategy optimization

## 🎨 UI Features

### Modern Design
- Gradient headers
- Metric cards with visual styling
- Color-coded status indicators
- Responsive layout
- Clean, professional interface

### Real-Time Updates
- Training progress bars
- Live metrics during paper trading
- Dynamic status updates
- Session state management

### User-Friendly
- Intuitive navigation
- Clear instructions
- Helpful tooltips
- Success/warning/error messages
- Organized tabs and sections

## 📁 File Structure

```
RLFI/
├── app.py                 # Main Streamlit application
├── run_ui.sh             # Launch script
├── models/               # Saved models directory
│   ├── *.zip            # Model files
│   └── *_metadata.json  # Model metadata
├── logs/                 # TensorBoard logs
└── data/                 # Data storage
```

## 🔧 Configuration

### Model Metadata
Each trained model saves metadata including:
- Model name
- Algorithm used
- Tickers traded
- Training date
- Total timesteps
- Initial capital

### Session State
The UI maintains state for:
- Currently loaded model
- Training status
- Paper trading status
- Configuration settings

## 💡 Usage Tips

### Training Models
1. Start with 2-4 tickers for faster training
2. Use 100k-1M timesteps for initial tests
3. Default learning rate (0.0003) works well for most cases
4. Monitor training progress in real-time
5. Models are automatically saved with timestamps

### Loading Models
1. Models appear as cards with metadata
2. Click "Load" to prepare for paper trading
3. Only one model can be loaded at a time
4. Unload before loading a different model

### Paper Trading
1. Load a model first
2. Configure trading parameters
3. Start paper trading to simulate
4. Monitor performance metrics
5. Stop anytime to review results

## 🎯 Workflow Example

**Complete Training to Paper Trading Flow:**

1. **Navigate to Train Model**
   - Enter tickers: `AAPL,MSFT,GOOGL`
   - Set date range: Last 2 years
   - Select indicators: MACD, RSI, CCI, DX
   - Download data

2. **Configure Training**
   - Algorithm: PPO
   - Timesteps: 100,000
   - Learning rate: 0.0003
   - Initial capital: $100,000

3. **Start Training**
   - Click "Start Training"
   - Monitor progress
   - Wait for completion
   - Model saved automatically

4. **Load Model**
   - Go to Load Model page
   - Find your trained model
   - Click "Load"
   - Verify model is loaded

5. **Paper Trade**
   - Go to Paper Trade page
   - Configure parameters
   - Start paper trading
   - Monitor performance

## 🔐 Security Notes

- No real trading occurs (paper trading only)
- No API keys required for basic features
- All data stored locally
- Models saved in `models/` directory

## 🐛 Troubleshooting

**UI won't start:**
```bash
# Check if port 8501 is available
lsof -i :8501

# Kill existing process if needed
pkill -f streamlit

# Restart UI
./run_ui.sh
```

**Training fails:**
- Check data downloaded successfully
- Verify sufficient disk space
- Review error messages in UI
- Check logs for details

**Model won't load:**
- Ensure model file exists in `models/`
- Check metadata file is present
- Verify algorithm matches model type

## 📊 Performance Tips

### For Faster Training
- Use CPU device (already configured)
- Reduce number of tickers
- Lower total timesteps for testing
- Use simpler algorithms (PPO is fastest)

### For Better Results
- Train with more timesteps (1M+)
- Use diverse tickers
- Include multiple technical indicators
- Test different algorithms
- Validate with longer date ranges

## 🎓 Next Steps

1. **Train Your First Model**
   - Start with 2 stocks
   - Use 50k timesteps for quick test
   - Monitor training progress

2. **Experiment with Parameters**
   - Try different algorithms
   - Adjust learning rates
   - Test various indicators

3. **Analyze Results**
   - Compare model performance
   - Review paper trading results
   - Refine strategy

4. **Scale Up**
   - Add more tickers
   - Increase training time
   - Optimize hyperparameters

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This is for learning RL and trading concepts
- **No Financial Advice**: Not intended for real trading decisions
- **Paper Trading Only**: No real money involved
- **Past Performance**: Does not guarantee future results
- **Risk Warning**: Real trading involves significant risk

---

**Enjoy exploring reinforcement learning for trading! 🚀📈**
