# Major System Updates

## 1. Expanded Stock Coverage (35-40+ stocks)

**Problem:** Training 2M-3M steps on only 5 stocks (5,845 observations) = 1:513 ratio (severe overfitting)

**Solution:** Updated stock baskets with 8-10 stocks each:
- tech_mega_large: 10 stocks (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, CRM, ADBE)
- tech_semis_large: 10 stocks (NVDA, AMD, INTC, AVGO, QCOM, TXN, MU, LRCX, KLAC, MRVL)
- tech_software: 10 stocks (enterprise software focus)
- financials_large: 10 stocks (major banks + asset managers)
- healthcare_large: 10 stocks (pharma + biotech + devices)
- consumer_large: 10 stocks (retail + autos + restaurants)
- industrials: 10 stocks (manufacturing + transport)
- energy_materials: 10 stocks (oil + gas + mining + chemicals)
- communications: 10 stocks (media + telecom)
- etf_broad: 10 ETFs (market exposure)
- etf_sector: 10 sector ETFs
- metals_commodities: 10 (precious metals + energy + agriculture)
- fixed_income: 10 bond ETFs
- safe_haven_mix: 10 defensive assets
- international: 10 developed international

**Impact:** 
- With 35-40 stocks: ~41k-47k observations
- New ratio: 1:43 to 1:49 (much healthier)
- More diverse market regimes captured
- Better for grokking (diverse data = less memorization needed)

## 2. Walk-Forward Validation System

**Problem:** Static train/test splits become stale over time and don't adapt to current market conditions

**Solution:** New `src/data/walk_forward_validator.py` implements:

### Rolling Window System
```python
WalkForwardValidator(
    train_window_days=1260,      # 5 years training data
    val_window_days=252,         # 1 year validation
    test_window_days=126,        # 6 months testing
    expanding_window=True,       # Training window grows over time
)
```

### How It Works
1. **Reference Date**: Uses "today" as the anchor point (not hardcoded dates)
2. **Recent Data Priority**: Validation and testing always use most recent data
3. **Expanding Window**: Training window starts from earliest data and grows
4. **Automatic Window Management**: As time passes, windows automatically shift forward

### Benefits
- **Future-Proof**: Always uses relevant, recent market data
- **No Stale Models**: Training never uses 5-year-old test criteria
- **Market Regime Adaptation**: Automatically adapts to bull/bear/volatile periods
- **Rolling Validation**: Can create multiple folds for robust evaluation

## 3. Integration Points

### Config Updates (`config/autotest_config.yaml`)
- All 15 baskets updated with 8-10 stocks each
- No more 5-stock baskets (too little diversification)
- Total universe: ~100+ unique stocks across all baskets

### Training Pipeline Updates (`src/autotest/automated_trainer.py`)
- Integrated `WalkForwardValidator` into data splitting
- Replaces static `train_ratio/val_ratio/test_ratio` splits
- Automatically reserves last 90 days as holdout (untouched)
- Logs window boundaries for transparency

### New Module (`src/data/walk_forward_validator.py`)
- Standalone validation system
- Can be used independently for experiments
- Supports multiple validation strategies (rolling, expanding, fixed)

## 4. Expected Outcomes

### Data Quality
- **Before**: 5,845 observations (5 stocks × 1,169 days)
- **After**: 40,915+ observations (35 stocks × 1,169 days)
- **Ratio**: Improved from 1:513 to 1:49 (10× better)

### Training Quality
- **Before**: Severe overfitting on 5 stocks, memorizing specific paths
- **After**: Models learn generalizable patterns across 35+ stocks
- **Grokking**: Easier to achieve with diverse data (less memorization needed)

### Validation Quality
- **Before**: Static split (2018-2022 train, 2023-2024 test) - becomes stale
- **After**: Always uses recent 6 months for testing, 1 year for validation
- **Adaptation**: Automatically adjusts to current market regime

## 5. Next Steps

1. **Tonight's Training**: System will automatically use:
   - Walk-forward splits based on current date
   - Larger stock baskets (8-10 stocks each)
   - More diverse training data

2. **Monitoring**: Watch for:
   - Better generalization in backtests
   - Lower overfitting (grokking detection scores)
   - More stable paper trading performance

3. **Future Enhancement**: Once we have more models, we can implement:
   - Multiple walk-forward folds per model
   - Cross-validation across different time periods
   - Market regime-specific training

## Summary

These changes address the core data scarcity issue while maintaining the grokking detection system. More stocks + walk-forward validation = better generalization without sacrificing training quality.
