# RL Training Best Practices for Trading Models

## Research Summary

### Training Duration

**Industry Standards:**
- **Simple Tasks** (CartPole, MountainCar): 100k-500k timesteps
- **Medium Complexity** (Atari games): 1M-10M timesteps
- **Complex Tasks** (Trading, Robotics): 10M-50M timesteps
- **State-of-the-art** (AlphaGo, Dota 2): 100M+ timesteps

**For Trading Specifically:**
- **Minimum**: 1M timesteps for basic convergence
- **Recommended**: 2-5M timesteps for stable performance
- **Optimal**: 10M+ timesteps for sophisticated strategies
- **Current AUTOTEST**: 500k timesteps (likely undertrained)

### Transfer Learning & Warm Starting

**Benefits:**
- Preserve learned patterns
- Faster adaptation to new data
- Build on successful strategies
- Reduce training time for refinements

**Best Practices:**
- Continue training top performers
- Fine-tune on recent market data
- Adjust learning rate (lower for fine-tuning)
- Monitor for catastrophic forgetting

### Population-Based Training (PBT)

**Concept** (DeepMind, 2017):
- Maintain population of models
- Periodically evaluate all models
- Replace worst performers with copies of best
- Mutate hyperparameters of copies
- Continue training

**Advantages:**
- Automatic hyperparameter tuning
- Exploits good models while exploring variations
- No wasted compute on poor performers
- Proven effective in RL (AlphaStar, IMPALA)

### Curriculum Learning

**Approach:**
- Start with simpler scenarios
- Gradually increase difficulty
- Adapt to changing market conditions
- Progressive task complexity

**For Trading:**
- Start: Single stock, simple indicators
- Progress: Multiple stocks, complex indicators
- Advanced: Portfolio management, risk constraints
- Expert: Multi-timeframe, regime detection

### Model Versioning & Lineage

**Track:**
- Parent model (if continuation)
- Generation number
- Training history
- Performance evolution
- Parameter changes

**Benefits:**
- Understand what improvements work
- Rollback if performance degrades
- Identify successful lineages
- Scientific reproducibility

## Recommended Training Strategy

### Hybrid Approach: 60% Exploration + 40% Exploitation

```
┌─────────────────────────────────────────────────────────┐
│ WEEK N TRAINING ALLOCATION                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ EXPLORATION (60% compute)                               │
│ ├─ 15 new models × 1.5M timesteps = 22.5M timesteps    │
│ └─ Test new parameter combinations                      │
│                                                          │
│ EXPLOITATION (40% compute)                              │
│ ├─ Top 5 from last week × 2M timesteps = 10M timesteps │
│ ├─ Champion model × 3M timesteps = 3M timesteps        │
│ └─ Build on proven strategies                           │
│                                                          │
│ TOTAL: ~35M timesteps/week                              │
└─────────────────────────────────────────────────────────┘
```

### Model Categories

#### 1. **Explorers** (New Models)
- Fresh random initialization
- Varied hyperparameters
- Different ticker combinations
- Goal: Discover new strategies

#### 2. **Refiners** (Continued Models)
- Top 5 performers from previous week
- Continue training with recent data
- Slightly adjusted hyperparameters
- Goal: Improve proven strategies

#### 3. **Champions** (Elite Models)
- Best 1-2 models across all weeks
- Extensive continued training
- Fine-tuned for current market
- Goal: Maintain peak performance

### Training Timesteps by Category

```yaml
explorers:
  count: 15
  timesteps: 1500000  # 1.5M each
  total: 22500000     # 22.5M total

refiners:
  count: 5
  base_timesteps: 500000      # Already trained
  additional_timesteps: 2000000  # 2M more
  total: 10000000     # 10M total

champions:
  count: 1
  base_timesteps: 2500000     # Already trained
  additional_timesteps: 3000000  # 3M more
  total: 3000000      # 3M total

GRAND TOTAL: 35.5M timesteps/week
```

## Implementation Recommendations

### Phase 1: Immediate Improvements (Week 1)

**Increase Training Duration:**
```yaml
training:
  base_timesteps: 1500000  # Up from 500k
  num_models_to_train: 20  # Down from 30 (same total compute)
```

**Benefits:**
- Better convergence
- More stable strategies
- Minimal code changes

### Phase 2: Model Retention (Week 2-3)

**Add Model Archive:**
```yaml
model_retention:
  enabled: true
  keep_top_n: 5
  archive_dir: "champion_models"
  versioning: true
```

**Track Lineage:**
- Parent model ID
- Generation number
- Total training timesteps
- Performance history

### Phase 3: Continuation Training (Week 4+)

**Hybrid Training:**
```yaml
training_strategy:
  new_models:
    count: 15
    timesteps: 1500000
  
  continued_models:
    count: 5
    source: "top_from_last_week"
    additional_timesteps: 2000000
    learning_rate_multiplier: 0.5  # Lower LR for fine-tuning
  
  champion_models:
    count: 1
    source: "best_all_time"
    additional_timesteps: 3000000
    learning_rate_multiplier: 0.3
```

### Phase 4: Population-Based Training (Future)

**Full PBT Implementation:**
- Maintain population of 20 models
- Evaluate every 500k timesteps
- Replace bottom 20% with top 20% copies
- Mutate hyperparameters (±20%)
- Continue training

## Training Schedule Optimization

### Current: All at Once
```
Sunday 2AM-4AM: Train all 30 models
Problem: 2 hours insufficient for quality training
```

### Proposed: Distributed Training
```
Sunday 2AM-8AM: Train all models (6 hours)
OR
Continuous: Train throughout the week
  - Monday: Train explorers (15 models)
  - Tuesday: Continue refiners (5 models)
  - Wednesday: Train champion (1 model)
  - Thursday-Friday: Backtest all
  - Saturday: Paper trade preparation
```

## Learning Rate Schedules

### For New Models
```python
learning_rate: 0.0003  # Standard
schedule: "linear"     # Decay to 0
```

### For Continued Models
```python
learning_rate: 0.0001  # Lower (0.3-0.5x original)
schedule: "constant"   # Or very slow decay
```

### For Champions
```python
learning_rate: 0.00003  # Very low (0.1x original)
schedule: "constant"    # Preserve learned patterns
```

## Catastrophic Forgetting Prevention

**Strategies:**
1. **Lower Learning Rates** - Smaller updates preserve knowledge
2. **Replay Buffer** - Mix old and new experiences
3. **Elastic Weight Consolidation** - Protect important weights
4. **Progressive Neural Networks** - Add capacity without forgetting

**For Trading:**
- Mix recent data (70%) with historical data (30%)
- Lower learning rate for continuation
- Monitor performance on validation set
- Rollback if performance degrades >10%

## Market Adaptation

### Rolling Window Training
```
Week 1: Train on 2023-2024 data
Week 2: Train on 2023-2024 + continue on 2024-2025 data
Week 3: Train on 2024-2025 data (fresh) + continue best
```

### Regime Detection
- Bull market strategies
- Bear market strategies
- High volatility strategies
- Low volatility strategies
- Switch models based on detected regime

## Compute Budget Allocation

### Current (30 models × 500k = 15M timesteps)
```
100% Exploration
0% Exploitation
Result: Constant discovery, no refinement
```

### Recommended (35M timesteps)
```
64% Exploration (15 new × 1.5M)
28% Exploitation (5 continued × 2M)
8% Champions (1 elite × 3M)
Result: Balanced discovery + refinement
```

### Aggressive Refinement (30M timesteps)
```
50% Exploration (10 new × 1.5M)
40% Exploitation (8 continued × 1.5M)
10% Champions (2 elite × 1.5M)
Result: Focus on improving proven strategies
```

## Success Metrics

### Training Convergence
- **Value Loss**: Should decrease and stabilize
- **Policy Loss**: Should decrease and stabilize
- **Entropy**: Should decrease (more confident actions)
- **Explained Variance**: Should approach 1.0

### Performance Metrics
- **Sharpe Ratio**: >1.0 good, >2.0 excellent
- **Max Drawdown**: <20% acceptable, <10% good
- **Win Rate**: >50% baseline, >60% good
- **Consistency**: Low variance across runs

### Improvement Metrics
- **Week-over-Week**: Should show upward trend
- **Generation-over-Generation**: Children should outperform parents
- **Champion Longevity**: How long does best model stay best?

## Recommended Reading

1. **"Population Based Training of Neural Networks"** (DeepMind, 2017)
2. **"Proximal Policy Optimization Algorithms"** (Schulman et al., 2017)
3. **"FinRL: Deep Reinforcement Learning Framework"** (Liu et al., 2021)
4. **"Overcoming Catastrophic Forgetting"** (Kirkpatrick et al., 2017)
5. **"Curriculum Learning"** (Bengio et al., 2009)

## Summary

**Key Takeaways:**

1. ✅ **Increase training duration** to 1.5-2M timesteps minimum
2. ✅ **Retain and continue training** top performers
3. ✅ **Balance exploration and exploitation** (60/40 split)
4. ✅ **Track model lineage** and versioning
5. ✅ **Use lower learning rates** for fine-tuning
6. ✅ **Monitor for catastrophic forgetting**
7. ✅ **Adapt to market conditions** with rolling windows

**Expected Improvements:**
- Better convergence and stability
- Continuous strategy refinement
- Reduced wasted compute
- Higher peak performance
- More consistent results

**Next Steps:**
1. Implement Phase 1 (increase timesteps)
2. Add model retention system
3. Implement continuation training
4. Track and compare generations
5. Consider PBT for advanced optimization
