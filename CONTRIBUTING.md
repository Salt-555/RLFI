# Contributing to RL Trading Bot

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/yourusername/RLFI.git
cd RLFI
```

3. Set up development environment:
```bash
./setup.sh
source venv/bin/activate
```

## Project Structure

```
RLFI/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── environment/       # Trading environment
│   ├── agents/            # RL agents and training
│   ├── evaluation/        # Backtesting and metrics
│   └── trading/           # Paper/live trading
├── scripts/               # Executable scripts
├── config/                # Configuration files
├── examples/              # Usage examples
└── tests/                 # Unit tests (TODO)
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use docstrings for all public functions

### Example:
```python
def calculate_sharpe_ratio(
    portfolio_values: List[float], 
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio for portfolio.
    
    Args:
        portfolio_values: List of portfolio values over time
        risk_free_rate: Annual risk-free rate (default: 0.0)
    
    Returns:
        Sharpe ratio (annualized)
    """
    # Implementation
    pass
```

## Adding New Features

### 1. New RL Algorithm

Add to `src/agents/trainer.py`:

```python
elif self.model_name == 'new_algo':
    self.model = NewAlgo(
        **common_params,
        # Algorithm-specific parameters
    )
```

Update `config/default_config.yaml`:

```yaml
training:
  algorithms:
    - ppo
    - a2c
    - new_algo
  
  new_algo:
    learning_rate: 0.0003
    # Other hyperparameters
```

### 2. New Technical Indicator

Add to `src/data/feature_engineer.py`:

```python
def add_custom_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add custom technical indicator."""
    # Implementation
    return df
```

### 3. New Evaluation Metric

Add to `src/evaluation/metrics.py`:

```python
def calculate_new_metric(portfolio_values: List[float]) -> float:
    """Calculate new performance metric."""
    # Implementation
    return metric_value
```

## Testing

### Running Tests
```bash
# TODO: Add pytest tests
pytest tests/
```

### Writing Tests
```python
def test_sharpe_ratio():
    portfolio_values = [100, 105, 110, 108, 115]
    sharpe = calculate_sharpe_ratio(portfolio_values)
    assert sharpe > 0
```

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Test thoroughly
4. Commit with clear messages:
```bash
git commit -m "Add: New feature description"
```

5. Push to your fork:
```bash
git push origin feature/your-feature-name
```

6. Create Pull Request on GitHub

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Tested on sample data

## Areas for Contribution

### High Priority
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] More RL algorithms (SAC, TD3)
- [ ] Real-time data streaming
- [ ] Advanced risk management

### Medium Priority
- [ ] Web dashboard for monitoring
- [ ] More data sources (Alpha Vantage, Polygon)
- [ ] Multi-timeframe support
- [ ] Options trading support
- [ ] Sentiment analysis integration

### Low Priority
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Mobile notifications
- [ ] Portfolio optimization
- [ ] Tax reporting

## Code Review Guidelines

### What We Look For
- Clean, readable code
- Proper error handling
- Efficient algorithms
- Good documentation
- Test coverage

### What We Avoid
- Hardcoded values
- Magic numbers
- Overly complex logic
- Poor variable names
- Missing error handling

## Questions?

- Open an issue for bugs
- Start a discussion for features
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🚀
