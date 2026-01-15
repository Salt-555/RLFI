"""
Test using actual FinRL components instead of custom implementations
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO

print("="*60)
print("FINRL WORKFLOW TEST")
print("="*60)

print("\n1. Downloading data using FinRL's YahooDownloader...")
df = YahooDownloader(
    start_date='2022-01-01',
    end_date='2024-01-01',
    ticker_list=['AAPL', 'MSFT']
).fetch_data()

print(f"Downloaded data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

print("\n2. Feature engineering using FinRL's FeatureEngineer...")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False
)

processed = fe.preprocess_data(df)
print(f"Processed data shape: {processed.shape}")
print(f"Columns: {processed.columns.tolist()}")

print("\n3. Splitting data...")
unique_dates = processed['date'].unique()
train_end = int(len(unique_dates) * 0.7)
val_end = int(len(unique_dates) * 0.85)

train_dates = unique_dates[:train_end]
val_dates = unique_dates[train_end:val_end]
test_dates = unique_dates[val_end:]

train = processed[processed['date'].isin(train_dates)].reset_index(drop=True)
val = processed[processed['date'].isin(val_dates)].reset_index(drop=True)
test = processed[processed['date'].isin(test_dates)].reset_index(drop=True)

print(f"Train: {len(train)} rows, {len(train_dates)} days")
print(f"Val: {len(val)} rows, {len(val_dates)} days")
print(f"Test: {len(test)} rows, {len(test_dates)} days")

print("\n4. Creating FinRL StockTradingEnv...")
stock_dimension = len(train['tic'].unique())
state_space = 1 + 2*stock_dimension + len(['macd', 'rsi_30', 'cci_30', 'dx_30'])*stock_dimension
print(f"Stock dimension: {stock_dimension}")
print(f"State space: {state_space}")

# FinRL expects data indexed by day number, not date
train_indexed = train.copy()
train_indexed.index = train_indexed['date'].factorize()[0]

env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "num_stock_shares": [0] * stock_dimension,
    "buy_cost_pct": [0.001] * stock_dimension,
    "sell_cost_pct": [0.001] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": ['macd', 'rsi_30', 'cci_30', 'dx_30'],
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockTradingEnv(df=train_indexed, **env_kwargs)

print("\n5. Training PPO agent (10k timesteps - quick test)...")
model = PPO("MlpPolicy", e_train_gym, verbose=1, tensorboard_log="./logs/", device='cpu')
model.learn(total_timesteps=10000, progress_bar=True)

print("\n6. Saving model...")
os.makedirs('models', exist_ok=True)
model.save('models/ppo_finrl_test.zip')

print("\n7. Testing on test data...")
test_indexed = test.copy()
test_indexed.index = test_indexed['date'].factorize()[0]
e_test_gym = StockTradingEnv(df=test_indexed, turbulence_threshold=None, **env_kwargs)
obs, _ = e_test_gym.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = e_test_gym.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"\nTest complete!")
print(f"Total reward: {total_reward}")
print(f"Final portfolio value: ${e_test_gym.state_memory[-1][0]:,.2f}")

print("\n" + "="*60)
print("FINRL WORKFLOW TEST COMPLETE!")
print("="*60)
