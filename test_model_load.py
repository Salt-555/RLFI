"""
Test model loading to diagnose the issue
"""
from stable_baselines3 import PPO
import os

print("Testing model loading...")

model_path = "models/ppo_finrl_test.zip"

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

print(f"Loading model from {model_path}...")

try:
    model = PPO.load(model_path, device='cpu')
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Policy type: {type(model.policy)}")
    
    # Test prediction with dummy observation
    import numpy as np
    dummy_obs = np.random.randn(13)  # state_space = 13
    
    print("\nTesting prediction...")
    action, _states = model.predict(dummy_obs, deterministic=True)
    print(f"✅ Prediction successful!")
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    import traceback
    traceback.print_exc()
