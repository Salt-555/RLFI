#!/usr/bin/env python3
"""Test Alpaca API keys"""
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

print(f"Testing API Key: {api_key[:10]}...")
print(f"Testing Secret: {secret_key[:10]}...")

try:
    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    print(f"\n✅ SUCCESS!")
    print(f"Account Number: {account.account_number}")
    print(f"Cash: ${account.cash}")
    print(f"Portfolio Value: ${account.portfolio_value}")
    print(f"Buying Power: ${account.buying_power}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
