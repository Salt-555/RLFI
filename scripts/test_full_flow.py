#!/usr/bin/env python3
"""
Full Flow Test - Complete Pipeline Test
Tests training, grokking detection, database tracking, and reporting
"""
import sys
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

from src.autotest.automated_trainer import AutomatedTrainer
from src.autotest.strategy_database import StrategyDatabase
from datetime import datetime
import yaml

print('='*80)
print('FULL FLOW TEST - Complete Pipeline')
print('='*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Initialize trainer
trainer = AutomatedTrainer()

# Create test parameter sets (2 models to test multi-model flow)
parameter_sets = [
    {
        'model_id': 'FLOW_TEST_001',
        'algorithm': 'ppo',
        'tickers': ['AAPL', 'MSFT', 'GOOGL'],
        'learning_rate': 0.0003,
        'timesteps': 15000,
        'eval_freq': 5000,
        'tech_indicators': ['macd', 'rsi_14', 'atr'],
        'n_steps': 1024,
        'batch_size': 64,
        'gamma': 0.99,
        'net_arch': [[256, 128], [256, 128]]
    },
    {
        'model_id': 'FLOW_TEST_002',
        'algorithm': 'ppo',
        'tickers': ['NVDA', 'AMD', 'INTC'],
        'learning_rate': 0.0001,
        'timesteps': 15000,
        'eval_freq': 5000,
        'tech_indicators': ['macd', 'rsi_14', 'bollinger'],
        'n_steps': 1024,
        'batch_size': 64,
        'gamma': 0.99,
        'net_arch': [[256, 128], [256, 128]]
    }
]

print(f"Testing with {len(parameter_sets)} models...")
print()

# Run full training flow
results = trainer.train_all_models(parameter_sets)

print()
print('='*80)
print('TEST SUMMARY')
print('='*80)

# Check results
successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]

print(f"Models Trained: {len(results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if failed:
    print(f"\nFailed Models:")
    for r in failed:
        print(f"  - {r['model_id']}: {r.get('error', 'Unknown error')}")

print()
print('GROKKING METRICS:')
print('-'*80)

# Check database
db = StrategyDatabase()
grok_stats = db.get_grokking_statistics()

print(f"Database Records: {grok_stats['total_models']}")
print(f"Grokking Rate: {grok_stats['grokking_percentage']:.1f}%")
print(f"Avg Improvement: {grok_stats['avg_improvement_pct']:.1f}%")

if grok_stats.get('phase_breakdown'):
    print(f"\nPhase Distribution:")
    for phase_info in grok_stats['phase_breakdown']:
        print(f"  {phase_info['phase']}: {phase_info['count']} models")

# Check individual model details
print()
print('MODEL DETAILS:')
print('-'*80)

for result in successful:
    model_id = result['model_id']
    metadata_path = result['metadata_path']
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    grok = metadata.get('grokking_analysis', {})
    print(f"\n{model_id}:")
    print(f"  Phase: {grok.get('phase', 'unknown')}")
    print(f"  Grokked: {grok.get('has_grokked', False)}")
    improvement = grok.get('relative_improvement_pct', 0)
    if improvement != 0:
        print(f"  Improvement: {improvement:.1f}%")
    else:
        print(f"  Improvement: N/A")
    print(f"  Score: {grok.get('score', 0)}")
    print(f"  State Dim: {metadata.get('state_dimension')}")
    print(f"  Market Context: {metadata.get('market_context_enabled')}")

print()
print('='*80)
print('FULL FLOW TEST COMPLETE')
print('='*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Quick verification
all_working = len(successful) == len(parameter_sets)
status_msg = "ALL SYSTEMS WORKING" if all_working else "SOME ISSUES DETECTED"
print(f"\nStatus: {status_msg}")
print(f"Success Rate: {len(successful)}/{len(parameter_sets)}")
