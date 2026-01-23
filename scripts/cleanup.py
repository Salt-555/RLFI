#!/usr/bin/env python3
"""
RLFI Data Cleanup Script
Manages data retention to prevent disk bloat over time.

Run manually: ./scripts/cleanup.py
Or via cron: 0 3 * * 0 /path/to/cleanup.py  (Sunday 3AM)
"""
import os
import sys
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
RLFI_ROOT = Path(__file__).parent.parent
DATA_DIR = RLFI_ROOT / "data" / "raw"
MODELS_DIR = RLFI_ROOT / "autotest_models"
LOGS_DIR = RLFI_ROOT / "logs"
RESULTS_DIR = RLFI_ROOT / "autotest_results"
AUTOTEST_LOGS_DIR = RLFI_ROOT / "autotest_logs"

# Retention policies (days)
RETENTION_POLICIES = {
    'market_data': 7,      # Keep market data CSVs for 7 days (re-downloaded as needed)
    'culled_models': 30,   # Keep culled model files for 30 days
    'tensorboard_logs': 14, # Keep TensorBoard logs for 14 days
    'training_logs': 90,   # Keep training summaries for 90 days
    'backtest_results': 90, # Keep backtest results for 90 days
}


def get_file_age_days(filepath: Path) -> float:
    """Get file age in days."""
    if not filepath.exists():
        return 0
    mtime = filepath.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / 86400


def cleanup_market_data():
    """Remove old market data CSVs (they get re-downloaded as needed)."""
    if not DATA_DIR.exists():
        return 0, 0
    
    removed_count = 0
    removed_bytes = 0
    max_age = RETENTION_POLICIES['market_data']
    
    for csv_file in DATA_DIR.glob("*.csv"):
        age = get_file_age_days(csv_file)
        if age > max_age:
            size = csv_file.stat().st_size
            csv_file.unlink()
            removed_count += 1
            removed_bytes += size
            print(f"  Removed: {csv_file.name} ({age:.1f} days old, {size/1024:.1f} KB)")
    
    return removed_count, removed_bytes


def cleanup_culled_models():
    """Remove model files for models that were culled more than N days ago."""
    if not MODELS_DIR.exists():
        return 0, 0
    
    removed_count = 0
    removed_bytes = 0
    max_age = RETENTION_POLICIES['culled_models']
    
    # Load lifecycle database to find culled models
    try:
        import sqlite3
        db_path = RLFI_ROOT / "autotest_strategies.db"
        if not db_path.exists():
            return 0, 0
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Find models culled more than max_age days ago
        cutoff_date = (datetime.now() - timedelta(days=max_age)).isoformat()
        cursor.execute('''
            SELECT model_id, model_path FROM model_lifecycle
            WHERE current_state = 'culled' AND paper_trading_ended_at < ?
        ''', (cutoff_date,))
        
        culled_models = cursor.fetchall()
        conn.close()
        
        for model_id, model_path in culled_models:
            # Remove model zip
            if model_path and os.path.exists(model_path):
                size = os.path.getsize(model_path)
                os.remove(model_path)
                removed_count += 1
                removed_bytes += size
                print(f"  Removed culled model: {model_path} ({size/1024:.1f} KB)")
            
            # Remove associated files
            for pattern in [f"{model_id}_*.yaml", f"{model_id}_*.json"]:
                for f in MODELS_DIR.glob(pattern):
                    size = f.stat().st_size
                    f.unlink()
                    removed_bytes += size
                    print(f"  Removed: {f.name}")
    
    except Exception as e:
        print(f"  Warning: Could not cleanup culled models: {e}")
    
    return removed_count, removed_bytes


def cleanup_tensorboard_logs():
    """Remove old TensorBoard log directories."""
    if not LOGS_DIR.exists():
        return 0, 0
    
    removed_count = 0
    removed_bytes = 0
    max_age = RETENTION_POLICIES['tensorboard_logs']
    
    for log_dir in LOGS_DIR.iterdir():
        if log_dir.is_dir():
            age = get_file_age_days(log_dir)
            if age > max_age:
                size = sum(f.stat().st_size for f in log_dir.rglob("*") if f.is_file())
                shutil.rmtree(log_dir)
                removed_count += 1
                removed_bytes += size
                print(f"  Removed: {log_dir.name} ({age:.1f} days old, {size/1024:.1f} KB)")
    
    return removed_count, removed_bytes


def cleanup_old_logs():
    """Remove old training summary and result files."""
    removed_count = 0
    removed_bytes = 0
    
    # Training logs
    if AUTOTEST_LOGS_DIR.exists():
        max_age = RETENTION_POLICIES['training_logs']
        for log_file in AUTOTEST_LOGS_DIR.glob("*.yaml"):
            age = get_file_age_days(log_file)
            if age > max_age:
                size = log_file.stat().st_size
                log_file.unlink()
                removed_count += 1
                removed_bytes += size
    
    # Backtest results
    if RESULTS_DIR.exists():
        max_age = RETENTION_POLICIES['backtest_results']
        for result_file in RESULTS_DIR.glob("*"):
            if result_file.is_file():
                age = get_file_age_days(result_file)
                if age > max_age:
                    size = result_file.stat().st_size
                    result_file.unlink()
                    removed_count += 1
                    removed_bytes += size
    
    return removed_count, removed_bytes


def get_disk_usage():
    """Get current disk usage for RLFI directories."""
    total_size = 0
    breakdown = {}
    
    dirs_to_check = [
        ('Market Data', DATA_DIR),
        ('Models', MODELS_DIR),
        ('TensorBoard Logs', LOGS_DIR),
        ('Training Logs', AUTOTEST_LOGS_DIR),
        ('Results', RESULTS_DIR),
    ]
    
    for name, dir_path in dirs_to_check:
        if dir_path.exists():
            size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
            breakdown[name] = size
            total_size += size
        else:
            breakdown[name] = 0
    
    return total_size, breakdown


def main():
    print("=" * 60)
    print("RLFI DATA CLEANUP")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Show current usage
    print("\nCurrent Disk Usage:")
    total_before, breakdown = get_disk_usage()
    for name, size in breakdown.items():
        print(f"  {name}: {size/1024/1024:.2f} MB")
    print(f"  TOTAL: {total_before/1024/1024:.2f} MB")
    
    # Run cleanup
    print("\nRunning cleanup...")
    
    print("\n[1/4] Market Data (>{} days):".format(RETENTION_POLICIES['market_data']))
    count, bytes_removed = cleanup_market_data()
    total_removed = bytes_removed
    if count == 0:
        print("  Nothing to clean")
    
    print("\n[2/4] Culled Models (>{} days):".format(RETENTION_POLICIES['culled_models']))
    count, bytes_removed = cleanup_culled_models()
    total_removed += bytes_removed
    if count == 0:
        print("  Nothing to clean")
    
    print("\n[3/4] TensorBoard Logs (>{} days):".format(RETENTION_POLICIES['tensorboard_logs']))
    count, bytes_removed = cleanup_tensorboard_logs()
    total_removed += bytes_removed
    if count == 0:
        print("  Nothing to clean")
    
    print("\n[4/4] Old Log Files (>{} days):".format(RETENTION_POLICIES['training_logs']))
    count, bytes_removed = cleanup_old_logs()
    total_removed += bytes_removed
    if count == 0:
        print("  Nothing to clean")
    
    # Show results
    total_after, _ = get_disk_usage()
    
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Space freed: {total_removed/1024/1024:.2f} MB")
    print(f"Before: {total_before/1024/1024:.2f} MB")
    print(f"After:  {total_after/1024/1024:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
