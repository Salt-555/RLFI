#!/bin/bash
# AUTOTEST Setup Script
# Sets up AUTOTEST to run automatically on Sunday nights (2AM-4AM)

echo "=========================================="
echo "AUTOTEST Setup Script"
echo "=========================================="
echo ""

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project directory: $PROJECT_DIR"

# Check if running as root for systemd setup
if [ "$EUID" -eq 0 ]; then 
    USE_SYSTEMD=true
    echo "Running as root - will set up systemd service"
else
    USE_SYSTEMD=false
    echo "Not running as root - will set up cron job"
fi

# Make autotest.py executable
chmod +x "$PROJECT_DIR/scripts/autotest.py"
echo "✓ Made autotest.py executable"

# Create necessary directories
mkdir -p "$PROJECT_DIR/autotest_models"
mkdir -p "$PROJECT_DIR/autotest_results"
mkdir -p "$PROJECT_DIR/autotest_logs"
echo "✓ Created AUTOTEST directories"

# Setup based on privileges
if [ "$USE_SYSTEMD" = true ]; then
    # Create systemd service file
    SERVICE_FILE="/etc/systemd/system/autotest.service"
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=AUTOTEST - Automated AI Trading System
After=network.target

[Service]
Type=simple
User=$SUDO_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 $PROJECT_DIR/scripts/autotest.py --daemon
Restart=always
RestartSec=3600

[Install]
WantedBy=multi-user.target
EOF
    
    echo "✓ Created systemd service file: $SERVICE_FILE"
    
    # Reload systemd
    systemctl daemon-reload
    echo "✓ Reloaded systemd"
    
    # Enable and start service
    echo ""
    echo "To enable AUTOTEST service:"
    echo "  sudo systemctl enable autotest.service"
    echo ""
    echo "To start AUTOTEST service:"
    echo "  sudo systemctl start autotest.service"
    echo ""
    echo "To check status:"
    echo "  sudo systemctl status autotest.service"
    echo ""
    echo "To view logs:"
    echo "  sudo journalctl -u autotest.service -f"
    
else
    # Setup cron job
    echo ""
    echo "Setting up cron job..."
    echo "The cron job will run AUTOTEST every Sunday at 2:00 AM"
    echo ""
    
    # Create cron entry
    CRON_ENTRY="0 2 * * 0 cd $PROJECT_DIR && /usr/bin/python3 $PROJECT_DIR/scripts/autotest.py --force >> $PROJECT_DIR/autotest_logs/cron.log 2>&1"
    
    # Check if cron entry already exists
    if crontab -l 2>/dev/null | grep -q "autotest.py"; then
        echo "⚠ Cron entry for AUTOTEST already exists"
    else
        # Add cron entry
        (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
        echo "✓ Added cron entry"
    fi
    
    echo ""
    echo "Cron job installed. AUTOTEST will run every Sunday at 2:00 AM"
    echo ""
    echo "To view current cron jobs:"
    echo "  crontab -l"
    echo ""
    echo "To remove the cron job:"
    echo "  crontab -e  # then delete the autotest.py line"
fi

echo ""
echo "=========================================="
echo "Manual Usage"
echo "=========================================="
echo ""
echo "Run AUTOTEST immediately (skip schedule):"
echo "  python3 scripts/autotest.py --force"
echo ""
echo "Run only training phase:"
echo "  python3 scripts/autotest.py --train-only"
echo ""
echo "Run only backtesting phase:"
echo "  python3 scripts/autotest.py --backtest-only"
echo ""
echo "Run as daemon (continuously check for scheduled time):"
echo "  python3 scripts/autotest.py --daemon"
echo ""
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo ""
echo "Edit configuration file:"
echo "  $PROJECT_DIR/config/autotest_config.yaml"
echo ""
echo "Key settings:"
echo "  - Schedule: Day and time window"
echo "  - Training: Number of models, parameter variations"
echo "  - Backtesting: Metrics and ranking weights"
echo "  - Paper Trading: Duration and risk parameters"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
