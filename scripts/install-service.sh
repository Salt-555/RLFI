#!/bin/bash
# RLFI Service Installation Script
# Run with: sudo ./scripts/install-service.sh

set -e

SERVICE_FILE="/home/salt/CodingProjects/RLFI/rlfi.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "=========================================="
echo "RLFI Service Installation"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo ./scripts/install-service.sh"
    exit 1
fi

# Create logs directory
mkdir -p /home/salt/CodingProjects/RLFI/logs
chown salt:salt /home/salt/CodingProjects/RLFI/logs

# Copy service file
echo "Installing service file..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/rlfi.service"

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service (auto-start on boot)
echo "Enabling service..."
systemctl enable rlfi.service

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start rlfi"
echo "  Stop:    sudo systemctl stop rlfi"
echo "  Status:  sudo systemctl status rlfi"
echo "  Logs:    tail -f /home/salt/CodingProjects/RLFI/logs/rlfi.log"
echo "  Errors:  tail -f /home/salt/CodingProjects/RLFI/logs/rlfi.error.log"
echo ""
echo "The service will auto-start on boot."
echo "To start now: sudo systemctl start rlfi"
