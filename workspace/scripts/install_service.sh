#!/bin/bash
# Install face-recognition as a systemd service on Raspberry Pi 5.
# Run once: sudo bash scripts/install_service.sh
# After install: sudo systemctl enable --now face-recognition

set -e

SERVICE_NAME="face-recognition"
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"
ENV_FILE="${WORKSPACE_DIR}/.env"

echo "Installing ${SERVICE_NAME} service..."
echo "  Workspace : ${WORKSPACE_DIR}"
echo "  Python    : ${PYTHON_BIN}"
echo "  Env file  : ${ENV_FILE}"

# ── Create the unit file ────────────────────────────────────────────────────
cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=Family Face Recognition System
After=network.target
Wants=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${WORKSPACE_DIR}
EnvironmentFile=-${ENV_FILE}
ExecStart=${PYTHON_BIN} ${WORKSPACE_DIR}/src/main.py --headless --no-alerts
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}
KillMode=process
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF

# ── Reload and enable ────────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}.service

echo ""
echo "Service installed. Commands:"
echo "  sudo systemctl start   ${SERVICE_NAME}   # start now"
echo "  sudo systemctl stop    ${SERVICE_NAME}   # stop"
echo "  sudo systemctl status  ${SERVICE_NAME}   # check status"
echo "  sudo journalctl -u ${SERVICE_NAME} -f    # follow logs"
echo ""
echo "To run with alerts enabled:"
echo "  Edit ExecStart in /etc/systemd/system/${SERVICE_NAME}.service"
echo "  Then: sudo systemctl daemon-reload && sudo systemctl restart ${SERVICE_NAME}"
