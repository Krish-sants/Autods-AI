#!/usr/bin/env bash
# One-shot deployment script for AutoDS-AI on a fresh Ubuntu server.
# Run as the ubuntu user (has sudo). Do NOT run as root.
set -euo pipefail

REPO_URL="https://github.com/Krish-sants/Autods-AI.git"
APP_DIR="$HOME/autods-ai"

echo "==> Installing Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to re-login for group membership."
fi

echo "==> Opening firewall ports..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT 2>/dev/null || true
sudo apt-get install -y -q iptables-persistent netfilter-persistent 2>/dev/null || true
sudo netfilter-persistent save 2>/dev/null || true

echo "==> Cloning repo..."
if [ -d "$APP_DIR" ]; then
    echo "   Directory exists — pulling latest..."
    git -C "$APP_DIR" pull
else
    git clone "$REPO_URL" "$APP_DIR"
fi

echo "==> Configuring environment..."
if [ ! -f "$APP_DIR/backend/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/backend/.env"
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  Edit $APP_DIR/backend/.env before continuing:"
    echo "    DOMAIN=your-domain-or-ip"
    echo "    APP_ACCESS_PASSWORD=a-strong-password"
    echo "    GOOGLE_API_KEY=optional"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
    read -r -p "Press Enter after editing the .env file..." _
fi

# Export DOMAIN for docker-compose.prod.yml interpolation
DOMAIN=$(grep '^DOMAIN=' "$APP_DIR/backend/.env" | cut -d= -f2 | tr -d '[:space:]')
export DOMAIN

echo "==> Building and starting services (this takes ~10 min on first run)..."
cd "$APP_DIR"
docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "Done! App is starting up."
echo "  URL: https://$DOMAIN  (or http://$DOMAIN if using a bare IP)"
echo ""
echo "Useful commands:"
echo "  View logs:    docker compose -f docker-compose.prod.yml logs -f"
echo "  Restart:      docker compose -f docker-compose.prod.yml restart"
echo "  Update:       git pull && docker compose -f docker-compose.prod.yml up -d --build"
echo "  Stop:         docker compose -f docker-compose.prod.yml down"
