#!/bin/bash

# Voice Chat Roulette Server Deployment Script
# Usage: ./deploy.sh [start|stop|restart|status|update]

set -e

# Configuration
SERVER_DIR="/opt/voice-chat-server"
APP_NAME="voice-chat-server"
VENV_DIR="$SERVER_DIR/venv"
REPO_URL="https://github.com/your-username/voice-chat-roulette.git"
BRANCH="main"
PORT="8000"
HOST="0.0.0.0"
LOG_FILE="$SERVER_DIR/server.log"
PID_FILE="$SERVER_DIR/server.pid"
ENV_FILE="$SERVER_DIR/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_root() {
    if [ "$EUID" -eq 0 ]; then
        error "Please do not run as root"
        exit 1
    fi
}

check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if ! command -v python3.8 &> /dev/null; then
        error "Python 3.8+ is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        error "git is required but not installed"
        exit 1
    fi
    
    success "All requirements met"
}

setup_directory() {
    log "Setting up server directory..."
    
    if [ ! -d "$SERVER_DIR" ]; then
        sudo mkdir -p "$SERVER_DIR"
        sudo chown $(whoami):$(whoami) "$SERVER_DIR"
        success "Created server directory: $SERVER_DIR"
    else
        warning "Server directory already exists: $SERVER_DIR"
    fi
    
    cd "$SERVER_DIR"
}

setup_venv() {
    log "Setting up Python virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        success "Created virtual environment"
    else
        warning "Virtual environment already exists"
    fi
    
    source "$VENV_DIR/bin/activate"
}

install_dependencies() {
    log "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -U pip
        pip install -r requirements.txt
        success "Dependencies installed"
    else
        error "requirements.txt not found"
        exit 1
    fi
}

create_env_file() {
    log "Creating environment file..."
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Database Configuration
MONGO_URL=mongodb://localhost:27017/
DB_NAME=voice_chat

# Server Configuration
HOST=$HOST
PORT=$PORT

# Security
SECRET_KEY=$(openssl rand -hex 32)

# Logging
LOG_LEVEL=INFO
LOG_FILE=$LOG_FILE

# CORS
ALLOWED_ORIGINS=*
EOF
        success "Environment file created: $ENV_FILE"
        warning "Please edit $ENV_FILE with your actual MongoDB credentials"
    else
        warning "Environment file already exists: $ENV_FILE"
    fi
}

download_code() {
    log "Downloading server code..."
    
    if [ ! -d "$SERVER_DIR/.git" ]; then
        git clone "$REPO_URL" .
        git checkout "$BRANCH"
        success "Code downloaded from $REPO_URL"
    else
        git pull origin "$BRANCH"
        success "Code updated from $REPO_URL"
    fi
}

start_server() {
    log "Starting server..."
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            warning "Server is already running (PID: $PID)"
            return
        fi
    fi
    
    cd "$SERVER_DIR"
    source "$VENV_DIR/bin/activate"
    
    nohup uvicorn main:app --host $HOST --port $PORT \
        --workers 4 \
        --timeout-keep-alive 60 \
        --log-level info \
        --access-log \
        >> "$LOG_FILE" 2>&1 &
    
    echo $! > "$PID_FILE"
    sleep 3
    
    if ps -p $! > /dev/null 2>&1; then
        success "Server started successfully (PID: $!)"
        log "Server is running on http://$HOST:$PORT"
        log "Check logs: tail -f $LOG_FILE"
    else
        error "Failed to start server"
        exit 1
    fi
}

stop_server() {
    log "Stopping server..."
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID
            rm "$PID_FILE"
            success "Server stopped (PID: $PID)"
        else
            warning "No running server found (PID: $PID)"
            rm "$PID_FILE"
        fi
    else
        warning "No PID file found: $PID_FILE"
        # Try to find and kill by port
        PIDS=$(lsof -ti:$PORT || true)
        if [ ! -z "$PIDS" ]; then
            echo "$PIDS" | xargs kill
            success "Killed processes on port $PORT"
        else
            warning "No processes found on port $PORT"
        fi
    fi
}

restart_server() {
    stop_server
    sleep 2
    start_server
}

server_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            success "Server is running (PID: $PID)"
            echo "Port: $PORT"
            echo "Log file: $LOG_FILE"
            echo "Uptime: $(ps -p $PID -o etime=)"
            return 0
        else
            error "Server is not running (stale PID: $PID)"
            rm "$PID_FILE"
            return 1
        fi
    else
        error "Server is not running"
        return 1
    fi
}

view_logs() {
    if [ -f "$LOG_FILE" ]; then
        log "Showing last 50 lines of log file:"
        tail -50 "$LOG_FILE"
    else
        warning "Log file not found: $LOG_FILE"
    fi
}

update_server() {
    log "Updating server..."
    
    stop_server
    download_code
    setup_venv
    install_dependencies
    start_server
    
    success "Server updated successfully"
}

setup_firewall() {
    log "Configuring firewall..."
    
    if command -v ufw &> /dev/null; then
        sudo ufw allow $PORT
        sudo ufw allow ssh
        success "Firewall configured for port $PORT"
    else
        warning "ufw not installed, skipping firewall configuration"
    fi
}

setup_nginx() {
    log "Setting up Nginx reverse proxy (optional)..."
    
    if ! command -v nginx &> /dev/null; then
        warning "Nginx not installed, skipping reverse proxy setup"
        return
    fi
    
    # Create nginx config
    NGINX_CONFIG="/etc/nginx/sites-available/$APP_NAME"
    
    if [ ! -f "$NGINX_CONFIG" ]; then
        sudo tee "$NGINX_CONFIG" > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /ws/ {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOF
        
        sudo ln -sf "$NGINX_CONFIG" "/etc/nginx/sites-enabled/"
        sudo nginx -t
        sudo systemctl reload nginx
        
        success "Nginx configured. Don't forget to:"
        warning "1. Replace 'your-domain.com' with your actual domain"
        warning "2. Setup SSL certificate with certbot"
    else
        warning "Nginx config already exists: $NGINX_CONFIG"
    fi
}

setup_systemd() {
    log "Setting up systemd service (optional)..."
    
    SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
    
    if [ ! -f "$SERVICE_FILE" ]; then
        sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Voice Chat Roulette Server
After=network.target

[Service]
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$SERVER_DIR
Environment=PATH=$VENV_DIR/bin:/usr/bin:/bin
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/uvicorn main:app --host $HOST --port $PORT --workers 4
Restart=always
RestartSec=5
StandardOutput=file:$LOG_FILE
StandardError=file:$LOG_FILE

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable $APP_NAME
        
        success "Systemd service created. You can now use:"
        echo "  sudo systemctl start $APP_NAME"
        echo "  sudo systemctl stop $APP_NAME"
        echo "  sudo systemctl status $APP_NAME"
    else
        warning "Systemd service already exists: $SERVICE_FILE"
    fi
}

show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the server"
    echo "  stop      - Stop the server"
    echo "  restart   - Restart the server"
    echo "  status    - Check server status"
    echo "  logs      - View server logs"
    echo "  update    - Update server code and restart"
    echo "  install   - Full installation"
    echo "  setup     - Setup additional services (nginx, systemd)"
    echo ""
    echo "Examples:"
    echo "  $0 install   # First-time installation"
    echo "  $0 start     # Start server"
    echo "  $0 status    # Check status"
}

full_installation() {
    log "Starting full installation..."
    
    check_root
    check_requirements
    setup_directory
    download_code
    setup_venv
    install_dependencies
    create_env_file
    setup_firewall
    
    success "Installation completed!"
    echo ""
    warning "IMPORTANT: Please edit $ENV_FILE and set your MongoDB credentials"
    echo ""
    echo "Next steps:"
    echo "  1. Edit $ENV_FILE"
    echo "  2. Run: $0 start"
    echo "  3. Run: $0 setup (optional, for production)"
}

# Main execution
case "${1:-}" in
    "start")
        start_server
        ;;
    "stop")
        stop_server
        ;;
    "restart")
        restart_server
        ;;
    "status")
        server_status
        ;;
    "logs")
        view_logs
        ;;
    "update")
        update_server
        ;;
    "install")
        full_installation
        ;;
    "setup")
        setup_nginx
        setup_systemd
        ;;
    "")
        show_usage
        ;;
    *)
        error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac