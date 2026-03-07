#!/bin/bash

# pm2_setup.sh
# 自动部署并启动 PM2 监控的服务

echo "Checking node/npm installation..."
if ! command -v npm &> /dev/null; then
    echo "npm could not be found. Installing Node.js..."
    # 针对 Ubuntu/Debian 系统
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

echo "Checking PM2 installation..."
if ! command -v pm2 &> /dev/null; then
    echo "PM2 could not be found. Installing PM2 globally..."
    npm install -g pm2
fi

echo "Setting up logs directory..."
mkdir -p /root/detectPaperLLM/depoly-pytorch/logs

echo "Killing existing uvicorn processes to free up port 9000..."
pkill -f uvicorn

echo "Starting service with PM2..."
cd /root/detectPaperLLM/depoly-pytorch
pm2 start ecosystem.config.js

echo "Configuring PM2 to start on boot..."
pm2 startup
# 此命令生成的启动脚本需要保存当前进程列表
pm2 save

echo "Deployment complete! Use 'pm2 status' to check service health."
