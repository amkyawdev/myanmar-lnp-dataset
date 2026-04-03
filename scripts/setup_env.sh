#!/bin/bash
# Environment setup script for Myanmar LNP Dataset
# Usage: ./scripts/setup_env.sh

set -e

echo "Setting up Myanmar LNP Dataset environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher required"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw data/processed data/external
mkdir -p checkpoints/best_model checkpoints/logs

# Create .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

echo "Setup complete! Activate the environment with: source venv/bin/activate"