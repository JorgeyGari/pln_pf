#!/bin/bash

# Check if Python 3 and pip are installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3."
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip."
    exit 1
fi

# Create virtual environment named .venv
echo "Creating virtual environment .venv..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

echo "Setup completed."
