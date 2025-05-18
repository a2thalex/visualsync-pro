#!/bin/bash

echo "VisualSync Professional Launcher"
echo "=============================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.8 or higher from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
    echo "Error: Python 3.8 or higher is required."
    echo "Your current Python version is: $python_version"
    exit 1
fi

# Make the script executable
chmod +x run.py

# Run the application
./run.py

exit 0
