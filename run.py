#!/usr/bin/env python
"""
Run script for VisualSync Professional
This script sets up the environment and launches the application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up"""
    # Check if Python version is compatible
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"Error: Python 3.8 or higher is required. You have {python_version.major}.{python_version.minor}")
        return False
    
    # Check if required directories exist
    required_dirs = [
        "assets/samples",
        "assets/generated/images",
        "assets/generated/videos",
        "assets/generated/stems"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating directory: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("Warning: .env file not found. Creating a template .env file.")
        with open(env_file, "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key\n")
            f.write("HUGGINGFACE_TOKEN=your_huggingface_token\n")
        print("Please edit the .env file and add your API keys.")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Checking and installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_application():
    """Run the VisualSync Professional application"""
    print("Starting VisualSync Professional...")
    try:
        # Change to the src directory
        os.chdir("src")
        # Run the application
        subprocess.run([sys.executable, "app.py"])
    except Exception as e:
        print(f"Error running application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("VisualSync Professional Setup")
    print("============================")
    
    if not check_environment():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    if not run_application():
        sys.exit(1)
    
    sys.exit(0)
