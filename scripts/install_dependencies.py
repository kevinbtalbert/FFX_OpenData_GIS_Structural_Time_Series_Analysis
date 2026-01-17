#!/usr/bin/env python3
"""
Install Dependencies Script
Installs all required packages and sets up the environment.
"""

import sys
import os
import subprocess

def main():
    print("Installing dependencies...")
    print("=" * 60)
    
    # Upgrade pip
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Prophet removed - using simple projection model instead
    
    # Install sts package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    print(f"\nPython version: {sys.version}")
    
    # Test critical imports
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly
        from sklearn import linear_model
        print(" All core packages installed successfully!")
    except ImportError as e:
        print(f" Import error: {e}")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/forecasts', exist_ok=True)
    os.makedirs('data/regressors', exist_ok=True)
    print(" Data directories created")
    
    # Validate CSV files
    csv_dir = 'csvs'
    required_files = [
        'Tax_Administration_s_Real_Estate_-_Assessed_Values.csv',
    ]
    
    for filename in required_files:
        filepath = os.path.join(csv_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f" Found {filename} ({size_mb:.1f} MB)")
        else:
            print(f" Warning: {filename} not found in csvs/ directory")
    
    print("\n" + "=" * 60)
    print("Installation complete! Ready to launch application.")
    print("=" * 60)

if __name__ == '__main__':
    main()
