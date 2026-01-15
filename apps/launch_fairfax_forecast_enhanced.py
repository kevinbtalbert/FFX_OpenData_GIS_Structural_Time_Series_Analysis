#!/usr/bin/env python3
# ###########################################################################
#
#  Launcher for Enhanced Fairfax County Real Estate Forecast Application
#  Python 3.13+ | Modern UI | Dark Mode Support
#
# ###########################################################################

import os
import subprocess
import sys

# Verify Python version
if sys.version_info < (3, 13):
    print("⚠️  Warning: This application is optimized for Python 3.13+")
    print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
    print("   Some features may not work as expected.")
    print()

# Set environment variables if needed
# Uncomment and configure these if not using .env file
# os.environ['AZURE_OPENAI_ENDPOINT'] = 'your-endpoint-here'
# os.environ['AZURE_OPENAI_API_KEY'] = 'your-key-here'
# os.environ['AZURE_OPENAI_DEPLOYMENT'] = 'gpt-4'

# Launch Streamlit app
port = os.getenv('CDSW_APP_PORT', '8501')

print("🚀 Launching Fairfax County Real Estate Forecast (Enhanced UI)...")
print(f"📍 Port: {port}")
print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print()

subprocess.run([
    'streamlit',
    'run',
    'apps/fairfax_forecast_enhanced.py',
    '--server.port', port,
    '--server.address', '0.0.0.0',
    '--theme.base', 'light',
    '--theme.primaryColor', '#667eea',
    '--theme.backgroundColor', '#ffffff',
    '--theme.secondaryBackgroundColor', '#f8fafc',
    '--theme.textColor', '#1e293b'
])
