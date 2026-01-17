#!/usr/bin/env python3
# ###########################################################################
#
#  Launcher for Fairfax County Real Estate Forecast Application
#  Python 3.13+ | Modern UI | Dark Mode Support
#
# ###########################################################################

import os
import subprocess
import sys

# Verify Python version
if sys.version_info < (3, 13):
    print("  Warning: This application is optimized for Python 3.13+")
    print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
    print("   Some features may not work as expected.")
    print()

# Environment variables are set during AMP deployment via .project-metadata.yaml
# No manual configuration needed - they're automatically available

# Launch Streamlit app
# CML sets CDSW_APP_PORT for applications
port = int(os.getenv('CDSW_APP_PORT', '8090'))

print(" Launching Fairfax County Real Estate Forecast...")
print(f" Binding to port: {port}")
print(f" Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print()

# Launch Streamlit with error capture
print(" Starting Streamlit with debug output...")
result = subprocess.run([
    'streamlit',
    'run',
    'apps/app.py',
    '--server.port', str(port),
    '--server.address', '127.0.0.1',
    '--server.headless', 'true',
    '--server.enableCORS', 'false',
    '--server.enableXsrfProtection', 'false'
], capture_output=False, text=True)

if result.returncode != 0:
    print(f" Streamlit exited with code: {result.returncode}")
    sys.exit(result.returncode)
