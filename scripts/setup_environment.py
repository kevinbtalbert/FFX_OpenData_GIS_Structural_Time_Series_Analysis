#!/usr/bin/env python3
# ###########################################################################
#
#  Environment Setup Script
#  Validates installation and prepares the environment
#
# ###########################################################################

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f" Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f" Python 3.7+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    required_packages = [
        'pandas',
        'numpy',
        'streamlit',
        'plotly',
        'prophet'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package} installed")
        except ImportError:
            print(f" {package} not found")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_csv_files():
    """Check if required CSV files exist."""
    print("\nChecking CSV files...")
    csv_dir = Path('csvs')
    
    if not csv_dir.exists():
        print(f" CSV directory not found: {csv_dir}")
        return False
    
    required_files = [
        'Tax_Administration_s_Real_Estate_-_Assessed_Values.csv',
    ]
    
    all_found = True
    for filename in required_files:
        filepath = csv_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f" {filename} ({size_mb:.1f} MB)")
        else:
            print(f" {filename} not found")
            all_found = False
    
    return all_found

def check_azure_openai_config():
    """Check if Azure OpenAI is configured."""
    print("\nChecking Azure OpenAI configuration...")
    
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
    
    if endpoint and api_key:
        print(" Azure OpenAI credentials found")
        print(f"  Endpoint: {endpoint}")
        print(f"  Deployment: {deployment}")
        return True
    else:
        print(" Azure OpenAI not configured (will use mock chatbot)")
        print("  These are set during AMP deployment via .project-metadata.yaml")
        print("  The application will work with mock chatbot for testing")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = [
        'data/models',
        'data/forecasts',
        'data/regressors'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f" {directory}")
    
    return True

def install_sts_module():
    """Install the sts module in development mode."""
    print("\nInstalling sts module...")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', '.'],
            check=True,
            capture_output=True
        )
        print(" sts module installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed to install sts module: {e}")
        return False

def main():
    """Main setup function."""
    print_header("Fairfax County Real Estate Forecast - Environment Setup")
    
    # Track overall status
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        all_checks_passed = False
        print(f"\n Missing packages: {', '.join(missing)}")
        print("Install with: pip3 install -r requirements.txt")
        print("              pip3 install prophet==1.1.5")
    
    # Check CSV files
    if not check_csv_files():
        all_checks_passed = False
        print("\n Please ensure CSV files are in the csvs/ directory")
        print("Download from: https://data.fairfaxcounty.gov/")
    
    # Check Azure OpenAI (optional)
    check_azure_openai_config()
    
    # Create directories
    create_directories()
    
    # Install sts module
    if not install_sts_module():
        all_checks_passed = False
    
    # Summary
    print_header("Setup Summary")
    
    if all_checks_passed:
        print(" All checks passed! You're ready to go.")
        print("\nNext steps:")
        print("  1. Launch the app: streamlit run apps/app.py")
        print("  2. Or use the API: python3 -c 'from sts.models.projection_api import ProjectionAPI'")
        print("\nFor more information, see README.md or START_HERE.md")
    else:
        print(" Some checks failed, but continuing with deployment.")
        print("The application may still work with reduced functionality.")
        print("\nFor help, see:")
        print("  - README.md (Troubleshooting section)")
        print("  - START_HERE.md")
    
    # Always return 0 to not block deployment
    return 0

if __name__ == '__main__':
    main()
