"""
Packaging script for creating a standalone executable of the Synthetic Data App.

This script uses PyInstaller to create a standalone executable that can be distributed
to users without requiring them to install Python or any dependencies.

Usage:
    python setup.py

Requirements:
    - PyInstaller (`pip install pyinstaller`)
    - All dependencies required by the synthetic_data_gui.py script
"""

import os
import sys
import subprocess
import platform
import shutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed."""
    try:
        import PyInstaller
        logger.info("PyInstaller is installed.")
    except ImportError:
        logger.error("PyInstaller is not installed. Please install it using: pip install pyinstaller")
        sys.exit(1)
    
    try:
        import pandas
        logger.info("pandas is installed.")
    except ImportError:
        logger.error("pandas is not installed. Please install it using: pip install pandas")
        sys.exit(1)
    
    try:
        import sdv
        logger.info(f"SDV is installed. Version: {sdv.__version__}")
    except ImportError:
        logger.error("SDV is not installed. Please install it using: pip install sdv")
        sys.exit(1)
    
    try:
        import scipy
        logger.info("scipy is installed.")
    except ImportError:
        logger.error("scipy is not installed. Please install it using: pip install scipy")
        sys.exit(1)
    
    try:
        import sklearn
        logger.info("scikit-learn is installed.")
    except ImportError:
        logger.error("scikit-learn is not installed. Please install it using: pip install scikit-learn")
        sys.exit(1)
    
    try:
        import matplotlib
        logger.info("matplotlib is installed.")
    except ImportError:
        logger.error("matplotlib is not installed. Please install it using: pip install matplotlib")
        sys.exit(1)
    
    try:
        import seaborn
        logger.info("seaborn is installed.")
    except ImportError:
        logger.error("seaborn is not installed. Please install it using: pip install seaborn")
        sys.exit(1)
    
    try:
        import tabulate
        logger.info("tabulate is installed.")
    except ImportError:
        logger.error("tabulate is not installed. Please install it using: pip install tabulate")
        sys.exit(1)

def create_executable():
    """Create standalone executable using PyInstaller."""
    logger.info("Creating standalone executable...")
    
    # Determine platform-specific settings
    is_windows = platform.system() == "Windows"
    is_mac = platform.system() == "Darwin"
    
    # Application name
    app_name = "Pseudomorph"
    
    # Ensure synthetic_data_gui.py exists
    if not os.path.exists("synthetic_data_gui.py"):
        logger.error("synthetic_data_gui.py not found in the current directory.")
        sys.exit(1)
    
    # Create a directory for the build artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_dir = f"build_{timestamp}"
    os.makedirs(build_dir, exist_ok=True)
    
    # Get a list of all Python files in the current directory for inclusion
    python_files = [f for f in os.listdir(".") if f.endswith(".py")]
    
    # Basic PyInstaller command
    cmd = [
        "pyinstaller",
        f"--name={app_name}",
        "--onedir" if is_mac else "--onefile",  # onedir for macOS (to create app bundle), onefile for Windows
        "--windowed",  # Do not open console window
        f"--distpath={build_dir}/dist",
        f"--workpath={build_dir}/build",
        f"--specpath={build_dir}",
    ]
    
    # Add Python files to include
    for py_file in python_files:
        if py_file != "setup.py":  # Skip the setup.py itself
            if is_windows:
                cmd.append(f"--add-data={py_file};.")
            else:
                cmd.append(f"--add-data={py_file}:.")
    
    # Add icon if available
    if is_windows and os.path.exists("icon.ico"):
        cmd.append("--icon=icon.ico")
    elif is_mac and os.path.exists("icon.icns"):
        cmd.append("--icon=icon.icns")
    
    # Add the main script
    cmd.append("synthetic_data_gui.py")
    
    # Log the command
    logger.info("Running PyInstaller with the following command:")
    logger.info(" ".join(cmd))
    
    # Run PyInstaller
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("PyInstaller completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("PyInstaller failed:")
        logger.error(e.stdout)
        logger.error(e.stderr)
        sys.exit(1)
    
    # Get the path to the executable
    if is_windows:
        executable_path = os.path.join(build_dir, "dist", f"{app_name}.exe")
    elif is_mac:
        executable_path = os.path.join(build_dir, "dist", f"{app_name}.app")
    else:  # Linux
        executable_path = os.path.join(build_dir, "dist", f"{app_name}")
    
    # Check if the executable was created
    if os.path.exists(executable_path):
        logger.info(f"Executable created at: {executable_path}")
        
        # Create a distribution zip/dmg file
        dist_dir = "dist"
        os.makedirs(dist_dir, exist_ok=True)
        
        platform_name = platform.system().lower()
        
        if is_mac:
            # For macOS, create a DMG file if possible
            try:
                # Check if create-dmg is installed
                dmg_installed = subprocess.run(["which", "create-dmg"], capture_output=True).returncode == 0
                
                if dmg_installed:
                    dmg_path = os.path.join(dist_dir, f"{app_name}-{platform_name}.dmg")
                    subprocess.run([
                        "create-dmg",
                        "--volname", f"{app_name} Installer",
                        "--volicon", "icon.icns" if os.path.exists("icon.icns") else "",
                        "--background", "background.png" if os.path.exists("background.png") else "",
                        "--window-pos", "200", "120",
                        "--window-size", "800", "400",
                        "--icon-size", "100",
                        "--icon", f"{app_name}.app", "200", "200",
                        "--hide-extension", f"{app_name}.app",
                        "--app-drop-link", "600", "200",
                        dmg_path,
                        executable_path
                    ], check=True)
                    logger.info(f"DMG file created at: {dmg_path}")
                else:
                    # Fall back to creating a zip file
                    logger.info("create-dmg not found, creating zip file instead")
                    zip_filename = f"{app_name}-{platform_name}.zip"
                    zip_path = os.path.join(dist_dir, zip_filename)
                    shutil.make_archive(
                        os.path.splitext(zip_path)[0],
                        'zip',
                        os.path.dirname(executable_path),
                        os.path.basename(executable_path)
                    )
                    logger.info(f"Distribution zip file created at: {zip_path}")
            except Exception as e:
                logger.error(f"Error creating DMG: {e}")
                # Fall back to creating a zip file
                zip_filename = f"{app_name}-{platform_name}.zip"
                zip_path = os.path.join(dist_dir, zip_filename)
                shutil.make_archive(
                    os.path.splitext(zip_path)[0],
                    'zip',
                    os.path.dirname(executable_path),
                    os.path.basename(executable_path)
                )
                logger.info(f"Distribution zip file created at: {zip_path}")
        else:
            # For Windows and Linux, create a zip file
            zip_filename = f"{app_name}-{platform_name}.zip"
            zip_path = os.path.join(dist_dir, zip_filename)
            
            try:
                shutil.make_archive(
                    os.path.splitext(zip_path)[0],
                    'zip',
                    os.path.dirname(executable_path),
                    os.path.basename(executable_path)
                )
                logger.info(f"Distribution zip file created at: {zip_path}")
            except Exception as e:
                logger.error(f"Error creating zip file: {e}")
    else:
        logger.error(f"Executable not found at: {executable_path}")

def create_readme():
    """Create a README file with instructions for using the application."""
    logger.info("Creating README file...")
    
    readme_content = """# Pseudomorph - Synthetic Data Generator & Evaluator

A standalone application for generating and evaluating synthetic data.

## Overview

This application allows you to:
1. Generate synthetic data from a real dataset using either Gaussian Copula or CTGAN methods
2. Evaluate the quality of synthetic data compared to real data
3. Visualize distributions, correlations, and data structure

## Requirements

This is a standalone application. No installation of Python or any other dependencies is required.

## Usage

### Generate Synthetic Data

1. Launch the application by double-clicking the executable file
2. In the "Generate Synthetic Data" tab:
   - Select your input CSV file containing real data
   - Choose the method (Gaussian, CTGAN, or both)
   - Set the number of rows to generate
   - Set the number of epochs for CTGAN training (more epochs = better quality but slower)
   - Choose preprocessing options
   - Select an output directory
   - Click "Generate Synthetic Data"

### Evaluate Synthetic Data

1. In the "Evaluate Synthetic Data" tab:
   - Select your real data CSV file
   - Select your synthetic data CSV file
   - Select an output directory for evaluation results
   - Click "Evaluate Synthetic Data"
   - When evaluation is complete, click "Open Evaluation Report" to view the results

## Understanding Evaluation Results

The evaluation report includes:
- Basic statistics comparison
- Column distribution analysis
- Correlation analysis
- PCA visualization of data structure
- Privacy assessment

Higher scores indicate better similarity between real and synthetic data. A score of 1.0 is perfect.

## Known Issues

- On macOS, you may need to right-click the app and select "Open" to bypass security restrictions the first time you run it.
- On Windows, you may need to allow the application through your antivirus software.
"""
    
    # Write the README file
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    logger.info("README file created.")

def main():
    """Main function to run the packaging script."""
    logger.info("Starting packaging process...")
    
    # Check requirements
    check_requirements()
    
    # Create README file
    create_readme()
    
    # Create executable
    create_executable()
    
    logger.info("Packaging process completed.")

if __name__ == "__main__":
    main()