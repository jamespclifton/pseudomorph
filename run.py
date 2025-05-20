#!/usr/bin/env python
"""
Synthetic Data App - Direct Executor

This script provides a direct way to run the synthetic data generator and evaluator
without dealing with Python module import issues.
"""

import os
import sys
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get the current directory (where this script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the modules directly from file paths
generator_path = os.path.join(current_dir, 'generator.py')
evaluator_path = os.path.join(current_dir, 'evaluator.py')
app_path = os.path.join(current_dir, 'app.py')

# Import the modules
generator = import_module_from_file('generator', generator_path)
evaluator = import_module_from_file('evaluator', evaluator_path)
app = import_module_from_file('app', app_path)

# Make the imported functions and classes available to app.py
sys.modules['generator'] = generator
sys.modules['evaluator'] = evaluator

if __name__ == "__main__":
    # Run the main function from app.py
    app.main()