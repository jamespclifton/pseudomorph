#!/usr/bin/env python
"""
Synthetic Data App - Easy Evaluation

This script provides a simple way to evaluate synthetic data without complex dependencies.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate and evaluate synthetic data.')
    
    # Input files
    parser.add_argument('--real-data', '-r', required=True, 
                      help='Path to the real data CSV file')
    
    parser.add_argument('--synthetic-data', '-s', required=True,
                      help='Path to the synthetic data CSV file')
    
    # Output directory
    parser.add_argument('--output-dir', '-o', default=None,
                      help='Directory to save evaluation results (default: timestamped directory)')
    
    return parser.parse_args()

def main():
    """Main function to run the synthetic data evaluation."""
    print("\n=== Simple Synthetic Data Evaluator ===\n")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Import the SimpleEvaluator
    try:
        # First try to import directly
        try:
            from simple_evaluator import SimpleEvaluator
        except ImportError:
            # If that doesn't work, try with the full path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            
            from simple_evaluator import SimpleEvaluator
    except ImportError as e:
        logger.error(f"Error importing SimpleEvaluator: {str(e)}")
        logger.error("Make sure simple_evaluator.py is in the same directory.")
        sys.exit(1)
    
    try:
        # Create evaluator
        evaluator = SimpleEvaluator(
            real_data_path=args.real_data,
            synthetic_data_path=args.synthetic_data,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        report_path = evaluator.evaluate()
        
        print("\nEvaluation complete!")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()