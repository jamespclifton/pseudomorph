"""
Synthetic Data App - Main Application

This script provides a command-line interface for generating and evaluating synthetic data.
"""

import os
import sys
import pandas as pd
import argparse
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import generator and evaluator modules
try:
    # Try different import approaches
    try:
        from generator import generate_synthetic_data, preprocess_data
        from evaluator import SyntheticDataEvaluator
    except ImportError:
        # Try with module name
        from synthetic_data_app.generator import generate_synthetic_data, preprocess_data
        from synthetic_data_app.evaluator import SyntheticDataEvaluator
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    logger.error("Make sure generator.py and evaluator.py are in the same directory.")
    # Print additional debugging info
    import os
    logger.error(f"Current directory: {os.getcwd()}")
    logger.error(f"Files in directory: {os.listdir('.')}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate and evaluate synthetic data.')
    
    # Input file
    parser.add_argument('--input', '-i', required=True, 
                      help='Path to the input CSV file containing real data')
    
    # Output directory
    parser.add_argument('--output-dir', '-o', default='output',
                      help='Directory to save output files (default: output)')
    
    # Synthetic data generation options
    parser.add_argument('--rows', '-r', type=int, default=None,
                      help='Number of synthetic rows to generate (default: same as input)')
    
    parser.add_argument('--method', '-m', choices=['gaussian', 'ctgan', 'both'], default='both',
                      help='Synthetic data generation method (default: both)')
    
    parser.add_argument('--epochs', '-e', type=int, default=300,
                      help='Number of training epochs for CTGAN (default: 300)')
    
    parser.add_argument('--id-column', default=None,
                      help='Column name to be treated as ID (if any)')
    
    # Preprocessing options
    parser.add_argument('--clean-column-names', action='store_true',
                      help='Clean column names (lowercase, replace spaces with underscores)')
    
    parser.add_argument('--handle-blanks', action='store_true',
                      help='Convert blank strings to NaN values')
    
    # Skip evaluation
    parser.add_argument('--skip-eval', action='store_true',
                      help='Skip evaluation step')
    
    return parser.parse_args()

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the synthetic data generation and evaluation."""
    print("\n=== Synthetic Data Generator and Evaluator ===\n")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    setup_output_directory(output_dir)
    
    # Save arguments for reproducibility
    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        # Convert args to dict
        args_dict = vars(args)
        json.dump(args_dict, f, indent=2)
    
    try:
        # Load and preprocess data
        logger.info(f"Loading real data from: {args.input}")
        
        real_data = pd.read_csv(args.input)
        original_shape = real_data.shape
        logger.info(f"Successfully loaded data. Original Shape: {original_shape}")
        
        # Preprocess data
        preprocessed_data = preprocess_data(
            real_data, 
            clean_column_names=args.clean_column_names,
            handle_blanks=args.handle_blanks
        )
        
        # Save preprocessed data
        preprocessed_path = os.path.join(output_dir, 'preprocessed_data.csv')
        preprocessed_data.to_csv(preprocessed_path, index=False)
        logger.info(f"Preprocessed data saved to: {preprocessed_path}")
        
        # Determine number of rows to generate
        num_rows = args.rows if args.rows is not None else len(preprocessed_data)
        
        # Generate synthetic data
        synthetic_data = {}
        
        # Identify ID column if specified
        id_column = args.id_column
        
        if args.method in ['gaussian', 'both']:
            logger.info(f"Generating {num_rows} rows using GaussianCopulaSynthesizer...")
            gaussian_data = generate_synthetic_data(
                preprocessed_data, 
                num_rows=num_rows,
                synthesizer_type='gaussian',
                id_column=id_column
            )
            
            # Save Gaussian synthetic data
            gaussian_path = os.path.join(output_dir, 'synthetic_data_gaussian.csv')
            gaussian_data.to_csv(gaussian_path, index=False)
            logger.info(f"Gaussian synthetic data saved to: {gaussian_path}")
            
            synthetic_data['gaussian'] = gaussian_data
        
        if args.method in ['ctgan', 'both']:
            logger.info(f"Generating {num_rows} rows using CTGANSynthesizer (epochs: {args.epochs})...")
            ctgan_data = generate_synthetic_data(
                preprocessed_data, 
                num_rows=num_rows,
                synthesizer_type='ctgan',
                epochs=args.epochs,
                id_column=id_column
            )
            
            # Save CTGAN synthetic data
            ctgan_path = os.path.join(output_dir, 'synthetic_data_ctgan.csv')
            ctgan_data.to_csv(ctgan_path, index=False)
            logger.info(f"CTGAN synthetic data saved to: {ctgan_path}")
            
            synthetic_data['ctgan'] = ctgan_data
        
        # Skip evaluation if requested
        if args.skip_eval:
            logger.info("Evaluation step skipped as requested.")
            print("\nSynthetic data generation complete!")
            print(f"Output files saved to: {output_dir}")
            return
        
        # Evaluation
        logger.info("Starting evaluation of synthetic data...")
        
        # Create subdirectories for evaluation results
        for method, data in synthetic_data.items():
            eval_dir = os.path.join(output_dir, f'evaluation_{method}')
            
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            
            # Initialize evaluator
            evaluator = SyntheticDataEvaluator(
                real_data=preprocessed_data,
                synthetic_data=data
            )
            
            # Set results directory
            evaluator.results_dir = eval_dir
            
            # Generate human-readable report
            report = evaluator.generate_human_readable_report()
            
            # Print summary
            print(f"\n--- Evaluation results for {method.upper()} method ---")
            print(report)
            
            logger.info(f"Evaluation results for {method} method saved to: {eval_dir}")
        
        print("\nSynthetic data generation and evaluation complete!")
        print(f"Output files saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()