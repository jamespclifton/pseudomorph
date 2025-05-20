"""
Script to regenerate synthetic data with more rows.
"""

import os
import sys
import logging
from generator import generate_synthetic_data, preprocess_data, import_sdv_modules
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Generate larger synthetic datasets."""
    if len(sys.argv) < 2:
        print("Usage: python generate_more_data.py <input_csv_file> [num_rows=1000]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    logger.info(f"Generating {num_rows} rows of synthetic data from {input_file}")
    
    # Import SDV modules
    import_sdv_modules()
    
    # Load and preprocess data
    real_data = pd.read_csv(input_file)
    preprocessed_data = preprocess_data(
        real_data, 
        clean_column_names=True,
        handle_blanks=True
    )
    
    # Generate data with GaussianCopulaSynthesizer
    logger.info(f"Generating {num_rows} rows using GaussianCopulaSynthesizer...")
    gaussian_data = generate_synthetic_data(
        preprocessed_data, 
        num_rows=num_rows,
        synthesizer_type='gaussian'
    )
    
    # Save Gaussian data
    gaussian_output = 'synthetic_data_gaussian_large.csv'
    gaussian_data.to_csv(gaussian_output, index=False)
    logger.info(f"Gaussian synthetic data saved to: {gaussian_output}")
    
    # Generate data with CTGANSynthesizer
    logger.info(f"Generating {num_rows} rows using CTGANSynthesizer (epochs: 300)...")
    ctgan_data = generate_synthetic_data(
        preprocessed_data, 
        num_rows=num_rows,
        synthesizer_type='ctgan',
        epochs=300  # More epochs for better quality
    )
    
    # Save CTGAN data
    ctgan_output = 'synthetic_data_ctgan_large.csv'
    ctgan_data.to_csv(ctgan_output, index=False)
    logger.info(f"CTGAN synthetic data saved to: {ctgan_output}")
    
    logger.info("Synthetic data generation complete!")

if __name__ == "__main__":
    main()