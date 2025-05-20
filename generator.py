"""
Synthetic Data Generator Module

This module provides functions to generate synthetic data from real datasets
using different synthesizers from the SDV library.
"""

import pandas as pd
import numpy as np
import logging
import sys
import re
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print diagnostic information
def print_diagnostics():
    """Print diagnostic information about the Python environment."""
    logger.info("--- SCRIPT DIAGNOSTICS ---")
    logger.info(f"Python executable being used by this script: {sys.executable}")
    logger.info(f"Python version for this script: {sys.version}")
    logger.info("sys.path for this script:")
    for path in sys.path:
        logger.info(path)
    logger.info("--- END SCRIPT DIAGNOSTICS ---")

# Import SDV and related modules
def import_sdv_modules():
    """Import SDV modules and check their availability."""
    logger.info("Attempting to import SDV modules...\n")
    
    try:
        import sdv
        logger.info(f"Successfully imported 'sdv'. Location: {sdv.__file__}")
        logger.info(f"SDV Version: {sdv.__version__}")
        
        try:
            from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
            logger.info("Successfully imported GaussianCopulaSynthesizer and CTGANSynthesizer from sdv.single_table")
        except ImportError as e:
            logger.error(f"Failed to import synthesizers from sdv.single_table: {e}")
            raise
        
        try:
            from sdv.metadata import SingleTableMetadata
            logger.info("Successfully imported SingleTableMetadata from sdv.metadata")
        except ImportError as e:
            logger.error(f"Failed to import SingleTableMetadata from sdv.metadata: {e}")
            raise
        
        return sdv, GaussianCopulaSynthesizer, CTGANSynthesizer, SingleTableMetadata
    
    except ImportError as e:
        logger.error(f"Failed to import sdv: {e}")
        logger.error("Please install SDV using: pip install sdv")
        raise

# Data preprocessing function
def preprocess_data(data, clean_column_names=True, handle_blanks=True, columns_to_check=None):
    """
    Preprocess the data for synthetic data generation.
    
    Args:
        data: pandas DataFrame to preprocess
        clean_column_names: Whether to clean column names (lowercase, replace spaces)
        handle_blanks: Whether to convert blank strings to NaN
        columns_to_check: Specific columns to check for blanks, if None check all
        
    Returns:
        Preprocessed pandas DataFrame
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Save original column names
    original_cols = df.columns.tolist()
    logger.info(f"Original column names: {original_cols}")
    
    # Clean column names if requested
    if clean_column_names:
        # Convert to lowercase and replace spaces/special chars with underscores
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col.lower()) for col in df.columns]
        logger.info(f"Cleaned column names: {df.columns.tolist()}")
    
    # Handle blank values if requested
    if handle_blanks:
        logger.info("Checking specific columns for blank values...")
        
        # If columns not specified, use all string columns
        if columns_to_check is None:
            columns_to_check = df.select_dtypes(include=['object']).columns
        
        for col in columns_to_check:
            if col in df.columns:
                logger.info(f"Ensuring blanks are treated as NaN in column: '{col}'")
                # Replace empty strings with NaN
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
        
        # Log missing value counts for important columns
        missing_counts = {col: df[col].isna().sum() for col in columns_to_check if col in df.columns}
        logger.info("Missing value counts:")
        for col, count in missing_counts.items():
            if count > 0:
                logger.info(f"  '{col}': {count} missing values")
    
    logger.info("Preprocessing complete.")
    return df

# Generate synthetic data
def generate_synthetic_data(data, num_rows=None, synthesizer_type='gaussian', epochs=300, id_column=None):
    """
    Generate synthetic data using the specified synthesizer.
    
    Args:
        data: pandas DataFrame containing the real data
        num_rows: Number of synthetic rows to generate (default: same as input)
        synthesizer_type: Type of synthesizer to use ('gaussian' or 'ctgan')
        epochs: Number of training epochs for CTGAN
        id_column: Column name to be treated as ID (if any)
        
    Returns:
        pandas DataFrame containing synthetic data
    """
    # Import SDV modules
    sdv, GaussianCopulaSynthesizer, CTGANSynthesizer, SingleTableMetadata = import_sdv_modules()
    
    # Determine number of rows to generate
    if num_rows is None:
        num_rows = len(data)
    
    # Create metadata
    logger.info("--- Creating Metadata ---")
    metadata = SingleTableMetadata()
    logger.info("Detecting metadata from dataframe...")
    metadata.detect_from_dataframe(data)
    
    # If ID column is specified, set its type to ID
    if id_column and id_column in data.columns:
        logger.info(f"Setting column '{id_column}' as ID type")
        metadata.update_column(column_name=id_column, sdtype='id')
    
    # Log detected column types
    logger.info("Detected column data types:")
    for col, props in metadata.columns.items():
        logger.info(f"  '{col}': {props['sdtype']}")
    
    # Create and fit synthesizer
    if synthesizer_type.lower() == 'gaussian':
        synthesizer = GaussianCopulaSynthesizer(metadata)
        logger.info("Fitting GaussianCopulaSynthesizer to data...")
    elif synthesizer_type.lower() == 'ctgan':
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs)
        logger.info(f"Fitting CTGANSynthesizer to data (epochs: {epochs})...")
    else:
        raise ValueError(f"Unsupported synthesizer type: {synthesizer_type}")
    
    synthesizer.fit(data)
    
    # Sample synthetic data
    logger.info(f"Sampling {num_rows} rows of synthetic data...")
    synthetic_data = synthesizer.sample(num_rows)
    logger.info(f"{synthesizer_type.capitalize()}Synthesizer data generation complete.")
    
    # Handle ID column generation if needed
    if id_column:
        if id_column in synthetic_data.columns:
            # If SDV didn't generate proper IDs, create sequential ones
            if synthetic_data[id_column].duplicated().any() or synthetic_data[id_column].isna().any():
                logger.info(f"Generating sequential UIDs for column '{id_column}'")
                synthetic_data[id_column] = [f"UID{i+1}" for i in range(len(synthetic_data))]
    
    return synthetic_data

# Direct test function
def direct_test(input_file, output_dir='.'):
    """Run a direct test of the synthetic data generation functions."""
    logger.info("--- Starting direct test of generator.py ---")
    
    # Load data
    logger.info(f"Loading real data from: {input_file}")
    data = pd.read_csv(input_file)
    logger.info(f"Successfully loaded data. Original Shape: {data.shape}")
    
    # Preprocess data
    preprocessed_data = preprocess_data(
        data, 
        clean_column_names=True,
        handle_blanks=True,
        columns_to_check=['hs_course_grade', 'final_exam_grade']
    )
    
    # Generate synthetic data with GaussianCopulaSynthesizer
    logger.info("--- Testing GaussianCopulaSynthesizer ---")
    logger.info("Generating 50 rows using GaussianCopulaSynthesizer...")
    gaussian_data = generate_synthetic_data(
        preprocessed_data, 
        num_rows=50,
        synthesizer_type='gaussian'
    )
    
    # Display sample
    logger.info("Generated GaussianCopula Data (first 5 rows):")
    print(gaussian_data.head().to_string())
    
    # Save to file
    gaussian_output = os.path.join(output_dir, 'synthetic_data_gaussian_test.csv')
    gaussian_data.to_csv(gaussian_output, index=False)
    logger.info(f"Data saved successfully to {gaussian_output}")
    
    # Generate synthetic data with CTGANSynthesizer
    logger.info("--- Testing CTGANSynthesizer ---")
    logger.info("Generating 50 rows using CTGANSynthesizer (epochs: 5)...")
    ctgan_data = generate_synthetic_data(
        preprocessed_data, 
        num_rows=50,
        synthesizer_type='ctgan',
        epochs=5  # Using fewer epochs for quick testing
    )
    
    # Display sample
    logger.info("Generated CTGAN Data (first 5 rows):")
    print(ctgan_data.head().to_string())
    
    # Save to file
    ctgan_output = os.path.join(output_dir, 'synthetic_data_ctgan_test.csv')
    ctgan_data.to_csv(ctgan_output, index=False)
    logger.info(f"Data saved successfully to {ctgan_output}")
    
    logger.info("--- Direct test of generator.py finished ---")

# Main function
def main():
    """Main function to run when script is executed directly."""
    print_diagnostics()
    
    import_sdv_modules()
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
        direct_test(input_file, output_dir)
    else:
        logger.info("No input file specified. Please provide a CSV file path as an argument.")
        logger.info("Example: python generator.py input.csv [output_dir]")

if __name__ == "__main__":
    main()