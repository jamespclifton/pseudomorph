# Pseudomorph - Synthetic Data Generator & Evaluator

A Python application for generating and evaluating synthetic data from real datasets, with both command-line and graphical interfaces.

## Overview

Pseudomorph allows you to:

1. Generate synthetic data from a real dataset using different methods:
   - Gaussian Copula synthesis
   - CTGAN (Conditional Tabular GAN) synthesis

2. Evaluate the quality of synthetic data compared to real data through:
   - Statistical distribution comparison
   - Correlation analysis
   - Principal Component Analysis (PCA) visualization
   - Privacy assessment

## Features

- **Multiple interfaces**: Both command-line and graphical user interfaces
- **Flexible generation options**: Control the number of rows, epochs, and preprocessing steps
- **Comprehensive evaluation**: Compare synthetic data to real data across multiple dimensions
- **Visual reports**: Generate detailed evaluation reports with visualizations
- **No-dependency option**: Basic functionality works without SDV dependency

## Installation

### Prerequisites

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy

### Optional Dependencies

- SDV (Synthetic Data Vault) for advanced synthesis methods:
  ```
  pip install sdv
  ```

### Basic Installation

1. Clone the repository:
   ```
   git clone https://github.com/jamespclifton/pseudomorph.git
   cd pseudomorph
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface

Generate synthetic data with the command-line interface:

```bash
python app.py --input data.csv --method both --rows 1000 --epochs 300
```

#### Options:

- `--input, -i`: Path to input CSV file (required)
- `--output-dir, -o`: Directory to save output files (default: output)
- `--rows, -r`: Number of synthetic rows to generate (default: same as input)
- `--method, -m`: Synthesis method: gaussian, ctgan, or both (default: both)
- `--epochs, -e`: Number of training epochs for CTGAN (default: 300)
- `--id-column`: Column to be treated as ID
- `--clean-column-names`: Clean column names (lowercase, replace spaces with underscores)
- `--handle-blanks`: Convert blank strings to NaN values
- `--skip-eval`: Skip evaluation step

### Graphical User Interface

Launch the GUI for a more user-friendly experience:

```bash
python synthetic_data_gui.py
```

The GUI provides tabs for:
1. **Generate Synthetic Data**: Select input file, method, and output options
2. **Evaluate Synthetic Data**: Compare real and synthetic datasets

### Simple Evaluation

For a quick evaluation of existing synthetic data:

```bash
python easy_evaluate.py --real-data real.csv --synthetic-data synthetic.csv
```

## Example Dataset

The repository includes `mock_hs_course_articulation_data.csv` as an example dataset. This is synthetic educational data with the following columns:

- articulated_course_year
- hs_grad_year
- hs_grad_status
- gender
- ethnicity
- high_school
- subject
- course
- hs_course_grade
- final_exam_grade
- credit_recommended
- credit_status
- id
- took_mcc_course_as_non_hs_student
- took_mcc_course_in_same_discipline
- earned_degree
- earned_coa
- earned_cop

## Building a Standalone Executable

To create a standalone executable:

```bash
python setup.py
```

This creates a distributable application that doesn't require a Python installation.

## File Structure

- `app.py`: Command-line interface
- `synthetic_data_gui.py`: Graphical user interface
- `generator.py`: Core synthetic data generation module
- `evaluator.py`: Comprehensive evaluation module (requires SDV)
- `enhanced_simple_evaluator.py`: Enhanced evaluator without SDV dependencies
- `simple_evaluator.py`: Basic evaluator without SDV dependencies
- `easy_evaluate.py`: Simplified evaluation script
- `run.py`: Helper script to run the app directly
- `setup.py`: Script to create standalone executable
- `generate_more_data.py`: Script to generate larger synthetic datasets

## Notes

- When using CTGAN, more epochs generally leads to better quality but slower generation
- The evaluation report provides a similarity score between 0 and 1, where higher is better
- Privacy scores close to 1.0 indicate synthetic data that can't be distinguished from real data

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
