"""
Synthetic Data Evaluator Module

This module provides functions to evaluate the quality of synthetic data compared to real data
using various statistical methods and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.metrics.tabular import KSTest, CSTest, LogisticDetection

# Create our own plotting functions since get_column_plot is not available
def create_column_plot(real_data, synthetic_data, column_name):
    """
    Create a plot comparing distributions of a single column between real and synthetic data.
    
    Args:
        real_data: DataFrame containing the real data
        synthetic_data: DataFrame containing the synthetic data
        column_name: Name of the column to visualize
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine if column is categorical or numerical
    if pd.api.types.is_numeric_dtype(real_data[column_name]):
        # Numerical column - use histograms
        ax.hist(real_data[column_name].dropna(), alpha=0.5, bins=20, label='Real')
        ax.hist(synthetic_data[column_name].dropna(), alpha=0.5, bins=20, label='Synthetic')
        ax.set_xlabel(column_name)
        ax.set_ylabel('Frequency')
    else:
        # Categorical column - use bar plots
        real_counts = real_data[column_name].value_counts(normalize=True)
        synth_counts = synthetic_data[column_name].value_counts(normalize=True)
        
        # Combine and get all categories
        all_cats = pd.concat([real_counts, synth_counts], axis=1).fillna(0)
        all_cats.columns = ['Real', 'Synthetic']
        
        all_cats.plot(kind='bar', ax=ax)
        ax.set_xlabel(column_name)
        ax.set_ylabel('Proportion')
    
    ax.set_title(f'Distribution of {column_name}')
    ax.legend()
    plt.tight_layout()
    
    return fig

def create_column_pair_plot(real_data, synthetic_data, column_names):
    """
    Create a plot comparing the joint distribution of two columns.
    
    Args:
        real_data: DataFrame containing the real data
        synthetic_data: DataFrame containing the synthetic data
        column_names: List of two column names to visualize
        
    Returns:
        matplotlib Figure object
    """
    if len(column_names) != 2:
        raise ValueError("column_names must contain exactly 2 column names")
    
    col1, col2 = column_names
    
    # Create a figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determine plot type based on column types
    if pd.api.types.is_numeric_dtype(real_data[col1]) and pd.api.types.is_numeric_dtype(real_data[col2]):
        # Both columns are numerical - use scatter plots
        ax1.scatter(real_data[col1], real_data[col2], alpha=0.5)
        ax1.set_title('Real Data')
        ax1.set_xlabel(col1)
        ax1.set_ylabel(col2)
        
        ax2.scatter(synthetic_data[col1], synthetic_data[col2], alpha=0.5, color='orange')
        ax2.set_title('Synthetic Data')
        ax2.set_xlabel(col1)
        ax2.set_ylabel(col2)
    
    elif not pd.api.types.is_numeric_dtype(real_data[col1]) and not pd.api.types.is_numeric_dtype(real_data[col2]):
        # Both columns are categorical - use heatmaps
        real_crosstab = pd.crosstab(real_data[col1], real_data[col2], normalize=True)
        synth_crosstab = pd.crosstab(synthetic_data[col1], synthetic_data[col2], normalize=True)
        
        sns.heatmap(real_crosstab, annot=True, cmap='Blues', ax=ax1, fmt='.2f')
        ax1.set_title('Real Data')
        
        sns.heatmap(synth_crosstab, annot=True, cmap='Oranges', ax=ax2, fmt='.2f')
        ax2.set_title('Synthetic Data')
    
    else:
        # Mixed types - one categorical, one numerical
        # Determine which is which
        if pd.api.types.is_numeric_dtype(real_data[col1]):
            num_col, cat_col = col1, col2
        else:
            num_col, cat_col = col2, col1
        
        # Use box plots
        sns.boxplot(x=cat_col, y=num_col, data=real_data, ax=ax1)
        ax1.set_title('Real Data')
        
        sns.boxplot(x=cat_col, y=num_col, data=synthetic_data, ax=ax2)
        ax2.set_title('Synthetic Data')
    
    plt.tight_layout()
    return fig

# Custom implementation of evaluate since it's not available in your SDV version
def custom_evaluate(synthetic_data, real_data, metadata=None):
    """
    Custom implementation of SDV's evaluate function
    """
    # Create a quality score based on column-wise similarity
    column_scores = {}
    for col in real_data.columns:
        try:
            if pd.api.types.is_numeric_dtype(real_data[col]):
                # Numerical column - use KSTest
                test = KSTest()
                score = test.compute(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    column_name=col
                )
            else:
                # Categorical column - use CSTest
                test = CSTest()
                score = test.compute(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    column_name=col
                )
            column_scores[col] = score
        except Exception:
            # Skip columns that can't be evaluated
            continue
    
    # Calculate an overall quality score (average of column scores)
    if column_scores:
        quality_score = sum(column_scores.values()) / len(column_scores)
    else:
        quality_score = 0.0
    
    # Create a result dictionary similar to SDV's evaluate
    result = {
        'quality_score': quality_score,
        'column_scores': column_scores
    }
    
    return result
from sklearn.decomposition import PCA
import os
import json
from typing import Dict, Tuple, List, Union, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticDataEvaluator:
    """Class for evaluating synthetic data quality against real data."""
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata=None):
        """
        Initialize the evaluator with real and synthetic datasets.
        
        Args:
            real_data: DataFrame containing the original real data
            synthetic_data: DataFrame containing the synthetic data
            metadata: Optional SDV metadata object describing the data
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metadata = metadata
        self.results_dir = "evaluation_results"
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # Make sure the datasets have the same columns
        self._validate_data()
        
        # Remove ID columns for statistical comparison
        self.real_data_no_id = self._remove_id_columns(self.real_data)
        self.synthetic_data_no_id = self._remove_id_columns(self.synthetic_data)
        
        logger.info("Evaluator initialized successfully")
    
    def _validate_data(self):
        """Ensure both datasets have the same columns and compatible types."""
        real_cols = set(self.real_data.columns)
        synth_cols = set(self.synthetic_data.columns)
        
        if real_cols != synth_cols:
            missing = real_cols - synth_cols
            extra = synth_cols - real_cols
            error_msg = []
            if missing:
                error_msg.append(f"Columns in real data missing from synthetic data: {missing}")
            if extra:
                error_msg.append(f"Extra columns in synthetic data: {extra}")
            
            raise ValueError("\n".join(error_msg))
        
        logger.info("Data validation passed - both datasets have matching columns")
    
    def _remove_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ID columns for statistical comparison.
        
        If using SDV metadata, use ID column types.
        Otherwise, use heuristics to identify ID columns.
        """
        df_copy = df.copy()
        
        if self.metadata:
            # Use metadata to identify ID columns
            id_columns = [
                col for col in df.columns 
                if self.metadata.columns.get(col, {}).get('sdtype') == 'id'
            ]
        else:
            # Use heuristics to identify ID columns
            id_columns = [col for col in df.columns if 'id' in col.lower()]
        
        if id_columns:
            logger.info(f"Removing ID columns for statistical comparison: {id_columns}")
            df_copy = df_copy.drop(columns=id_columns)
        
        return df_copy
    
    def run_overall_evaluation(self) -> Dict:
        """
        Run overall evaluation to get a quality score.
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Running overall evaluation")
        
        try:
            # Use our custom evaluate function instead of SDV's
            evaluation_results = custom_evaluate(
                synthetic_data=self.synthetic_data_no_id,
                real_data=self.real_data_no_id,
                metadata=self.metadata
            )
            
            # Save results
            results_path = os.path.join(self.results_dir, 'overall_evaluation.json')
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logger.info(f"Overall evaluation complete. Quality score: {evaluation_results['quality_score']}")
            return evaluation_results
        
        except Exception as e:
            logger.error(f"Error in overall evaluation: {str(e)}")
            return {"error": str(e)}
    
    def run_column_distribution_tests(self, columns: Optional[List[str]] = None) -> Dict:
        """
        Run distribution tests for individual columns.
        
        Args:
            columns: List of column names to test. If None, all columns are tested.
            
        Returns:
            Dictionary with column names as keys and test results as values
        """
        if columns is None:
            # Use all columns except those with too many categories (limit to 20 unique values)
            columns = [
                col for col in self.real_data_no_id.columns
                if self.real_data_no_id[col].nunique() <= 20 or pd.api.types.is_numeric_dtype(self.real_data_no_id[col])
            ]
        
        logger.info(f"Running column distribution tests for {len(columns)} columns")
        
        results = {}
        
        # Use KSTest for numerical columns, CSTest for categorical
        for col in columns:
            try:
                fig_path = os.path.join(self.results_dir, f'column_dist_{col}.png')
                
                # Create and save column distribution plot using our custom function
                fig = create_column_plot(
                    real_data=self.real_data_no_id,
                    synthetic_data=self.synthetic_data_no_id,
                    column_name=col
                )
                fig.savefig(fig_path)
                plt.close(fig)
                
                # Determine if column is categorical or numerical
                if pd.api.types.is_numeric_dtype(self.real_data_no_id[col]):
                    # Numerical column - use KSTest
                    test = KSTest()
                    test_result = test.compute(
                        real_data=self.real_data_no_id,
                        synthetic_data=self.synthetic_data_no_id,
                        column_name=col
                    )
                else:
                    # Categorical column - use CSTest
                    test = CSTest()
                    test_result = test.compute(
                        real_data=self.real_data_no_id,
                        synthetic_data=self.synthetic_data_no_id,
                        column_name=col
                    )
                
                results[col] = {
                    'score': test_result,
                    'plot_path': fig_path
                }
                
                logger.info(f"Column {col} distribution test complete. Score: {test_result}")
                
            except Exception as e:
                logger.warning(f"Error testing column {col}: {str(e)}")
                results[col] = {'error': str(e)}
        
        # Save results
        results_path = os.path.join(self.results_dir, 'column_distribution_tests.json')
        with open(results_path, 'w') as f:
            # Convert to a serializable format
            serializable_results = {
                k: {**v, 'plot_path': str(v.get('plot_path', ''))} 
                for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=2)
        
        return results
    
    def run_correlation_analysis(self) -> Dict:
        """
        Analyze and compare correlation matrices between real and synthetic data.
        
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info("Running correlation analysis")
        
        try:
            # Get numerical columns for correlation analysis
            numerical_cols = self.real_data_no_id.select_dtypes(include=['number']).columns
            
            if len(numerical_cols) < 2:
                logger.warning("Not enough numerical columns for correlation analysis")
                return {"error": "Not enough numerical columns for correlation analysis"}
            
            # Calculate correlation matrices
            real_corr = self.real_data_no_id[numerical_cols].corr()
            synth_corr = self.synthetic_data_no_id[numerical_cols].corr()
            
            # Calculate difference between correlation matrices
            diff_corr = real_corr - synth_corr
            
            # Generate plots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Real data correlation
            sns.heatmap(real_corr, annot=True, cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1)
            axes[0].set_title('Real Data Correlation')
            
            # Synthetic data correlation
            sns.heatmap(synth_corr, annot=True, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
            axes[1].set_title('Synthetic Data Correlation')
            
            # Difference
            sns.heatmap(diff_corr, annot=True, cmap='coolwarm', ax=axes[2], vmin=-1, vmax=1)
            axes[2].set_title('Correlation Difference (Real - Synthetic)')
            
            plt.tight_layout()
            corr_path = os.path.join(self.results_dir, 'correlation_comparison.png')
            plt.savefig(corr_path)
            plt.close(fig)
            
            # Calculate correlation similarity score (Frobenius norm of difference matrix)
            # Lower value means more similar
            from numpy.linalg import norm
            frob_norm = norm(diff_corr.fillna(0).values, 'fro')
            similarity_score = 1.0 / (1.0 + frob_norm)  # Transform to 0-1 range
            
            results = {
                'correlation_similarity_score': similarity_score,
                'plot_path': corr_path
            }
            
            # Save results
            results_path = os.path.join(self.results_dir, 'correlation_analysis.json')
            with open(results_path, 'w') as f:
                # Convert to a serializable format
                serializable_results = {**results, 'plot_path': str(results['plot_path'])}
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Correlation analysis complete. Similarity score: {similarity_score}")
            return results
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {"error": str(e)}
    
    def run_pca_visualization(self) -> Dict:
        """
        Run PCA to visualize how well synthetic data captures the structure of real data.
        
        Returns:
            Dictionary with PCA visualization results
        """
        logger.info("Running PCA visualization")
        
        try:
            # Get numerical columns for PCA
            numerical_cols = self.real_data_no_id.select_dtypes(include=['number']).columns
            
            if len(numerical_cols) < 2:
                logger.warning("Not enough numerical columns for PCA visualization")
                return {"error": "Not enough numerical columns for PCA visualization"}
            
            # Prepare data for PCA
            real_numerical = self.real_data_no_id[numerical_cols].fillna(0)
            synth_numerical = self.synthetic_data_no_id[numerical_cols].fillna(0)
            
            # Fit PCA on combined data to ensure same transformation
            combined_data = pd.concat([real_numerical, synth_numerical])
            
            # Standardize the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            combined_data_scaled = scaler.fit_transform(combined_data)
            
            # Apply PCA
            pca = PCA(n_components=2)
            combined_pca = pca.fit_transform(combined_data_scaled)
            
            # Split back into real and synthetic
            real_pca = combined_pca[:len(real_numerical)]
            synth_pca = combined_pca[len(real_numerical):]
            
            # Create PCA plot
            plt.figure(figsize=(10, 8))
            plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real Data', color='blue')
            plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label='Synthetic Data', color='red')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Comparison of Real vs Synthetic Data')
            plt.legend()
            plt.grid(alpha=0.3)
            
            pca_path = os.path.join(self.results_dir, 'pca_visualization.png')
            plt.savefig(pca_path)
            plt.close()
            
            # Calculate overlap metric
            # Using a simple approach: ratio of synthetic points that fall within real data space
            from scipy.spatial import ConvexHull
            
            # Function to check if a point is inside a convex hull
            def is_inside_hull(point, hull):
                new_hull = ConvexHull(np.vstack((hull.points, point)))
                return len(new_hull.vertices) == len(hull.vertices)
            
            try:
                # Create convex hull of real data
                real_hull = ConvexHull(real_pca)
                
                # Count how many synthetic points fall inside real data hull
                points_inside = 0
                for point in synth_pca:
                    if is_inside_hull(point, real_hull):
                        points_inside += 1
                
                overlap_score = points_inside / len(synth_pca)
            except Exception as hull_error:
                logger.warning(f"Could not compute convex hull: {hull_error}")
                overlap_score = None
            
            results = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'pca_overlap_score': overlap_score,
                'plot_path': pca_path
            }
            
            # Save results
            results_path = os.path.join(self.results_dir, 'pca_visualization.json')
            with open(results_path, 'w') as f:
                # Convert to a serializable format
                serializable_results = {
                    **results,
                    'plot_path': str(results['plot_path']),
                    'explained_variance': [float(x) for x in results['explained_variance']]
                }
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"PCA visualization complete. Overlap score: {overlap_score}")
            return results
            
        except Exception as e:
            logger.error(f"Error in PCA visualization: {str(e)}")
            return {"error": str(e)}
    
    def run_detectability_test(self) -> Dict:
        """
        Test how easily a machine learning model can distinguish between real and synthetic data.
        
        Returns:
            Dictionary with detectability test results
        """
        logger.info("Running detectability test")
        
        try:
            # Use SDV's LogisticDetection metric
            detection = LogisticDetection()
            
            # Compute detection score
            detection_score = detection.compute(
                real_data=self.real_data_no_id,
                synthetic_data=self.synthetic_data_no_id
            )
            
            # A score of 0.5 is ideal (indistinguishable)
            # A score of 1.0 means perfectly distinguishable
            # We can transform this to a "privacy score" where higher is better
            privacy_score = 1.0 - abs(detection_score - 0.5) * 2
            
            results = {
                'detection_score': detection_score,
                'privacy_score': privacy_score,
                'interpretation': {
                    'detection_score': 'Value of 0.5 means real and synthetic data are indistinguishable',
                    'privacy_score': 'Higher is better (1.0 = perfect)'
                }
            }
            
            # Save results
            results_path = os.path.join(self.results_dir, 'detectability_test.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Detectability test complete. Detection score: {detection_score}, Privacy score: {privacy_score}")
            return results
            
        except Exception as e:
            logger.error(f"Error in detectability test: {str(e)}")
            return {"error": str(e)}
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a complete summary report of all evaluations.
        
        Returns:
            Dictionary with summary of all evaluation results
        """
        logger.info("Generating summary report")
        
        try:
            # Run all evaluations
            overall = self.run_overall_evaluation()
            column_tests = self.run_column_distribution_tests()
            correlation = self.run_correlation_analysis()
            pca = self.run_pca_visualization()
            detectability = self.run_detectability_test()
            
            # Compile summary
            summary = {
                'overall_quality_score': overall.get('quality_score', None),
                'column_distribution_scores': {
                    k: v.get('score', None) for k, v in column_tests.items() if 'score' in v
                },
                'correlation_similarity_score': correlation.get('correlation_similarity_score', None),
                'pca_overlap_score': pca.get('pca_overlap_score', None),
                'privacy_score': detectability.get('privacy_score', None),
                'plots': {
                    'correlation_plot': str(correlation.get('plot_path', '')),
                    'pca_plot': str(pca.get('plot_path', '')),
                    'column_plots': {
                        k: str(v.get('plot_path', '')) for k, v in column_tests.items() if 'plot_path' in v
                    }
                }
            }
            
            # Calculate an overall combined score (weighted average)
            scores = [
                (overall.get('quality_score', 0), 0.4),  # 40% weight to SDV's quality score
                (correlation.get('correlation_similarity_score', 0), 0.2),  # 20% weight to correlation similarity
                (detectability.get('privacy_score', 0), 0.2),  # 20% weight to privacy score
                (pca.get('pca_overlap_score', 0) if pca.get('pca_overlap_score') is not None else 0, 0.2)  # 20% weight to PCA overlap
            ]
            
            # Filter out None values
            valid_scores = [(score, weight) for score, weight in scores if score is not None]
            
            if valid_scores:
                combined_score = sum(score * weight for score, weight in valid_scores) / sum(weight for _, weight in valid_scores)
                summary['combined_score'] = combined_score
            else:
                summary['combined_score'] = None
            
            # Save summary report
            summary_path = os.path.join(self.results_dir, 'summary_report.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary report complete. Combined score: {summary.get('combined_score')}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return {"error": str(e)}
    
    def generate_human_readable_report(self) -> str:
        """
        Generate a human-readable text report summarizing the evaluation results.
        
        Returns:
            String containing the formatted report
        """
        logger.info("Generating human-readable report")
        
        try:
            # Run summary report to get all results
            summary = self.generate_summary_report()
            
            if 'error' in summary:
                return f"Error generating report: {summary['error']}"
            
            # Format the report
            report = [
                "# Synthetic Data Quality Report",
                "",
                f"## Overall Quality",
                f"Combined quality score: {summary.get('combined_score', 'N/A'):.2f} out of 1.0",
                f"SDV quality score: {summary.get('overall_quality_score', 'N/A'):.2f} out of 1.0",
                "",
                f"## Privacy Assessment",
                f"Privacy score: {summary.get('privacy_score', 'N/A'):.2f} out of 1.0",
                "A privacy score of 1.0 means the synthetic data is indistinguishable from real data.",
                "",
                "## Column Distribution Analysis",
                "Scores closer to 1.0 indicate better matching distributions:",
            ]
            
            # Add column scores, sorted by score (descending)
            if 'column_distribution_scores' in summary:
                col_scores = summary['column_distribution_scores']
                sorted_cols = sorted(col_scores.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
                
                for col, score in sorted_cols:
                    if score is not None:
                        report.append(f"- {col}: {score:.2f}")
            
            # Add correlation and PCA info
            report.extend([
                "",
                "## Correlation Structure",
                f"Correlation similarity score: {summary.get('correlation_similarity_score', 'N/A'):.2f} out of 1.0",
                "This measures how well the synthetic data preserves relationships between variables.",
                "",
                "## Data Structure",
                f"PCA overlap score: {summary.get('pca_overlap_score', 'N/A'):.2f if summary.get('pca_overlap_score') is not None else 'N/A'} out of 1.0",
                "This measures how well the synthetic data captures the overall structure of the real data."
            ])
            
            # Add recommendations section
            report.extend([
                "",
                "## Recommendations",
            ])
            
            combined_score = summary.get('combined_score', 0)
            
            if combined_score is None:
                report.append("Unable to calculate recommendations due to missing scores.")
            elif combined_score >= 0.8:
                report.append("✓ The synthetic data is high quality and ready for use!")
                report.append("✓ The synthetic data closely resembles the statistical properties of the real data.")
            elif combined_score >= 0.6:
                report.append("✓ The synthetic data is of good quality.")
                report.append("  Consider reviewing the column distribution scores to identify specific areas for improvement.")
            else:
                report.append("⚠ The synthetic data quality is below optimal levels.")
                report.append("  Consider:")
                report.append("  - Increasing the number of training epochs for the CTGAN synthesizer")
                report.append("  - Using different synthesizer parameters")
                report.append("  - Pre-processing the real data to handle outliers or missing values")
            
            # Join the report lines
            full_report = "\n".join(report)
            
            # Save report
            report_path = os.path.join(self.results_dir, 'human_readable_report.md')
            with open(report_path, 'w') as f:
                f.write(full_report)
            
            logger.info("Human-readable report complete")
            return full_report
            
        except Exception as e:
            logger.error(f"Error generating human-readable report: {str(e)}")
            return f"Error generating human-readable report: {str(e)}"

def main():
    """Example usage of the evaluator module."""
    try:
        # Load real and synthetic data
        real_data_path = input("Enter path to real data CSV: ")
        synth_data_path = input("Enter path to synthetic data CSV: ")
        
        real_data = pd.read_csv(real_data_path)
        synth_data = pd.read_csv(synth_data_path)
        
        # Initialize evaluator
        evaluator = SyntheticDataEvaluator(real_data, synth_data)
        
        # Generate report
        report = evaluator.generate_human_readable_report()
        
        print("\n" + "="*50)
        print(report)
        print("="*50)
        print(f"\nDetailed results saved in: {evaluator.results_dir}/")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()