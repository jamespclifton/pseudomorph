"""
Simple Synthetic Data Evaluator - Enhanced Version

This is a simplified version of the evaluator that evaluates all columns in a dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Union, Optional
from scipy.stats import ks_2samp, chi2_contingency

# Configure matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
# Limit the number of open figures to avoid warnings
matplotlib.rcParams['figure.max_open_warning'] = 50

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleEvaluator:
    """A simplified class for evaluating synthetic data quality."""
    
    def __init__(self, real_data_path, synthetic_data_path, output_dir=None):
        """
        Initialize the evaluator with paths to real and synthetic data.
        
        Args:
            real_data_path: Path to CSV file containing real data
            synthetic_data_path: Path to CSV file containing synthetic data
            output_dir: Directory to save evaluation results (default: timestamped directory)
        """
        self.real_data_path = real_data_path
        self.synthetic_data_path = synthetic_data_path
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"evaluation_results_{timestamp}"
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Load data
        self.real_data = pd.read_csv(real_data_path)
        self.synthetic_data = pd.read_csv(synthetic_data_path)
        
        # Normalize column names (convert to lowercase and replace dashes with underscores)
        self.real_data.columns = [col.lower().replace('-', '_') for col in self.real_data.columns]
        self.synthetic_data.columns = [col.lower().replace('-', '_') for col in self.synthetic_data.columns]
        
        # Log basic info
        logger.info(f"Loaded real data: {self.real_data.shape}")
        logger.info(f"Loaded synthetic data: {self.synthetic_data.shape}")
        
        # Verify that both datasets have the same columns after normalization
        real_cols = set(self.real_data.columns)
        synth_cols = set(self.synthetic_data.columns)
        
        if real_cols != synth_cols:
            logger.warning("Real and synthetic data have different columns!")
            logger.warning(f"Columns only in real data: {real_cols - synth_cols}")
            logger.warning(f"Columns only in synthetic data: {synth_cols - real_cols}")
            
            # Use intersection of columns
            common_cols = list(real_cols.intersection(synth_cols))
            self.real_data = self.real_data[common_cols]
            self.synthetic_data = self.synthetic_data[common_cols]
            logger.info(f"Using {len(common_cols)} common columns for evaluation")
    
    def evaluate(self):
        """Run the complete evaluation process."""
        logger.info("Starting evaluation...")
        
        # Create a summary report file
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        
        # Close all open figures as a precaution
        plt.close('all')
        
        with open(report_path, "w") as f:
            f.write("# Synthetic Data Evaluation Report\n\n")
            f.write(f"- Real data: {self.real_data_path}\n")
            f.write(f"- Synthetic data: {self.synthetic_data_path}\n")
            f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic statistics
            f.write("## 1. Basic Statistics\n\n")
            self._evaluate_basic_stats(f)
            
            # Column distributions
            f.write("\n## 2. Column Distributions\n\n")
            self._evaluate_column_distributions(f)
            
            # Correlation analysis
            f.write("\n## 3. Correlation Analysis\n\n")
            self._evaluate_correlations(f)
            
            # PCA visualization
            f.write("\n## 4. Data Structure (PCA)\n\n")
            self._evaluate_pca(f)
            
            # Privacy assessment
            f.write("\n## 5. Privacy Assessment\n\n")
            self._evaluate_privacy(f)
            
            # Summary
            f.write("\n## 6. Summary\n\n")
            f.write("The synthetic data evaluation is complete. Key findings:\n\n")
            f.write("- The synthetic data captures the statistical properties of the real data with varying degrees of accuracy.\n")
            f.write("- Check the individual column distributions to identify specific areas for improvement.\n")
            f.write("- Review the correlation heatmaps to see how well relationships between variables are preserved.\n")
        
        # Make sure all figures are closed
        plt.close('all')
        
        logger.info(f"Evaluation complete! Report saved to: {report_path}")
        return report_path
    
    def _evaluate_basic_stats(self, file):
        """Evaluate and report basic statistics."""
        file.write("### Basic Shape and Data Types\n\n")
        file.write(f"- Real data shape: {self.real_data.shape}\n")
        file.write(f"- Synthetic data shape: {self.synthetic_data.shape}\n\n")
        
        # Compare data types
        real_dtypes = self.real_data.dtypes
        synth_dtypes = self.synthetic_data.dtypes
        
        file.write("### Data Type Comparison\n\n")
        file.write("| Column | Real Data Type | Synthetic Data Type | Match |\n")
        file.write("|--------|---------------|---------------------|-------|\n")
        
        for col in self.real_data.columns:
            real_type = real_dtypes[col]
            synth_type = synth_dtypes[col]
            match = "✓" if real_type == synth_type else "✗"
            file.write(f"| {col} | {real_type} | {synth_type} | {match} |\n")
        
        # Summary statistics for numerical columns
        numerical_cols = self.real_data.select_dtypes(include=['number']).columns
        
        if len(numerical_cols) > 0:
            file.write("\n### Summary Statistics (Numerical Columns)\n\n")
            
            for col in numerical_cols:
                file.write(f"#### {col}\n\n")
                
                real_stats = self.real_data[col].describe()
                synth_stats = self.synthetic_data[col].describe()
                
                stats_df = pd.DataFrame({
                    'Real': real_stats,
                    'Synthetic': synth_stats,
                    'Difference': synth_stats - real_stats,
                    'Percent Diff': ((synth_stats - real_stats) / real_stats * 100).round(2)
                })
                
                file.write(stats_df.to_markdown() + "\n\n")
                
        # Value counts for categorical columns
        categorical_cols = self.real_data.select_dtypes(exclude=['number']).columns
        
        if len(categorical_cols) > 0:
            file.write("\n### Top Categories (Categorical Columns)\n\n")
            
            for col in categorical_cols[:10]:  # Limit to first 10 categorical columns to keep report manageable
                file.write(f"#### {col}\n\n")
                
                real_counts = self.real_data[col].value_counts(normalize=True).head(5)
                synth_counts = self.synthetic_data[col].value_counts(normalize=True).head(5)
                
                file.write("Top 5 categories in real data:\n\n")
                file.write(real_counts.to_markdown() + "\n\n")
                
                file.write("Top 5 categories in synthetic data:\n\n")
                file.write(synth_counts.to_markdown() + "\n\n")
    
    def _evaluate_column_distributions(self, file):
        """Evaluate and visualize column distributions."""
        # Select columns to evaluate - use all columns
        numerical_cols = self.real_data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = [
            col for col in self.real_data.select_dtypes(exclude=['number']).columns
            if self.real_data[col].nunique() <= 30  # Limit to categories with reasonable number of unique values
        ]
        
        # Combine columns
        columns = numerical_cols + categorical_cols
        
        file.write(f"Evaluating distributions for {len(columns)} columns.\n\n")
        
        scores = {}
        
        # Close any existing figures to avoid memory issues
        plt.close('all')
        
        for col in columns:
            file.write(f"### Column: {col}\n\n")
            
            # Create distribution plot
            fig_path = os.path.join(self.output_dir, f"dist_{col}.png")
            
            try:
                # Create a new figure for each column
                plt.figure(figsize=(10, 6))
                
                if pd.api.types.is_numeric_dtype(self.real_data[col]):
                    # Numerical column - use histograms
                    plt.hist(self.real_data[col].dropna(), alpha=0.5, bins=20, label='Real')
                    plt.hist(self.synthetic_data[col].dropna(), alpha=0.5, bins=20, label='Synthetic')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    
                    # Calculate KS test using scipy directly
                    stat, p_value = ks_2samp(
                        self.real_data[col].dropna(),
                        self.synthetic_data[col].dropna()
                    )
                    
                    # Transform p-value to a score (higher p-value = better similarity)
                    # Score of 1 means distributions are identical
                    # Score close to 0 means distributions are very different
                    score = min(1.0, p_value * 20)  # Scale p-value to get a more intuitive score
                    scores[col] = score
                    
                    file.write(f"Kolmogorov-Smirnov test:\n")
                    file.write(f"- Statistic: {stat:.4f}\n")
                    file.write(f"- p-value: {p_value:.4f}\n")
                    file.write(f"- Similarity score: {score:.4f} (higher is better)\n\n")
                else:
                    # Categorical column - use bar plots
                    real_counts = self.real_data[col].value_counts(normalize=True)
                    synth_counts = self.synthetic_data[col].value_counts(normalize=True)
                    
                    # Combine and get all categories
                    all_cats = pd.concat([real_counts, synth_counts], axis=1).fillna(0)
                    all_cats.columns = ['Real', 'Synthetic']
                    
                    all_cats.plot(kind='bar')
                    plt.xlabel(col)
                    plt.ylabel('Proportion')
                    
                    # Use Chi-Square test for categorical columns
                    try:
                        # Create contingency table with all categories
                        all_categories = list(set(self.real_data[col].dropna().unique()) | 
                                             set(self.synthetic_data[col].dropna().unique()))
                        
                        real_counts_full = self.real_data[col].value_counts()
                        synth_counts_full = self.synthetic_data[col].value_counts()
                        
                        # Ensure all categories are present in both counts
                        for cat in all_categories:
                            if cat not in real_counts_full:
                                real_counts_full[cat] = 0
                            if cat not in synth_counts_full:
                                synth_counts_full[cat] = 0
                        
                        # Convert to DataFrame and sort by category
                        counts_df = pd.DataFrame({
                            'real': real_counts_full,
                            'synthetic': synth_counts_full
                        }).fillna(0)
                        
                        # Chi-square test
                        chi2, p_value, _, _ = chi2_contingency(counts_df.T)
                        
                        # Transform p-value to a score
                        score = min(1.0, p_value * 20)
                        scores[col] = score
                        
                        file.write(f"Chi-square test:\n")
                        file.write(f"- Statistic: {chi2:.4f}\n")
                        file.write(f"- p-value: {p_value:.4f}\n")
                        file.write(f"- Similarity score: {score:.4f} (higher is better)\n\n")
                    except Exception as e:
                        file.write(f"Could not perform Chi-square test: {str(e)}\n\n")
                        # Fallback score based on Jensen-Shannon divergence
                        js_div = self._jensen_shannon_divergence(real_counts, synth_counts)
                        score = 1.0 - min(1.0, js_div)
                        scores[col] = score
                        file.write(f"Jensen-Shannon divergence: {js_div:.4f}\n")
                        file.write(f"Similarity score: {score:.4f} (higher is better)\n\n")
                
                plt.title(f'Distribution of {col}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()  # Explicitly close figure after saving
                
                file.write(f"![Distribution of {col}]({os.path.basename(fig_path)})\n\n")
                
            except Exception as e:
                file.write(f"Error generating distribution for {col}: {str(e)}\n\n")
                plt.close()  # Make sure to close the figure even if there's an error
        
        # Calculate average score
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            file.write(f"\n### Overall Distribution Score: {avg_score:.4f}\n\n")
            file.write("*Higher scores indicate better similarity between real and synthetic distributions.*\n\n")
    
    def _jensen_shannon_divergence(self, p, q):
        """
        Calculate Jensen-Shannon divergence between two probability distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            Jensen-Shannon divergence
        """
        # Ensure distributions have the same categories
        all_categories = set(p.index) | set(q.index)
        p_full = pd.Series({cat: p.get(cat, 0) for cat in all_categories})
        q_full = pd.Series({cat: q.get(cat, 0) for cat in all_categories})
        
        # Calculate Kullback-Leibler divergence
        def kl_divergence(p, q):
            return np.sum(p * np.log2(p / q + np.finfo(float).eps))
        
        # Calculate Jensen-Shannon divergence
        m = 0.5 * (p_full + q_full)
        return 0.5 * (kl_divergence(p_full, m) + kl_divergence(q_full, m))
    
    def _evaluate_correlations(self, file):
        """Evaluate and visualize correlations between variables."""
        # Get numerical columns for correlation analysis
        numerical_cols = self.real_data.select_dtypes(include=['number']).columns
        
        if len(numerical_cols) < 2:
            file.write("Not enough numerical columns for correlation analysis.\n\n")
            return
        
        # Maximum number of columns to include in visualization to avoid overflow errors
        # This can be adjusted based on screen size and readability
        max_cols = 12
        
        # Create two visualizations: one summary with all columns, and one detailed with a subset
        file.write("### Full Correlation Matrix\n\n")
        file.write("The correlation analysis is split into two parts:\n")
        file.write("1. A summary matrix showing all numerical variables.\n")
        file.write("2. A detailed matrix showing correlations for a subset of important variables.\n\n")
        
        # ---- FULL CORRELATION MATRIX (SUMMARY) ----
        try:
            # Create a summary correlation matrix first to give an overview
            # Calculate correlation matrices
            real_corr = self.real_data[numerical_cols].corr()
            synth_corr = self.synthetic_data[numerical_cols].corr()
            diff_corr = real_corr - synth_corr
            
            # Replace NaN with 0 to avoid plotting issues
            real_corr = real_corr.fillna(0)
            synth_corr = synth_corr.fillna(0)
            diff_corr = diff_corr.fillna(0)
            
            # Create a simplified visualization for the correlation summary
            summary_path = os.path.join(self.output_dir, "correlation_summary.png")
            
            plt.figure(figsize=(10, 8))
            
            # Use a mask to show only the upper triangle
            mask = np.triu(np.ones_like(diff_corr, dtype=bool), k=1)
            
            # Plot absolute differences to clearly see where the models differ
            # Use RdBu_r colormap where red = large difference, blue = small difference
            abs_diff = np.abs(diff_corr)
            sns.heatmap(abs_diff, mask=mask, cmap='RdBu_r', vmin=0, vmax=1, 
                        annot=False, square=True, linewidths=.5)
            plt.title('Correlation Difference (absolute) - Full Matrix')
            plt.tight_layout()
            plt.savefig(summary_path)
            plt.close()
            
            file.write(f"![Correlation Summary - All Variables]({os.path.basename(summary_path)})\n\n")
            file.write("*The heatmap shows absolute differences between real and synthetic correlations. Redder areas indicate larger differences.*\n\n")
            
            # Calculate correlation similarity score for all columns
            from numpy.linalg import norm
            frob_norm = norm(diff_corr.values, 'fro')
            max_norm = norm(np.ones_like(diff_corr) * 2, 'fro')  # Max possible difference is 2 (-1 to 1)
            full_similarity_score = 1 - (frob_norm / max_norm)  # Scale to 0-1 range
            
            file.write(f"Full correlation similarity score: {full_similarity_score:.4f} (higher is better)\n\n")
            
        except Exception as e:
            file.write(f"Error generating full correlation summary: {str(e)}\n\n")
        
        # ---- DETAILED CORRELATION MATRIX ----
        file.write("### Detailed Correlation Analysis\n\n")
        
        # Select a subset of columns for detailed visualization
        # Strategy: Use high-importance variables or those with highest variance
        selected_cols = []
        
        # Always include ID columns if they exist
        id_cols = [col for col in numerical_cols if 'id' in col.lower() or 'key' in col.lower()]
        selected_cols.extend(id_cols[:2])  # Include up to 2 ID columns
        
        # Add other important columns
        remaining_cols = [col for col in numerical_cols if col not in selected_cols]
        
        # If there are date/year columns, prioritize them
        date_cols = [col for col in remaining_cols if any(x in col.lower() for x in ['year', 'date', 'time', 'grad'])]
        selected_cols.extend(date_cols[:3])  # Add up to 3 date-related columns
        
        # For remaining slots, add columns with highest variance
        remaining_cols = [col for col in numerical_cols if col not in selected_cols]
        if remaining_cols and len(selected_cols) < max_cols:
            # Calculate variance for remaining columns
            variances = self.real_data[remaining_cols].var()
            # Sort columns by variance (highest first)
            sorted_cols = variances.sort_values(ascending=False).index.tolist()
            # Add highest variance columns until we reach max_cols
            selected_cols.extend(sorted_cols[:max_cols - len(selected_cols)])
        
        # If we still don't have enough columns, just add the first few numerical columns
        if len(selected_cols) == 0:
            selected_cols = numerical_cols[:min(max_cols, len(numerical_cols))]
        elif len(selected_cols) < min(5, len(numerical_cols)):
            # Add more columns if we have less than 5 (but not more than available)
            remaining = [col for col in numerical_cols if col not in selected_cols]
            selected_cols.extend(remaining[:min(5, len(remaining))])
        
        file.write(f"Detailed correlation analysis for {len(selected_cols)} numerical columns:\n")
        file.write(", ".join(selected_cols) + "\n\n")
        
        try:
            # Calculate correlation matrices for selected columns
            real_corr = self.real_data[selected_cols].corr()
            synth_corr = self.synthetic_data[selected_cols].corr()
            
            # Replace NaN with 0 for visualization
            real_corr = real_corr.fillna(0)
            synth_corr = synth_corr.fillna(0)
            
            # Calculate difference
            diff_corr = real_corr - synth_corr
            
            # Generate plots
            corr_path = os.path.join(self.output_dir, "correlation_comparison.png")
            
            # Create a larger figure for more columns
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            # Real data correlation
            sns.heatmap(real_corr, annot=True, cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1, fmt='.2f')
            axes[0].set_title('Real Data Correlation')
            
            # Synthetic data correlation
            sns.heatmap(synth_corr, annot=True, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1, fmt='.2f')
            axes[1].set_title('Synthetic Data Correlation')
            
            # Difference
            sns.heatmap(diff_corr, annot=True, cmap='coolwarm', ax=axes[2], vmin=-1, vmax=1, fmt='.2f')
            axes[2].set_title('Correlation Difference (Real - Synthetic)')
            
            plt.tight_layout()
            plt.savefig(corr_path)
            plt.close(fig)  # Explicitly close the figure
            
            file.write(f"![Correlation Comparison]({os.path.basename(corr_path)})\n\n")
            
            # Calculate correlation similarity score (Frobenius norm of difference matrix)
            # Lower value means more similar
            from numpy.linalg import norm
            frob_norm = norm(diff_corr.values, 'fro')
            max_norm = norm(np.ones_like(diff_corr) * 2, 'fro')  # Max possible difference is 2 (-1 to 1)
            similarity_score = 1 - (frob_norm / max_norm)  # Scale to 0-1 range
            
            file.write(f"### Detailed Correlation Similarity Score: {similarity_score:.4f}\n\n")
            file.write("*Higher score indicates better preservation of relationships between variables.*\n\n")
            
        except Exception as e:
            file.write(f"Error generating detailed correlation analysis: {str(e)}\n\n")
            import traceback
            file.write(f"```\n{traceback.format_exc()}\n```\n\n")
    
    def _evaluate_pca(self, file):
        """Evaluate data structure using PCA visualization."""
        # Get numerical columns for PCA
        numerical_cols = self.real_data.select_dtypes(include=['number']).columns
        
        if len(numerical_cols) < 2:
            file.write("Not enough numerical columns for PCA visualization.\n\n")
            return
        
        # Prepare data for PCA
        try:
            # Close any existing figures to avoid memory issues
            plt.close('all')
            
            # Filter out constant columns that can cause issues with PCA
            real_numerical = self.real_data[numerical_cols].copy()
            synth_numerical = self.synthetic_data[numerical_cols].copy()
            
            # Find columns with variance = 0 in either dataset
            real_var = real_numerical.var()
            synth_var = synth_numerical.var()
            zero_var_cols = real_var[real_var == 0].index.tolist() + synth_var[synth_var == 0].index.tolist()
            zero_var_cols = list(set(zero_var_cols))  # Remove duplicates
            
            if zero_var_cols:
                file.write(f"Removing {len(zero_var_cols)} constant columns from PCA analysis: {', '.join(zero_var_cols)}\n\n")
                valid_cols = [col for col in numerical_cols if col not in zero_var_cols]
                real_numerical = real_numerical[valid_cols]
                synth_numerical = synth_numerical[valid_cols]
            
            # Handle NaN values by filling with column means
            real_numerical = real_numerical.fillna(real_numerical.mean())
            synth_numerical = synth_numerical.fillna(synth_numerical.mean())
            
            # Standardize the data
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_numerical)
            synth_scaled = scaler.transform(synth_numerical)  # Use same scaler
            
            # Check for NaN or infinity values and replace them
            real_scaled = np.nan_to_num(real_scaled)
            synth_scaled = np.nan_to_num(synth_scaled)
            
            # Apply PCA
            pca = PCA(n_components=2)
            real_pca = pca.fit_transform(real_scaled)
            synth_pca = pca.transform(synth_scaled)
            
            # Create PCA plot
            pca_path = os.path.join(self.output_dir, "pca_visualization.png")
            
            plt.figure(figsize=(10, 8))
            plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real Data', color='blue')
            plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label='Synthetic Data', color='red')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Comparison of Real vs Synthetic Data')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(pca_path)
            plt.close()  # Explicitly close the figure
            
            file.write(f"![PCA Visualization]({os.path.basename(pca_path)})\n\n")
            
            # Calculate distance between centroids
            real_centroid = np.mean(real_pca, axis=0)
            synth_centroid = np.mean(synth_pca, axis=0)
            centroid_distance = np.linalg.norm(real_centroid - synth_centroid)
            
            # Calculate standard deviations
            real_std = np.std(real_pca, axis=0)
            real_avg_std = np.mean(real_std)
            
            # Scale distance by standard deviation to get a relative measure
            relative_distance = centroid_distance / real_avg_std
            
            # Calculate a similarity score (1 for identical, 0 for very different)
            pca_similarity = 1 / (1 + relative_distance)
            
            file.write(f"### PCA Similarity Score: {pca_similarity:.4f}\n\n")
            file.write("*This score measures how well the synthetic data captures the overall structure of the real data.*\n\n")
            file.write(f"- Explained variance by first two components: {pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]:.2%}\n")
            
        except Exception as e:
            file.write(f"Error in PCA visualization: {str(e)}\n\n")
            import traceback
            file.write(f"```\n{traceback.format_exc()}\n```\n\n")
    
    def _evaluate_privacy(self, file):
        """Evaluate privacy by checking how easily real and synthetic data can be distinguished."""
        try:
            # Create combined dataset with labels
            real_data_labeled = self.real_data.copy()
            real_data_labeled['is_synthetic'] = 0
            
            synth_data_labeled = self.synthetic_data.copy()
            synth_data_labeled['is_synthetic'] = 1
            
            # Combine datasets
            combined_data = pd.concat([real_data_labeled, synth_data_labeled], ignore_index=True)
            
            # Handle potential encoding issues with categorical variables
            # Get dummies but first fill NAs
            for col in combined_data.select_dtypes(include=['object']):
                combined_data[col] = combined_data[col].fillna('NA')
            
            # Prepare features and target
            combined_data_encoded = pd.get_dummies(combined_data.drop('is_synthetic', axis=1), drop_first=True)
            y = combined_data['is_synthetic']
            
            # Check for any lingering NaNs and replace with 0
            X = combined_data_encoded.fillna(0)
            
            # Train a random forest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Cross-validation score
            cv_scores = cross_val_score(clf, X, y, cv=5)
            mean_cv_score = np.mean(cv_scores)
            
            # Calculate privacy score
            # If mean_cv_score is 0.5, it means the classifier is just guessing randomly
            # (can't distinguish between real and synthetic)
            # If mean_cv_score is 1.0, it means the classifier can perfectly distinguish them
            # Higher privacy score is better
            privacy_score = 1.0 - (2 * abs(mean_cv_score - 0.5))
            
            file.write(f"### Privacy Score: {privacy_score:.4f}\n\n")
            file.write("*A score of 1.0 means the synthetic data is indistinguishable from real data (50% classifier accuracy).*\n\n")
            file.write(f"- Mean classifier accuracy: {mean_cv_score:.4f}\n")
            file.write(f"- Cross-validation scores: {', '.join(f'{score:.4f}' for score in cv_scores)}\n\n")
            
            # Interpret the results
            if privacy_score > 0.8:
                file.write("- **Excellent Privacy**: The synthetic data is very hard to distinguish from real data.\n")
            elif privacy_score > 0.6:
                file.write("- **Good Privacy**: The synthetic data has good privacy properties but some patterns might be distinguishable.\n")
            elif privacy_score > 0.4:
                file.write("- **Moderate Privacy**: The synthetic data has moderate privacy properties.\n")
            else:
                file.write("- **Poor Privacy**: The synthetic data is easily distinguishable from real data.\n")
                
        except Exception as e:
            file.write(f"Error in privacy assessment: {str(e)}\n\n")
            
            # Provide alternative privacy assessment method
            file.write("Attempting alternative privacy assessment method...\n\n")
            
            try:
                # Check distribution overlap for key columns
                numerical_cols = self.real_data.select_dtypes(include=['number']).columns
                categorical_cols = self.real_data.select_dtypes(exclude=['number']).columns
                
                overlap_scores = []
                
                # Check numerical columns
                for col in numerical_cols[:3]:  # Limit to first 3 columns
                    real_min = self.real_data[col].min()
                    real_max = self.real_data[col].max()
                    
                    # Calculate percentage of synthetic values within real data range
                    in_range = ((self.synthetic_data[col] >= real_min) & 
                                (self.synthetic_data[col] <= real_max)).mean()
                    
                    overlap_scores.append(in_range)
                    file.write(f"- Column '{col}': {in_range:.2%} of synthetic values within real data range\n")
                
                # Check categorical columns
                for col in categorical_cols[:3]:  # Limit to first 3 columns
                    real_cats = set(self.real_data[col].unique())
                    synth_cats = set(self.synthetic_data[col].unique())
                    
                    # Calculate Jaccard similarity between category sets
                    intersection = len(real_cats.intersection(synth_cats))
                    union = len(real_cats.union(synth_cats))
                    
                    if union > 0:
                        similarity = intersection / union
                        overlap_scores.append(similarity)
                        file.write(f"- Column '{col}': Category set similarity: {similarity:.2f}\n")
                
                # Calculate average overlap score
                if overlap_scores:
                    avg_overlap = sum(overlap_scores) / len(overlap_scores)
                    file.write(f"\nAverage distribution overlap: {avg_overlap:.4f}\n")
                    
                    # Estimate privacy score based on overlap
                    # Moderate overlap is good for privacy
                    estimated_privacy = 1.0 - abs(avg_overlap - 0.7) / 0.3
                    estimated_privacy = max(0.0, min(1.0, estimated_privacy))
                    
                    file.write(f"Estimated privacy score: {estimated_privacy:.4f}\n")
                    
            except Exception as e2:
                file.write(f"Error in alternative privacy assessment: {str(e2)}\n\n")

def main():
    """Command line interface for the simple evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate synthetic data quality.')
    parser.add_argument('--real', required=True, help='Path to real data CSV file')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--output-dir', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluator = SimpleEvaluator(
        real_data_path=args.real,
        synthetic_data_path=args.synthetic,
        output_dir=args.output_dir
    )
    
    report_path = evaluator.evaluate()
    print(f"Evaluation complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main()