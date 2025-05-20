import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import pandas as pd
from datetime import datetime
import logging

# Configure matplotlib to use a non-interactive backend
# This MUST come before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require GUI thread

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import custom modules
try:
    from generator import generate_synthetic_data, preprocess_data, import_sdv_modules
    from fixed_enhanced_evaluator import SimpleEvaluator  # Using the fixed enhanced evaluator
    logger.info("Successfully imported fixed enhanced evaluator")
except ImportError as e:
    # Try alternative imports for the evaluator
    try:
        from enhanced_simple_evaluator import SimpleEvaluator  # Try the original enhanced evaluator
        logger.info("Successfully imported enhanced evaluator")
    except ImportError:
        try:
            from simple_evaluator import SimpleEvaluator  # Fallback to original evaluator
            logger.info("Using standard evaluator. For more comprehensive evaluation, make sure enhanced_simple_evaluator.py is available.")
        except ImportError as e2:
            logger.error(f"Error importing modules: {str(e2)}")
            logger.error("Make sure generator.py and evaluator files are in the same directory.")
            sys.exit(1)

class ConsoleRedirector:
    """Redirects console output to a tkinter Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        
    def write(self, text):
        self.buffer += text
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)
        
    def flush(self):
        pass

class SyntheticDataApp:
    """Main GUI application for the Synthetic Data App."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pseudomorph - Synthetic Data Generator & Evaluator")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Remember last directory accessed
        self.last_directory = os.path.expanduser("~")  # Start with home directory
        
        # Check SDV installation at startup
        self.check_sdv_installation()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tab control
        self.tab_control = ttk.Notebook(self.main_frame)
        
        # Create tabs
        self.generate_tab = ttk.Frame(self.tab_control)
        self.evaluate_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.generate_tab, text="Generate Synthetic Data")
        self.tab_control.add(self.evaluate_tab, text="Evaluate Synthetic Data")
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Bind tab change event
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)
        
        # Setup Generate tab
        self.setup_generate_tab()
        
        # Setup Evaluate tab
        self.setup_evaluate_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Flag to track if there's an ongoing operation
        self.processing = False
        
        # Stored report path
        self.report_path = None
    
    def on_tab_change(self, event):
        """Handle tab change events."""
        selected_tab = self.tab_control.index(self.tab_control.select())
        if selected_tab == 1:  # Evaluate tab (index 1)
            # When switching to evaluate tab, ensure real_data_var matches input_file_var
            if self.input_file_var.get():
                self.real_data_var.set(self.input_file_var.get())
    
    def check_sdv_installation(self):
        """Check if SDV is installed and display warning if not."""
        try:
            import_sdv_modules()
        except ImportError:
            messagebox.showwarning(
                "SDV Not Found", 
                "The Synthetic Data Vault (SDV) package is not installed. "
                "Please install it using: pip install sdv"
            )
    
    def setup_generate_tab(self):
        """Setup the Generate tab UI elements."""
        # Create frames
        input_frame = ttk.LabelFrame(self.generate_tab, text="Input Data", padding="10")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        options_frame = ttk.LabelFrame(self.generate_tab, text="Generation Options", padding="10")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        output_frame = ttk.LabelFrame(self.generate_tab, text="Output", padding="10")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        console_frame = ttk.LabelFrame(self.generate_tab, text="Log", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Input file selection
        ttk.Label(input_frame, text="Input CSV File:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.input_file_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_file_var, width=50).grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input_file).grid(column=2, row=0, padx=5, pady=5)
        
        # Generation options
        ttk.Label(options_frame, text="Method:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.method_var = tk.StringVar(value="gaussian")
        method_combo = ttk.Combobox(options_frame, textvariable=self.method_var, state="readonly")
        method_combo["values"] = ("gaussian", "ctgan", "both")
        method_combo.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Number of Rows:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.rows_var = tk.StringVar(value="1000")
        ttk.Entry(options_frame, textvariable=self.rows_var, width=10).grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="CTGAN Epochs:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="30")
        ttk.Entry(options_frame, textvariable=self.epochs_var, width=10).grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Preprocessing options
        self.clean_columns_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Clean Column Names (lowercase, replace spaces)", 
                        variable=self.clean_columns_var).grid(column=0, row=3, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        self.handle_blanks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Handle Blank Values (convert to NaN)", 
                        variable=self.handle_blanks_var).grid(column=0, row=4, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Output directory
        ttk.Label(output_frame, text="Output Directory:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir_var = tk.StringVar(value="output")
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50).grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir).grid(column=2, row=0, padx=5, pady=5)
        
        # Generate button
        self.generate_button = ttk.Button(output_frame, text="Generate Synthetic Data", command=self.generate_data)
        self.generate_button.grid(column=0, row=1, columnspan=3, pady=10)
        
        # Console output
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=10)
        self.console_text.pack(fill=tk.BOTH, expand=True)
        self.console_text.config(state=tk.DISABLED)
    
    def setup_evaluate_tab(self):
        """Setup the Evaluate tab UI elements."""
        # Create frames
        data_frame = ttk.LabelFrame(self.evaluate_tab, text="Data Selection", padding="10")
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        output_frame = ttk.LabelFrame(self.evaluate_tab, text="Output", padding="10")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        console_frame = ttk.LabelFrame(self.evaluate_tab, text="Log", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Real data selection
        ttk.Label(data_frame, text="Real Data CSV:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.real_data_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.real_data_var, width=50).grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(data_frame, text="Browse...", command=self.browse_real_data).grid(column=2, row=0, padx=5, pady=5)
        
        # Synthetic data selection
        ttk.Label(data_frame, text="Synthetic Data CSV:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.synthetic_data_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.synthetic_data_var, width=50).grid(column=1, row=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(data_frame, text="Browse...", command=self.browse_synthetic_data).grid(column=2, row=1, padx=5, pady=5)
        
        # Output directory
        ttk.Label(output_frame, text="Output Directory:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.eval_output_dir_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.eval_output_dir_var, width=50).grid(column=1, row=0, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse...", command=self.browse_eval_output_dir).grid(column=2, row=0, padx=5, pady=5)
        
        # Evaluate button
        self.evaluate_button = ttk.Button(output_frame, text="Evaluate Synthetic Data", command=self.evaluate_data)
        self.evaluate_button.grid(column=0, row=1, columnspan=3, pady=10)
        
        # Open report button
        self.open_report_button = ttk.Button(output_frame, text="Open Evaluation Report", command=self.open_report)
        self.open_report_button.grid(column=0, row=2, columnspan=3, pady=10)
        self.open_report_button.config(state=tk.DISABLED)
        
        # Console output
        self.eval_console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=10)
        self.eval_console_text.pack(fill=tk.BOTH, expand=True)
        self.eval_console_text.config(state=tk.DISABLED)
    
    def browse_input_file(self):
        """Browse for input CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select Input CSV File",
            initialdir=self.last_directory,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.input_file_var.set(file_path)
            # Always update the real data path in the Evaluate tab
            self.real_data_var.set(file_path)
            # Remember this directory
            self.last_directory = os.path.dirname(file_path)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.last_directory
        )
        if dir_path:
            self.output_dir_var.set(dir_path)
            # Remember this directory, but DO NOT update real_data_var
            self.last_directory = dir_path
    
    def browse_real_data(self):
        """Browse for real data CSV file."""
        # Start in the directory of the input file if available
        if self.input_file_var.get() and os.path.exists(self.input_file_var.get()):
            initial_dir = os.path.dirname(self.input_file_var.get())
        else:
            initial_dir = self.last_directory
            
        file_path = filedialog.askopenfilename(
            title="Select Real Data CSV File",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.real_data_var.set(file_path)
            # Remember this directory
            self.last_directory = os.path.dirname(file_path)
    
    def browse_synthetic_data(self):
        """Browse for synthetic data CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select Synthetic Data CSV File",
            initialdir=self.last_directory,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.synthetic_data_var.set(file_path)
            # Remember this directory
            self.last_directory = os.path.dirname(file_path)
    
    def browse_eval_output_dir(self):
        """Browse for evaluation output directory."""
        dir_path = filedialog.askdirectory(
            title="Select Evaluation Output Directory",
            initialdir=self.last_directory
        )
        if dir_path:
            self.eval_output_dir_var.set(dir_path)
            # Remember this directory
            self.last_directory = dir_path
    
    def generate_data(self):
        """Generate synthetic data based on input parameters."""
        # Prevent multiple operations
        if self.processing:
            messagebox.showinfo("Processing", "An operation is already in progress. Please wait.")
            return
            
        # Validate inputs
        if not self.input_file_var.get():
            messagebox.showerror("Error", "Please select an input CSV file.")
            return
        
        if not os.path.exists(self.input_file_var.get()):
            messagebox.showerror("Error", f"Input file not found: {self.input_file_var.get()}")
            return
        
        # Update real_data_var with input_file_var value to ensure it's set correctly
        self.real_data_var.set(self.input_file_var.get())
        
        try:
            rows = int(self.rows_var.get())
            if rows <= 0:
                messagebox.showerror("Error", "Number of rows must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of rows must be a valid integer.")
            return
        
        try:
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                messagebox.showerror("Error", "Number of epochs must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of epochs must be a valid integer.")
            return
        
        # Set processing flag
        self.processing = True
        
        # Disable the generate button while processing
        self.generate_button.config(state=tk.DISABLED)
        self.status_var.set("Generating synthetic data...")
        
        # Clear console output
        self.console_text.config(state=tk.NORMAL)
        self.console_text.delete(1.0, tk.END)
        self.console_text.config(state=tk.DISABLED)
        
        # Redirect stdout/stderr to console
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text)
        sys.stderr = ConsoleRedirector(self.console_text)
        
        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self._generate_data_thread)
        thread.daemon = True
        thread.start()
        
        # Check if thread is still running
        self.root.after(100, lambda: self._check_thread_status(thread, original_stdout, original_stderr))
    
    def _generate_data_thread(self):
        """Worker thread for data generation."""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir_var.get()):
                os.makedirs(self.output_dir_var.get())
            
            # Load and preprocess data
            print(f"Loading real data from: {self.input_file_var.get()}")
            real_data = pd.read_csv(self.input_file_var.get())
            print(f"Successfully loaded data. Original Shape: {real_data.shape}")
            
            # Preprocess data
            preprocessed_data = preprocess_data(
                real_data, 
                clean_column_names=self.clean_columns_var.get(),
                handle_blanks=self.handle_blanks_var.get()
            )
            
            # Save preprocessed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir_var.get(), timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            preprocessed_path = os.path.join(output_dir, 'preprocessed_data.csv')
            preprocessed_data.to_csv(preprocessed_path, index=False)
            print(f"Preprocessed data saved to: {preprocessed_path}")
            
            # Generate synthetic data
            method = self.method_var.get()
            rows = int(self.rows_var.get())
            epochs = int(self.epochs_var.get())
            
            # Store paths to generated files
            generated_files = {}
            
            if method in ['gaussian', 'both']:
                print(f"Generating {rows} rows using GaussianCopulaSynthesizer...")
                gaussian_data = generate_synthetic_data(
                    preprocessed_data, 
                    num_rows=rows,
                    synthesizer_type='gaussian'
                )
                
                # Save Gaussian synthetic data
                gaussian_path = os.path.join(output_dir, 'synthetic_data_gaussian.csv')
                gaussian_data.to_csv(gaussian_path, index=False)
                print(f"Gaussian synthetic data saved to: {gaussian_path}")
                generated_files['gaussian'] = gaussian_path
            
            if method in ['ctgan', 'both']:
                print(f"Generating {rows} rows using CTGANSynthesizer (epochs: {epochs})...")
                ctgan_data = generate_synthetic_data(
                    preprocessed_data, 
                    num_rows=rows,
                    synthesizer_type='ctgan',
                    epochs=epochs
                )
                
                # Save CTGAN synthetic data
                ctgan_path = os.path.join(output_dir, 'synthetic_data_ctgan.csv')
                ctgan_data.to_csv(ctgan_path, index=False)
                print(f"CTGAN synthetic data saved to: {ctgan_path}")
                generated_files['ctgan'] = ctgan_path
            
            print("\nSynthetic data generation complete!")
            print(f"Output files saved to: {output_dir}")
            
            # Always set the real data path in the evaluate tab
            self.root.after(0, lambda: self.real_data_var.set(self.input_file_var.get()))
            
            # Remember the output directory for evaluation
            if not self.eval_output_dir_var.get():
                # Use the main thread to update GUI elements
                self.root.after(0, lambda: self.eval_output_dir_var.set(output_dir))
            
            # Set the synthetic data path in the evaluate tab
            if method == 'gaussian' and 'gaussian' in generated_files:
                self.root.after(0, lambda: self.synthetic_data_var.set(generated_files['gaussian']))
            elif method == 'ctgan' and 'ctgan' in generated_files:
                self.root.after(0, lambda: self.synthetic_data_var.set(generated_files['ctgan']))
            elif method == 'both' and 'gaussian' in generated_files:
                # Default to gaussian when both are generated
                self.root.after(0, lambda: self.synthetic_data_var.set(generated_files['gaussian']))
            
        except Exception as e:
            import traceback
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
    
    def evaluate_data(self):
        """Evaluate synthetic data against real data."""
        # Prevent multiple operations
        if self.processing:
            messagebox.showinfo("Processing", "An operation is already in progress. Please wait.")
            return
            
        # Validate inputs
        if not self.real_data_var.get():
            messagebox.showerror("Error", "Please select a real data CSV file.")
            return
        
        if not os.path.exists(self.real_data_var.get()):
            messagebox.showerror("Error", f"Real data file not found: {self.real_data_var.get()}")
            return
        
        if not self.synthetic_data_var.get():
            messagebox.showerror("Error", "Please select a synthetic data CSV file.")
            return
        
        if not os.path.exists(self.synthetic_data_var.get()):
            messagebox.showerror("Error", f"Synthetic data file not found: {self.synthetic_data_var.get()}")
            return
        
        # Set processing flag
        self.processing = True
        
        # Disable the evaluate button while processing
        self.evaluate_button.config(state=tk.DISABLED)
        self.open_report_button.config(state=tk.DISABLED)
        self.status_var.set("Evaluating synthetic data...")
        
        # Clear console output
        self.eval_console_text.config(state=tk.NORMAL)
        self.eval_console_text.delete(1.0, tk.END)
        self.eval_console_text.config(state=tk.DISABLED)
        
        # Redirect stdout/stderr to console
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.eval_console_text)
        sys.stderr = ConsoleRedirector(self.eval_console_text)
        
        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self._evaluate_data_thread)
        thread.daemon = True
        thread.start()
        
        # Check if thread is still running
        self.root.after(100, lambda: self._check_eval_thread_status(thread, original_stdout, original_stderr))
    
    def _evaluate_data_thread(self):
        """Worker thread for data evaluation."""
        try:
            # Create output directory if specified
            output_dir = self.eval_output_dir_var.get()
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Initialize evaluator
            evaluator = SimpleEvaluator(
                real_data_path=self.real_data_var.get(),
                synthetic_data_path=self.synthetic_data_var.get(),
                output_dir=output_dir
            )
            
            # Run evaluation
            self.report_path = evaluator.evaluate()
            
            print("\nEvaluation complete!")
            print(f"Report saved to: {self.report_path}")
            
        except Exception as e:
            import traceback
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            self.report_path = None
    
    def open_report(self):
        """Open the evaluation report in the default application."""
        if hasattr(self, 'report_path') and self.report_path and os.path.exists(self.report_path):
            # Use the appropriate command based on the platform
            import platform
            import subprocess
            
            try:
                if platform.system() == 'Windows':
                    os.startfile(self.report_path)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', self.report_path])
                else:  # Linux
                    subprocess.call(['xdg-open', self.report_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open report: {str(e)}")
        else:
            messagebox.showerror("Error", "Evaluation report not found.")
    
    def _check_thread_status(self, thread, original_stdout, original_stderr):
        """Check if the worker thread has completed."""
        if thread.is_alive():
            # Thread still running, check again later
            self.root.after(100, lambda: self._check_thread_status(thread, original_stdout, original_stderr))
        else:
            # Thread completed, restore stdout/stderr and enable button
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.generate_button.config(state=tk.NORMAL)
            self.status_var.set("Ready")
            self.processing = False
            
            # Make sure the real_data_var is still set to input_file_var (might have been changed elsewhere)
            self.real_data_var.set(self.input_file_var.get())
            
            # Switch to evaluate tab
            self.tab_control.select(1)  # Select evaluate tab
    
    def _check_eval_thread_status(self, thread, original_stdout, original_stderr):
        """Check if the evaluation worker thread has completed."""
        if thread.is_alive():
            # Thread still running, check again later
            self.root.after(100, lambda: self._check_eval_thread_status(thread, original_stdout, original_stderr))
        else:
            # Thread completed, restore stdout/stderr and enable buttons
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.evaluate_button.config(state=tk.NORMAL)
            self.processing = False
            
            # Enable the open report button if report was generated
            if hasattr(self, 'report_path') and self.report_path:
                self.open_report_button.config(state=tk.NORMAL)
            
            self.status_var.set("Ready")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = SyntheticDataApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()