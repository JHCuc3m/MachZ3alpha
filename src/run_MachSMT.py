#!/usr/bin/env python3

import os
import sys
import glob
import random
import time  
import csv

# Add root root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
root_root_dir = os.path.dirname(root_dir)
sys.path.append(root_root_dir)

# Now we can import alphasmt
from alphasmt.evaluator import SolverRunner
from machsmt import MachSMT, Benchmark

def calculate_metrics(results, timeout):
    """
    Calculate PAR2 score and solving rate from results.
    
    Args:
        results: List of (solved, runtime, result) tuples
        timeout: Timeout value in seconds
    
    Returns:
        tuple: (par2_score, solving_rate)
    """
    if not results:
        return 0.0, 0.0
        
    num_solved = sum(1 for solved, _, _ in results if solved)
    total_instances = len(results)
    
    # Calculate PAR2 score
    par2_times = []
    for solved, runtime, _ in results:
        if solved:
            par2_times.append(runtime)
        else:
            par2_times.append(2 * timeout)  # PAR2 penalty
    
    avg_par2_score = sum(par2_times) / len(par2_times)
    solving_rate = (num_solved / total_instances) * 100 if total_instances > 0 else 0
    
    return avg_par2_score, solving_rate

def load_config(config_file):
    """Load configuration from a .env file"""
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

def adjust_csv_paths(input_csv, output_csv, benchmark_root):
    """Adjust the benchmark paths in the CSV to be absolute paths"""
    with open(input_csv, 'r') as infile, open(output_csv, 'w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        header = next(reader)
        writer.writerow(header)
        
        # Process rows
        for row in reader:
            if len(row) > 0:  # Ensure row has data
                # benchmark path is in the first column
                benchmark_path = row[0]
                absolute_path = os.path.join(benchmark_root, benchmark_path)
                row[0] = absolute_path
                writer.writerow(row)
    return output_csv

def evaluate_strategy_on_benchmark(
    solver_path: str,
    benchmark_path: str, 
    strategy: str,
    timeout: int = 300,
    tmp_dir: str = "/tmp/"
):
    """
    Evaluates a Z3 strategy on a single benchmark instance using proper thread handling.
    """
    try:
        # Create thread
        runnerThread = SolverRunner(solver_path, benchmark_path, timeout, 0, strategy, tmp_dir)
        runnerThread.start()
        
        # Time the execution
        time_start = time.time()
        time_left = max(0, timeout - (time.time() - time_start))
        runnerThread.join(time_left)
        
        # Collect results
        id, resTask, timeTask, pathTask = runnerThread.collect()
        solved = True if (resTask == 'sat' or resTask == 'unsat') else False
        
        return solved, timeTask, resTask
        
    except Exception as e:
        print(f"Error in evaluate_strategy_on_benchmark: {e}")
        return False, timeout, "error"

def load_strategy_mapping(mapping_file):
    """Load the strategy mapping from CSV file."""
    strategies = {}
    with open(mapping_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategies[row['solver']] = row['strategy']
    return strategies

def main():

    # Load configuration
    config_file = os.path.join(current_dir, "../training_data/config_files/hycomp_config.env")
    config = load_config(config_file)
    
    # Get configuration values with defaults
    z3_path = config.get('Z3_PATH', '/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/z3')
    timeout = int(config.get('TIMEOUT', '300'))
    tmp_dir = config.get('TMP_DIR', '/tmp/')
    num_benchmarks = int(config.get('NUM_BENCHMARKS', '100'))

    training_dir = "../" + config.get("TRAINING_DIR")

    # Set paths relative to current directory
    training_file = os.path.join(current_dir, training_dir + "machsmt_training.csv")
    adjusted_training_file = os.path.join(current_dir, training_dir + "adjusted_training.csv")

    output_dir = "../" + config.get("OUTPUT_DIR")

    output_file = os.path.join(current_dir, output_dir + "output.txt")
    mapping_file = os.path.join(current_dir, training_dir + "z3Alpha_strategy_mapping.csv")
    
    # Create adjusted training file with correct paths
    print("Adjusting benchmark paths in training data...")
    adjust_csv_paths(training_file, adjusted_training_file, root_dir)
    
    print(f"Root directory: {root_dir}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Training file: {adjusted_training_file}")
    print(f"Using Z3 path: {z3_path}")
    print(f"Timeout: {timeout} seconds")
    
    # Load strategy mapping
    strategy_mapping = load_strategy_mapping(mapping_file)
    
    # Clear/create output file
    with open(output_file, 'w') as f:
        pass
    
    # 1) Build MachSMT model with training data
    print("Building MachSMT model with Z3Alpha training data...")
    machsmt = MachSMT([adjusted_training_file])
    machsmt.save("lib")
    
    # 2) Evaluate using k-fold cross validation
    print("Evaluating model using k-fold cross validation...")
    machsmt.eval()
    
    # 3) Test on sample benchmarks
    print("Testing model on sample benchmarks...")
    
    # Get list of benchmark files
    benchmark_subpath = config.get('BENCHMARK_DIR', "benchmarks/fastsmt_exp/qf_nra/hycomp/test")
    benchmark_path = os.path.join(root_dir, benchmark_subpath)
    print(f"Looking for benchmarks in: {benchmark_path}")
    
    benchmark_files = glob.glob(os.path.join(benchmark_path, "**/*.smt2"), recursive=True)
    if not benchmark_files:
        print(f"Warning: No benchmark files found in {benchmark_path}")
        return
        
    print(f"Found {len(benchmark_files)} benchmark files")
    
    # Randomly select benchmarks
    selected_benchmarks = random.sample(benchmark_files, max(num_benchmarks, len(benchmark_files)))
    
    # Load model for predictions
    prediction_model = MachSMT.load("lib", with_db=False)
    
    # Load default strategy
    default_strat_file = os.path.join(current_dir, training_dir + "z3alpha_strat.txt")
    with open(default_strat_file, 'r') as f:
        default_strategy = f.read().strip()

    if len(default_strategy) == 0: 
        default_strategy = None
    
    # Lists to store results for both approaches
    machsmt_results = []
    default_results = []

    # Process each benchmark
    for i, benchmark_path in enumerate(selected_benchmarks, 1):
        print(f"\n{i}/{max(num_benchmarks, len(benchmark_files))} Processing {benchmark_path}: ")
        
        try:
            # Create and parse benchmark
            benchmark = Benchmark(benchmark_path)
            benchmark.parse()
            
            # Get MachSMT predictions
            predictions, scores = prediction_model.predict([benchmark], include_predictions=True)
            selected_solver = predictions[0].get_name()
            
            # Get the corresponding strategy
            machsmt_strategy = strategy_mapping.get(selected_solver, "Strategy not found")
            
            if machsmt_strategy == "Strategy not found":
                print(f"Warning: No strategy found for solver {selected_solver}")
                continue
            
            # Evaluate MachSMT strategy
            solved_machsmt, runtime_machsmt, result_machsmt = evaluate_strategy_on_benchmark(
                solver_path=z3_path,
                benchmark_path=benchmark_path,
                strategy=machsmt_strategy,
                timeout=timeout,
                tmp_dir=tmp_dir
            )
            
            # Evaluate default strategy
            solved_default, runtime_default, result_default = evaluate_strategy_on_benchmark(
                solver_path=z3_path,
                benchmark_path=benchmark_path,
                strategy=default_strategy,
                timeout=timeout,
                tmp_dir=tmp_dir
            )
            
            # Store results for metrics calculation
            machsmt_results.append((solved_machsmt, runtime_machsmt, result_machsmt))
            default_results.append((solved_default, runtime_default, result_default))
            
            # Save results to file
            with open(output_file, 'a') as f:
                f.write(f"Benchmark: {benchmark_path}\n")
                f.write("\nMachSMT Strategy:\n")
                f.write(f"Selected solver: {selected_solver}\n")
                f.write(f"Strategy: {machsmt_strategy}\n")
                f.write(f"Solved: {solved_machsmt}\n")
                f.write(f"Runtime: {runtime_machsmt:.2f}s\n")
                f.write(f"Result: {result_machsmt}\n")
                f.write("\nDefault Strategy:\n")
                f.write(f"Strategy: {default_strategy}\n")
                f.write(f"Solved: {solved_default}\n")
                f.write(f"Runtime: {runtime_default:.2f}s\n")
                f.write(f"Result: {result_default}\n")
                f.write("---\n")
            
            # Print results
            print(f"\nMachSMT Strategy Results:")
            print(f"Selected solver: {selected_solver}")
            print(f"Strategy: {machsmt_strategy}")
            print(f"Solved: {solved_machsmt}")
            print(f"Runtime: {runtime_machsmt:.2f}s")
            print(f"Result: {result_machsmt}")
            
            print(f"\nDefault Strategy Results:")
            print(f"Strategy: {default_strategy}")
            print(f"Solved: {solved_default}")
            print(f"Runtime: {runtime_default:.2f}s")
            print(f"Result: {result_default}")
            
        except Exception as e:
            print(f"Error processing benchmark {benchmark_path}: {str(e)}")
            continue
    
    # Calculate metrics for both approaches
    par2_machsmt, solving_rate_machsmt = calculate_metrics(machsmt_results, timeout)
    par2_default, solving_rate_default = calculate_metrics(default_results, timeout)
    
    print("\nOverall Performance Metrics:")
    print("\nMachSMT Strategy:")
    print(f"Average PAR2 Score: {par2_machsmt:.2f}")
    print(f"Solving Rate: {solving_rate_machsmt:.2f}%")
    
    print("\nDefault Strategy:")
    print(f"Average PAR2 Score: {par2_default:.2f}")
    print(f"Solving Rate: {solving_rate_default:.2f}%")
    
    # Save metrics to file
    with open(output_file, 'a') as f:
        f.write("\nOverall Performance Metrics:\n")
        f.write("\nMachSMT Strategy:\n")
        f.write(f"Average PAR2 Score: {par2_machsmt:.2f}\n")
        f.write(f"Solving Rate: {solving_rate_machsmt:.2f}%\n")
        f.write("\nDefault Strategy:\n")
        f.write(f"Average PAR2 Score: {par2_default:.2f}\n")
        f.write(f"Solving Rate: {solving_rate_default:.2f}%\n")
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main()
