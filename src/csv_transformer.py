import pandas as pd
import os

def transform_csv_for_machsmt(input_file, output_file, mapping_file):
    """
    Transform CSV data from the input format to MachSMT training format and create strategy mapping.
    
    Input format:
    strat,benchmark1,benchmark2,...
    strategy1,score1,score2,...
    
    Output files:
    1. MachSMT format: benchmark,solver,score
    2. Mapping format: solver,strategy
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Get the strategy column name and benchmark columns
    strat_col = df.columns[0]  # First column is 'strat'
    benchmark_cols = df.columns[1:]  # Rest are benchmark columns
    
    # Initialize lists to store the transformed data
    benchmarks = []
    solvers = []
    scores = []
    
    # Create mapping DataFrame
    mapping_data = []
    
    # Process each row (strategy) and column (benchmark)
    for idx, row in df.iterrows():
        strategy = row[strat_col]
        solver_name = f"Z3_strat_{idx}"
        
        # Add to mapping data
        mapping_data.append({
            'solver': solver_name,
            'strategy': strategy
        })
        
        for benchmark in benchmark_cols:
            score = row[benchmark]
            # Handle par2 score: if negative, double it and make positive
            if score < 0:
                score = abs(score) * 2
                
            benchmarks.append(benchmark)
            solvers.append(solver_name)
            scores.append(score)
    
    # Create the output DataFrame
    output_df = pd.DataFrame({
        'benchmark': benchmarks,
        'solver': solvers,
        'score': scores
    })
    
    # Create the mapping DataFrame
    mapping_df = pd.DataFrame(mapping_data)
    
    # Save both files
    output_df.to_csv(output_file, index=False)
    mapping_df.to_csv(mapping_file, index=False)
    
    print(f"Transformed data saved to {output_file}")
    print(f"Strategy mapping saved to {mapping_file}")
    
    # Print some statistics
    print("\nTransformation Statistics:")
    print(f"Number of benchmarks: {len(benchmark_cols)}")
    print(f"Number of strategies/solvers: {len(df)}")
    print(f"Total data points: {len(output_df)}")

if __name__ == "__main__":
    input_file = "../training_data/hycomp/ln_res.csv"
    output_file = "../training_data/hycomp/machsmt_training.csv"
    mapping_file = "../training_data/hycomp/z3Alpha_strategy_mapping.csv"
    transform_csv_for_machsmt(input_file, output_file, mapping_file)