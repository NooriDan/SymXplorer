import numpy as np
import pandas as pd
import argparse

def unwrap_phase(input_file, output_file, phase_column):
    """
    Reads a CSV file, unwraps the phase in the specified column, and writes to a new CSV.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        phase_column (str): Name of the column containing the phase data to unwrap.
    """
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    
    # Check if the specified phase column exists
    if phase_column not in df.columns:
        print(f"Error: Column '{phase_column}' not found in the CSV file.")
        return
    
    # Unwrap the phase
    try:
        df[f"{phase_column}_unwrapped"] = np.unwrap(df[phase_column].to_numpy())
    except Exception as e:
        print(f"Error unwrapping phase: {e}")
        return

    # Save the updated DataFrame to a new CSV
    df.to_csv(output_file, index=False)
    print(f"Phase unwrapped and saved to '{output_file}'.")

if __name__ == "__main__":
    # Argument parser for command-line use
    # parser = argparse.ArgumentParser(description="Unwrap the phase column in a CSV file.")
    # parser.add_argument("input_file", help="Path to the input CSV file.")
    # parser.add_argument("output_file", help="Path to the output CSV file.")
    # parser.add_argument("phase_column", help="Name of the column containing the phase data.")
    
    # args = parser.parse_args()
    # unwrap_phase(args.input_file, args.output_file, args.phase_column)

    unwrap_phase("AC_sim.csv", "unwrapped_data.csv", "phase")