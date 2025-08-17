import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
OUTPUT_DIR = 'plots_and_tables'

# Define the metrics to extract and their desired column names
# Format: 'new_column_name': ('json_garment_key', 'json_metric_key')
METRICS_TO_EXTRACT = {
    'Outer Removal Success Rate': ('outer', 'succesfully_removed'),
    'Outer Removal Quality': ('outer', 'removal_quality'),
    'Inner Removal Success Rate': ('inner', 'succesfully_removed'),
    'Inner Removal Quality': ('inner', 'removal_quality'),
}

def load_and_process_data(file_path):
    """
    Loads data from the new JSON format and processes it into a pandas DataFrame.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    processed_rows = []
    for scan_id, scan_data in data.items():
        indices = scan_data.get('indices', [])
        if not indices:
            continue

        for i, index_val in enumerate(indices):
            row = {'scan': scan_id, 'index': index_val}
            
            # Process the defined metrics for both 'inner' and 'outer'
            for col_name, (garment_key, metric_key) in METRICS_TO_EXTRACT.items():
                try:
                    # Navigate to the correct list of values
                    values = scan_data.get(garment_key, {}).get(metric_key, [])
                    val = values[i]
                    
                    # Convert boolean 'true'/'false' to 1/0 for averaging
                    if isinstance(val, bool):
                        row[col_name] = int(val)
                    else:
                        row[col_name] = float(val)

                except (IndexError, TypeError, KeyError):
                    # If data is missing or list is empty, record as NaN
                    row[col_name] = np.nan
            
            processed_rows.append(row)

    df = pd.DataFrame(processed_rows)
    # Reorder columns for logical presentation
    df = df[['scan', 'index'] + list(METRICS_TO_EXTRACT.keys())]
    
    return df

def analyze_and_plot(df, group_by_col, analysis_name, generate_bar_plots=True):
    """
    Analyzes data by grouping, prints a table, optionally generates bar plots, 
    and returns the averaged DataFrame.
    """
    print(f"\n--- Averages per {analysis_name} ---\n")
    
    # Calculate mean, ignoring NaN values and non-numeric columns
    avg_df = df.groupby(group_by_col).mean(numeric_only=True)

    # Save and print the table
    table_path = os.path.join(OUTPUT_DIR, f'table_avg_per_{analysis_name}.csv')
    avg_df.round(3).to_csv(table_path)
    print(f"Table of averages per {analysis_name}:")
    print(avg_df.round(3).to_string())
    print(f"\nFull table saved to {table_path}")

    # Generate and save bar plots if requested
    if generate_bar_plots:
        print(f"Generating bar plots for averages per {analysis_name}...")
        for column in avg_df.columns:
            plt.figure(figsize=(10, 6))
            avg_df[column].plot(kind='bar')
            plt.title(f'Average {column} per {analysis_name}')
            plt.xlabel(analysis_name)
            plt.ylabel(f'Average {column}')
            plt.tight_layout()
            
            safe_col_name = "".join(c for c in column if c.isalnum() or c in (' ', '-', '_')).rstrip()
            bar_path = os.path.join(OUTPUT_DIR, f'bar_avg_{safe_col_name}_per_{analysis_name}.png')
            plt.savefig(bar_path)
            plt.close()
        print("Bar plots saved.")
    
    return avg_df


def analyze_overall_metrics(df):
    """
    Calculates and prints the overall average for each metric.
    """
    print("\n--- Overall Averages Across All Scans and Indices ---\n")
    overall_avg = df.drop(columns=['scan', 'index']).mean()
    
    table_path = os.path.join(OUTPUT_DIR, 'table_overall_averages.csv')
    overall_avg.round(4).to_csv(table_path)

    print(overall_avg.round(4).to_string())
    print(f"\nOverall averages table saved to {table_path}")
    

def generate_boxplots(df, suffix):
    """
    Generates box plots for the main metrics from the given DataFrame.
    """
    print(f"\n--- Generating Box Plots for {suffix} ---\n")
    # Automatically get metric columns from the DataFrame, excluding identifiers
    metric_cols = [col for col in df.columns if col not in ['scan', 'index']]
    
    if not metric_cols:
        print("No metric columns found, skipping box plots.")
        return

    # Use a 2x2 layout for the 4 metrics
    df[metric_cols].plot(kind='box', subplots=True, layout=(2, 2), figsize=(12, 10), sharey=False)
    
    plt.suptitle(f'Distribution of Metrics ({suffix})', fontsize=18, y=1.0)
    plt.tight_layout(pad=2.0)
    
    plot_path = os.path.join(OUTPUT_DIR, f'boxplot_metric_distributions_{suffix.replace(" ", "_").lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Metric distribution box plots saved to {plot_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <json_file>")
        sys.exit(1)

    JSON_FILE = sys.argv[1]
    # Create a more generic output directory name from the json file name
    method = os.path.splitext(os.path.basename(JSON_FILE))[0]
    OUTPUT_DIR = f"./plots_gemma/{method}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and process data using the new function
    full_df = load_and_process_data(JSON_FILE)
    full_df.round(4).to_csv(os.path.join(OUTPUT_DIR, 'full.csv'))
    print("Data loaded and processed successfully.")

    # --- Run Analyses ---
    # 1. Averages per Scan (with bar plots)
    avg_per_scan_df = analyze_and_plot(full_df, 'scan', 'Scan', generate_bar_plots=False)
    generate_boxplots(avg_per_scan_df, "Per Scan Averages")

    # 2. Averages per Index (NO bar plots)
    avg_per_index_df = analyze_and_plot(full_df, 'index', 'Index', generate_bar_plots=True)
    generate_boxplots(avg_per_index_df, "Per Index Averages")

    # 3. Overall Averages
    analyze_overall_metrics(full_df)

    print(f"\n✅ Analysis complete. All outputs are saved in the '{OUTPUT_DIR}' directory.")
