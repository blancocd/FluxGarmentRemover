import pandas as pd
import argparse

def analyze_scan_data(filepath, n=1):
    """
    Analyzes scan data from a CSV file to find the scans with the
    lowest success and quality rates for inner and outer removals.

    Args:
        filepath (str): The path to the CSV file.
        n (int): The number of top 'n' scans with the lowest scores to retrieve.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # --- Pre-process the data ---
    # Separate the DataFrame into 'Inner' and 'Outer' scans based on the 'scan' column
    outer_scans_df = df[df['scan'].str.contains('Outer', case=False, na=False)].copy()
    inner_scans_df = df[df['scan'].str.contains('Inner', case=False, na=False)].copy()

    # --- Analyze Outer Scans ---
    print("--- Analysis for Outer Scans ---")
    
    # Find the n lowest for 'Outer Removal Success Rate'
    lowest_outer_success = outer_scans_df.nsmallest(n, 'Outer Removal Success Rate')
    print(f"\n# Top {n} Scans with LEAST 'Outer Removal Success Rate':")
    print(lowest_outer_success[['scan', 'Outer Removal Success Rate']])

    # Find the n lowest for 'Outer Removal Quality'
    lowest_outer_quality = outer_scans_df.nsmallest(n, 'Outer Removal Quality')
    print(f"\n# Top {n} Scans with LEAST 'Outer Removal Quality':")
    print(lowest_outer_quality[['scan', 'Outer Removal Quality']])

    print("\n" + "="*40 + "\n") # Separator

    # --- Analyze Inner Scans ---
    print("--- Analysis for Inner Scans ---")

    # Find the n lowest for 'Inner Removal Success Rate'
    lowest_inner_success = inner_scans_df.nsmallest(n, 'Inner Removal Success Rate')
    print(f"\n# Top {n} Scans with LEAST 'Inner Removal Success Rate':")
    print(lowest_inner_success[['scan', 'Inner Removal Success Rate']])

    # Find the n lowest for 'Inner Removal Quality'
    lowest_inner_quality = inner_scans_df.nsmallest(n, 'Inner Removal Quality')
    print(f"\n# Top {n} Scans with LEAST 'Inner Removal Quality':")
    print(lowest_inner_quality[['scan', 'Inner Removal Quality']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze scan data for lowest performing scans.")
    parser.add_argument("filepath", type=str, help="Path to the CSV file to analyze.")
    parser.add_argument("-n", type=int, default=3, help="Number of lowest-scoring scans to show (default: 3).")
    args = parser.parse_args()

    analyze_scan_data(args.filepath, n=args.n)