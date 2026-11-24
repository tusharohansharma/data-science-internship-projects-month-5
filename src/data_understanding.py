"""
Flight Delay Prediction - Data Understanding

This module performs an initial assessment of the
airline delay datasets.It examines dataset dimensions,
column information, missing values, duplicate records and
descriptive statistics.

The analysis uses chunk-based processing
for large datasets to reduce memory consumption.
"""

from pathlib import Path
import pandas as pd

# Project Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS_FILES = {
    "Flights": RAW_DATA_DIR / "flights.csv",
    "Airlines": RAW_DATA_DIR / "airlines.csv",
    "Airports": RAW_DATA_DIR / "airports.csv"
}

CHUNK_SIZE = 200_000
SAMPLE_SIZE = 10_000

# Dataset Validation

def validate_datasets():
    # Check whether all required dataset files exist.

    missing_files = [
        file_path
        for file_path in DATASETS_FILES.values()
        if not file_path.exists()
    ]

    if missing_files:
        missing_names = [
            file_path.name for file_path in missing_files
        ]

        raise FileNotFoundError(f"Required dataset files not found: {missing_names}")

# Dataset Inspection

def inspect_dataset(dataset_name, file_path):
    # Inspect dataset structure and basic information.

    print("\n" + "=" * 65)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 65)

    total_rows = 0
    missing_counts = None
    sample_df = None

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        total_rows += len(chunk)

        chunk_missing = chunk.isna().sum()

        if missing_counts is None:
            missing_counts = chunk_missing
        else:
            missing_counts = missing_counts.add(
                chunk_missing,
                fill_value=0
            )

        if sample_df is None:
            sample_df = chunk.head(SAMPLE_SIZE).copy()

    if sample_df is None:
        raise ValueError(f"{dataset_name} dataset is empty.")

    total_columns = len(sample_df.columns)

    print(f"\nDataset Dimensions: {total_rows} rows x {total_columns} columns")

    print("\nColumn names:")
    print(sample_df.columns.tolist())

    print("\nData Types:")
    print(sample_df.dtypes)

    print("\nMissing Values:")
    missing_summary = pd.DataFrame({
        "missing_values": missing_counts.astype(int),
        "missing_percentage": (missing_counts / total_rows * 100).round(2)
    })

    print(missing_summary.sort_values(
        by="missing_percentage",
        ascending=False
    ))

    print("\nSample Records:")

    print(sample_df.head(5).to_string(index=False))

    print("\nDuplicate Records in Sample:")
    print(sample_df.duplicated().sum())

    print("\nNumerical Summary:")
    numerical_data = sample_df.select_dtypes(include="number")
    if numerical_data.empty:
        print("No numerical columns available.")
    else:
        print(numerical_data.describe().round(2))

    return {
        "dataset": dataset_name,
        "rows": total_rows,
        "columns": total_columns,
        "missing_values": missing_summary
    }

# Complete Dataset Analysis

def analyze_datasets():
    # Perform data understanding across all datasets.

    validate_datasets()

    analysis_results = []

    for dataset_name, file_path in DATASETS_FILES.items():
        result = inspect_dataset(
            dataset_name,
            file_path
        )

        analysis_results.append(result)

    print("\n" + "=" * 65)
    print("DATA UNDERSTANDING COMPLETED")
    print("=" * 65)

    return analysis_results

# Main Execution

def main():
    # Execute the dataset understanding workflow.

    analyze_datasets()

if __name__ == "__main__":
    main()