"""
Flight Delay Prediction - Data Preprocessing

This module prepares the raw flight dataset for machine learning.
It identifies columns with excessive missing values, removes unsuitable
columns, performs basic data cleaning, and optimizes numerical data types.

Chunk-based processing is used to handle the large flight dataset
without loading the entire file into memory.
"""

from pathlib import Path
import pandas as pd

# Project Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flights.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = (PROCESSED_DATA_DIR / "processed_flight_data.csv")

CHUNK_SIZE = 200_000
MISSING_THRESHOLD = 0.40

# Required Column Configuration

PROTECTED_COLUMNS = {
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "DISTANCE",
    "ARRIVAL_DELAY",
    "CANCELLED",
    "DIVERTED"
}

DUPLICATE_KEY_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE"
]

# Dataset Validation

def validate_dataset():
    # Validate the availability of the raw flight dataset.

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Flight dataset not found: {RAW_DATA_PATH}"
        )
    
# Identify Columns with Excessive Missing Values

def identify_missing_columns():
    # Calculate missing-value percentages across the dataset.

    total_rows = 0
    missing_counts = None

    for chunk in pd.read_csv(
        RAW_DATA_PATH,
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

    if total_rows == 0:
        raise ValueError("The flight dataset is empty.")

    missing_percentage = (missing_counts / total_rows)

    columns_to_drop = [
        column
        for column, percentage in missing_percentage.items()
        if percentage > MISSING_THRESHOLD
        and column not in PROTECTED_COLUMNS
    ]

    print("\nColumns Removed Due to Excessive Missing Values:")
    print(columns_to_drop)

    return columns_to_drop

# Data Type Optimization

def optimize_data_types(df):
    # Reduce numerical memory consumption.

    integer_columns = df.select_dtypes(
        include=["int64"]
    ).columns

    float_columns = df.select_dtypes(
        include=["float64"]
    ).columns

    for column in integer_columns:
        df[column] = pd.to_numeric(
            df[column],
            downcast="integer"
        )

    for column in float_columns:
        df[column] = pd.to_numeric(
            df[column],
            downcast="float"
        )

    return df

# Clean Individual Data Chunks

def clean_chunk(df, columns_to_drop):
    # Clean and optimize an individual dataset chunk.

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # Remove duplicate flight records within the current chunk.
    duplicate_columns = [
        column
        for column in DUPLICATE_KEY_COLUMNS
        if column in df.columns
    ]

    if duplicate_columns:
        df = df.drop_duplicates(
            subset=duplicate_columns
        )

    # Standardize categorical identifiers.
    categorical_columns = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT"
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    df = optimize_data_types(df)

    return df

# Process and Save Flight Dataset

def preprocess_flight_data():
    # Preprocess the complete flight dataset using chunks.

    validate_dataset()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columns_to_drop = identify_missing_columns()

    if PROCESSED_DATA_PATH.exists():
        PROCESSED_DATA_PATH.unlink()

    total_processed_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(
        RAW_DATA_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        processed_chunk = clean_chunk(
            chunk,
            columns_to_drop
        )

        processed_chunk.to_csv(
            PROCESSED_DATA_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        total_processed_rows += len(processed_chunk)
        first_chunk = False

        print(
            f"Processed records: {total_processed_rows:,}"
        )

    if first_chunk:
        raise ValueError(
            "No records were available for preprocessing."
        )

    print("\n" + "=" * 65)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 65)

    print(f"Processed Records: {total_processed_rows:,}")
    print(f"Saved Dataset: {PROCESSED_DATA_PATH}")

    return PROCESSED_DATA_PATH

# Main Execution

def main():
    """Execute the data preprocessing workflow."""

    preprocess_flight_data()

if __name__ == "__main__":
    main()