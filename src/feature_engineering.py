"""
Flight Delay Prediction - Feature Engineering

This module transforms the preprocessed flight dataset into a
machine-learning-ready dataset.

It creates a binary flight-delay target, extracts date and scheduled
time features, removes unsuitable flight records, and eliminates
post-flight information to prevent data leakage.
"""

from pathlib import Path
import pandas as pd

# Project Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_flight_data.csv"
)

FEATURED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_flight_data.csv"
)

CHUNK_SIZE = 200_000

DELAY_THRESHOLD = 15


# Data Leakage Configuration

LEAKAGE_COLUMNS = [
    "ARRIVAL_DELAY",
    "DEPARTURE_DELAY",
    "ARRIVAL_TIME",
    "DEPARTURE_TIME",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_OUT",
    "TAXI_IN",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DIVERTED",
    "CANCELLED",
    "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "TAIL_NUMBER",
    "FLIGHT_NUMBER"
]


# Dataset Validation

def validate_dataset():
    # Validate the availability of the preprocessed dataset.

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA_PATH}"
        )
    
# Flight Date Preparation

def extract_date_features(df):
    # Create date-based features from year, month, and day.

    required_columns = [
        "YEAR",
        "MONTH",
        "DAY"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing date columns: {missing_columns}"
        )

    flight_date = pd.to_datetime(
        {
            "year": df["YEAR"],
            "month": df["MONTH"],
            "day": df["DAY"]
        },
        errors="coerce"
    )

    df["DAY_OF_WEEK"] = flight_date.dt.dayofweek + 1
    df["MONTH"] = flight_date.dt.month

    return df


# Scheduled Time Feature Extraction

def extract_scheduled_time_features(df):
    # Extract hour and minute from scheduled flight times.

    scheduled_time_columns = [
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_ARRIVAL"
    ]

    for column in scheduled_time_columns:
        if column not in df.columns:
            continue

        scheduled_time = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Convert HHMM format into separate hour and minute features.
        df[f"{column}_HOUR"] = (
            scheduled_time // 100
        )

        df[f"{column}_MINUTE"] = (
            scheduled_time % 100
        )

        invalid_time = (
            (scheduled_time < 0)
            | (scheduled_time > 2359)
            | (scheduled_time % 100 >= 60)
        )

        df.loc[
            invalid_time,
            [
                f"{column}_HOUR",
                f"{column}_MINUTE"
            ]
        ] = pd.NA

    return df


# Target Variable Creation

def create_target_variable(df):
    # Create a binary target for flight delay prediction.

    required_columns = [
        "ARRIVAL_DELAY",
        "CANCELLED",
        "DIVERTED"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required target columns missing: {missing_columns}"
        )

    df["ARRIVAL_DELAY"] = pd.to_numeric(
        df["ARRIVAL_DELAY"],
        errors="coerce"
    )

    df["CANCELLED"] = pd.to_numeric(
        df["CANCELLED"],
        errors="coerce"
    )

    df["DIVERTED"] = pd.to_numeric(
        df["DIVERTED"],
        errors="coerce"
    )

    # Exclude cancelled and diverted flights.
    df = df[
        (df["CANCELLED"] == 0)
        & (df["DIVERTED"] == 0)
    ].copy()

    # Remove records without a valid arrival delay.
    df = df.dropna(
        subset=["ARRIVAL_DELAY"]
    )

    # Create binary classification target.
    df["DELAYED"] = (
        df["ARRIVAL_DELAY"] > DELAY_THRESHOLD
    ).astype("int8")

    return df

# Remove Leakage and Unnecessary Features

def remove_leakage_features(df):
    # Remove target-related and post-flight information.

    df = df.drop(
        columns=LEAKAGE_COLUMNS,
        errors="ignore"
    )

    # Remove original date components after feature extraction.
    df = df.drop(
        columns=["YEAR", "DAY"],
        errors="ignore"
    )

    return df


# Feature Preparation

def prepare_features(df):
    # Apply target creation and feature engineering.

    df = create_target_variable(df)

    df = extract_date_features(df)

    df = extract_scheduled_time_features(df)

    df = remove_leakage_features(df)

    # Remove records with invalid dates.
    df = df.dropna(
        subset=["DAY_OF_WEEK", "MONTH"]
    )

    # Reset the index after filtering.
    df = df.reset_index(drop=True)

    return df


# Process Dataset Using Chunks

def engineer_features():
    # Generate model-ready features using chunk-based processing.

    validate_dataset()

    FEATURED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if FEATURED_DATA_PATH.exists():
        FEATURED_DATA_PATH.unlink()

    total_processed_rows = 0
    total_delayed = 0
    total_on_time = 0
    first_chunk = True

    for chunk in pd.read_csv(
        PROCESSED_DATA_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        featured_chunk = prepare_features(chunk)

        if featured_chunk.empty:
            continue

        featured_chunk.to_csv(
            FEATURED_DATA_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        total_processed_rows += len(featured_chunk)

        total_delayed += int(
            featured_chunk["DELAYED"].sum()
        )

        total_on_time += int(
            (featured_chunk["DELAYED"] == 0).sum()
        )

        first_chunk = False

        print(
            f"Feature-engineered records: "
            f"{total_processed_rows:,}"
        )

    if first_chunk:
        raise ValueError(
            "No valid flight records remained after feature engineering."
        )

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 65)

    print(f"Total Records: {total_processed_rows:,}")
    print(f"Delayed Flights: {total_delayed:,}")
    print(f"On-Time Flights: {total_on_time:,}")

    print("\nTarget Distribution:")

    target_distribution = pd.DataFrame({
        "Flight Status": ["On Time", "Delayed"],
        "Count": [total_on_time, total_delayed]
    })

    target_distribution["Percentage"] = (
        target_distribution["Count"]
        / total_processed_rows
        * 100
    ).round(2)

    print(target_distribution.to_string(index=False))

    print(f"\nSaved Dataset: {FEATURED_DATA_PATH}")

    return FEATURED_DATA_PATH

# Main Execution

def main():
    # Execute the feature engineering workflow.

    engineer_features()

if __name__ == "__main__":
    main()