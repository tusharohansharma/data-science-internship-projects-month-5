"""
Flight Delay Prediction - Model Training

This module prepares training and testing datasets, builds preprocessing
pipelines, and trains Logistic Regression and Random Forest classifiers.

A reproducible sample is used to manage computational requirements
associated with the large flight dataset.

Each trained model includes its preprocessing pipeline to ensure
consistent transformations during prediction.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_flight_data.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

LOGISTIC_MODEL_PATH = (
    MODEL_DIR / "logistic_regression_model.pkl"
)

RANDOM_FOREST_MODEL_PATH = (
    MODEL_DIR / "random_forest_model.pkl"
)

TARGET_COLUMN = "DELAYED"

CHUNK_SIZE = 200_000
SAMPLE_FRACTION = 0.05
MAX_TRAINING_ROWS = 300_000

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Feature Configuration

NUMERICAL_FEATURES = [
    "MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
    "SCHEDULED_DEPARTURE_HOUR",
    "SCHEDULED_DEPARTURE_MINUTE",
    "SCHEDULED_ARRIVAL_HOUR",
    "SCHEDULED_ARRIVAL_MINUTE"
]

CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT"
]

FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)

# Dataset Validation

def validate_dataset():
    # Validate the availability of the feature-engineered dataset.

    if not FEATURED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature-engineered dataset not found: "
            f"{FEATURED_DATA_PATH}"
        )

# Load Training Dataset

def load_training_data():
    # Load a reproducible sample of the feature-engineered dataset.

    validate_dataset()

    sampled_chunks = []

    for chunk in pd.read_csv(
        FEATURED_DATA_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        if TARGET_COLUMN not in chunk.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' is missing."
            )

        missing_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in chunk.columns
        ]

        if missing_features:
            raise ValueError(
                f"Required features missing: {missing_features}"
            )

        sampled_chunk = chunk.sample(
            frac=SAMPLE_FRACTION,
            random_state=RANDOM_STATE
        )

        if not sampled_chunk.empty:
            sampled_chunks.append(sampled_chunk)

    if not sampled_chunks:
        raise ValueError(
            "No records were available for model training."
        )

    sampled_df = pd.concat(
        sampled_chunks,
        ignore_index=True
    )

    if len(sampled_df) > MAX_TRAINING_ROWS:
        sampled_df = sampled_df.sample(
            n=MAX_TRAINING_ROWS,
            random_state=RANDOM_STATE
        )

    sampled_df = sampled_df.dropna(
        subset=[TARGET_COLUMN]
    )

    sampled_df[TARGET_COLUMN] = pd.to_numeric(
        sampled_df[TARGET_COLUMN],
        errors="coerce"
    )

    sampled_df = sampled_df.dropna(
        subset=[TARGET_COLUMN]
    )

    sampled_df[TARGET_COLUMN] = (
        sampled_df[TARGET_COLUMN].astype("int8")
    )

    if not set(sampled_df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            "Target variable must contain only binary values 0 and 1."
        )

    if sampled_df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Both target classes are required for classification."
        )

    print("\nTraining Dataset Loaded")
    print("-" * 50)
    print(f"Selected Records: {len(sampled_df):,}")
    print(f"Selected Features: {len(FEATURE_COLUMNS)}")

    print("\nTarget Distribution:")
    print(
        sampled_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    return sampled_df

# Prepare Training and Testing Data

def prepare_train_test_data(df):
    # Separate features and target and perform stratified splitting.

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Convert numerical features into numeric data types.
    for column in NUMERICAL_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Standardize categorical feature representations.
    for column in CATEGORICAL_FEATURES:
        X[column] = (
            X[column]
            .astype("string")
            .str.strip()
        )

        X[column] = X[column].replace(
            "",
            pd.NA
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain-Test Split Completed")
    print("-" * 50)
    print(f"Training Records: {len(X_train):,}")
    print(f"Testing Records: {len(X_test):,}")

    return X_train, X_test, y_train, y_test

# Create Categorical Encoder

def create_one_hot_encoder():
    # Create a OneHotEncoder compatible with different sklearn versions.

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )

# Build Preprocessing Pipeline

def build_preprocessor():
    # Create numerical and categorical preprocessing pipelines.

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            create_one_hot_encoder()
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor

# Build Classification Models

def build_models():
    # Create classification models with preprocessing pipelines.

    logistic_regression_pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])

    random_forest_pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

    return {
        "Logistic Regression": logistic_regression_pipeline,
        "Random Forest": random_forest_pipeline
    }

# Train Classification Models

def train_models(X_train, y_train):
    # Train and save both classification models.

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    models = build_models()
    trained_models = {}

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"TRAINING: {model_name.upper()}")
        print("=" * 60)

        model.fit(
            X_train,
            y_train
        )

        trained_models[model_name] = model

        if model_name == "Logistic Regression":
            model_path = LOGISTIC_MODEL_PATH
        else:
            model_path = RANDOM_FOREST_MODEL_PATH

        joblib.dump(
            model,
            model_path,
            compress=3
        )

        print(f"{model_name} training completed.")
        print(f"Saved Model: {model_path}")

    return trained_models

# Complete Model Training Workflow

def run_model_training():
    # Execute dataset loading, splitting, and model training.

    df = load_training_data()

    X_train, X_test, y_train, y_test = (
        prepare_train_test_data(df)
    )

    trained_models = train_models(
        X_train,
        y_train
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)

    return {
        "models": trained_models,
        "X_test": X_test,
        "y_test": y_test,
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

# Main Execution

def main():
    # Execute the model training workflow.

    run_model_training()

if __name__ == "__main__":
    main()