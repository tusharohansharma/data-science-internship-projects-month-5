"""
Flight Delay Prediction - End-to-End Pipeline

This module orchestrates the complete flight delay prediction workflow.

It integrates data preprocessing, feature engineering, model training,
and model evaluation into a reusable execution pipeline.

The pipeline processes the Airline Delay Dataset and trains two
classification algorithms: Logistic Regression and Random Forest.

The best-performing model is selected using F1-score and saved
for future prediction tasks.

Main Workflow:
    1. Preprocess the raw flight dataset.
    2. Generate model-ready features.
    3. Train classification models.
    4. Evaluate model performance.
    5. Compare model results.
    6. Select and save the best model.
"""

from pathlib import Path
import sys

# Project Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import Project Modules

from data_preprocessing import preprocess_flight_data
from feature_engineering import engineer_features
from model_training import run_model_training
from model_evaluation import evaluate_models

# Data Preprocessing Stage

def run_data_preprocessing():
    # Execute raw flight dataset preprocessing.

    print("\n" + "=" * 65)
    print("STAGE 1: DATA PREPROCESSING")
    print("=" * 65)

    processed_path = preprocess_flight_data()

    return processed_path

# Feature Engineering Stage

def run_feature_engineering():
    # Generate model-ready flight features.

    print("\n" + "=" * 65)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 65)

    featured_path = engineer_features()

    return featured_path

# Model Training Stage

def run_training_stage():
    # Train Logistic Regression and Random Forest models.

    print("\n" + "=" * 65)
    print("STAGE 3: MODEL TRAINING")
    print("=" * 65)

    training_results = run_model_training()

    return training_results

# Model Evaluation Stage

def run_evaluation_stage(training_results):
    # Evaluate trained models and select the best classifier.

    print("\n" + "=" * 65)
    print("STAGE 4: MODEL EVALUATION")
    print("=" * 65)

    evaluation_results = evaluate_models(
        models=training_results["models"],
        X_test=training_results["X_test"],
        y_test=training_results["y_test"]
    )

    return evaluation_results

# Complete Project Pipeline

def run_pipeline():
    # Execute the complete flight delay prediction workflow.

    print("\n" + "=" * 65)
    print("FLIGHT DELAY PREDICTION PIPELINE")
    print("=" * 65)

    try:
        # Stage 1: Preprocess raw flight data.
        processed_path = run_data_preprocessing()

        # Stage 2: Generate model-ready features.
        featured_path = run_feature_engineering()

        # Stage 3: Train classification models.
        training_results = run_training_stage()

        # Stage 4: Evaluate and select the best model.
        evaluation_results = run_evaluation_stage(
            training_results
        )

        print("\n" + "=" * 65)
        print("FLIGHT DELAY PREDICTION PIPELINE COMPLETED")
        print("=" * 65)

        print(f"Processed Dataset: {processed_path}")
        print(f"Featured Dataset: {featured_path}")

        print(
            f"Training Records: "
            f"{training_results['training_records']:,}"
        )

        print(
            f"Testing Records: "
            f"{training_results['testing_records']:,}"
        )

        print(
            f"Best Model: "
            f"{evaluation_results['best_model_name']}"
        )

        print("\nFinal Model Comparison:")
        print(
            evaluation_results["comparison"].to_string(
                index=False
            )
        )

        return {
            "training_results": training_results,
            "evaluation_results": evaluation_results
        }

    except FileNotFoundError as error:
        print(f"\nFile Error: {error}")
        raise

    except ValueError as error:
        print(f"\nData Validation Error: {error}")
        raise

    except Exception as error:
        print(f"\nPipeline Execution Error: {error}")
        raise

# Main Execution

def main():
    # Execute the end-to-end flight delay prediction pipeline.

    run_pipeline()

if __name__ == "__main__":
    main()