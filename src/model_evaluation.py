"""
Flight Delay Prediction - Model Evaluation

This module evaluates trained classification models using standard
classification metrics and visualization techniques.

It generates confusion matrices, ROC curves, classification reports,
and a comparative performance summary.

The best-performing model is selected using F1-score and saved
for future prediction tasks.
"""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Project Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_COMPARISON_PATH = (
    OUTPUT_DIR / "model_comparison.csv"
)

CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR / "classification_report.txt"
)

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"

# Output Directory Setup

def create_output_directories():
    # Create required output and model directories.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

# Model Performance Metrics

def calculate_metrics(model, X_test, y_test):
    # Calculate classification performance metrics.

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob
        )
    }

    return y_pred, y_prob, metrics

# Confusion Matrix Visualization

def plot_confusion_matrix(
    y_test,
    y_pred,
    model_name,
    is_best_model=False
):
    # Generate and save a confusion matrix heatmap.

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["On Time", "Delayed"],
        yticklabels=["On Time", "Delayed"],
        ax=axis
    )

    axis.set_title(
        f"{model_name} - Confusion Matrix"
    )

    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("Actual Label")

    plt.tight_layout()

    filename = (
        "confusion_matrix.png"
        if is_best_model
        else f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    )

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"Confusion Matrix Saved: {output_path}")

# ROC Curve Visualization

def plot_roc_curve(
    y_test,
    y_prob,
    model_name,
    is_best_model=False
):
    # Generate and save the ROC curve.

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc_score:.4f}"
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    axis.set_title(
        f"{model_name} - ROC Curve"
    )

    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")

    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)

    plt.tight_layout()

    filename = (
        "roc_curve.png"
        if is_best_model
        else f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    )

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"ROC Curve Saved: {output_path}")

# Classification Report

def generate_classification_report(
    y_test,
    y_pred,
    model_name
):
    # Generate a detailed classification report.

    report = classification_report(
        y_test,
        y_pred,
        target_names=["On Time", "Delayed"],
        zero_division=0
    )

    return (
        f"\n{'=' * 60}\n"
        f"{model_name.upper()}\n"
        f"{'=' * 60}\n"
        f"{report}\n"
    )

# Evaluate Individual Model

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
    is_best_model=False
):
    # Evaluate one classification model.

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name.upper()}")
    print("=" * 60)

    y_pred, y_prob, metrics = calculate_metrics(
        model,
        X_test,
        y_test
    )

    print("\nPerformance Metrics:")

    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name}: {metric_value:.4f}"
        )

    plot_confusion_matrix(
        y_test,
        y_pred,
        model_name,
        is_best_model
    )

    plot_roc_curve(
        y_test,
        y_prob,
        model_name,
        is_best_model
    )

    classification_report_text = (
        generate_classification_report(
            y_test,
            y_pred,
            model_name
        )
    )

    print("\nClassification Report:")
    print(classification_report_text)

    return {
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_prob,
        "classification_report": classification_report_text
    }

# Save Model Comparison

def save_model_comparison(comparison_df):
    # Save the comparative model performance results.

    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False
    )

    print(
        f"\nModel Comparison Saved: {MODEL_COMPARISON_PATH}"
    )

# Save Classification Reports

def save_classification_reports(report_text):
    # Save classification reports for all trained models.

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report_text)

    print(
        f"Classification Report Saved: "
        f"{CLASSIFICATION_REPORT_PATH}"
    )

# Select and Save Best Model

def select_best_model(
    models,
    evaluation_results
):
    # Select the best model based on F1-score.

    best_model_name = max(
        evaluation_results,
        key=lambda name: (
            evaluation_results[name]["metrics"]["F1-Score"],
            evaluation_results[name]["metrics"]["ROC-AUC"]
        )
    )

    best_model = models[best_model_name]

    joblib.dump(
        best_model,
        BEST_MODEL_PATH,
        compress=3
    )

    print("\n" + "=" * 60)
    print("BEST MODEL SELECTED")
    print("=" * 60)

    print(f"Selected Model: {best_model_name}")
    print(
        f"F1-Score: "
        f"{evaluation_results[best_model_name]['metrics']['F1-Score']:.4f}"
    )

    print(f"Saved Model: {BEST_MODEL_PATH}")

    return best_model_name, best_model

# Complete Model Evaluation Workflow

def evaluate_models(models, X_test, y_test):
    # Evaluate all trained models and compare their performance.

    create_output_directories()

    evaluation_results = {}
    report_sections = []

    # Evaluate each model before selecting the best one.
    for model_name, model in models.items():

        result = evaluate_model(
            model,
            X_test,
            y_test,
            model_name
        )

        evaluation_results[model_name] = result

        report_sections.append(
            result["classification_report"]
        )

    # Create comparative performance summary.
    comparison_data = []

    for model_name, result in evaluation_results.items():

        comparison_data.append({
            "Model": model_name,
            **result["metrics"]
        })

    comparison_df = pd.DataFrame(
        comparison_data
    )

    comparison_df = comparison_df.sort_values(
        by="F1-Score",
        ascending=False
    ).reset_index(drop=True)

    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))

    save_model_comparison(
        comparison_df
    )

    # Select the best model using the predefined criterion.
    best_model_name, best_model = select_best_model(
        models,
        evaluation_results
    )

    # Generate final visualizations for the selected model.
    best_result = evaluation_results[best_model_name]

    plot_confusion_matrix(
        y_test,
        best_result["predictions"],
        best_model_name,
        is_best_model=True
    )

    plot_roc_curve(
        y_test,
        best_result["probabilities"],
        best_model_name,
        is_best_model=True
    )

    # Save reports for both models.
    save_classification_reports(
        "\n".join(report_sections)
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 60)

    return {
        "comparison": comparison_df,
        "evaluation_results": evaluation_results,
        "best_model_name": best_model_name,
        "best_model": best_model
    }


def main():
    # Evaluate the saved models using the training workflow's reproducible test split.

    model_paths = {
        "Logistic Regression": LOGISTIC_MODEL_PATH,
        "Random Forest": RANDOM_FOREST_MODEL_PATH
    }
    missing_models = [
        path for path in model_paths.values()
        if not path.exists()
    ]
    if missing_models:
        missing_names = [path.name for path in missing_models]
        raise FileNotFoundError(
            f"Trained model files not found: {missing_names}. "
            "Run model_training.py before model_evaluation.py."
        )

    print("\nPreparing the evaluation dataset...")
    from model_training import load_training_data, prepare_train_test_data

    sampled_data = load_training_data()
    _, X_test, _, y_test = prepare_train_test_data(sampled_data)
    del sampled_data

    print("\nLoading trained models...")
    models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in model_paths.items()
    }
    evaluate_models(models, X_test, y_test)


if __name__ == "__main__":
    main()