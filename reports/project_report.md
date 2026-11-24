# Project Report

## Project Title

Flight Delay Prediction Using Machine Learning

## Objective

The objective of this project is to develop a binary classification system that predicts whether a flight will arrive more than 15 minutes late using information intended to be available before departure.

## Dataset Information

The project uses three files from the Airline Delay dataset:

- `flights.csv` — Flight-level operational records.
- `airlines.csv` — Airline code and name reference.
- `airports.csv` — Airport code and location reference.

The dataset preview indicated approximately 5.8 million flight records. The exact dimensions should be confirmed using the data-understanding module.

## Data Understanding

The data-understanding module examines:

- Dataset dimensions.
- Column names.
- Data types.
- Missing-value counts.
- Sample records.
- Duplicate records in the inspected sample.
- Numerical summaries.

Chunk-based reading is used to accommodate the large flight dataset.

## Data Preprocessing

The preprocessing module performs the following operations:

- Reads flight records in chunks.
- Calculates missing-value percentages.
- Removes selected columns exceeding the configured 40% missing-value threshold.
- Removes duplicate records within each chunk.
- Standardizes categorical identifiers.
- Reduces numeric memory usage where appropriate.

The processed dataset is saved under `data/processed/`.

**Limitation:** Duplicate removal is performed within individual chunks and does not guarantee global duplicate removal across chunk boundaries.

## Feature Engineering

The feature-engineering module performs the following operations:

- Excludes cancelled and diverted flights.
- Removes records without usable arrival-delay values.
- Creates the binary target variable `DELAYED`.
- Assigns `1` when arrival delay exceeds 15 minutes.
- Assigns `0` when arrival delay is 15 minutes or less.
- Extracts weekday information.
- Extracts scheduled departure and arrival hours and minutes.
- Removes post-flight information to reduce target leakage.

The feature-engineered dataset is saved under `data/processed/`.

## Model Development

Two machine learning classification algorithms are used:

1. Logistic Regression
2. Random Forest

Numerical features are median-imputed and standardized.

Categorical features are imputed using the most frequent value and one-hot encoded.

The preprocessing operations are integrated into Scikit-learn pipelines to maintain consistent transformations during training and prediction.

A reproducible sample capped at 300,000 records is used for model development.

The dataset is divided using an 80/20 stratified train-test split.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Classification Report

The implementation selects the model with the highest F1-score, using ROC-AUC as a tie-breaker.

## Results

Evaluation used the stratified 20% holdout from the reproducible sample: 57,140 flights (46,936 on time and 10,204 delayed). The positive class is arrival delay greater than 15 minutes.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.5936 | **0.6158** |
| Precision (delayed) | 0.2445 | **0.2551** |
| Recall (delayed) | **0.6105** | 0.6000 |
| F1-score (delayed) | 0.3492 | **0.3580** |
| ROC-AUC | 0.6382 | **0.6544** |

Random Forest is selected by the configured highest-F1 rule (0.3580 vs. 0.3492). It also has higher accuracy, precision, and ROC-AUC. Logistic Regression has slightly higher recall, identifying 108 more delayed flights in this holdout. Both models have low precision, so many flights predicted as delayed are actually on time.

Confusion-matrix counts (actual rows, predicted columns; class order On Time, Delayed):

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 27,686 | 19,250 | 3,974 | 6,230 |
| Random Forest | 29,062 | 17,874 | 4,082 | 6,122 |

Both ROC-AUC values are modestly above the 0.50 random-ranking baseline (0.6382 for Logistic Regression and 0.6544 for Random Forest). The selected model is not necessarily the best choice for every application: the slightly higher Logistic Regression recall may matter more when missed delays are especially costly, while both models produce many false alerts.

Evaluation artifacts: [model comparison](../outputs/model_comparison.csv), [classification report](../outputs/classification_report.txt), [Logistic Regression confusion matrix](../outputs/confusion_matrix_logistic_regression.png), [Random Forest confusion matrix](../outputs/confusion_matrix_random_forest.png), [Logistic Regression ROC curve](../outputs/roc_curve_logistic_regression.png), [Random Forest ROC curve](../outputs/roc_curve_random_forest.png).

## Limitations

- Model development uses a sample capped at 300,000 records rather than the complete dataset.
- Duplicate removal is performed within chunks rather than globally.
- Remaining missing feature values are handled through the model pipelines.
- Weather forecasts and real-time weather observations are not included.
- The current train-test split is stratified but not chronological.
- Model performance may vary across airlines, airports, routes, and time periods.

## Conclusion

This project establishes a modular workflow for flight-delay classification. On the sampled stratified holdout set, Random Forest is the selected model, with an F1-score of 0.3580 and ROC-AUC of 0.6544. Logistic Regression has slightly higher recall for delays (0.6105 vs. 0.6000). Both models miss a substantial share of actual delays and produce many false positives. Because this evaluation uses a random rather than chronological split, performance should be validated on future periods and across airlines, airports, and routes before operational use.