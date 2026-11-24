# Flight Delay Prediction Using Machine Learning

A machine learning project that predicts whether a flight is likely to arrive more than 15 minutes late using historical airline flight information.

## Project Objective

The objective of this project is to develop a binary classification system that predicts flight delays using historical flight information, including airline, route, distance, and scheduled departure details.

The target variable is defined as follows:

- `DELAYED = 1`: Arrival delay is greater than 15 minutes.
- `DELAYED = 0`: Arrival delay is 15 minutes or less.

Cancelled and diverted flights are excluded from the classification dataset.

## Dataset

This project uses the **2015 Flight Delays and Cancellations** dataset, published on Kaggle by the U.S. Department of Transportation (`usdot`).

- **Dataset:** [2015 Flight Delays and Cancellations – Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Publisher/Uploader:** U.S. Department of Transportation (`usdot`)
- **Original Data Source:** U.S. Department of Transportation (DOT), Bureau of Transportation Statistics (BTS).
- **License:** CC0: Public Domain.

### Dataset Files

| File | Description |
|---|---|
| `flights.csv` | Contains flight schedules, airline information, departure and arrival details, and delay information. |
| `airlines.csv` | Contains airline codes and corresponding airline names. |
| `airports.csv` | Contains airport codes and corresponding airport information. |

## Dataset Setup

The original dataset is not included in this GitHub repository because of its large file size.

To obtain the dataset:

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/usdot/flight-delays).
2. Download the 2015 Flight Delays and Cancellations dataset.
3. Extract the downloaded archive.
4. Place the following CSV files inside the `data/raw/` directory:

```text
data/
└── raw/
    ├── flights.csv
    ├── airlines.csv
    └── airports.csv
```

**Note:** Raw datasets, processed datasets, and trained machine learning models are excluded from this repository using `.gitignore`. Download the original dataset and run the project pipeline to generate the required processed files and models.

## Project Structure

```text
Data-Science-Internship-Projects-Month-5/
│
├── data/
│   ├── raw/
│   │   ├── flights.csv
│   │   ├── airlines.csv
│   │   └── airports.csv
│   │
│   └── processed/
│       ├── processed_flights.csv
│       └── featured_flights.csv
│
├── models/
│   ├── logistic_regression_pipeline.pkl
│   └── random_forest_pipeline.pkl
│
├── notebooks/
│   └── flight_delay_prediction.ipynb
│
├── outputs/
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── logistic_regression_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── roc_curve_comparison.png
│
├── reports/
│   ├── project_report.md
│   └── model_evaluation_report.md
│
├── src/
│   ├── data_understanding.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── flight_delay_pipeline.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The filenames under `data/processed/`, `models/`, and `outputs/` represent the intended generated artifacts. Ensure they match the actual filenames configured in the Python modules.*

## Technologies Used

- **Python** — Main programming language.
- **Pandas** — Data manipulation and preprocessing.
- **NumPy** — Numerical operations.
- **Scikit-learn** — Machine learning algorithms and evaluation.
- **Matplotlib** — Data visualization.
- **Seaborn** — Statistical visualization.
- **Joblib** — Model serialization.
- **Jupyter Notebook** — Interactive development and analysis.

## Installation and Setup

Use the Python interpreter selected in Visual Studio Code. A separate virtual environment is not required for this project.

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the Python installation:

```bash
python --version
```

Ensure that Visual Studio Code is using the same Python interpreter in which the dependencies were installed.

## Running the Project

Run the complete workflow from the project root directory:

```bash
python src/flight_delay_pipeline.py
```

The pipeline coordinates the following stages:

1. Data understanding.
2. Data preprocessing.
3. Feature engineering.
4. Model training.
5. Model evaluation.

Individual data preparation modules can also be executed separately:

```bash
python src/data_understanding.py
python src/data_preprocessing.py
python src/feature_engineering.py
```

## Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the following notebook:

```text
notebooks/flight_delay_prediction.ipynb
```

Run the notebook cells in sequence to inspect the dataset, execute the preparation workflow, train the models, and review evaluation results.

## Project Methodology

### 1. Data Understanding

The data-understanding stage examines the dataset's structure and quality.

Activities include:

- Inspecting dataset dimensions.
- Identifying column names and data types.
- Analyzing missing values.
- Reviewing sample records.
- Examining numerical summaries.
- Inspecting duplicate records.

Chunk-based reading is used to accommodate the large flight dataset.

### 2. Data Preprocessing

The preprocessing stage prepares the raw flight records for further analysis.

Operations include:

- Reading flight records in chunks.
- Calculating missing-value percentages.
- Removing selected columns exceeding the configured 40% missing-value threshold.
- Removing duplicate records within individual chunks.
- Standardizing categorical identifiers.
- Reducing numeric memory usage where appropriate.

The processed dataset is saved in `data/processed/`.

### 3. Feature Engineering

The feature-engineering stage creates the features required for classification.

Operations include:

- Excluding cancelled and diverted flights.
- Removing records without usable arrival-delay values.
- Creating the binary target variable `DELAYED`.
- Extracting scheduled departure and arrival hours and minutes.
- Extracting weekday information.
- Removing post-flight information to reduce target leakage.

The feature-engineered dataset is saved in `data/processed/`.

### 4. Model Training

Two machine learning classification algorithms are used:

- Logistic Regression
- Random Forest

The training workflow applies:

- Median imputation for numerical features.
- Numerical feature standardization.
- Most-frequent-value imputation for categorical features.
- One-hot encoding for categorical features.

Preprocessing and model training are integrated into Scikit-learn pipelines to maintain consistent transformations.

A reproducible sample capped at 300,000 records is used for model development.

The dataset is divided using an 80/20 stratified train-test split.

### 5. Model Evaluation

The trained models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional evaluation outputs include:

- Confusion matrices.
- ROC curves.
- Classification reports.
- Model comparison results.

The implementation selects the model with the highest F1-score, using ROC-AUC as a tie-breaker.

## Generated Outputs

The project generates the following artifacts:

### Processed Datasets

| File | Description |
|---|---|
| `processed_flights.csv` | Cleaned flight dataset |
| `featured_flights.csv` | Feature-engineered dataset containing the target variable |

### Trained Models

| File | Description |
|---|---|
| `logistic_regression_pipeline.pkl` | Serialized Logistic Regression pipeline |
| `random_forest_pipeline.pkl` | Serialized Random Forest pipeline |

### Evaluation Outputs

| File | Description |
|---|---|
| `model_comparison.csv` | Comparative evaluation metrics |
| `classification_report.txt` | Classification performance details |
| `logistic_regression_confusion_matrix.png` | Logistic Regression confusion matrix |
| `random_forest_confusion_matrix.png` | Random Forest confusion matrix |
| `roc_curve_comparison.png` | ROC curve comparison of both models |

Generated datasets and serialized models are excluded from Git and can be recreated by running the project pipeline.

## Model Evaluation Results

The results below were produced by `src/model_evaluation.py` on a stratified holdout set of 57,140 flights (46,936 on-time and 10,204 delayed). Precision, recall, and F1-score are for the delayed class (`DELAYED = 1`).

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.5936 | **0.6158** |
| Precision | 0.2445 | **0.2551** |
| Recall | **0.6105** | 0.6000 |
| F1-score | 0.3492 | **0.3580** |
| ROC-AUC | 0.6382 | **0.6544** |

**Selected Model: Random Forest.** It has the higher F1-score, so it is selected by the evaluation code's F1-first rule (ROC-AUC is only a tie-breaker). It also has higher accuracy, precision, and ROC-AUC. Logistic Regression has slightly higher recall, identifying 108 more delayed flights in this test set.

Confusion-matrix counts (actual rows, predicted columns; class order On Time, Delayed):

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 27,686 | 19,250 | 3,974 | 6,230 |
| Random Forest | 29,062 | 17,874 | 4,082 | 6,122 |

Both models identify about 60% of delayed flights, but delayed-class precision is low: only about 24–26% of flights predicted as delayed were delayed. The ROC-AUC values are above the 0.50 random-ranking baseline but indicate modest discrimination. These metrics describe this sampled, random holdout set and should not be treated as estimates of future operational performance.

Evaluation figures: [Logistic Regression confusion matrix](outputs/confusion_matrix_logistic_regression.png), [Random Forest confusion matrix](outputs/confusion_matrix_random_forest.png), [Logistic Regression ROC curve](outputs/roc_curve_logistic_regression.png), and [Random Forest ROC curve](outputs/roc_curve_random_forest.png).

## Project Reports

Detailed documentation is available in the `reports/` directory:

- `project_report.md` — Overall project methodology, implementation, and conclusions.
- `model_evaluation_report.md` — Model comparison, evaluation metrics, and performance analysis.

## Limitations

- Model development uses a sample capped at 300,000 records rather than the complete dataset.
- Duplicate removal is performed within individual chunks and does not guarantee global duplicate removal.
- Remaining missing feature values are handled through the model preprocessing pipelines.
- Weather forecasts and real-time weather observations are not included.
- The current train-test split is stratified but not chronological.
- Model performance may vary across airlines, airports, routes, and time periods.

## Future Improvements

Potential improvements include:

- Introducing a chronological train-test split.
- Implementing global duplicate handling where required.
- Incorporating reliable pre-departure weather forecast information.
- Optimizing classification thresholds according to operational requirements.
- Evaluating model performance across different airlines, airports, and routes.
- Exploring additional machine learning algorithms.
- Improving scalability for large-scale flight datasets.

## Conclusion

This project establishes a modular machine learning workflow for flight delay prediction, covering dataset understanding, preprocessing, feature engineering, model training, and evaluation.

On the evaluated holdout set, Random Forest was selected with an F1-score of 0.3580 and ROC-AUC of 0.6544. Logistic Regression had slightly higher recall for delayed flights (0.6105 versus 0.6000). Both models miss a substantial share of actual delays and generate many false alerts, so their results are a baseline rather than evidence of deployment readiness. Further validation should use a chronological split and consider the operational costs of missed delays versus false positives.