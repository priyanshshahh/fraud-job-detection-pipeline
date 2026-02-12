# Fraud Job Detection Pipeline

This project implements an end-to-end machine learning pipeline for detecting fraudulent job postings. It includes data preprocessing, feature engineering, model training, evaluation, and an interactive dashboard for prediction.

## Overview

The pipeline consists of the following modules:
1.  **Preprocessing (`preprocess.py`)**: Cleans and tokenizes text data.
2.  **Feature Engineering (`featureengineering.py`)**: Extracts TF-IDF, sentiment, and metadata features.
3.  **Model Training (`modeltraining.py`)**: Trains a Multinomial Naïve Bayes classifier.
4.  **Model Evaluation (`modelevaluation.py`)**: Generates detailed metrics and plots.
5.  **Orchestrator (`detectionpipelines.py`)**: Runs the entire pipeline sequentially.
6.  **Dashboard (`dashboard.py`)**: A Dash web application for interactive use.

## Setup

1.  **Clone the repository.**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Note: Ensure you have Python 3.8+ installed.

3.  **Data**:
    The pipeline expects a CSV file named `fake_job_postings.csv` in the root directory.
    -   If you have the dataset (e.g., from Kaggle), place it in the root folder.
    -   For testing purposes, a script `create_dummy_data.py` is provided (or you can create one manually) to generate a sample file.

    If `fake_job_postings.csv` is missing, you can create a dummy one by running:
    ```python
    python create_dummy_data.py
    ```

## Usage

### Run the Pipeline

To run the complete pipeline (preprocessing -> features -> training -> evaluation):

```bash
python detectionpipelines.py
```

This will generate:
-   `preprocessed_job_postings.csv`: Preprocessed text data.
-   `data/features/`: Extracted features and artifacts.
-   `models/`: Trained model and metrics.
-   `evaluation/`: Evaluation report and plots.

### Run the Dashboard

To launch the interactive dashboard:

```bash
python dashboard.py
```

Open your browser and navigate to `http://127.0.0.1:8050/`.
-   **Single Prediction**: Enter job details to check if it's fraudulent.
-   **Evaluation Plots**: View performance metrics like ROC curve, Confusion Matrix, etc.

## Directory Structure

```
.
├── dashboard.py            # Interactive web app
├── detectionpipelines.py   # Main pipeline orchestrator
├── featureengineering.py   # Feature extraction logic
├── modelevaluation.py      # Evaluation and reporting
├── modeltraining.py        # Model training logic
├── preprocess.py           # Text preprocessing logic
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── fake_job_postings.csv   # Input dataset (user provided)
```

## Notes

-   The pipeline downloads necessary NLTK data automatically.
-   Ensure `fake_job_postings.csv` is properly formatted (standard CSV with headers).
