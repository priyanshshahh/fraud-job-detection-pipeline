#!/usr/bin/env python3
"""
End-to-end pipeline for fraudulent job posting detection.
Runs preprocessing, feature extraction, model training, and evaluation.
"""

import os

# Step 1: Text preprocessing
from preprocess import preprocess_dataset

# Step 2: Feature engineering
from featureengineering import feature_extraction

# Step 3: Model training utilities
import modeltraining
from modeltraining import load_features, prepare_features, train_model, evaluate_model, cross_validate, save_artifacts

# Step 4: Detailed evaluation
from modelevaluation import evaluate_all

def main():
    """Run the entire pipeline via command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Execute full fraud detection pipeline")
    parser.add_argument("raw_csv", help="Path to the raw job postings CSV")
    parser.add_argument("output_dir", help="Directory to store all artifacts")
    args = parser.parse_args()

    raw_csv = os.path.abspath(args.raw_csv)
    output_root = os.path.abspath(args.output_dir)
    preproc_csv   = os.path.join(output_root, "preprocessed_job_postings.csv")
    features_dir  = os.path.join(output_root, "data", "features")
    models_dir    = os.path.join(output_root, "models")
    evaluation_dir= os.path.join(output_root, "evaluation")

    # Ensure output folders exist
    os.makedirs(features_dir,    exist_ok=True)
    os.makedirs(models_dir,      exist_ok=True)
    os.makedirs(evaluation_dir,  exist_ok=True)

    # ─── 1) Preprocessing ────────────────────────────────────────────────────
    print("=== STEP 1: Preprocessing ===")
    preprocess_dataset(raw_csv, preproc_csv)

    # ─── 2) Feature Extraction ───────────────────────────────────────────────
    print("\n=== STEP 2: Feature Extraction ===")
    feature_extraction(preproc_csv, features_dir)

    # ─── 3) Model Training ───────────────────────────────────────────────────
    print("\n=== STEP 3: Model Training ===")
    # 3a) Load the feature blocks and target
    tfidf, sentiment_df, metadata_df, target_df = load_features(features_dir)
    y = target_df['fraudulent'].values

    # 3b) Prepare the combined sparse matrix
    X = prepare_features(tfidf, sentiment_df, metadata_df)

    # 3c) Train / test split, fit model
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # 3d) Evaluate on hold‐out
    metrics   = evaluate_model(model, X_test, y_test)

    # 3e) Cross‐validate on full data
    cv_scores = cross_validate(model, X, y)

    # 3f) Save the model and metrics artifacts
    save_artifacts(model, metrics, cv_scores, models_dir)

    # ─── 4) Detailed Evaluation ─────────────────────────────────────────────
    print("\n=== STEP 4: Detailed Evaluation & Reporting ===")
    evaluate_all(features_dir, models_dir, evaluation_dir)

    print(
        f"\n🎉 Pipeline complete! Check outputs under:\n"
        f"  • Preprocessed CSV: {preproc_csv}\n"
        f"  • Features:        {features_dir}\n"
        f"  • Models:          {models_dir}\n"
        f"  • Evaluation:      {evaluation_dir}"
    )

if __name__ == "__main__":
    main()
