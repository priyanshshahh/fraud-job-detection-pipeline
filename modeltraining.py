#!/usr/bin/env python3
"""
Model training module for fraudulent job posting detection.
This version loads your precomputed features, fixes negative sentiment values,
trains a Multinomial Naïve Bayes on sparse data, evaluates, cross‐validates,
and saves all artifacts without ever converting the full TF–IDF matrix to dense.
"""

import os
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_features(features_dir):
    """
    Load TF–IDF, sentiment, metadata features and the target variable,
    trying both tfidf_matrix.joblib and tfidf_features.joblib.
    """
    # 1) TF–IDF (sparse matrix)
    candidates = [
        os.path.join(features_dir, 'tfidf_matrix.joblib'),
        os.path.join(features_dir, 'tfidf_features.joblib')
    ]
    for path in candidates:
        if os.path.exists(path):
            tfidf = joblib.load(path)
            print("Loaded TF–IDF matrix:", tfidf.shape)
            break
    else:
        raise FileNotFoundError(f"None of {candidates} found.")
    # 2) Sentiment features
    sentiment = pd.read_csv(os.path.join(features_dir, 'sentiment_features.csv'))
    print("Loaded sentiment features:", sentiment.shape)
    # 3) Metadata features
    metadata = pd.read_csv(os.path.join(features_dir, 'metadata_features.csv'))
    print("Loaded metadata features:", metadata.shape)
    # 4) Target variable
    target = pd.read_csv(os.path.join(features_dir, 'target.csv'))
    print("Loaded target:", target.shape)
    return tfidf, sentiment, metadata, target

def prepare_features(tfidf, sentiment_df, metadata_df):
    """
    Convert sentiment and metadata to sparse float matrices,
    shift negative sentiment compound scores to non-negative,
    then horizontally stack with TF–IDF.
    """
    # --- Fix and sparsify sentiment ---
    sent = sentiment_df.copy()
    # ensure numeric & fill NaN
    sent = sent.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # shift any compound columns by +1
    for col in sent.columns:
        if col.endswith('_compound'):
            sent[col] += 1.0
    sent_sparse = csr_matrix(sent.values.astype(float))

    # --- One‐hot encode and sparsify metadata ---
    meta = pd.get_dummies(metadata_df, drop_first=True).fillna(0.0)
    meta = meta.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    meta_sparse = csr_matrix(meta.values.astype(float))

    # --- Combine all features ---
    X = hstack([tfidf, sent_sparse, meta_sparse]).tocsr()
    print("Combined feature matrix shape:", X.shape)
    return X

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Split data and fit a MultinomialNB on sparse features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Predict on X_test and compute performance metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    print("\nTest Metrics:")
    for k, v in metrics.items():
        if k in ('confusion_matrix', 'classification_report'):
            print(f"\n{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")
    return metrics

def cross_validate(model, X, y, n_splits=5, random_state=42):
    """
    Perform stratified k-fold cross-validation on sparse data.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = {
        'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
        'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
        'recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
        'f1': cross_val_score(model, X, y, cv=cv, scoring='f1'),
        'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    }
    print(f"\n{n_splits}-fold CV results:")
    for k, arr in scores.items():
        print(f"{k}: {arr.mean():.4f} ± {arr.std():.4f}")
    return scores

def plot_confusion(cm, output_dir):
    """
    Save a heatmap of the confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_artifacts(model, metrics, cv_scores, output_dir):
    """
    Dump model, metrics, CV scores, and confusion plot to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'nb_model.joblib'))
    joblib.dump(metrics, os.path.join(output_dir, 'metrics.joblib'))
    joblib.dump(cv_scores, os.path.join(output_dir, 'cv_scores.joblib'))
    plot_confusion(metrics['confusion_matrix'], output_dir)
    print("All artifacts saved to:", output_dir)

def main():
    features_dir = "/Users/priyansh/Desktop/fraud_job_detection/data/features"
    output_dir   = "/Users/priyansh/Desktop/fraud_job_detection/models"

    # 1) Load
    tfidf, sentiment, metadata, target_df = load_features(features_dir)
    y = target_df['fraudulent'].values

    # 2) Prepare features
    X = prepare_features(tfidf, sentiment, metadata)

    # 3) Train
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # 4) Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # 5) Cross-validate
    cv_scores = cross_validate(model, X, y)

    # 6) Save results
    save_artifacts(model, metrics, cv_scores, output_dir)

if __name__ == "__main__":
    main()
