#modelevaluation
#!/usr/bin/env python3
"""
Model evaluation module for fraudulent job posting detection.
Reconstructs the test set, loads the trained model, recomputes metrics,
creates all plots, and writes a Markdown report.
"""

import os
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1) Load & Prepare Features ---

def load_features(features_dir):
    """
    Load TF–IDF, sentiment, metadata, and target from disk.
    """
    # 1a) TF–IDF sparse matrix
    path1 = os.path.join(features_dir, 'tfidf_matrix.joblib')
    path2 = os.path.join(features_dir, 'tfidf_features.joblib')
    if os.path.exists(path1):
        tfidf = joblib.load(path1)
    elif os.path.exists(path2):
        tfidf = joblib.load(path2)
    else:
        raise FileNotFoundError(f"No TF–IDF file under {path1} or {path2}")
    print("Loaded TF–IDF shape:", tfidf.shape)

    # 1b) Sentiment
    sentiment = pd.read_csv(os.path.join(features_dir, 'sentiment_features.csv'))
    print("Loaded sentiment shape:", sentiment.shape)

    # 1c) Metadata
    metadata = pd.read_csv(os.path.join(features_dir, 'metadata_features.csv'))
    print("Loaded metadata shape:", metadata.shape)

    # 1d) Target
    target = pd.read_csv(os.path.join(features_dir, 'target.csv'))
    print("Loaded target shape:", target.shape)

    return tfidf, sentiment, metadata, target['fraudulent'].values

def prepare_features(tfidf, sentiment_df, metadata_df):
    """
    Convert sentiment & metadata to sparse floats, shift negatives +1,
    then hstack with TF–IDF to rebuild X.
    """
    # 2a) Sentiment → float array → shift compound cols → sparse
    sent = sentiment_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    for c in sent.columns:
        if c.endswith('_compound'):
            sent[c] += 1.0
    sent_sparse = csr_matrix(sent.values.astype(float))

    # 2b) Metadata → one-hot → float array → sparse
    meta = pd.get_dummies(metadata_df, drop_first=True)
    meta = meta.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    meta_sparse = csr_matrix(meta.values.astype(float))

    # 2c) Combine
    X = hstack([tfidf, sent_sparse, meta_sparse]).tocsr()
    print("Rebuilt feature matrix:", X.shape)
    return X

# --- 2) Plots ---

def plot_roc(y_true, y_prob, outdir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'roc_curve.png'))
    plt.close()

def plot_pr(y_true, y_prob, outdir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    base_rate = y_true.mean()
    plt.axhline(base_rate, linestyle='--', color='gray', label="Baseline")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'pr_curve.png'))
    plt.close()

def plot_thresholds(y_true, y_prob, outdir):
    thresholds = [i/20 for i in range(1,20)]
    records = []
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        records.append({
            'threshold': thr,
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall': recall_score(y_true, preds),
            'f1': f1_score(y_true, preds)
        })
    df = pd.DataFrame(records)
    plt.figure(figsize=(10,6))
    for metric,color in [('accuracy','blue'),('precision','green'),('recall','red'),('f1','orange')]:
        plt.plot(df['threshold'], df[metric], label=metric.capitalize(), color=color)
    plt.title("Metrics vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'threshold_analysis.png'))
    plt.close()
    best = df.loc[df['f1'].idxmax()]
    return best['threshold'], df

def plot_confusion(cm, outdir):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(outdir, 'confusion_matrix.png'))
    plt.close()

def plot_cv(cv_scores, outdir):
    names = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    means = [cv_scores[k].mean() for k in ['accuracy','precision','recall','f1','roc_auc']]
    stds  = [cv_scores[k].std()  for k in ['accuracy','precision','recall','f1','roc_auc']]
    plt.figure(figsize=(8,6))
    bars = plt.bar(names, means, yerr=stds, capsize=5)
    plt.title("5-Fold CV Scores")
    plt.ylabel("Score")
    plt.ylim(0,1.05)
    plt.grid(axis='y', alpha=0.3)
    for b,m in zip(bars,means):
        plt.text(b.get_x()+b.get_width()/2, m+0.01, f"{m:.3f}", ha='center')
    plt.savefig(os.path.join(outdir, 'cv_scores.png'))
    plt.close()

# --- 3) Main Evaluation Workflow ---

def evaluate_all(features_dir, models_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # A) Reconstruct X & y
    tfidf, sentiment, metadata, y = load_features(features_dir)
    X = prepare_features(tfidf, sentiment, metadata)

    # B) Train/Test split exactly as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # C) Load trained model
    model = joblib.load(os.path.join(models_dir, 'nb_model.joblib'))

    # D) Predictions & probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # E) Compute metrics
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    # F) Cross-validation (reuse saved if available)
    cv_path = os.path.join(models_dir, 'cv_scores.joblib')
    if os.path.exists(cv_path):
        cv_scores = joblib.load(cv_path)
    else:
        # fallback: run 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {
            'accuracy':  cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
            'recall':    cross_val_score(model, X, y, cv=cv, scoring='recall'),
            'f1':        cross_val_score(model, X, y, cv=cv, scoring='f1'),
            'roc_auc':   cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        }

    # G) Create plots
    plot_roc(y_test, y_prob, output_dir)
    plot_pr(y_test, y_prob, output_dir)
    thr, thr_df = plot_thresholds(y_test, y_prob, output_dir)
    plot_confusion(cm, output_dir)
    plot_cv(cv_scores, output_dir)

    # H) Write Markdown report
    md = []
    md.append("# Model Evaluation Report\n")
    md.append("## Test Set Metrics\n")
    md.append(f"- **Accuracy:** {acc:.4f}")
    md.append(f"- **Precision:** {prec:.4f}")
    md.append(f"- **Recall:** {rec:.4f}")
    md.append(f"- **F1 Score:** {f1:.4f}\n")
    md.append("```\n" + cr + "\n```\n")
    md.append("## Optimal Threshold\n")
    md.append(f"- Best F1 at threshold **{thr:.2f}**\n")
    md.append("## Cross-Validation (5-fold)\n")
    for metric in ['accuracy','precision','recall','f1','roc_auc']:
        mean, std = cv_scores[metric].mean(), cv_scores[metric].std()
        md.append(f"- **{metric.capitalize()}:** {mean:.4f} ± {std:.4f}")
    md.append("\n")
    with open(os.path.join(output_dir, 'evaluation_report.md'), 'w') as f:
        f.write("\n".join(md))

    print("Evaluation complete. Outputs in:", output_dir)

def main():
    """CLI entry point for model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("features_dir", help="Directory containing feature files")
    parser.add_argument("models_dir", help="Directory containing trained model")
    parser.add_argument("output_dir", help="Directory to store evaluation outputs")
    args = parser.parse_args()

    evaluate_all(args.features_dir, args.models_dir, args.output_dir)


if __name__ == "__main__":
    main()
