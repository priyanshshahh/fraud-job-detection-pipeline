#feature engineering
#!/usr/bin/env python3
"""
Feature extraction module for fraudulent job posting detection.
This module implements TF–IDF vectorization, sentiment analysis,
and metadata encoding on preprocessed job posting data.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def combine_text_columns(df, text_columns):
    """
    Combine multiple text columns into one long string per row.
    """
    # Replace missing values with empty strings
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Join the columns with spaces
    return df[text_columns].apply(lambda row: ' '.join(row), axis=1)

def extract_tfidf_features(text_series, max_features=5000, ngram_range=(1, 2)):
    """
    Fit a TF–IDF vectorizer on the text and return both the vectorizer and the matrix.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=5,
        max_df=0.8,
        sublinear_tf=True
    )
    X = tfidf.fit_transform(text_series)
    return tfidf, X

def extract_sentiment_features(df, text_columns):
    """
    Use VADER to compute sentiment scores for each text column.
    Returns a DataFrame with neg/neu/pos/compound for each.
    """
    sid = SentimentIntensityAnalyzer()
    out = pd.DataFrame(index=df.index)

    for col in text_columns:
        if col in df.columns:
            scores = df[col].fillna('').apply(lambda txt: sid.polarity_scores(txt))
            out[f'{col}_neg']      = scores.apply(lambda s: s['neg'])
            out[f'{col}_neu']      = scores.apply(lambda s: s['neu'])
            out[f'{col}_pos']      = scores.apply(lambda s: s['pos'])
            out[f'{col}_compound'] = scores.apply(lambda s: s['compound'])

    return out

def extract_metadata_features(df, metadata_columns):
    """
    Select and clean metadata columns (no encoding here).
    Missing categorical → 'unknown'; missing numeric → 0.
    """
    meta = df[metadata_columns].copy()
    for col in metadata_columns:
        if col in meta.columns:
            if meta[col].dtype == 'object':
                meta[col] = meta[col].fillna('unknown')
            else:
                meta[col] = meta[col].fillna(0)
    return meta

def feature_extraction(input_file, output_dir):
    """
    Run all feature‐extraction steps and save outputs to disk.
    """
    # Load data
    df = pd.read_csv(input_file)

    # Define which columns to use
    proc_text_cols = [
        'title_processed',
        'company_profile_processed',
        'description_processed',
        'requirements_processed',
        'benefits_processed'
    ]
    orig_text_cols = ['title','company_profile','description','requirements','benefits']
    meta_cols      = [
        'telecommuting','has_company_logo','has_questions',
        'employment_type','required_experience','required_education',
        'industry','function'
    ]

    # 1) TF–IDF on combined processed text
    combined = combine_text_columns(df, proc_text_cols)
    tfidf_vec, tfidf_mat = extract_tfidf_features(combined)

    # 2) Sentiment on the original text
    sentiment_df = extract_sentiment_features(df, orig_text_cols)

    # 3) Metadata cleaning
    metadata_df = extract_metadata_features(df, meta_cols)

    # 4) Target variable
    target = df['fraudulent']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save everything
    joblib.dump(tfidf_vec, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(tfidf_mat, os.path.join(output_dir, 'tfidf_matrix.joblib'))
    sentiment_df.to_csv(os.path.join(output_dir, 'sentiment_features.csv'), index=False)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata_features.csv'), index=False)
    target.to_csv(os.path.join(output_dir, 'target.csv'), index=False)

    return tfidf_mat, sentiment_df, metadata_df, target

if __name__ == "__main__":
    # Download VADER data once
    nltk.download('vader_lexicon')

    inp  = "/Users/priyansh/Desktop/fraud_job_detection/preprocessed_job_postings.csv"
    outd = "/Users/priyansh/Desktop/fraud_job_detection/data/features"
    feature_extraction(inp, outd)
    print("Feature extraction complete!")
