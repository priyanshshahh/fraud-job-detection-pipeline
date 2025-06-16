#!/usr/bin/env python3
"""
Preprocessing module for fraudulent job posting detection.
This module handles text preprocessing including tokenization,
stopword removal, and text normalization.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def clean_text(text):
    """
    Clean and normalize a single text string.
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def tokenize_text(text):
    """
    Split a cleaned string into word tokens.
    """
    if isinstance(text, str):
        return word_tokenize(text)
    return []

def remove_stopwords(tokens):
    """
    Remove common English stopwords from a list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]

def lemmatize_tokens(tokens):
    """
    Reduce tokens to their base (lemma) form.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_text(text):
    """
    Full pipeline for a single text: clean, tokenize, remove stopwords, lemmatize.
    """
    if pd.isna(text) or text is None:
        return ""
    cleaned     = clean_text(text)
    tokens      = tokenize_text(cleaned)
    filtered    = remove_stopwords(tokens)
    lemmatized  = lemmatize_tokens(filtered)
    return ' '.join(lemmatized)

def preprocess_dataset(input_file, output_file):
    """
    Apply preprocessing to each text column in the CSV and save the result.
    """
    df = pd.read_csv(input_file)
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']

    for col in text_columns:
        if col in df.columns:
            df[f'{col}_processed'] = df[col].apply(preprocess_text)

    df.to_csv(output_file, index=False)
    return df

def main():
    """CLI entry point for preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess job posting data")
    parser.add_argument("input_csv", help="Path to raw job postings CSV")
    parser.add_argument("output_csv", help="Where to save the processed CSV")
    args = parser.parse_args()

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    preprocess_dataset(args.input_csv, args.output_csv)
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
