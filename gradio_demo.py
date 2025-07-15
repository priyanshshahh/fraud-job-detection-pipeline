#!/usr/bin/env python3
"""Gradio demo for fraudulent job posting detection."""
import os
import numpy as np
import pandas as pd
import joblib
import nltk
import gradio as gr
from scipy.sparse import hstack, csr_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from preprocess import preprocess_text

# Ensure VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(ROOT_DIR, 'data', 'features')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

TFIDF_PATH = os.path.join(FEATURES_DIR, 'tfidf_vectorizer.joblib')
MODEL_PATH = os.path.join(MODELS_DIR, 'nb_model.joblib')
METADATA_CSV = os.path.join(FEATURES_DIR, 'metadata_features.csv')

for path, desc in [(TFIDF_PATH, 'TF–IDF vectorizer'), (MODEL_PATH, 'Model'), (METADATA_CSV, 'Metadata CSV')]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {desc} at {path}")

# Load artifacts
TFIDF = joblib.load(TFIDF_PATH)
MODEL = joblib.load(MODEL_PATH)
VADER = SentimentIntensityAnalyzer()
meta_template = pd.read_csv(METADATA_CSV)
METADATA_COLS = pd.get_dummies(meta_template, drop_first=True).columns.tolist()

TEXT_FIELDS = ['title','company_profile','description','requirements','benefits']
META_FIELDS = ['telecommuting','has_company_logo','has_questions','employment_type',
               'required_experience','required_education','industry','function']

def predict_job(title, company_profile, description, requirements, benefits,
                telecommuting, has_company_logo, has_questions,
                employment_type, required_experience, required_education,
                industry, function):
    job = {
        'title': title or '',
        'company_profile': company_profile or '',
        'description': description or '',
        'requirements': requirements or '',
        'benefits': benefits or '',
        'telecommuting': 1 if telecommuting else 0,
        'has_company_logo': 1 if has_company_logo else 0,
        'has_questions': 1 if has_questions else 0,
        'employment_type': employment_type or '',
        'required_experience': required_experience or '',
        'required_education': required_education or '',
        'industry': industry or '',
        'function': function or ''
    }

    processed = {f: preprocess_text(job[f]) for f in TEXT_FIELDS}
    tfidf_vec = TFIDF.transform([' '.join(processed.values())])

    sent = []
    for f in TEXT_FIELDS:
        scores = VADER.polarity_scores(job[f])
        sent += [scores['neg'], scores['neu'], scores['pos'], scores['compound'] + 1.0]
    sent_sp = csr_matrix(np.array(sent).reshape(1, -1))

    meta_df = pd.DataFrame([{k: job[k] for k in META_FIELDS}])
    meta_enc = pd.get_dummies(meta_df, drop_first=True)
    meta_enc = meta_enc.reindex(columns=METADATA_COLS, fill_value=0)
    meta_sp = csr_matrix(meta_enc.values.astype(float))

    features = hstack([tfidf_vec, sent_sp, meta_sp])
    expect = MODEL.n_features_in_
    if features.shape[1] < expect:
        pad = csr_matrix((1, expect - features.shape[1]))
        features = hstack([features, pad]).tocsr()
    elif features.shape[1] > expect:
        features = features[:, :expect]

    prob = float(MODEL.predict_proba(features)[:, 1][0])
    label = 'FRAUDULENT' if prob >= 0.60 else 'LEGITIMATE'
    return label, prob

inputs = [
    gr.Textbox(label='Title'),
    gr.Textbox(label='Company Profile'),
    gr.Textbox(label='Description'),
    gr.Textbox(label='Requirements'),
    gr.Textbox(label='Benefits'),
    gr.Checkbox(label='Telecommuting'),
    gr.Checkbox(label='Has Company Logo'),
    gr.Checkbox(label='Has Questions'),
    gr.Textbox(label='Employment Type'),
    gr.Textbox(label='Required Experience'),
    gr.Textbox(label='Required Education'),
    gr.Textbox(label='Industry'),
    gr.Textbox(label='Function')
]

outputs = [
    gr.Textbox(label='Prediction'),
    gr.Number(label='Fraud Probability')
]

demo = gr.Interface(
    fn=predict_job,
    inputs=inputs,
    outputs=outputs,
    title='Fraudulent Job Posting Detector',
    description='Enter job details to predict whether the posting is fraudulent.'
)

if __name__ == '__main__':
    demo.launch()
