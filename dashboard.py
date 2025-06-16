#!/usr/bin/env python3
"""
Dashboard for Fraudulent Job Posting Detection.
Interactive web app for single-case predictions and evaluation plot display.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import nltk
from dash import Dash, dcc, html, Input, Output, State
from scipy.sparse import hstack, csr_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Define directories
ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR     = os.path.join(ROOT_DIR, 'data', 'features')
MODELS_DIR       = os.path.join(ROOT_DIR, 'models')
METADATA_CSV     = os.path.join(FEATURES_DIR, 'metadata_features.csv')
ASSETS_DIR       = os.path.join(ROOT_DIR, 'assets')  # Evaluation plots here

# Paths to artifacts
tfidf_path = os.path.join(FEATURES_DIR, 'tfidf_vectorizer.joblib')
model_path = os.path.join(MODELS_DIR,   'nb_model.joblib')

# Validate existence
for path, desc in [(tfidf_path, 'TF–IDF vectorizer'), (model_path, 'Model'), (METADATA_CSV, 'Metadata CSV')]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {desc} at {path}")

# Load artifacts
tfidf = joblib.load(tfidf_path)
model = joblib.load(model_path)
vader = SentimentIntensityAnalyzer()

# Prepare metadata encoding template
metadata_template = pd.read_csv(METADATA_CSV)
metadata_cols     = pd.get_dummies(metadata_template, drop_first=True).columns.tolist()

# Initialize Dash app
external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_css)
app.title = "Fraudulent Job Detector"

# Layout
app.layout = html.Div([
    html.H1("Fraudulent Job Posting Detector"),
    dcc.Tabs([
        dcc.Tab(label='Single Prediction', children=[
            html.Div([
                html.Label('Title'), dcc.Input(id='title', type='text', style={'width':'100%'}), html.Br(), html.Br(),
                html.Label('Company Profile'), dcc.Textarea(id='company_profile', style={'width':'100%'}), html.Br(), html.Br(),
                html.Label('Description'), dcc.Textarea(id='description', style={'width':'100%'}), html.Br(), html.Br(),
                html.Label('Requirements'), dcc.Textarea(id='requirements', style={'width':'100%'}), html.Br(), html.Br(),
                html.Label('Benefits'), dcc.Textarea(id='benefits', style={'width':'100%'}), html.Br(), html.Br(),
                html.H4('Metadata'),
                dcc.Checklist(id='telecommuting', options=[{'label':'Telecommuting','value':1}], value=[]),
                dcc.Checklist(id='has_company_logo', options=[{'label':'Has Logo','value':1}], value=[]),
                dcc.Checklist(id='has_questions', options=[{'label':'Has Questions','value':1}], value=[]), html.Br(),
                html.Label('Employment Type'),    dcc.Input(id='employment_type', type='text'), html.Br(), html.Br(),
                html.Label('Required Experience'), dcc.Input(id='required_experience', type='text'), html.Br(), html.Br(),
                html.Label('Required Education'),  dcc.Input(id='required_education', type='text'), html.Br(), html.Br(),
                html.Label('Industry'),            dcc.Input(id='industry', type='text'), html.Br(), html.Br(),
                html.Label('Function'),            dcc.Input(id='function', type='text'), html.Br(), html.Br(),
                html.Button('Predict', id='predict_btn'),
                html.Div(id='output_div', style={'marginTop':'20px','fontSize':'18px'})
            ], style={'padding':'20px'})
        ]),
        dcc.Tab(label='Evaluation Plots', children=[
            html.Div([html.Img(src=app.get_asset_url('roc_curve.png'), style={'width':'45%','padding':'10px'}),
                      html.Img(src=app.get_asset_url('pr_curve.png'), style={'width':'45%','padding':'10px'})]),
            html.Div([html.Img(src=app.get_asset_url('threshold_analysis.png'), style={'width':'45%','padding':'10px'}),
                      html.Img(src=app.get_asset_url('cv_scores.png'), style={'width':'45%','padding':'10px'})])
        ])
    ])
])

# Helper to build and sanitize job dict
def build_job_dict(values):
    keys = ['title','company_profile','description','requirements','benefits',
            'telecommuting','has_company_logo','has_questions',
            'employment_type','required_experience','required_education',
            'industry','function']
    job = dict(zip(keys, values))
    # Convert checklists (list) to binary
    for chk in ['telecommuting','has_company_logo','has_questions']:
        val = job.get(chk)
        job[chk] = 1 if isinstance(val, list) and 1 in val else 0
    return job

# Prediction callback
@app.callback(
    Output('output_div','children'),
    Input('predict_btn','n_clicks'),
    State('title','value'),
    State('company_profile','value'),
    State('description','value'),
    State('requirements','value'),
    State('benefits','value'),
    State('telecommuting','value'),
    State('has_company_logo','value'),
    State('has_questions','value'),
    State('employment_type','value'),
    State('required_experience','value'),
    State('required_education','value'),
    State('industry','value'),
    State('function','value')
)
def on_predict(n_clicks, *args):
    if not n_clicks:
        return ''
    job = build_job_dict(args)
    # Text preprocessing
    from preprocess import preprocess_text
    processed = {f: preprocess_text(job[f] or '') for f in ['title','company_profile','description','requirements','benefits']}
    tfidf_vec = tfidf.transform([' '.join(processed.values())])
    # Sentiment
    sent = []
    for f in ['title','company_profile','description','requirements','benefits']:
        scores = vader.polarity_scores(job[f] or '')
        sent += [scores['neg'], scores['neu'], scores['pos'], scores['compound']+1.0]
    sent_sp = csr_matrix(np.array(sent).reshape(1,-1))
    # Metadata
    meta_df = pd.DataFrame([{k: job[k] for k in ['telecommuting','has_company_logo','has_questions',
                                               'employment_type','required_experience',
                                               'required_education','industry','function']}])
    meta_enc = pd.get_dummies(meta_df, drop_first=True)
    meta_enc = meta_enc.reindex(columns=metadata_cols, fill_value=0)
    meta_sp = csr_matrix(meta_enc.values.astype(float))
    # Combine and align to model input size
    features = hstack([tfidf_vec, sent_sp, meta_sp])
    expect = model.n_features_in_
    if features.shape[1] < expect:
        pad = csr_matrix((1, expect - features.shape[1]))
        features = hstack([features, pad]).tocsr()
    elif features.shape[1] > expect:
        features = features[:, :expect]
    # Predict
    prob = float(model.predict_proba(features)[:,1][0])
    label = 'FRAUDULENT' if prob >= 0.60 else 'LEGITIMATE'
    return html.Div([html.H4(f"Prediction: {label}"), html.P(f"Fraud Probability: {prob:.4f}")])

if __name__=='__main__':
    app.run(debug=True)
