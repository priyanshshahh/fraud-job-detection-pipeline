# Fraud Job Detection Pipeline

This repository contains an end-to-end workflow for detecting fraudulent job postings.

## Setup

1. Install Python 3.8 or higher.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK resources:
   ```bash
   python -m nltk.downloader punkt stopwords wordnet omw-1.4 vader_lexicon
   ```

## Usage

Run the full pipeline on a raw CSV and store all outputs in an output directory:

```bash
python detectionpipelines.py path/to/fake_job_postings.csv output_dir
```

Steps can also be executed individually:

```bash
python preprocess.py input.csv preprocessed.csv
python featureengineering.py preprocessed.csv features_dir
python modeltraining.py features_dir models_dir
python modelevaluation.py features_dir models_dir evaluation_dir
```

## Dashboard

After training, launch the dashboard to inspect evaluation plots and perform single-job predictions:

```bash
python dashboard.py
```

Ensure the `models` directory and `data/features` folder referenced above exist inside your chosen output directory.

## Gradio Demo

After training your model, you can launch a lightweight Gradio interface for quick demonstrations:

```bash
python gradio_demo.py
```

This interface accepts the same job posting fields as the dashboard and returns a prediction label and fraud probability.
