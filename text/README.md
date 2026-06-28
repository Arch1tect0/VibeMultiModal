# Text Modality - Crime Report NLP Pipeline

## Overview

This folder contains the NLP pipeline developed for the text modality of the AI Incident Reporting System project. The goal of this component is to process unstructured social media crime reports and convert them into a structured format that can be integrated with the outputs from the audio, PDF, image, and video modalities.

## Dataset

* **Dataset:** CrimeReport (Kaggle)
* **Data Type:** Social media posts and crime-related news reports

## Tools Used

* Python
* pandas
* spaCy (`en_core_web_sm`)
* HuggingFace Transformers
* Jupyter Notebook

## Features

The pipeline performs the following tasks:

1. Reads raw crime-related text data.
2. Cleans and preprocesses text by removing noise and normalizing content.
3. Tokenizes text and removes common stop words.
4. Uses spaCy to extract named entities such as people, organizations, locations, and dates.
5. Uses HuggingFace models for sentiment analysis.
6. Uses HuggingFace zero-shot classification to categorize incidents into:

   * Theft / Robbery
   * Assault / Violence
   * Fire / Arson
   * Traffic Accident
   * Public Disturbance
   * Other
7. Exports the final structured dataset as `Text.csv`.

## Output

The final CSV contains the following columns:

* `Text_ID`
* `Source`
* `Raw_Text`
* `Sentiment`
* `Entities`
* `Topic`

## How to Run

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

### 2. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 3. Run the notebook

Open and run:

```text
text_nlp_pipeline_requirements_student_style.ipynb
```

## Output File

The pipeline generates:

```text
Text.csv
```
