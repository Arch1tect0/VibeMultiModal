# VibeMultiModal

### Multimodal Crime / Incident Report Analyzer

---

## What This Repository Is About

Every day, emergency departments receive incident reports from many different sources: **911 audio calls**, **police PDF documents**, **crime-scene photographs**, and **social media or news text**. Each source stores information in a different, unstructured format. Analysts must manually read and compare all of it, which is slow, inconsistent, and makes it harder to respond quickly.

**VibeMultiModal** is our group's prototype **Multimodal Incident Report Analyzer**. The goal is to:

1. Process each type of unstructured data with AI
2. Extract structured fields (event type, location, time, sentiment, confidence, etc.)
3. Merge all outputs into one unified incident dataset
4. Provide a dashboard where users can explore incidents and drill down to the original source record

This repository contains the full pipeline — one folder per data modality, plus an integration folder for the merged dataset and Streamlit dashboard.

> **Team note:** Our group was smaller than the full six-student assignment. We completed four modalities (Audio, PDF, Image, Text) plus Integration and Dashboard. The Video modality was not implemented.

---

## Quick Reference Links

| Deliverable | Location |
|---|---|
| Final merged dataset | [integration/final_integrated_incidents.csv](integration/final_integrated_incidents.csv) |
| Integration notebook | [integration/final_integration.ipynb](integration/final_integration.ipynb) |
| Incident Intelligence Dashboard | [integration/app.py](integration/app.py) |
| Architecture diagram | [integration/multimodal_pipeline_architecture.pdf](integration/multimodal_pipeline_architecture.pdf) |

---

## End-to-End Pipeline

```
 STAGE 1                STAGE 2                    STAGE 3                 STAGE 4 & 5
 Ingestion              AI Processing              Extraction              Integration + Dashboard
 ----------------       -------------------        ----------------        ---------------------------

 911 audio calls   -->  audio/                 -->  audio_output.csv  --\
 Police PDFs       -->  pdf/                   -->  pdf_output.csv    ---+-->  final_integrated_incidents.csv  -->  Streamlit Dashboard
 Scene images      -->  image/                 -->  results.csv       --/
 Crime text posts  -->  text/                  -->  Text.csv          --/
```

**How integration works:**

- Each modality assigns its own prefixed `Incident_ID` (`AUD-`, `DOC-`, `IMG-`, `TXT-`)
- Each modality keeps its own original ID column (`Call_ID`, `Report_ID`, `Image_ID`, `Text_ID`)
- All tables are combined with `pandas.concat` (UNION) into one shared schema
- Missing values are filled with `Unknown` or `N/A`
- A severity label (`Low`, `Medium`, `High`) is computed from each row's score

This is **not** a join across modalities. Each row represents one source record from one modality.

---

## Repository Structure

Each folder is self-contained and includes its **own README** with full setup instructions, datasets, models, and how to reproduce that modality's output.

| Folder | Role | What It Does | Key Output | Detailed Docs |
|---|---|---|---|---|
| [audio/](audio/) | Student 1 — Audio Analyst | Transcribes 911 calls, extracts events, locations, sentiment, and urgency | `audio_output.csv` | [audio/README.md](audio/README.md) |
| [pdf/](pdf/) | Student 2 — Document Analyst | OCR and field extraction from police PDF documents | `pdf_output.csv` | [pdf/README.md](pdf/README.md) |
| [image/](image/) | Student 3 — Image Analyst | Fire/smoke detection, scene classification, object detection, OCR | `outputs/results.csv` | [image/README.md](image/README.md) |
| [text/](text/) | Student 5 — Text Analyst | NLP on crime text: cleaning, NER, sentiment, topic classification | `Text.csv` | [text/README.md](text/README.md) |
| [integration/](integration/) | Student 6 — Integration Lead | Merges all CSVs, assigns severity, builds dashboard | `final_integrated_incidents.csv`, `app.py` | [integration/README.md](integration/README.md) |

**Start here for the big picture. Open each folder's README when you need step-by-step instructions for that component.**

---

## Modality Summaries

### Audio (`audio/`)

- **Input:** 911 emergency call audio (`.wav`)
- **Tools:** OpenAI Whisper (speech-to-text), spaCy (location NER), keyword matching (event type), custom urgency scoring
- **Output columns:** `Call_ID`, `Transcript`, `Extracted_Event`, `Location`, `Sentiment`, `Urgency_Score`
- **Example events detected:** Homicide, Shooting, Fire, Traffic Accident, Medical Emergency

See [audio/README.md](audio/README.md) for Kaggle setup and notebook instructions.

---

### PDF (`pdf/`)

- **Input:** Scanned police PDF documents (Arkansas LESO 1033 training plan proposals)
- **Tools:** pdf2image, Tesseract OCR, regex-based field extraction
- **Output columns:** `Report_ID`, `Incident_Type`, `Date`, `Location`, `Agency`, `Officer`, `Summary`, `Suspect_Description`, `Outcome`, `Source_Pages`
- **Example document types:** Training Lesson Plan, Policy/Correspondence, Vendor Capability Statement

See [pdf/README.md](pdf/README.md) for OCR setup and notebook instructions.

---

### Image (`image/`)

- **Input:** Fire and incident scene photographs
- **Tools:** YOLOv8, CLIP, ViT fire classifier, OpenCV, Tesseract OCR
- **Output columns:** `Image_ID`, `Scene_Type`, `Objects_Detected`, `Text_Extracted`, confidence scores
- **Example scene types:** Forest Fire, Structure Fire, Grass Fire, Vehicle Fire

See [image/README.md](image/README.md) for detector configuration and run commands.

---

### Text (`text/`)

- **Input:** Crime-related social media posts and news reports (CrimeReport dataset)
- **Tools:** spaCy NER, HuggingFace Transformers (sentiment + zero-shot topic classification)
- **Output columns:** `Text_ID`, `Source`, `Raw_Text`, `Sentiment`, `Entities`, `Topic`
- **Example topics:** Assault / Violence, Theft / Robbery, Public Disturbance, Fire / Arson, Traffic Accident, Other

See [text/README.md](text/README.md) for notebook and dependency setup.

---

### Integration and Dashboard (`integration/`)

- **Input:** All four modality CSV files
- **Tools:** pandas, Jupyter, Streamlit, Plotly
- **Outputs:**
  - `final_integrated_incidents.csv` — unified master dataset
  - `app.py` — Incident Intelligence Dashboard with Overview and Drill Through tabs

See [integration/README.md](integration/README.md) for dashboard features and integration notebook details.

---

## Final Integrated Dataset

**File:** [integration/final_integrated_incidents.csv](integration/final_integrated_incidents.csv)

This is the main deliverable: a single merged CSV containing structured information extracted from all implemented modalities.

### Schema

| Column | Description | Example |
|---|---|---|
| `Incident_ID` | Modality-prefixed unique ID | `AUD-001`, `DOC-001`, `IMG-001`, `TXT-001` |
| `Source` | Data modality | Audio, PDF, Image, Text |
| `Event` | Extracted incident or event type | Homicide, Forest Fire, Assault / Violence |
| `Location` | Location when available | `the Hudson River`, `Unknown`, `N/A` |
| `Time` | Timestamp when available (best-effort) | `May 26, 2015`, `Unknown` |
| `Severity` | Final severity classification | Low, Medium, High |
| `Original_ID` | ID in the original modality CSV | `AUD-001`, `RPT_001`, image filename, `TXT_1` |
| `Score` | Normalized confidence or urgency score (0–10) | `6.7`, `5.0`, `9.0` |

### Example rows

| Incident_ID | Source | Event | Location | Time | Severity | Score |
|---|---|---|---|---|---|---|
| AUD-001 | Audio | Homicide | Unknown | Unknown | Medium | 6.7 |
| DOC-001 | PDF | Vendor Capability Statement | Bentonville, Arkansas | May 26, 2015 | Medium | 5.0 |
| IMG-001 | Image | Forest Fire | Unknown | Unknown | High | 9.11 |
| TXT-001 | Text | Assault / Violence | Unknown | Unknown | High | 9.0 |

### Severity rules

| Score range | Severity |
|---|---|
| 0 to less than 3 | Low |
| 3 to less than 7 | Medium |
| 7 to 10 | High |

### How scores are calculated per modality

| Modality | Score source |
|---|---|
| Audio | `Urgency_Score x 10` |
| PDF | Fixed score of `5` for administrative documents |
| Image | `Scene_Decision_Confidence x 10` |
| Text | Topic-based mapping (e.g. Assault/Violence = 9, Public Disturbance = 4) |

---

## Run the Dashboard

The dashboard is the recommended way to explore the final dataset and demonstrate the project.

### Step 1 — Clone the repository

```bash
git clone https://github.com/Arch1tect0/VibeMultiModal.git
cd VibeMultiModal/integration
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Start the app

```bash
streamlit run app.py
```

If the `streamlit` command is not found:

```bash
python -m streamlit run app.py
```

### Step 4 — Open in browser

Go to: **http://localhost:8501**

### Dashboard features

**Overview tab**

- KPI cards: total incidents, high severity count, source count, average score
- Charts: incidents by source, incidents by severity
- High severity review table
- Full filtered incident table
- Download filtered results as CSV

**Incident Drill Through tab**

- Search incidents by ID, source, event, location, or severity
- Select an incident from a filtered dropdown
- View the integrated incident summary
- Inspect the original source record (transcript, PDF text, image detections, article text)

The dashboard automatically reads source files from sibling folders:

| Incident prefix | Source file |
|---|---|
| `AUD-` | `../audio/audio_output.csv` |
| `DOC-` | `../pdf/pdf_output.csv` |
| `IMG-` | `../image/outputs/results.csv` |
| `TXT-` | `../text/Text.csv` |

---

## Run Individual Modality Pipelines

Use these when you need to regenerate a single modality's output. Full details are in each folder's README.

### Audio

```bash
cd audio
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Open and run `wav2vec2.ipynb` (Kaggle recommended with GPU).

### PDF

```bash
cd pdf
pip install -r requirements.txt
brew install tesseract poppler   # macOS only
```

Open and run `leso_extraction_pipeline.ipynb`.

### Image

```bash
cd image
pip install -r requirements.txt
python -m src.analyze_images --image-dir path/to/images --csv-path outputs/results.csv
```

### Text

```bash
cd text
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Open and run `Text_Finaloutput.ipynb`.

### Integration (rebuild merged dataset)

```bash
cd integration
pip install pandas jupyter
jupyter notebook final_integration.ipynb
```

Run all cells to regenerate `final_integrated_incidents.csv`.

---

## Assignment Deliverables

| # | Deliverable | Location in this repo |
|---|---|---|
| 1 | AI Pipeline Architecture Diagram | [integration/multimodal_pipeline_architecture.pdf](integration/multimodal_pipeline_architecture.pdf) |
| 2 | Code Repository (organized by modality) | This repository |
| 3 | Final Structured Dataset | [integration/final_integrated_incidents.csv](integration/final_integrated_incidents.csv) |
| 4 | Project Report | Submitted separately on Canvas (PDF) |
| 5 | Demonstration video | Linked in Project Report (Google Drive) |
| 6 | Individual Contributions table | Included in Project Report |

---

## Technologies Used

| Layer | Primary tools |
|---|---|
| Audio | OpenAI Whisper, spaCy, pandas |
| PDF | pdf2image, pytesseract, regex extraction |
| Image | YOLOv8, CLIP, ViT, OpenCV, Tesseract |
| Text | spaCy, HuggingFace Transformers |
| Integration | pandas, Jupyter Notebook |
| Dashboard | Streamlit, Plotly |

---

## Requirements

- Python 3.9 or newer
- Jupyter Notebook (for `.ipynb` files)
- Streamlit, pandas, plotly (dashboard — see `integration/requirements.txt`)
- Modality-specific dependencies listed in each folder's `requirements.txt` or README
